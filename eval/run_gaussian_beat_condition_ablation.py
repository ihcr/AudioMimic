import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EDGE import EDGE
from dataset.dance_dataset import AISTPPDataset
from eval.g1_metrics import run_g1_motion_evaluation
from eval.write_g1_metric_comparison import write_comparison
from model.diffusion import cond_batch_size, move_cond_to_device, slice_cond


DEFAULT_VARIANTS = (
    "real",
    "shift_p10",
    "random",
    "constant",
    "no_beat_uncond",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a GaussianBeat-only checkpoint uses its 1-D beat "
            "condition by perturbing that condition at inference time."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats")
    parser.add_argument(
        "--processed_data_dir",
        default="data/finedance_g1_gaussian_beat_dataset_backups",
    )
    parser.add_argument(
        "--output_root",
        default="eval/EXP-20260522-gaussian-beat-condition-ablation",
    )
    parser.add_argument(
        "--real_motion_dir",
        default="eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/motions",
        help="Optional existing real-condition motion directory to re-score.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=DEFAULT_VARIANTS,
        default=[],
        help="Run a subset of variants. Defaults to all variants.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_eval_clips", type=int, default=0)
    parser.add_argument("--shift_frames", type=int, default=10)
    parser.add_argument("--feature_fusion", default="linear")
    parser.add_argument("--diagnostic_count", default=8, type=int)
    parser.add_argument("--enable_fk_metrics", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(payload):
    if isinstance(payload, dict):
        return {key: json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [json_safe(value) for value in payload]
    if isinstance(payload, tuple):
        return [json_safe(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, Path):
        return str(payload)
    return payload


def clear_motion_dir(motion_dir):
    motion_dir = Path(motion_dir)
    if motion_dir.exists():
        shutil.rmtree(motion_dir)
    motion_dir.mkdir(parents=True, exist_ok=True)
    return motion_dir


def shift_condition(cond, shift_frames):
    if shift_frames < 0:
        raise ValueError("shift_frames must be non-negative")
    if shift_frames == 0:
        return cond.clone()
    if cond.ndim != 2:
        raise ValueError(f"GaussianBeat condition expected [T, 1], got {tuple(cond.shape)}")
    if shift_frames >= cond.shape[0]:
        return cond[:1].expand_as(cond).clone()
    shifted = cond.clone()
    shifted[shift_frames:] = cond[:-shift_frames]
    shifted[:shift_frames] = cond[:1]
    return shifted


def constant_condition_like(cond, constant_value):
    return torch.full_like(cond, float(constant_value))


def compute_feature_mean(feature_dir):
    feature_paths = sorted(Path(feature_dir).glob("*.npy"))
    if not feature_paths:
        raise FileNotFoundError(f"No GaussianBeat feature files found in {feature_dir}")
    total = 0.0
    count = 0
    for path in tqdm(feature_paths, desc="Compute constant beat", unit="file"):
        features = np.load(path)
        total += float(features.sum())
        count += int(features.size)
    return total / max(count, 1)


def build_random_feature_paths(feature_paths, seed):
    if not feature_paths:
        raise ValueError("feature_paths must not be empty")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(feature_paths))
    rng.shuffle(indices)
    if len(indices) > 1:
        fixed_points = np.flatnonzero(indices == np.arange(len(indices)))
        for index in fixed_points:
            swap_with = (index + 1) % len(indices)
            indices[index], indices[swap_with] = indices[swap_with], indices[index]
    return [feature_paths[index] for index in indices]


class GaussianBeatAblationDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        variant,
        shift_frames=10,
        constant_value=0.0,
        random_feature_paths=None,
    ):
        self.base_dataset = base_dataset
        self.variant = variant
        self.shift_frames = int(shift_frames)
        self.constant_value = float(constant_value)
        self.random_feature_paths = list(random_feature_paths or [])
        if self.variant == "random" and len(self.random_feature_paths) != len(self.base_dataset):
            raise ValueError("random_feature_paths must match dataset length")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        pose, cond, filename, wavname = self.base_dataset[idx]
        if self.variant in {"real", "no_beat_uncond"}:
            return pose, cond, filename, wavname
        if self.variant == "shift_p10":
            cond = shift_condition(cond, self.shift_frames)
            return pose, cond, filename, wavname
        if self.variant == "constant":
            cond = constant_condition_like(cond, self.constant_value)
            return pose, cond, filename, wavname
        if self.variant == "random":
            cond = torch.from_numpy(
                np.asarray(np.load(self.random_feature_paths[idx]), dtype=np.float32)
            )
            return pose, cond, filename, wavname
        raise ValueError(f"Unsupported variant: {self.variant}")


def slice_batch(batch, stop):
    pose, cond, filename, wavname = batch
    return (
        pose[:stop],
        slice_cond(cond, slice(None, stop)),
        filename[:stop],
        wavname[:stop],
    )


def iter_limited_batches(loader, max_eval_clips=0):
    emitted = 0
    for batch in loader:
        if max_eval_clips and emitted >= max_eval_clips:
            break
        batch_size = cond_batch_size(batch[1])
        if max_eval_clips and emitted + batch_size > max_eval_clips:
            batch = slice_batch(batch, max_eval_clips - emitted)
            batch_size = cond_batch_size(batch[1])
        emitted += batch_size
        yield batch


def render_g1_dataset_batch(model, batch, render_dir, motion_dir, label):
    _, cond, _, wavname = batch
    render_count = cond_batch_size(cond)
    shape = (render_count, model.horizon, model.repr_dim)
    cond = move_cond_to_device(cond, model.accelerator.device)
    model.diffusion.render_sample(
        shape,
        slice_cond(cond, slice(None, render_count)),
        model.normalizer,
        label,
        render_dir,
        name=wavname[:render_count],
        sound=True,
        mode="normal",
        fk_out=str(motion_dir),
        render=False,
    )


def evaluate_motion_dir(args, variant, motion_dir, variant_dir):
    variant_dir = Path(variant_dir)
    render_dir = variant_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    return run_g1_motion_evaluation(
        motion_path=motion_dir,
        reference_motion_path=Path(args.data_path) / "test" / "motions_sliced",
        metrics_path=variant_dir / "metrics.json",
        g1_table_path=variant_dir / "g1_table.json",
        motion_audit_path=variant_dir / "motion_audit.json",
        paper_report_path=variant_dir / "paper_report.md",
        render_dir=render_dir,
        diagnostic_count=args.diagnostic_count,
        checkpoint=args.checkpoint,
        feature_type="gaussian_beat",
        use_beats=False,
        beat_rep="none",
        seed=args.seed,
        enable_fk_metrics=args.enable_fk_metrics,
        fk_model_path=args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
    )


def generate_variant(model, dataset, args, variant, variant_dir, guidance_weight):
    set_seed(args.seed)
    motion_dir = clear_motion_dir(Path(variant_dir) / "motions")
    render_dir = Path(variant_dir) / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    original_guidance = model.diffusion.guidance_weight
    model.diffusion.guidance_weight = float(guidance_weight)
    try:
        progress_total = args.max_eval_clips if args.max_eval_clips else len(dataset)
        with tqdm(
            total=progress_total,
            desc=f"Generate {variant}",
            unit="clip",
        ) as progress:
            for batch in iter_limited_batches(loader, max_eval_clips=args.max_eval_clips):
                render_g1_dataset_batch(
                    model,
                    batch,
                    render_dir,
                    motion_dir,
                    label=variant,
                )
                progress.update(cond_batch_size(batch[1]))
    finally:
        model.diffusion.guidance_weight = original_guidance
    return motion_dir


def load_base_dataset(args, model):
    return AISTPPDataset(
        data_path=args.data_path,
        backup_path=args.processed_data_dir,
        train=False,
        feature_type="gaussian_beat",
        normalizer=model.normalizer,
        use_beats=False,
        beat_rep="distance",
        motion_format="g1",
    )


def count_motion_files(path):
    return len(list(Path(path).glob("*.pkl")))


def run_ablation(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = args.variant or list(DEFAULT_VARIANTS)
    set_seed(args.seed)

    model = EDGE(
        "gaussian_beat",
        args.checkpoint,
        use_beats=False,
        lambda_beat=0.0,
        motion_format="g1",
        feature_fusion=args.feature_fusion,
    )
    model.eval()
    base_dataset = load_base_dataset(args, model)

    train_feature_dir = Path(args.data_path) / "train" / "gaussian_beat_feats"
    constant_value = compute_feature_mean(train_feature_dir)
    feature_paths = [Path(path) for path in base_dataset.data["filenames"]]
    random_feature_paths = build_random_feature_paths(feature_paths, args.seed)

    manifest = {
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "processed_data_dir": args.processed_data_dir,
        "output_root": str(output_root),
        "variants": variants,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_eval_clips": args.max_eval_clips,
        "shift_frames": args.shift_frames,
        "constant_value": constant_value,
        "random_permutation_seed": args.seed,
        "default_guidance_weight": float(model.diffusion.guidance_weight),
        "results": {},
    }

    comparison_entries = []
    for variant in variants:
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = variant_dir / "metrics.json"

        if variant == "real" and args.real_motion_dir and not args.max_eval_clips:
            real_motion_dir = Path(args.real_motion_dir)
            expected_count = args.max_eval_clips or len(base_dataset)
            if real_motion_dir.is_dir() and count_motion_files(real_motion_dir) >= expected_count:
                print(f"Re-scoring existing real motions: {real_motion_dir}")
                motion_dir = real_motion_dir
                generated = False
            else:
                print("Existing real motions are missing or incomplete; regenerating real variant.")
                ablated_dataset = GaussianBeatAblationDataset(
                    base_dataset,
                    variant,
                    shift_frames=args.shift_frames,
                    constant_value=constant_value,
                    random_feature_paths=random_feature_paths,
                )
                motion_dir = generate_variant(
                    model,
                    ablated_dataset,
                    args,
                    variant,
                    variant_dir,
                    guidance_weight=model.diffusion.guidance_weight,
                )
                generated = True
        else:
            ablated_dataset = GaussianBeatAblationDataset(
                base_dataset,
                variant,
                shift_frames=args.shift_frames,
                constant_value=constant_value,
                random_feature_paths=random_feature_paths,
            )
            guidance_weight = 0.0 if variant == "no_beat_uncond" else model.diffusion.guidance_weight
            motion_dir = generate_variant(
                model,
                ablated_dataset,
                args,
                variant,
                variant_dir,
                guidance_weight=guidance_weight,
            )
            generated = True

        metrics = evaluate_motion_dir(args, variant, motion_dir, variant_dir)
        comparison_entries.append((variant, metrics_path))
        manifest["results"][variant] = {
            "motion_dir": str(motion_dir),
            "metrics_path": str(metrics_path),
            "generated": generated,
            "num_motion_files": metrics.get("num_motion_files"),
            "guidance_weight": 0.0
            if variant == "no_beat_uncond"
            else float(model.diffusion.guidance_weight),
        }

    rows = write_comparison(
        comparison_entries,
        json_path=output_root / "comparison_g1_metrics.json",
        markdown_path=output_root / "comparison_g1_metrics.md",
    )
    manifest["comparison_json"] = str(output_root / "comparison_g1_metrics.json")
    manifest["comparison_markdown"] = str(output_root / "comparison_g1_metrics.md")
    manifest["comparison_rows"] = rows
    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, sort_keys=True)
    return rows


if __name__ == "__main__":
    run_ablation(parse_args())
