import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EDGE import EDGE
from dataset.dance_dataset import AISTPPDataset
from eval.g1_metrics import run_g1_motion_evaluation
from model.diffusion import cond_batch_size, move_cond_to_device, slice_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--feature_type", default="jukebox", type=str)
    parser.add_argument("--feature_fusion", default="linear", type=str)
    parser.add_argument("--data_path", default="data/g1_aistpp", type=str)
    parser.add_argument("--processed_data_dir", default="data/g1_dataset_backups", type=str)
    parser.add_argument("--render_dir", default="eval/g1/renders", type=str)
    parser.add_argument("--motion_save_dir", default="eval/g1/motions", type=str)
    parser.add_argument("--metrics_path", default="eval/g1/metrics.json", type=str)
    parser.add_argument("--g1_table_path", default="eval/g1/g1_table.json", type=str)
    parser.add_argument("--motion_audit_path", default="eval/g1/motion_audit.json", type=str)
    parser.add_argument("--paper_report_path", default="eval/g1/paper_report.md", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--max_eval_clips", default=0, type=int)
    parser.add_argument("--diagnostic_count", default=8, type=int)
    parser.add_argument("--enable_fk_metrics", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument(
        "--g1_root_quat_order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
    )
    parser.add_argument(
        "--g1_render_backend",
        choices=("mujoco", "stick"),
        default="mujoco",
    )
    parser.add_argument("--g1_render_width", default=960, type=int)
    parser.add_argument("--g1_render_height", default=720, type=int)
    parser.add_argument("--g1_mujoco_gl", default="egl", type=str)
    parser.add_argument(
        "--motion_energy_condition_variant",
        choices=(
            "auto",
            "oracle_gt",
            "oracle_controls",
            "pred_energy",
            "pred_controls",
            "flat_energy",
            "flat_intensity",
            "zero_energy",
            "zero_beatness",
            "zero_control",
            "zero_all_controls",
        ),
        default="auto",
        help=(
            "Condition variant for structured motion-control eval. auto uses "
            "predicted controls for wav2clip_motion_intensity_beatness and "
            "oracle GT energy for legacy wav2clip_motion_energy_beat."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_motion_dir(motion_dir):
    motion_dir = Path(motion_dir)
    if motion_dir.exists():
        shutil.rmtree(motion_dir)
    motion_dir.mkdir(parents=True, exist_ok=True)
    return motion_dir


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


def clone_structured_condition_with_control(cond, **updates):
    control = dict(cond["control"])
    for key, value in updates.items():
        if value is not None:
            control[key] = value
    return {
        "semantic": dict(cond["semantic"]),
        "control": control,
    }


def apply_motion_energy_condition_variant(model, cond, variant):
    if not isinstance(cond, dict) or "control" not in cond:
        if variant not in ("auto", "oracle_gt", "oracle_controls"):
            raise ValueError(f"{variant} requires structured motion-energy condition")
        return cond
    control = cond["control"]
    if variant == "auto":
        variant = "pred_controls" if "motion_intensity" in control else "oracle_gt"
    if variant in ("oracle_gt", "oracle_controls"):
        return cond
    gaussian_beat = control["gaussian_beat"]
    beat_energy = control.get("beat_energy_envelope")
    motion_intensity = control.get("motion_intensity")
    motion_beatness = control.get("motion_beatness")
    if variant == "pred_controls":
        if hasattr(model.diffusion.model, "predict_controls"):
            predictions = model.diffusion.model.predict_controls(cond)
            return clone_structured_condition_with_control(
                cond,
                motion_intensity=predictions["motion_intensity"].detach(),
                motion_beatness=predictions["motion_beatness"].detach(),
            )
        if hasattr(model.diffusion.model, "predict_energy") and beat_energy is not None:
            pred_energy = model.diffusion.model.predict_energy(cond).detach()
            return clone_structured_condition_with_control(cond, beat_energy_envelope=pred_energy)
        raise ValueError("pred_controls requires predict_controls or legacy predict_energy")
    if variant == "pred_energy":
        if not hasattr(model.diffusion.model, "predict_energy"):
            raise ValueError("pred_energy requires a model with predict_energy")
        pred_energy = model.diffusion.model.predict_energy(cond).detach()
        return clone_structured_condition_with_control(cond, beat_energy_envelope=pred_energy)
    if variant == "flat_energy":
        if beat_energy is None:
            raise ValueError("flat_energy requires beat_energy_envelope")
        flat_energy = beat_energy.mean(dim=1, keepdim=True).expand_as(beat_energy).clone()
        return clone_structured_condition_with_control(cond, beat_energy_envelope=flat_energy)
    if variant == "flat_intensity":
        if motion_intensity is None:
            raise ValueError("flat_intensity requires motion_intensity")
        flat_intensity = motion_intensity.mean(dim=1, keepdim=True).expand_as(motion_intensity).clone()
        return clone_structured_condition_with_control(cond, motion_intensity=flat_intensity)
    if variant == "zero_energy":
        if beat_energy is None:
            raise ValueError("zero_energy requires beat_energy_envelope")
        return clone_structured_condition_with_control(
            cond,
            beat_energy_envelope=torch.zeros_like(beat_energy),
        )
    if variant == "zero_beatness":
        if motion_beatness is None:
            raise ValueError("zero_beatness requires motion_beatness")
        return clone_structured_condition_with_control(
            cond,
            motion_beatness=torch.zeros_like(motion_beatness),
        )
    if variant == "zero_control":
        return clone_structured_condition_with_control(
            cond,
            gaussian_beat=torch.zeros_like(gaussian_beat),
            beat_energy_envelope=torch.zeros_like(beat_energy) if beat_energy is not None else None,
            motion_intensity=torch.zeros_like(motion_intensity) if motion_intensity is not None else None,
            motion_beatness=torch.zeros_like(motion_beatness) if motion_beatness is not None else None,
        )
    if variant == "zero_all_controls":
        return clone_structured_condition_with_control(
            cond,
            gaussian_beat=torch.zeros_like(gaussian_beat),
            beat_energy_envelope=torch.zeros_like(beat_energy) if beat_energy is not None else None,
            motion_intensity=torch.zeros_like(motion_intensity) if motion_intensity is not None else None,
            motion_beatness=torch.zeros_like(motion_beatness) if motion_beatness is not None else None,
        )
    raise ValueError(f"Unsupported motion_energy_condition_variant: {variant}")


def render_g1_dataset_batch(
    model,
    batch,
    render_dir,
    motion_dir,
    label="g1_eval",
    motion_energy_condition_variant="oracle_gt",
):
    _, cond, _, wavname = batch
    render_count = cond_batch_size(cond)
    shape = (render_count, model.horizon, model.repr_dim)
    cond = move_cond_to_device(cond, model.accelerator.device)
    cond = apply_motion_energy_condition_variant(
        model,
        cond,
        motion_energy_condition_variant,
    )
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


def run_g1_dataset_evaluation(args):
    set_seed(args.seed)
    motion_dir = clear_motion_dir(args.motion_save_dir)
    Path(args.render_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)

    model = EDGE(
        args.feature_type,
        args.checkpoint,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        lambda_beat=0.0,
        motion_format="g1",
        feature_fusion=args.feature_fusion,
    )
    model.eval()

    dataset = AISTPPDataset(
        data_path=args.data_path,
        backup_path=args.processed_data_dir,
        train=False,
        feature_type=args.feature_type,
        normalizer=model.normalizer,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        motion_format="g1",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    progress_total = args.max_eval_clips if args.max_eval_clips else len(dataset)
    with tqdm(total=progress_total, desc="G1 dataset eval", unit="clip") as progress:
        for batch in iter_limited_batches(loader, max_eval_clips=args.max_eval_clips):
            render_g1_dataset_batch(
                model,
                batch,
                args.render_dir,
                motion_dir,
                motion_energy_condition_variant=args.motion_energy_condition_variant,
            )
            progress.update(cond_batch_size(batch[1]))

    metrics = run_g1_motion_evaluation(
        motion_path=motion_dir,
        reference_motion_path=Path(args.data_path) / "test" / "motions_sliced",
        metrics_path=args.metrics_path,
        g1_table_path=args.g1_table_path,
        motion_audit_path=args.motion_audit_path,
        paper_report_path=args.paper_report_path,
        render_dir=args.render_dir,
        diagnostic_count=args.diagnostic_count,
        checkpoint=args.checkpoint,
        feature_type=args.feature_type,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        seed=args.seed,
        enable_fk_metrics=args.enable_fk_metrics,
        fk_model_path=args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
    )
    metrics["motion_energy_condition_variant"] = args.motion_energy_condition_variant
    with open(args.metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


if __name__ == "__main__":
    run_g1_dataset_evaluation(parse_args())
