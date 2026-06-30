import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.g1_latent_beat_dataset import G1LatentBeatDataset, G1MusicControlLatentDataset
from dataset.g1_motion_prior_dataset import G1MotionPriorNormalizer
from dataset.motion_representation import G1_YAW_DELTA_MOTION_FORMAT
from eval.g1_metrics import run_g1_motion_evaluation
from eval.run_g1_motion_prior_eval import build_model_from_checkpoint, save_decoded_g1_motion
from model.g1_latent_diffusion import (
    G1Beat8DLatentDenoiser,
    G1LatentDiffusion,
    G1MusicControlLatentDenoiser,
)


SUMMARY_METRIC_KEYS = (
    "G1BeatF1",
    "G1FKBAS",
    "G1FKRoboPerformBAS",
    "G1Dist",
    "G1Div",
    "G1NoNearSupportRate",
    "G1FootHighLiftRate",
    "G1GroundPenetration",
    "G1FootSliding",
    "G1WristJerkMean",
    "G1FootJerkMean",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--prior_checkpoint", default="", type=str)
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats", type=str)
    parser.add_argument(
        "--audio_data_path",
        default="",
        type=str,
        help=(
            "Dataset root that contains <split>/wavs_sliced/*.wav for beat metrics. "
            "Defaults to the AIST++ sibling of data_path."
        ),
    )
    parser.add_argument(
        "--motion_prior_processed_data_dir",
        default="data/finedance_g1_v6b_motion_prior_dataset_backups",
        type=str,
    )
    parser.add_argument(
        "--latent_processed_data_dir",
        default="data/finedance_g1_v6bc_music_control_latent_dataset_backups",
        type=str,
    )
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--motion_format", default=G1_YAW_DELTA_MOTION_FORMAT, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_eval_clips", default=0, type=int)
    parser.add_argument("--cache_limit_per_split", default=0, type=int)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--sampling_steps", default=50, type=int)
    parser.add_argument("--guidance_weight", default=1.0, type=float)
    parser.add_argument("--diagnostic_count", default=8, type=int)
    parser.add_argument("--enable_fk_metrics", action="store_true")
    parser.add_argument(
        "--metric_workers",
        default=8,
        type=int,
        help="Parallel CPU workers for per-motion beat/FK metric computation.",
    )
    parser.add_argument(
        "--resume_existing_metrics",
        action="store_true",
        help="Reuse completed <output_dir>/<variant>/metrics.json files for the same checkpoint.",
    )
    parser.add_argument(
        "--variants",
        default="auto",
        type=str,
    )
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    return parser.parse_args()


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_existing_variant_metrics(metrics_path, checkpoint_path):
    metrics_path = Path(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("checkpoint") != checkpoint_path:
        raise ValueError(
            f"Refusing to reuse {metrics_path}: checkpoint mismatch "
            f"{metrics.get('checkpoint')} != {checkpoint_path}"
        )
    return metrics


def _build_diffusion_from_checkpoint(checkpoint):
    config = checkpoint["config"]
    if config.get("model_type") == "g1_music_control_latent":
        denoiser = G1MusicControlLatentDenoiser(
            latent_dim=int(config.get("latent_dim", 128)),
            control_dim=8,
            semantic_dim=512,
            latent_frames=int(config.get("latent_frames", 75)),
            control_frames=int(config.get("beat_frames", 150)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            num_heads=int(config.get("num_heads", 4)),
            ff_size=int(config.get("ff_size", 1024)),
            dropout=float(config.get("dropout", 0.1)),
            use_wav2clip_semantic=bool(config.get("use_wav2clip_semantic", False)),
        )
    else:
        denoiser = G1Beat8DLatentDenoiser(
            latent_dim=int(config.get("latent_dim", 128)),
            beat_dim=8,
            latent_frames=int(config.get("latent_frames", 75)),
            beat_frames=int(config.get("beat_frames", 150)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            num_heads=int(config.get("num_heads", 4)),
            ff_size=int(config.get("ff_size", 1024)),
            dropout=float(config.get("dropout", 0.1)),
        )
    diffusion = G1LatentDiffusion(
        denoiser,
        timesteps=int(config.get("diffusion_steps", 1000)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        cond_drop_prob=float(config.get("cond_drop_prob", 0.1)),
    )
    diffusion.load_state_dict(checkpoint["diffusion"])
    return diffusion


def _random_condition(features):
    if features is None:
        return None
    if features.shape[0] > 1:
        return features[torch.randperm(features.shape[0], device=features.device)]
    return torch.roll(features, shifts=37, dims=1)


def _variant_beats(beat_features, variant):
    legacy_map = {
        "real_beat8d": "real",
        "shifted_beat8d": "shifted_control",
        "random_beat8d": "random_control",
        "zero_beat8d": "zero_control",
    }
    variant = legacy_map.get(variant, variant)
    control, _ = variant_conditions(beat_features, None, variant, use_wav2clip_semantic=False)
    return control


def variant_conditions(control_features, semantic_features, variant, use_wav2clip_semantic):
    if variant == "real":
        return control_features, semantic_features
    if variant == "shifted_control":
        return torch.roll(control_features, shifts=20, dims=1), semantic_features
    if variant == "random_control":
        return _random_condition(control_features), semantic_features
    if variant == "zero_control":
        return torch.zeros_like(control_features), semantic_features
    if variant == "zero_all":
        semantic = torch.zeros_like(semantic_features) if semantic_features is not None else None
        return torch.zeros_like(control_features), semantic
    if variant in ("random_semantic", "random_semantic_real_control"):
        if not use_wav2clip_semantic or semantic_features is None:
            raise ValueError(f"{variant} requires --use_wav2clip_semantic checkpoint")
        return control_features, _random_condition(semantic_features)
    if variant == "zero_semantic":
        if not use_wav2clip_semantic or semantic_features is None:
            raise ValueError("zero_semantic requires --use_wav2clip_semantic checkpoint")
        return control_features, torch.zeros_like(semantic_features)
    if variant == "real_semantic_random_control":
        if not use_wav2clip_semantic or semantic_features is None:
            raise ValueError("real_semantic_random_control requires --use_wav2clip_semantic checkpoint")
        return _random_condition(control_features), semantic_features
    raise ValueError(f"Unsupported eval variant: {variant}")


def default_variants(use_wav2clip_semantic):
    variants = ["real", "shifted_control", "random_control", "zero_control", "zero_all"]
    if use_wav2clip_semantic:
        variants.extend(
            [
                "random_semantic",
                "zero_semantic",
                "random_semantic_real_control",
                "real_semantic_random_control",
            ]
        )
    return variants


def _default_audio_data_path(data_path):
    data_path = Path(data_path)
    if data_path.name == "finedance_g1_fkbeats":
        return data_path.parent / "finedance_aistpp"
    return data_path


def _audio_path_for_stem(args, stem):
    audio_root = Path(args.audio_data_path) if args.audio_data_path else _default_audio_data_path(args.data_path)
    audio_path = audio_root / args.split / "wavs_sliced" / f"{stem}.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"missing audio slice for latent eval metrics: {audio_path}; "
            "set --audio_data_path to the dataset root containing <split>/wavs_sliced"
        )
    return str(audio_path)


def _set_pickle_audio_path(path, audio_path):
    import pickle

    path = Path(path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload["audio_path"] = audio_path
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle)
    tmp_path.replace(path)


@torch.inference_mode()
def _generate_variant(
    variant,
    diffusion,
    prior_model,
    prior_motion_normalizer,
    latent_normalizer,
    loader,
    device,
    output_dir,
    args,
    use_wav2clip_semantic=False,
):
    variant_dir = Path(output_dir) / variant
    motion_dir = variant_dir / "motions"
    target_dir = variant_dir / "targets"
    motion_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    mean, std = prior_motion_normalizer.tensors(device=device)
    records = []
    generated = 0
    for batch in tqdm(loader, desc=f"Generate {variant}", unit="batch"):
        if "control_features" in batch:
            control = batch["control_features"].to(device, non_blocking=True).float()
            semantic = (
                batch["semantic_features"].to(device, non_blocking=True).float()
                if "semantic_features" in batch
                else None
            )
        else:
            control = batch["beat_features"].to(device, non_blocking=True).float()
            semantic = None
        control, semantic = variant_conditions(
            control,
            semantic,
            variant,
            use_wav2clip_semantic=use_wav2clip_semantic,
        )
        if args.max_eval_clips and generated >= args.max_eval_clips:
            break
        batch_size = int(control.shape[0])
        if args.max_eval_clips and generated + batch_size > args.max_eval_clips:
            keep = int(args.max_eval_clips - generated)
            control = control[:keep]
            semantic = semantic[:keep] if semantic is not None else None
            batch = {
                key: value[:keep] if torch.is_tensor(value) else value[:keep]
                for key, value in batch.items()
            }
            batch_size = keep
        latent = diffusion.ddim_sample(
            control,
            shape=(batch_size, 75, 128),
            semantic_features=semantic,
            sampling_steps=args.sampling_steps,
            guidance_weight=args.guidance_weight,
        )
        latent = latent_normalizer.unnormalize_tensor(latent)
        motion_norm, _ = prior_model.decode(latent)
        motion_raw = motion_norm * std + mean
        for index in range(batch_size):
            stem = batch["stem"][index]
            source_path = batch["source_path"][index]
            motion_path = motion_dir / f"{stem}.pkl"
            target_path = target_dir / f"{stem}.pkl"
            audio_path = _audio_path_for_stem(args, stem)
            save_decoded_g1_motion(
                motion_raw[index].detach().cpu(),
                source_path,
                motion_path,
                args.motion_format,
            )
            _set_pickle_audio_path(motion_path, audio_path)
            shutil.copy2(source_path, target_path)
            _set_pickle_audio_path(target_path, audio_path)
            records.append(
                {
                    "stem": stem,
                    "source_path": source_path,
                    "audio_path": audio_path,
                    "motion_path": str(motion_path),
                    "target_path": str(target_path),
                }
            )
        generated += batch_size

    metrics = run_g1_motion_evaluation(
        motion_path=motion_dir,
        reference_motion_path=target_dir,
        metrics_path=variant_dir / "metrics.json",
        g1_table_path=variant_dir / "g1_table.json",
        motion_audit_path=variant_dir / "motion_audit.json",
        paper_report_path=variant_dir / "paper_report.md",
        render_dir=variant_dir / "diagnostics",
        diagnostic_count=args.diagnostic_count,
        checkpoint=args.checkpoint,
        feature_type=f"g1_latent_diffusion_{variant}",
        use_beats=True,
        beat_rep="v6bc_music_control",
        seed=1234,
        sample_limit=None,
        enable_fk_metrics=args.enable_fk_metrics,
        fk_model_path=args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
        failure_panel_path=variant_dir / "failure_panel.json",
        metric_workers=args.metric_workers,
    )
    _write_json({"records": records}, variant_dir / "manifest.json")
    return metrics


def run_g1_latent_diffusion_eval(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    prior_checkpoint = args.prior_checkpoint or config["prior_checkpoint"]
    use_music_control = config.get("model_type") == "g1_music_control_latent"
    use_wav2clip_semantic = bool(config.get("use_wav2clip_semantic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_kwargs = dict(
        data_path=args.data_path,
        motion_prior_processed_data_dir=args.motion_prior_processed_data_dir,
        latent_processed_data_dir=args.latent_processed_data_dir,
        prior_checkpoint=prior_checkpoint,
        split=args.split,
        motion_format=args.motion_format,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        cache_batch_size=args.batch_size,
        cache_device=device.type,
        cache_limit_per_split=args.cache_limit_per_split,
        rebuild_cache=args.rebuild_cache,
        data_len=args.max_eval_clips,
    )
    if use_music_control:
        dataset = G1MusicControlLatentDataset(
            use_wav2clip_semantic=use_wav2clip_semantic,
            **dataset_kwargs,
        )
    else:
        dataset = G1LatentBeatDataset(**dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    diffusion = _build_diffusion_from_checkpoint(checkpoint).to(device)
    diffusion.eval()
    prior_checkpoint_payload = torch.load(prior_checkpoint, map_location=device, weights_only=False)
    prior_model = build_model_from_checkpoint(prior_checkpoint_payload, args.motion_format).to(device)
    prior_model.eval()
    prior_motion_normalizer = G1MotionPriorNormalizer.from_state_dict(
        prior_checkpoint_payload["normalizer"]
    )
    latent_normalizer = dataset.normalizer

    if args.variants == "auto":
        variants = default_variants(use_wav2clip_semantic)
    else:
        legacy_map = {
            "real_beat8d": "real",
            "shifted_beat8d": "shifted_control",
            "random_beat8d": "random_control",
            "zero_beat8d": "zero_control",
        }
        variants = [
            legacy_map.get(item.strip(), item.strip())
            for item in args.variants.split(",")
            if item.strip()
        ]
    output_dir = Path(args.output_dir)
    summary = {
        "checkpoint": args.checkpoint,
        "prior_checkpoint": str(prior_checkpoint),
        "output_dir": str(output_dir),
        "num_eval_clips": len(dataset),
        "use_wav2clip_semantic": use_wav2clip_semantic,
        "sampling_steps": int(args.sampling_steps),
        "guidance_weight": float(args.guidance_weight),
        "variants": {},
    }
    for variant in variants:
        metrics_path = output_dir / variant / "metrics.json"
        if args.resume_existing_metrics and metrics_path.is_file():
            print(f"Reusing existing metrics for {variant}: {metrics_path}", flush=True)
            metrics = _load_existing_variant_metrics(metrics_path, args.checkpoint)
        else:
            metrics = _generate_variant(
                variant,
                diffusion,
                prior_model,
                prior_motion_normalizer,
                latent_normalizer,
                loader,
                device,
                output_dir,
                args,
                use_wav2clip_semantic=use_wav2clip_semantic,
            )
        summary["variants"][variant] = {
            "metrics_path": str(metrics_path),
            **{key: metrics.get(key) for key in SUMMARY_METRIC_KEYS},
        }
    _write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    run_g1_latent_diffusion_eval(parse_args())


if __name__ == "__main__":
    main()
