import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset.g1_latent_beat_dataset import G1LatentNormalizer
from dataset.g1_motion_prior_dataset import G1MotionPriorNormalizer
from dataset.motion_representation import G1_YAW_DELTA_MOTION_FORMAT, decode_g1_motion
from eval.g1_visualization import render_g1_motion
from eval.render_g1_checkpoint_comparison import (
    _extract_feature,
    _load_cached_features,
    _prepare_audio_slices,
    compose_comparison,
)
from eval.run_g1_latent_diffusion_eval import _build_diffusion_from_checkpoint
from eval.run_g1_motion_prior_eval import build_model_from_checkpoint
from feature_config import BEAT_FEATURES_8D_DIM, WAV2CLIP_DIM, WAV2CLIP_STFT_BEAT_DIM
from test import FPS, SLICE_LENGTH_FRAMES, SLICE_STRIDE_FRAMES, set_inference_seed


DEFAULT_CHECKPOINT = (
    "runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/"
    "weights/train-1500.pt"
)
DEFAULT_PRIOR_CHECKPOINT = (
    "runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/"
    "weights/train-500.pt"
)


def parse_compose_video(value):
    label, sep, video = value.partition("=")
    if not sep or not label or not video:
        raise argparse.ArgumentTypeError("--compose_video must be label=/path/to/video.mp4")
    if "/" in label or os.sep in label:
        raise argparse.ArgumentTypeError("compose label must be path-safe")
    return label, video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and render V6b-series latent-diffusion G1 dance from audio."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prior_checkpoint", default="")
    parser.add_argument("--music", required=True)
    parser.add_argument("--out_length", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--slice_start", type=int, default=None)
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--feature_source",
        choices=("auto", "cache", "extract"),
        default="auto",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="v6bb_1500")
    parser.add_argument("--motion_format", default=G1_YAW_DELTA_MOTION_FORMAT)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--guidance_weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--g1_root_quat_order", default="xyzw")
    parser.add_argument("--g1_render_backend", choices=("mujoco", "stick"), default="mujoco")
    parser.add_argument("--g1_render_width", type=int, default=960)
    parser.add_argument("--g1_render_height", type=int, default=720)
    parser.add_argument("--g1_mujoco_gl", default="egl")
    parser.add_argument("--comparison_width", type=int, default=640)
    parser.add_argument("--comparison_height", type=int, default=480)
    parser.add_argument(
        "--comparison_layout",
        choices=("auto", "horizontal", "grid2x2"),
        default="auto",
    )
    parser.add_argument("--comparison_path", default="")
    parser.add_argument(
        "--compose_self_index",
        type=int,
        default=0,
        help="Position of the current latent render in the composed comparison.",
    )
    parser.add_argument(
        "--compose_video",
        action="append",
        type=parse_compose_video,
        default=[],
        help="Repeatable video to place after V6b-B in the final comparison: label=path.",
    )
    return parser.parse_args()


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _save_pickle(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _extract_or_load_beat8d(selected_wavs, output_dir):
    from data.audio_extraction.beat_features_8d_features import extract

    feature_dir = output_dir / "features" / "beat_features_8d"
    feature_dir.mkdir(parents=True, exist_ok=True)
    features = []
    paths = []
    for wav_path in selected_wavs:
        save_path = feature_dir / f"{Path(wav_path).stem}.npy"
        result = extract(str(wav_path), skip_completed=False, dest_dir=str(feature_dir))
        if result is not None:
            feature, returned_path = result
            save_path = Path(returned_path)
            tmp_path = save_path.with_suffix(f"{save_path.suffix}.tmp")
            with open(tmp_path, "wb") as handle:
                np.save(handle, np.asarray(feature, dtype=np.float32))
            os.replace(tmp_path, save_path)
        feature = np.asarray(np.load(save_path), dtype=np.float32)
        if feature.shape != (SLICE_LENGTH_FRAMES, BEAT_FEATURES_8D_DIM):
            raise ValueError(
                f"{save_path} expected {(SLICE_LENGTH_FRAMES, BEAT_FEATURES_8D_DIM)}, "
                f"got {feature.shape}"
            )
        features.append(feature)
        paths.append(str(save_path))
    return torch.from_numpy(np.stack(features).astype(np.float32)), paths


def _load_beat8d_features(audio_plan, output_dir, args):
    use_cache = audio_plan["source"] == "cache" and args.feature_source in {"auto", "cache"}
    if use_cache:
        features = _load_cached_features(
            "beat_features_8d",
            audio_plan["selected"],
            args.data_path,
            args.split,
        )
        paths = [
            str(Path(args.data_path) / args.split / "beat_features_8d_feats" / f"{path.stem}.npy")
            for path in audio_plan["selected"]
        ]
        return features, paths
    return _extract_or_load_beat8d(audio_plan["selected"], output_dir)


def _extract_or_load_wav2clip_semantic(selected_wavs, output_dir):
    from data.audio_extraction.wav2clip_stft_beat_features import load_wav2clip_model

    feature_dir = output_dir / "features" / "wav2clip_stft_beat"
    features = []
    paths = []
    wav2clip_model = load_wav2clip_model()
    try:
        for wav_path in selected_wavs:
            feature = _extract_feature(
                "wav2clip_stft_beat",
                Path(wav_path),
                feature_dir,
                wav2clip_model=wav2clip_model,
            )
            feature = np.asarray(feature, dtype=np.float32)
            save_path = feature_dir / f"{Path(wav_path).stem}.npy"
            if feature.shape != (SLICE_LENGTH_FRAMES, WAV2CLIP_STFT_BEAT_DIM):
                raise ValueError(
                    f"{save_path} expected {(SLICE_LENGTH_FRAMES, WAV2CLIP_STFT_BEAT_DIM)}, "
                    f"got {feature.shape}"
                )
            features.append(feature[:, :WAV2CLIP_DIM])
            paths.append(str(save_path))
    finally:
        del wav2clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.from_numpy(np.stack(features).astype(np.float32)), paths


def _load_wav2clip_semantic_features(audio_plan, output_dir, args):
    use_cache = audio_plan["source"] == "cache" and args.feature_source in {"auto", "cache"}
    if use_cache:
        features = _load_cached_features(
            "wav2clip_stft_beat",
            audio_plan["selected"],
            args.data_path,
            args.split,
        )
        paths = [
            str(Path(args.data_path) / args.split / "wav2clip_stft_beat_feats" / f"{path.stem}.npy")
            for path in audio_plan["selected"]
        ]
        return features[:, :, :WAV2CLIP_DIM].contiguous(), paths
    return _extract_or_load_wav2clip_semantic(audio_plan["selected"], output_dir)


def _stitch_yaw_delta_samples(raw_motion, motion_format):
    if motion_format != G1_YAW_DELTA_MOTION_FORMAT:
        raise ValueError(f"V6b-B render expects {G1_YAW_DELTA_MOTION_FORMAT}, got {motion_format}")
    if raw_motion.ndim != 3:
        raise ValueError(f"raw_motion expected [slices, frames, dim], got {tuple(raw_motion.shape)}")
    if raw_motion.shape[1] != SLICE_LENGTH_FRAMES:
        raise ValueError(
            f"raw_motion expected {SLICE_LENGTH_FRAMES} frames per slice, got {raw_motion.shape[1]}"
        )
    if raw_motion.shape[0] == 1:
        return raw_motion[0]
    tail = raw_motion[1:, SLICE_STRIDE_FRAMES:, :].reshape(-1, raw_motion.shape[-1])
    return torch.cat([raw_motion[0], tail], dim=0)


def _decode_payload(raw_motion, audio_plan, args, feature_paths):
    stitched = _stitch_yaw_delta_samples(raw_motion, args.motion_format)
    decoded = decode_g1_motion(stitched.unsqueeze(0), motion_format=args.motion_format)
    root_pos = decoded["root_pos"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    root_rot = decoded["root_rot"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    dof_pos = decoded["dof_pos"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return {
        "motion_rep": "g1",
        "motion_format": args.motion_format,
        "fps": float(FPS),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "pos": root_pos,
        "q": np.concatenate([root_rot, dof_pos], axis=-1).astype(np.float32),
        "audio_path": str(audio_plan["music"]),
        "source_audio_slices": [str(path) for path in audio_plan["selected"]],
        "source_feature_paths": list(feature_paths),
        "stitch_stride_frames": int(SLICE_STRIDE_FRAMES),
    }


@torch.inference_mode()
def generate_v6b_motion(audio_plan, output_dir, args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prior_checkpoint_path = Path(args.prior_checkpoint or checkpoint["config"].get("prior_checkpoint") or DEFAULT_PRIOR_CHECKPOINT)
    if not prior_checkpoint_path.is_file():
        raise FileNotFoundError(f"prior checkpoint not found: {prior_checkpoint_path}")
    prior_checkpoint = torch.load(prior_checkpoint_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_inference_seed(args.seed)
    features, feature_paths = _load_beat8d_features(audio_plan, output_dir, args)
    control = features.to(device=device).float()
    use_wav2clip_semantic = bool(
        checkpoint.get("config", {}).get("use_wav2clip_semantic", False)
    )
    semantic = None
    semantic_paths = []
    if use_wav2clip_semantic:
        semantic_features, semantic_paths = _load_wav2clip_semantic_features(
            audio_plan,
            output_dir,
            args,
        )
        semantic = semantic_features.to(device=device).float()

    diffusion = _build_diffusion_from_checkpoint(checkpoint).to(device=device)
    diffusion.eval()
    prior_model = build_model_from_checkpoint(prior_checkpoint, args.motion_format).to(device=device)
    prior_model.eval()
    latent_normalizer = G1LatentNormalizer.from_state_dict(checkpoint["latent_normalizer"])
    motion_normalizer = G1MotionPriorNormalizer.from_state_dict(prior_checkpoint["normalizer"])
    mean, std = motion_normalizer.tensors(device=device)

    latent = diffusion.ddim_sample(
        control,
        shape=(control.shape[0], int(checkpoint["config"].get("latent_frames", 75)), int(checkpoint["config"].get("latent_dim", 128))),
        semantic_features=semantic,
        sampling_steps=args.sampling_steps,
        guidance_weight=args.guidance_weight,
    )
    latent = latent_normalizer.unnormalize_tensor(latent)
    motion_norm, _ = prior_model.decode(latent.float())
    raw_motion = motion_norm * std + mean
    payload = _decode_payload(raw_motion, audio_plan, args, feature_paths)
    return payload, {
        "checkpoint": str(checkpoint_path),
        "prior_checkpoint": str(prior_checkpoint_path),
        "device": str(device),
        "sampling_steps": int(args.sampling_steps),
        "guidance_weight": float(args.guidance_weight),
        "latent_shape": list(latent.shape),
        "raw_motion_shape": list(raw_motion.shape),
        "use_wav2clip_semantic": use_wav2clip_semantic,
        "control_feature_paths": list(feature_paths),
        "semantic_feature_paths": list(semantic_paths),
    }


def render_and_compose(payload, generation_meta, audio_plan, output_dir, args):
    render_dir = output_dir / "videos" / args.label
    motion_dir = output_dir / "motions" / args.label
    if args.overwrite:
        shutil.rmtree(render_dir, ignore_errors=True)
        shutil.rmtree(motion_dir, ignore_errors=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    music_stem = Path(args.music).stem
    motion_path = motion_dir / f"{args.label}_0_{music_stem}_g1.pkl"
    _save_pickle(payload, motion_path)

    video_path = None
    if not args.no_render:
        video_path = Path(
            render_g1_motion(
                payload,
                out=render_dir,
                epoch=args.label,
                num=0,
                name=[str(path) for path in audio_plan["selected"]],
                sound=True,
                stitch=True,
                model_path=args.g1_fk_model_path,
                root_quat_order=args.g1_root_quat_order,
                render_backend=args.g1_render_backend,
                width=args.g1_render_width,
                height=args.g1_render_height,
                mujoco_gl=args.g1_mujoco_gl,
            )
        )

    self_result = {
        "label": args.label,
        "video": str(video_path) if video_path is not None else None,
        "motion": str(motion_path),
    }
    render_results = []
    for label, video in args.compose_video:
        video_path_arg = Path(video)
        if not video_path_arg.is_file():
            raise FileNotFoundError(f"compose video for {label} not found: {video_path_arg}")
        render_results.append({"label": label, "video": str(video_path_arg), "motion": None})
    self_index = max(0, min(int(args.compose_self_index), len(render_results)))
    render_results.insert(self_index, self_result)

    comparison_path = Path(args.comparison_path) if args.comparison_path else output_dir / "comparison.mp4"
    comparison_command = None
    if args.overwrite and comparison_path.exists():
        comparison_path.unlink()
    if not args.no_render and len(render_results) > 1:
        comparison_command = compose_comparison(
            render_results,
            comparison_path,
            args.comparison_width,
            args.comparison_height,
            layout=args.comparison_layout,
        )

    manifest = {
        "music": str(audio_plan["music"]),
        "feature_source": args.feature_source,
        "audio_source": audio_plan["source"],
        "data_path": args.data_path,
        "split": args.split,
        "out_length": float(args.out_length),
        "seed": int(args.seed),
        "slice_start": audio_plan["start_idx"],
        "all_slice_count": audio_plan["all_slice_count"],
        "sample_size": audio_plan["sample_size"],
        "slice_step": audio_plan["slice_step"],
        "effective_stride_seconds": audio_plan["effective_stride_seconds"],
        "selected_slices": [str(path) for path in audio_plan["selected"]],
        "model": generation_meta,
        "render": {
            "label": args.label,
            "motion": str(motion_path),
            "video": str(video_path) if video_path is not None else None,
            "g1_render_backend": args.g1_render_backend,
            "g1_mujoco_gl": args.g1_mujoco_gl,
            "g1_render_width": args.g1_render_width,
            "g1_render_height": args.g1_render_height,
        },
        "comparison_inputs": render_results,
        "comparison": str(comparison_path) if comparison_path.is_file() else None,
        "comparison_command": comparison_command,
    }
    manifest_path = output_dir / "v6b_manifest.json"
    _write_json(manifest, manifest_path)
    print(f"Wrote V6b-B manifest: {manifest_path}")
    if video_path is not None:
        print(f"Wrote V6b-B video: {video_path}")
    if comparison_path.is_file():
        print(f"Wrote comparison: {comparison_path}")


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", args.g1_mujoco_gl)
    os.environ.setdefault("PYOPENGL_PLATFORM", args.g1_mujoco_gl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_plan = _prepare_audio_slices(
        args.music,
        output_dir,
        args.out_length,
        args.seed,
        args.slice_start,
        args.data_path,
        args.split,
        args.feature_source,
    )
    payload, generation_meta = generate_v6b_motion(audio_plan, output_dir, args)
    render_and_compose(payload, generation_meta, audio_plan, output_dir, args)


if __name__ == "__main__":
    main()
