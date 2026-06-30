import argparse
import json
import os
import pickle
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.slice import slice_audio
from eval.g1_visualization import _ffmpeg_exe
from eval.run_g1_dataset_eval import apply_motion_energy_condition_variant
from feature_config import (
    BEAT_FEATURES_8D_FEATURE_TYPE,
    BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE,
    BODY_INTENSITY_DIM,
    MOTION_BEATNESS_DIM,
    MOTION_ENERGY_DIM,
    MOTION_INTENSITY_DIM,
    SUPPORT_BEATNESS_DIM,
    SUPPORT_CONTACT_DIM,
    UPPER_BEATNESS_DIM,
    WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_DIM,
    WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE,
    WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
)
from model.diffusion import move_cond_to_device
from test import (
    FPS,
    SLICE_LENGTH_FRAMES,
    SLICE_STRIDE_FRAMES,
    build_beat_condition_slices,
    choose_slice_start,
    set_inference_seed,
    stringintkey,
)


CACHED_SLICE_STRIDE_SECONDS = 0.5
EXTRACT_SLICE_STRIDE_SECONDS = SLICE_STRIDE_FRAMES / FPS


DEFAULT_MODEL_SPECS = (
    "wav2clip_r02_2000:wav2clip_stft_beat:stream_adapter:"
    "runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter_resume600_to2000/weights/train-2000.pt",
    "gaussian_beat_1000:gaussian_beat:linear:"
    "runs/train/EXP-20260520-finedance-g1-gaussian-beat_r01_linear/weights/train-1000.pt",
    "librosa35_2000:baseline:linear:"
    "runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt",
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    feature_type: str
    feature_fusion: str
    checkpoint: str
    condition_variant: str = "auto"


def parse_model_spec(value):
    parts = value.split(":", 4)
    if len(parts) not in (4, 5) or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            "--model must be label:feature_type:feature_fusion:checkpoint[:condition_variant]"
        )
    if len(parts) == 4:
        label, feature_type, feature_fusion, checkpoint = parts
        condition_variant = "auto"
    else:
        label, feature_type, feature_fusion, checkpoint, condition_variant = parts
    if "/" in label or os.sep in label:
        raise argparse.ArgumentTypeError("model label must be a path-safe name")
    return ModelSpec(
        label=label,
        feature_type=feature_type,
        feature_fusion=feature_fusion,
        checkpoint=checkpoint,
        condition_variant=condition_variant,
    )


def _is_structured_motion_control_feature(feature_type):
    return feature_type in (
        BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE,
        WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    )


def _raw_feature_types_for_model_specs(model_specs):
    feature_types = set()
    for spec in model_specs:
        if spec.feature_type == BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE:
            feature_types.add(BEAT_FEATURES_8D_FEATURE_TYPE)
        elif _is_structured_motion_control_feature(spec.feature_type):
            feature_types.add("wav2clip_stft_beat")
            feature_types.add("gaussian_beat")
        else:
            feature_types.add(spec.feature_type)
    return feature_types


def _build_structured_motion_condition(feature_type, raw_features):
    if feature_type == BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE:
        beat_features = raw_features[BEAT_FEATURES_8D_FEATURE_TYPE]
        batch, frames = beat_features.shape[:2]
        return {
            "semantic": {"beat_features_8d": beat_features},
            "control": {
                "motion_beatness": torch.zeros(
                    (batch, frames, MOTION_BEATNESS_DIM),
                    dtype=beat_features.dtype,
                ),
            },
        }

    combined = raw_features["wav2clip_stft_beat"]
    gaussian_beat = raw_features["gaussian_beat"]
    batch, frames = combined.shape[:2]
    control = {"gaussian_beat": gaussian_beat}
    if feature_type == WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE:
        control["beat_energy_envelope"] = torch.zeros(
            (batch, frames, MOTION_ENERGY_DIM),
            dtype=combined.dtype,
        )
    elif feature_type in (
        WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    ):
        control["motion_intensity"] = torch.zeros(
            (batch, frames, MOTION_INTENSITY_DIM),
            dtype=combined.dtype,
        )
        control["motion_beatness"] = torch.zeros(
            (batch, frames, MOTION_BEATNESS_DIM),
            dtype=combined.dtype,
        )
    elif feature_type == WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE:
        control["body_intensity"] = torch.zeros(
            (batch, frames, BODY_INTENSITY_DIM),
            dtype=combined.dtype,
        )
        control["support_beatness"] = torch.zeros(
            (batch, frames, SUPPORT_BEATNESS_DIM),
            dtype=combined.dtype,
        )
        control["upper_beatness"] = torch.zeros(
            (batch, frames, UPPER_BEATNESS_DIM),
            dtype=combined.dtype,
        )
        control["support_contact"] = torch.zeros(
            (batch, frames, SUPPORT_CONTACT_DIM),
            dtype=combined.dtype,
        )
    else:
        raise ValueError(f"Unsupported structured feature_type: {feature_type}")
    return {
        "semantic": {"wav2clip": combined[:, :, :WAV2CLIP_DIM].contiguous()},
        "control": control,
    }


def _default_output_dir(args):
    music_stem = Path(args.music).stem
    source_suffix = "" if args.feature_source == "extract" else f"_{args.feature_source}"
    return (
        Path("renders")
        / "EXP-20260513-finedance-g1-wav2clip-stft-beat"
        / f"checkpoint_comparison_{music_stem}_{int(args.out_length)}s_seed{args.seed}{source_suffix}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render multiple G1 checkpoints on the same music segment and stack "
            "their videos into one comparison."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        default=[],
        help="Repeatable: label:feature_type:feature_fusion:checkpoint",
    )
    parser.add_argument("--music", default="data/finedance/music_wav/012.wav")
    parser.add_argument("--out_length", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--slice_start", type=int, default=None)
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--feature_source",
        choices=("auto", "cache", "extract"),
        default="auto",
        help=(
            "Use dataset cached wav/features when available, require them with "
            "'cache', or recompute features from sliced audio with 'extract'."
        ),
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--motion_format", default="g1")
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
        help="Use grid2x2 for four-tile GT/model comparisons, or horizontal legacy layout.",
    )
    parser.add_argument(
        "--no_gt",
        action="store_true",
        help="Do not prepend the ground-truth dance tile to the comparison render.",
    )
    parser.add_argument("--gt_label", default="gt")
    parser.add_argument(
        "--gt_motion_dir",
        default=None,
        help="Directory containing sliced G1 ground-truth motion pickles.",
    )
    args = parser.parse_args()
    if not args.model:
        args.model = [parse_model_spec(spec) for spec in DEFAULT_MODEL_SPECS]
    if len(args.model) < 1:
        parser.error("at least one --model entry is required")
    if args.output_dir is None:
        args.output_dir = str(_default_output_dir(args))
    return args


def _ensure_output_dir(path, overwrite=False):
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slice_step_for_cached_long_generation(
    cached_stride_seconds=CACHED_SLICE_STRIDE_SECONDS,
    long_stride_frames=SLICE_STRIDE_FRAMES,
    fps=FPS,
):
    long_stride_seconds = float(long_stride_frames) / float(fps)
    step = int(round(long_stride_seconds / float(cached_stride_seconds)))
    if step < 1 or not np.isclose(step * float(cached_stride_seconds), long_stride_seconds):
        raise ValueError(
            "cached slice stride must evenly divide the long-generation stride; "
            f"cached={cached_stride_seconds}s long={long_stride_seconds}s"
        )
    return step


def _choose_slice_window(file_list, sample_size, seed, slice_start, slice_step=1):
    slice_step = int(slice_step)
    if slice_step < 1:
        raise ValueError("slice_step must be positive")
    required_span = 1 + (int(sample_size) - 1) * slice_step
    if len(file_list) < required_span:
        raise ValueError(
            f"only {len(file_list)} slices are available; "
            f"{sample_size} slices with step {slice_step} require {required_span}"
        )

    rng = set_inference_seed(seed)
    max_start = len(file_list) - required_span
    start_idx = rng.randint(0, max_start)
    if slice_start is not None:
        if slice_start < 0 or slice_start + required_span > len(file_list):
            raise ValueError(
                f"--slice_start {slice_start} is outside valid range "
                f"0..{max_start}"
            )
        start_idx = slice_start
    selected = [file_list[start_idx + index * slice_step] for index in range(sample_size)]
    return start_idx, selected


def _cached_wav_slices(music_stem, data_path, split):
    slice_dir = Path(data_path) / split / "wavs_sliced"
    return sorted(
        slice_dir.glob(f"{music_stem}_slice*.wav"),
        key=lambda path: stringintkey(path.name),
    )


def _prepare_audio_slices(
    music_path,
    output_dir,
    out_length,
    seed,
    slice_start,
    data_path,
    split,
    feature_source,
):
    music_path = Path(music_path)
    if feature_source != "cache" and not music_path.is_file():
        raise FileNotFoundError(f"music file not found: {music_path}")
    sample_size = int(out_length / 2.5) - 1
    if sample_size < 1:
        raise ValueError("--out_length must be at least 5 seconds")

    if feature_source in {"auto", "cache"}:
        cached_wavs = _cached_wav_slices(music_path.stem, data_path, split)
        if cached_wavs:
            slice_step = _slice_step_for_cached_long_generation()
            start_idx, selected = _choose_slice_window(
                cached_wavs,
                sample_size,
                seed,
                slice_start,
                slice_step=slice_step,
            )
            return {
                "source": "cache",
                "music": music_path,
                "slice_dir": cached_wavs[0].parent,
                "all_slice_count": len(cached_wavs),
                "sample_size": sample_size,
                "slice_step": slice_step,
                "cached_slice_stride_seconds": CACHED_SLICE_STRIDE_SECONDS,
                "effective_stride_seconds": slice_step * CACHED_SLICE_STRIDE_SECONDS,
                "start_idx": start_idx,
                "selected": selected,
            }
        if feature_source == "cache":
            raise FileNotFoundError(
                f"no cached wav slices for {music_path.stem!r} under "
                f"{Path(data_path) / split / 'wavs_sliced'}"
            )

    if not music_path.is_file():
        raise FileNotFoundError(f"music file not found: {music_path}")

    slice_dir = output_dir / "audio_slices" / music_path.stem
    if slice_dir.exists():
        shutil.rmtree(slice_dir)
    slice_dir.mkdir(parents=True)
    slice_audio(str(music_path), EXTRACT_SLICE_STRIDE_SECONDS, 5.0, str(slice_dir))
    file_list = sorted(slice_dir.glob("*.wav"), key=lambda path: stringintkey(path.name))
    if len(file_list) < sample_size:
        raise ValueError(
            f"{music_path} only produced {len(file_list)} slices; "
            f"{sample_size} slices are required for {out_length:g}s"
        )

    start_idx, selected = _choose_slice_window(file_list, sample_size, seed, slice_start)
    return {
        "source": "extract",
        "music": music_path,
        "slice_dir": slice_dir,
        "all_slice_count": len(file_list),
        "sample_size": sample_size,
        "slice_step": 1,
        "cached_slice_stride_seconds": None,
        "effective_stride_seconds": EXTRACT_SLICE_STRIDE_SECONDS,
        "start_idx": start_idx,
        "selected": selected,
    }


def _extract_feature(feature_type, wav_path, dest_dir, wav2clip_model=None):
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_path = dest_dir / f"{wav_path.stem}.npy"
    if save_path.is_file():
        return np.load(save_path)

    if feature_type == "baseline":
        from data.audio_extraction.baseline_features import extract

        features, path = extract(str(wav_path), skip_completed=False, dest_dir=str(dest_dir))
    elif feature_type == "gaussian_beat":
        from data.audio_extraction.gaussian_beat_features import extract

        features, path = extract(str(wav_path), skip_completed=False, dest_dir=str(dest_dir))
    elif feature_type == "beat_features_8d":
        from data.audio_extraction.beat_features_8d_features import extract

        features, path = extract(str(wav_path), skip_completed=False, dest_dir=str(dest_dir))
    elif feature_type == "jukebox":
        from data.audio_extraction.jukebox_features import extract

        features, path = extract(str(wav_path), skip_completed=False, dest_dir=str(dest_dir))
    elif feature_type == "wav2clip_stft_beat":
        from data.audio_extraction.wav2clip_stft_beat_features import extract

        features, path = extract(
            str(wav_path),
            skip_completed=False,
            dest_dir=str(dest_dir),
            wav2clip_model=wav2clip_model,
        )
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    np.save(path, features)
    return np.asarray(features, dtype=np.float32)


def _feature_cache_path(feature_type, wav_path, data_path, split):
    return Path(data_path) / split / f"{feature_type}_feats" / f"{wav_path.stem}.npy"


def _load_cached_features(feature_type, selected_wavs, data_path, split):
    feature_paths = [
        _feature_cache_path(feature_type, wav_path, data_path, split)
        for wav_path in selected_wavs
    ]
    missing = [path for path in feature_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"cached {feature_type} features are missing; first missing file: {missing[0]}. "
            "Use --feature_source extract to recompute from audio instead."
        )
    cond_list = [np.load(path) for path in feature_paths]
    return torch.from_numpy(np.asarray(cond_list, dtype=np.float32)).float()


def _load_cached_beat_condition(selected_wavs, data_path, split, beat_rep):
    beat_paths = [
        Path(data_path) / split / "beat_feats" / f"{wav_path.stem}.npz"
        for wav_path in selected_wavs
    ]
    missing = [path for path in beat_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"cached beat features are missing; first missing file: {missing[0]}"
        )

    key = "audio_dist" if beat_rep == "distance" else "audio_mask"
    beat_list = []
    for path in beat_paths:
        with np.load(path) as beat_meta:
            beat_list.append(beat_meta[key])
    beat = torch.from_numpy(np.asarray(beat_list)).float()
    if beat_rep == "distance":
        return beat.long()
    return beat.unsqueeze(-1)


def build_conditions(model_specs, selected_wavs, output_dir, args, audio_plan):
    raw_features_by_type = {}
    use_cached_features = audio_plan["source"] == "cache" and args.feature_source in {
        "auto",
        "cache",
    }
    wav2clip_model = None
    try:
        for feature_type in sorted(_raw_feature_types_for_model_specs(model_specs)):
            if use_cached_features:
                raw_features_by_type[feature_type] = _load_cached_features(
                    feature_type,
                    selected_wavs,
                    args.data_path,
                    args.split,
                )
                continue

            if feature_type == "wav2clip_stft_beat":
                from data.audio_extraction.wav2clip_stft_beat_features import load_wav2clip_model

                wav2clip_model = load_wav2clip_model()
            feature_dir = output_dir / "features" / feature_type
            cond_list = [
                _extract_feature(
                    feature_type,
                    wav_path,
                    feature_dir,
                    wav2clip_model=wav2clip_model,
                )
                for wav_path in tqdm(
                    selected_wavs,
                    desc=f"Extract {feature_type}",
                    unit="slice",
                )
            ]
            raw_features_by_type[feature_type] = torch.from_numpy(np.array(cond_list)).float()
    finally:
        del wav2clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    features_by_type = {}
    for feature_type in sorted({spec.feature_type for spec in model_specs}):
        if _is_structured_motion_control_feature(feature_type):
            features_by_type[feature_type] = _build_structured_motion_condition(
                feature_type,
                raw_features_by_type,
            )
        else:
            features_by_type[feature_type] = raw_features_by_type[feature_type]
    return features_by_type


def _default_condition_variant(spec):
    if spec.condition_variant != "auto":
        return spec.condition_variant
    if spec.feature_type == WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE:
        return "pred_energy"
    if spec.feature_type in (
        WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    ):
        return "pred_controls"
    return "auto"


def _checkpoint_motion_format(checkpoint_path, fallback):
    from dataset.motion_representation import validate_motion_format

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        return validate_motion_format(
            checkpoint.get("config", {}).get("motion_format", fallback)
        )
    finally:
        del checkpoint


def render_model(spec, cond, audio_plan, args, output_dir):
    from EDGE import EDGE

    checkpoint = Path(spec.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found for {spec.label}: {checkpoint}")
    motion_format = _checkpoint_motion_format(checkpoint, args.motion_format)

    set_inference_seed(args.seed)
    render_dir = output_dir / "videos" / spec.label
    motion_dir = output_dir / "motions" / spec.label
    render_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)
    video_path = render_dir / f"{spec.label}_0_{Path(args.music).stem}_g1.mp4"
    motion_path = motion_dir / f"{spec.label}_0_{Path(args.music).stem}_g1.pkl"
    condition_variant = _default_condition_variant(spec)
    if not args.overwrite and motion_path.is_file() and (
        args.no_render or video_path.is_file()
    ):
        return {
            "label": spec.label,
            "condition_variant": condition_variant,
            "motion_format": motion_format,
            "g1_render_backend": args.g1_render_backend,
            "g1_mujoco_gl": args.g1_mujoco_gl,
            "video": str(video_path) if video_path.is_file() else None,
            "motion": str(motion_path),
        }

    model = EDGE(
        spec.feature_type,
        checkpoint_path=str(checkpoint),
        use_beats=False,
        lambda_beat=0.0,
        motion_format=motion_format,
        feature_fusion=spec.feature_fusion,
    )
    model.eval()
    model_cond = move_cond_to_device(cond, model.accelerator.device)
    model_cond = apply_motion_energy_condition_variant(model, model_cond, condition_variant)
    resolved_motion_format = model.motion_format
    if model.use_beats:
        if audio_plan["source"] == "cache" and args.feature_source in {"auto", "cache"}:
            beat_cond = _load_cached_beat_condition(
                audio_plan["selected"],
                args.data_path,
                args.split,
                model.beat_rep,
            )
        else:
            beat_cond = build_beat_condition_slices(
                beat_source="audio",
                beat_rep=model.beat_rep,
                wav_path=str(audio_plan["music"]),
                beat_file=None,
                total_slices=audio_plan["all_slice_count"],
                start_idx=audio_plan["start_idx"],
                num_slices=audio_plan["sample_size"],
                fps=FPS,
                horizon=SLICE_LENGTH_FRAMES,
                stride_frames=SLICE_STRIDE_FRAMES,
            )
        model_cond = {"music": model_cond, "beat": beat_cond}

    data_tuple = None, model_cond, [str(path) for path in audio_plan["selected"]]
    model.render_sample(
        data_tuple,
        spec.label,
        str(render_dir),
        render_count=-1,
        fk_out=str(motion_dir),
        render=not args.no_render,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        g1_render_backend=args.g1_render_backend,
        g1_render_width=args.g1_render_width,
        g1_render_height=args.g1_render_height,
        g1_mujoco_gl=args.g1_mujoco_gl,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.no_render and not video_path.is_file():
        matches = sorted(render_dir.glob("*.mp4"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one rendered video for {spec.label} in {render_dir}, "
                f"found {len(matches)}"
            )
        video_path = matches[0]
    return {
        "label": spec.label,
        "condition_variant": condition_variant,
        "motion_format": resolved_motion_format,
        "g1_render_backend": args.g1_render_backend,
        "g1_mujoco_gl": args.g1_mujoco_gl,
        "video": str(video_path) if video_path.is_file() else None,
        "motion": str(motion_path) if motion_path.is_file() else None,
    }


def _slice_index_from_path(path):
    stem = Path(path).stem
    marker = "_slice"
    if marker not in stem:
        raise ValueError(f"slice path does not contain {marker!r}: {path}")
    return int(stem.rsplit(marker, 1)[1])


def _gt_motion_path(wav_path, data_path, split, gt_motion_dir=None):
    motion_dir = (
        Path(gt_motion_dir)
        if gt_motion_dir is not None
        else Path(data_path) / split / "motions_sliced"
    )
    return motion_dir / f"{Path(wav_path).stem}.pkl"


def _gt_motion_path_for_audio_slice(wav_path, data_path, split, audio_plan, gt_motion_dir=None):
    if gt_motion_dir is not None or audio_plan["source"] != "extract":
        return _gt_motion_path(wav_path, data_path, split, gt_motion_dir)

    ratio = float(audio_plan["effective_stride_seconds"]) / CACHED_SLICE_STRIDE_SECONDS
    rounded_ratio = int(round(ratio))
    if rounded_ratio < 1 or not np.isclose(rounded_ratio, ratio):
        raise ValueError(
            "extract-mode GT lookup requires the extracted audio stride to be an "
            f"integer multiple of the cached motion stride; extract="
            f"{audio_plan['effective_stride_seconds']}s cached={CACHED_SLICE_STRIDE_SECONDS}s"
        )

    extract_index = _slice_index_from_path(wav_path)
    cached_index = extract_index * rounded_ratio
    motion_dir = Path(data_path) / split / "motions_sliced"
    music_stem = Path(audio_plan["music"]).stem
    return motion_dir / f"{music_stem}_slice{cached_index}.pkl"


def _stitch_g1_motion_payloads(payloads, stride_frames=SLICE_STRIDE_FRAMES):
    if not payloads:
        raise ValueError("cannot stitch an empty ground-truth motion list")

    first = payloads[0]
    horizon = int(first["root_pos"].shape[0])
    keep_from_next = horizon - int(stride_frames)
    if keep_from_next <= 0:
        raise ValueError("stride_frames must be smaller than the G1 slice horizon")

    def stitch_array(key):
        arrays = [np.asarray(payload[key]) for payload in payloads]
        return np.concatenate([arrays[0], *[array[-keep_from_next:] for array in arrays[1:]]])

    return {
        "motion_format": first.get("motion_format", "g1"),
        "motion_rep": first.get("motion_rep", first.get("motion_format", "g1")),
        "fps": float(first.get("fps", FPS)),
        "root_pos": stitch_array("root_pos").astype(np.float32),
        "root_rot": stitch_array("root_rot").astype(np.float32),
        "dof_pos": stitch_array("dof_pos").astype(np.float32),
    }


def render_ground_truth(audio_plan, args, output_dir):
    from eval.g1_visualization import render_g1_motion

    label = args.gt_label
    render_dir = output_dir / "videos" / label
    motion_dir = output_dir / "motions" / label
    render_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)
    video_path = render_dir / f"{label}_0_{Path(args.music).stem}_g1.mp4"
    motion_path = motion_dir / f"{label}_0_{Path(args.music).stem}_g1.pkl"
    gt_paths = [
        _gt_motion_path_for_audio_slice(
            path,
            args.data_path,
            args.split,
            audio_plan,
            args.gt_motion_dir,
        )
        for path in audio_plan["selected"]
    ]
    missing = [path for path in gt_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"ground-truth motion slice is missing; first missing file: {missing[0]}"
        )

    if not args.overwrite and motion_path.is_file() and (
        args.no_render or video_path.is_file()
    ):
        return {
            "label": label,
            "condition_variant": "ground_truth",
            "motion_format": "g1",
            "g1_render_backend": args.g1_render_backend,
            "g1_mujoco_gl": args.g1_mujoco_gl,
            "video": str(video_path) if video_path.is_file() else None,
            "motion": str(motion_path),
            "source_motions": [str(path) for path in gt_paths],
        }

    payloads = []
    for path in gt_paths:
        with open(path, "rb") as handle:
            payloads.append(pickle.load(handle))
    payload = _stitch_g1_motion_payloads(payloads)
    with open(motion_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)

    if not args.no_render:
        rendered = render_g1_motion(
            payload,
            out=render_dir,
            epoch=label,
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
        video_path = Path(rendered)

    return {
        "label": label,
        "condition_variant": "ground_truth",
        "motion_format": payload["motion_format"],
        "g1_render_backend": args.g1_render_backend,
        "g1_mujoco_gl": args.g1_mujoco_gl,
        "video": str(video_path) if video_path.is_file() else None,
        "motion": str(motion_path),
        "source_motions": [str(path) for path in gt_paths],
    }


def write_label_banner(labels, path, tile_width, banner_height=56):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to label comparison renders without ffmpeg drawtext."
        ) from exc

    width = int(tile_width) * len(labels)
    image = Image.new("RGB", (width, int(banner_height)), color=(12, 12, 12))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, label in enumerate(labels):
        x0 = index * int(tile_width)
        x1 = x0 + int(tile_width)
        if index > 0:
            draw.line((x0, 0, x0, banner_height), fill=(70, 70, 70), width=2)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (
                x0 + max((int(tile_width) - text_width) // 2, 8),
                max((int(banner_height) - text_height) // 2, 4),
            ),
            label,
            fill=(245, 245, 245),
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def write_label_overlay(labels, path, tile_width, tile_height, columns, banner_height=56):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to label comparison renders without ffmpeg drawtext."
        ) from exc

    columns = int(columns)
    rows = int(np.ceil(len(labels) / columns))
    tile_width = int(tile_width)
    tile_height = int(tile_height)
    banner_height = int(banner_height)
    image = Image.new(
        "RGBA",
        (tile_width * columns, tile_height * rows),
        color=(0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, label in enumerate(labels):
        row, column = divmod(index, columns)
        x0 = column * tile_width
        y0 = row * tile_height
        draw.rectangle(
            (x0, y0, x0 + tile_width, y0 + banner_height),
            fill=(12, 12, 12, 220),
        )
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (
                x0 + max((tile_width - text_width) // 2, 8),
                y0 + max((banner_height - text_height) // 2, 4),
            ),
            label,
            fill=(245, 245, 245, 255),
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _resolve_comparison_layout(layout, render_count):
    if layout == "auto":
        return "grid2x2" if int(render_count) == 4 else "horizontal"
    return layout


def compose_comparison(render_results, output_path, width, height, layout="auto"):
    videos = [result["video"] for result in render_results]
    if any(video is None for video in videos):
        raise ValueError("cannot compose comparison when rendering is disabled")

    labels = [result["label"] for result in render_results]
    resolved_layout = _resolve_comparison_layout(layout, len(render_results))
    if resolved_layout == "grid2x2" and len(render_results) != 4:
        raise ValueError("grid2x2 comparison layout requires exactly four rendered tiles")

    command = [_ffmpeg_exe(), "-loglevel", "error", "-y"]
    for video in videos:
        command.extend(["-i", video])

    filters = []
    video_labels = []
    for index, result in enumerate(render_results):
        filters.append(
            f"[{index}:v]"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
            f"[v{index}]"
        )
        video_labels.append(f"[v{index}]")
    if resolved_layout == "grid2x2":
        overlay_path = output_path.with_name("comparison_labels.png")
        write_label_overlay(labels, overlay_path, width, height, columns=2)
        command.extend(["-loop", "1", "-i", str(overlay_path)])
        overlay_index = len(render_results)
        filters.append(f"{video_labels[0]}{video_labels[1]}hstack=inputs=2[row0]")
        filters.append(f"{video_labels[2]}{video_labels[3]}hstack=inputs=2[row1]")
        filters.append("[row0][row1]vstack=inputs=2[stack]")
        filters.append(f"[stack][{overlay_index}:v]overlay=0:0:format=auto[v]")
    else:
        banner_path = output_path.with_name("comparison_labels.png")
        write_label_banner(labels, banner_path, width)
        command.extend(["-loop", "1", "-i", str(banner_path)])
        banner_index = len(render_results)
        total_width = int(width) * len(render_results)
        if len(video_labels) == 1:
            filters.append(f"{video_labels[0]}copy[stack]")
        else:
            filters.append(f"{''.join(video_labels)}hstack=inputs={len(video_labels)}[stack]")
        filters.append(f"[{banner_index}:v]scale={total_width}:56[banner]")
        filters.append("[banner][stack]vstack=inputs=2[v]")

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True)
    return command


def main():
    args = parse_args()
    output_dir = _ensure_output_dir(args.output_dir, overwrite=args.overwrite)
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
    model_specs = list(args.model)
    features_by_type = build_conditions(
        model_specs,
        audio_plan["selected"],
        output_dir,
        args,
        audio_plan,
    )

    render_results = []
    if not args.no_gt:
        print(f"Rendering {args.gt_label}")
        render_results.append(render_ground_truth(audio_plan, args, output_dir))
    for spec in model_specs:
        print(f"Rendering {spec.label}")
        render_results.append(
            render_model(
                spec,
                features_by_type[spec.feature_type],
                audio_plan,
                args,
                output_dir,
            )
        )

    comparison_path = output_dir / "comparison.mp4"
    comparison_command = None
    if not args.no_render:
        comparison_command = compose_comparison(
            render_results,
            comparison_path,
            args.comparison_width,
            args.comparison_height,
            layout=args.comparison_layout,
        )
        comparison_layout = _resolve_comparison_layout(
            args.comparison_layout,
            len(render_results),
        )
    else:
        comparison_layout = None

    manifest = {
        "music": str(audio_plan["music"]),
        "feature_source": args.feature_source,
        "audio_source": audio_plan["source"],
        "data_path": args.data_path,
        "split": args.split,
        "out_length": args.out_length,
        "seed": args.seed,
        "slice_start": audio_plan["start_idx"],
        "all_slice_count": audio_plan["all_slice_count"],
        "sample_size": audio_plan["sample_size"],
        "slice_step": audio_plan["slice_step"],
        "cached_slice_stride_seconds": audio_plan["cached_slice_stride_seconds"],
        "effective_stride_seconds": audio_plan["effective_stride_seconds"],
        "motion_format_arg": args.motion_format,
        "g1_render_backend": args.g1_render_backend,
        "g1_mujoco_gl": args.g1_mujoco_gl,
        "g1_render_width": args.g1_render_width,
        "g1_render_height": args.g1_render_height,
        "comparison_width": args.comparison_width,
        "comparison_height": args.comparison_height,
        "comparison_layout": comparison_layout,
        "include_gt": not args.no_gt,
        "gt_label": args.gt_label,
        "gt_motion_dir": args.gt_motion_dir,
        "selected_slices": [str(path) for path in audio_plan["selected"]],
        "models": [asdict(spec) for spec in model_specs],
        "renders": render_results,
        "comparison": str(comparison_path) if comparison_path.is_file() else None,
        "comparison_command": comparison_command,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote manifest: {manifest_path}")
    if comparison_path.is_file():
        print(f"Wrote comparison: {comparison_path}")


if __name__ == "__main__":
    main()
