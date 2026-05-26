import argparse
import json
import os
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
    MOTION_BEATNESS_DIM,
    MOTION_ENERGY_DIM,
    MOTION_INTENSITY_DIM,
    WAV2CLIP_DIM,
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
        WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE,
        WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    )


def _raw_feature_types_for_model_specs(model_specs):
    feature_types = set()
    for spec in model_specs:
        if _is_structured_motion_control_feature(spec.feature_type):
            feature_types.add("wav2clip_stft_beat")
            feature_types.add("gaussian_beat")
        else:
            feature_types.add(spec.feature_type)
    return feature_types


def _build_structured_motion_condition(feature_type, raw_features):
    combined = raw_features["wav2clip_stft_beat"]
    gaussian_beat = raw_features["gaussian_beat"]
    batch, frames = combined.shape[:2]
    control = {"gaussian_beat": gaussian_beat}
    if feature_type == WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE:
        control["beat_energy_envelope"] = torch.zeros(
            (batch, frames, MOTION_ENERGY_DIM),
            dtype=combined.dtype,
        )
    elif feature_type == WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE:
        control["motion_intensity"] = torch.zeros(
            (batch, frames, MOTION_INTENSITY_DIM),
            dtype=combined.dtype,
        )
        control["motion_beatness"] = torch.zeros(
            (batch, frames, MOTION_BEATNESS_DIM),
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
    args = parser.parse_args()
    if not args.model:
        args.model = [parse_model_spec(spec) for spec in DEFAULT_MODEL_SPECS]
    if len(args.model) < 2:
        parser.error("at least two --model entries are required")
    if args.output_dir is None:
        args.output_dir = str(_default_output_dir(args))
    return args


def _ensure_output_dir(path, overwrite=False):
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _choose_slice_window(file_list, sample_size, seed, slice_start):
    if len(file_list) < sample_size:
        raise ValueError(
            f"only {len(file_list)} slices are available; "
            f"{sample_size} slices are required"
        )

    rng = set_inference_seed(seed)
    start_idx = choose_slice_start(len(file_list), sample_size, rng)
    if slice_start is not None:
        if slice_start < 0 or slice_start + sample_size > len(file_list):
            raise ValueError(
                f"--slice_start {slice_start} is outside valid range "
                f"0..{len(file_list) - sample_size}"
            )
        start_idx = slice_start
    return start_idx, file_list[start_idx : start_idx + sample_size]


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
            start_idx, selected = _choose_slice_window(
                cached_wavs,
                sample_size,
                seed,
                slice_start,
            )
            return {
                "source": "cache",
                "music": music_path,
                "slice_dir": cached_wavs[0].parent,
                "all_slice_count": len(cached_wavs),
                "sample_size": sample_size,
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
    slice_audio(str(music_path), 2.5, 5.0, str(slice_dir))
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
    if spec.feature_type == WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE:
        return "pred_controls"
    return "auto"


def render_model(spec, cond, audio_plan, args, output_dir):
    from EDGE import EDGE

    checkpoint = Path(spec.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found for {spec.label}: {checkpoint}")

    set_inference_seed(args.seed)
    render_dir = output_dir / "videos" / spec.label
    motion_dir = output_dir / "motions" / spec.label
    render_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    model = EDGE(
        spec.feature_type,
        checkpoint_path=str(checkpoint),
        use_beats=False,
        lambda_beat=0.0,
        motion_format=args.motion_format,
        feature_fusion=spec.feature_fusion,
    )
    model.eval()
    model_cond = move_cond_to_device(cond, model.accelerator.device)
    condition_variant = _default_condition_variant(spec)
    model_cond = apply_motion_energy_condition_variant(model, model_cond, condition_variant)
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

    video_path = render_dir / f"{spec.label}_0_{Path(args.music).stem}_g1.mp4"
    motion_path = motion_dir / f"{spec.label}_0_{Path(args.music).stem}_g1.pkl"
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
        "video": str(video_path) if video_path.is_file() else None,
        "motion": str(motion_path) if motion_path.is_file() else None,
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


def compose_comparison(render_results, output_path, width, height):
    videos = [result["video"] for result in render_results]
    if any(video is None for video in videos):
        raise ValueError("cannot compose comparison when rendering is disabled")

    labels = [result["label"] for result in render_results]
    banner_path = output_path.with_name("comparison_labels.png")
    write_label_banner(labels, banner_path, width)

    command = [_ffmpeg_exe(), "-loglevel", "error", "-y"]
    for video in videos:
        command.extend(["-i", video])
    command.extend(["-loop", "1", "-i", str(banner_path)])

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
    banner_index = len(render_results)
    total_width = int(width) * len(render_results)
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
        )

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
