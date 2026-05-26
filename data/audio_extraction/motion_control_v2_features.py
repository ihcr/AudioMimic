import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from data.audio_extraction.wav2clip_stft_beat_features import TARGET_FRAMES
from model.g1_torch_kinematics import G1TorchKinematics


CACHE_VERSION = "motion_control_v2"
FPS = 30
DEFAULT_WINDOW_RADIUS = 6
DEFAULT_INTENSITY_SIGMA = 5.0
DEFAULT_BEATNESS_SPEED_SIGMA = 5.0
DEFAULT_BEATNESS_ENVELOPE_SIGMA = 3.0
DEFAULT_SHOULDER_INNER = 3
DEFAULT_SHOULDER_OUTER = 6
FEATURE_DIR_NAME = "motion_control_v2_feats"
METADATA_NAME = "motion_control_v2_metadata.json"
KEYPOINT_WEIGHTS = {
    "left_wrist_yaw_link": 0.35,
    "right_wrist_yaw_link": 0.35,
    "left_ankle_roll_link": 0.10,
    "right_ankle_roll_link": 0.10,
    "torso_link": 0.10,
}


def _split_dir(data_path, split):
    return Path(data_path) / split


def _motion_paths(data_path, split):
    return sorted((_split_dir(data_path, split) / "motions_sliced").glob("*.pkl"))


def _gaussian_beat_path(data_path, split, stem):
    return _split_dir(data_path, split) / "gaussian_beat_feats" / f"{stem}.npy"


def _feature_path(data_path, split, stem):
    return _split_dir(data_path, split) / FEATURE_DIR_NAME / f"{stem}.npz"


def _load_motion(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    root_pos = np.asarray(data.get("root_pos", data.get("pos")), dtype=np.float32)
    root_rot = data.get("root_rot")
    dof_pos = data.get("dof_pos")
    if root_rot is None or dof_pos is None:
        q = np.asarray(data["q"], dtype=np.float32)
        root_rot = q[:, :4]
        dof_pos = q[:, 4:]
    root_rot = np.asarray(root_rot, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    if root_pos.shape != (TARGET_FRAMES, 3):
        raise ValueError(f"{path} root_pos expected {(TARGET_FRAMES, 3)}, got {root_pos.shape}")
    if root_rot.shape != (TARGET_FRAMES, 4):
        raise ValueError(f"{path} root_rot expected {(TARGET_FRAMES, 4)}, got {root_rot.shape}")
    if dof_pos.shape != (TARGET_FRAMES, 29):
        raise ValueError(f"{path} dof_pos expected {(TARGET_FRAMES, 29)}, got {dof_pos.shape}")
    return root_pos, root_rot, dof_pos


def _load_motion_batch(paths):
    roots, rots, dofs = [], [], []
    for path in paths:
        root_pos, root_rot, dof_pos = _load_motion(path)
        roots.append(root_pos)
        rots.append(root_rot)
        dofs.append(dof_pos)
    return np.stack(roots, axis=0), np.stack(rots, axis=0), np.stack(dofs, axis=0)


def _load_audio_beat_frames(gaussian_path):
    beat = np.load(gaussian_path, mmap_mode="r")
    if beat.shape != (TARGET_FRAMES, 1):
        raise ValueError(f"{gaussian_path} expected {(TARGET_FRAMES, 1)}, got {beat.shape}")
    values = np.asarray(beat[:, 0], dtype=np.float32)
    left = np.r_[values[0], values[:-1]]
    right = np.r_[values[1:], values[-1]]
    frames = np.flatnonzero((values >= 1.0 - 1e-5) & (values >= left) & (values >= right))
    return frames.astype(np.int64)


def _resolve_keypoint_indices(kinematics):
    missing = [name for name in KEYPOINT_WEIGHTS if name not in kinematics.keypoint_names]
    if missing:
        raise ValueError(f"G1 keypoints missing from kinematics model: {missing}")
    return [kinematics.keypoint_names.index(name) for name in KEYPOINT_WEIGHTS]


@torch.inference_mode()
def _weighted_fk_speed_batch(kinematics, root_pos, root_rot, dof_pos, keypoint_indices, device):
    root_pos = torch.from_numpy(root_pos).to(device=device, dtype=torch.float32)
    root_rot = torch.from_numpy(root_rot).to(device=device, dtype=torch.float32)
    dof_pos = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32)
    result = kinematics(root_pos, root_rot, dof_pos)
    keypoints = result["keypoints"].index_select(
        -2,
        torch.tensor(keypoint_indices, device=device, dtype=torch.long),
    )
    weights = torch.tensor(
        [KEYPOINT_WEIGHTS[name] for name in KEYPOINT_WEIGHTS],
        device=device,
        dtype=keypoints.dtype,
    )
    frame_speed = keypoints.new_zeros(keypoints.shape[:2] + (keypoints.shape[-2],))
    frame_speed[:, 1:] = torch.linalg.vector_norm(
        keypoints[:, 1:] - keypoints[:, :-1],
        dim=-1,
    ) * FPS
    weighted_speed = (frame_speed * weights.view(1, 1, -1)).sum(dim=-1)
    return weighted_speed.detach().cpu().numpy().astype(np.float32)


def intensity_values_for_frames(weighted_speed, beat_frames, window_radius):
    values = []
    for beat_frame in beat_frames:
        start = max(int(beat_frame) - window_radius, 0)
        end = min(int(beat_frame) + window_radius + 1, weighted_speed.shape[0])
        values.append(float(np.max(weighted_speed[start:end])))
    return np.asarray(values, dtype=np.float32)


def beatness_values_for_frames(
    smoothed_speed,
    beat_frames,
    window_radius,
    shoulder_inner,
    shoulder_outer,
):
    values = []
    smoothed_speed = np.asarray(smoothed_speed, dtype=np.float32)
    for beat_frame in beat_frames:
        beat_frame = int(beat_frame)
        start = max(beat_frame - window_radius, 0)
        end = min(beat_frame + window_radius + 1, smoothed_speed.shape[0])
        window = smoothed_speed[start:end]
        valley_frame = start + int(np.argmin(window))
        valley_speed = float(smoothed_speed[valley_frame])

        left_start = max(valley_frame - shoulder_outer, 0)
        left_end = max(valley_frame - shoulder_inner + 1, left_start)
        right_start = min(valley_frame + shoulder_inner, smoothed_speed.shape[0])
        right_end = min(valley_frame + shoulder_outer + 1, smoothed_speed.shape[0])
        shoulders = []
        if left_start < left_end:
            shoulders.append(float(np.percentile(smoothed_speed[left_start:left_end], 75)))
        if right_start < right_end:
            shoulders.append(float(np.percentile(smoothed_speed[right_start:right_end], 75)))
        shoulder_speed = max(shoulders) if shoulders else valley_speed
        offset = float(valley_frame - beat_frame)
        offset_sigma = max(float(shoulder_inner), 1.0)
        offset_weight = float(np.exp(-0.5 * (offset / offset_sigma) ** 2))
        values.append(max(shoulder_speed - valley_speed, 0.0) * offset_weight)
    return np.asarray(values, dtype=np.float32)


def _normalize(values, p05, p95):
    if values.size == 0:
        return values.astype(np.float32)
    return np.clip((values - p05) / (p95 - p05), 0.0, 1.0).astype(np.float32)


def _build_envelope(beat_frames, normalized_values, sigma):
    frames = np.arange(TARGET_FRAMES, dtype=np.float32)
    envelope = np.zeros((TARGET_FRAMES,), dtype=np.float32)
    for beat_frame, value in zip(beat_frames, normalized_values):
        values = float(value) * np.exp(-0.5 * ((frames - float(beat_frame)) / sigma) ** 2)
        envelope = np.maximum(envelope, values.astype(np.float32))
    return envelope[:, None].astype(np.float32)


def _feature_file_is_valid(path):
    try:
        with np.load(path) as data:
            required = {
                "motion_intensity_envelope": (TARGET_FRAMES, 1),
                "motion_beatness_envelope": (TARGET_FRAMES, 1),
                "weighted_fk_speed": (TARGET_FRAMES,),
                "smoothed_weighted_fk_speed": (TARGET_FRAMES,),
            }
            for key, shape in required.items():
                if data[key].shape != shape or not np.isfinite(data[key]).all():
                    return False
            return (
                data["audio_beat_frames"].ndim == 1
                and data["intensity_peaks"].ndim == 1
                and data["beatness_peaks"].ndim == 1
                and np.isfinite(data["intensity_peaks"]).all()
                and np.isfinite(data["beatness_peaks"]).all()
            )
    except Exception:
        return False


def _iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collect_train_stats(
    data_path,
    kinematics,
    keypoint_indices,
    batch_size,
    device,
    window_radius,
    beatness_speed_sigma,
    shoulder_inner,
    shoulder_outer,
):
    train_paths = _motion_paths(data_path, "train")
    if not train_paths:
        raise ValueError(f"No train motions found under {data_path}")
    all_intensity = []
    all_beatness = []
    for batch_paths in tqdm(
        list(_iter_batches(train_paths, batch_size)),
        desc="Collect train motion-control stats",
        unit="batch",
    ):
        root_pos, root_rot, dof_pos = _load_motion_batch(batch_paths)
        weighted_speeds = _weighted_fk_speed_batch(
            kinematics,
            root_pos,
            root_rot,
            dof_pos,
            keypoint_indices,
            device,
        )
        for path, weighted_speed in zip(batch_paths, weighted_speeds):
            beat_frames = _load_audio_beat_frames(_gaussian_beat_path(data_path, "train", path.stem))
            smoothed = gaussian_filter1d(weighted_speed, sigma=beatness_speed_sigma).astype(np.float32)
            intensity = intensity_values_for_frames(weighted_speed, beat_frames, window_radius)
            beatness = beatness_values_for_frames(
                smoothed,
                beat_frames,
                window_radius,
                shoulder_inner,
                shoulder_outer,
            )
            if intensity.size:
                all_intensity.append(intensity)
            if beatness.size:
                all_beatness.append(beatness)
    if not all_intensity or not all_beatness:
        raise ValueError("No audio beat frames found in the train GaussianBeat cache.")
    intensity_values = np.concatenate(all_intensity, axis=0)
    beatness_values = np.concatenate(all_beatness, axis=0)
    intensity_p05, intensity_p95 = np.percentile(intensity_values, [5, 95]).astype(np.float32)
    beatness_p05, beatness_p95 = np.percentile(beatness_values, [5, 95]).astype(np.float32)
    for name, p05, p95 in (
        ("motion_intensity", intensity_p05, intensity_p95),
        ("motion_beatness", beatness_p05, beatness_p95),
    ):
        if not np.isfinite([p05, p95]).all() or p95 <= p05:
            raise ValueError(f"Invalid {name} normalization stats: p05={p05}, p95={p95}")
    return {
        "motion_intensity": {
            "p05": float(intensity_p05),
            "p95": float(intensity_p95),
            "train_peak_count": int(intensity_values.size),
        },
        "motion_beatness": {
            "p05": float(beatness_p05),
            "p95": float(beatness_p95),
            "train_peak_count": int(beatness_values.size),
        },
    }


def write_split_features(
    data_path,
    split,
    kinematics,
    keypoint_indices,
    batch_size,
    device,
    window_radius,
    intensity_sigma,
    beatness_speed_sigma,
    beatness_envelope_sigma,
    shoulder_inner,
    shoulder_outer,
    normalization,
    skip_completed=True,
):
    split_paths = _motion_paths(data_path, split)
    if not split_paths:
        raise ValueError(f"No {split} motions found under {data_path}")
    output_dir = _split_dir(data_path, split) / FEATURE_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for batch_paths in tqdm(
        list(_iter_batches(split_paths, batch_size)),
        desc=f"Write {split} motion-control features",
        unit="batch",
    ):
        pending = [
            path
            for path in batch_paths
            if not skip_completed or not _feature_file_is_valid(_feature_path(data_path, split, path.stem))
        ]
        skipped += len(batch_paths) - len(pending)
        if not pending:
            continue
        root_pos, root_rot, dof_pos = _load_motion_batch(pending)
        weighted_speeds = _weighted_fk_speed_batch(
            kinematics,
            root_pos,
            root_rot,
            dof_pos,
            keypoint_indices,
            device,
        )
        for path, weighted_speed in zip(pending, weighted_speeds):
            beat_frames = _load_audio_beat_frames(_gaussian_beat_path(data_path, split, path.stem))
            smoothed = gaussian_filter1d(weighted_speed, sigma=beatness_speed_sigma).astype(np.float32)
            raw_intensity = intensity_values_for_frames(weighted_speed, beat_frames, window_radius)
            raw_beatness = beatness_values_for_frames(
                smoothed,
                beat_frames,
                window_radius,
                shoulder_inner,
                shoulder_outer,
            )
            intensity_peaks = _normalize(
                raw_intensity,
                normalization["motion_intensity"]["p05"],
                normalization["motion_intensity"]["p95"],
            )
            beatness_peaks = _normalize(
                raw_beatness,
                normalization["motion_beatness"]["p05"],
                normalization["motion_beatness"]["p95"],
            )
            intensity_envelope = _build_envelope(beat_frames, intensity_peaks, intensity_sigma)
            beatness_envelope = _build_envelope(beat_frames, beatness_peaks, beatness_envelope_sigma)
            save_path = _feature_path(data_path, split, path.stem)
            tmp_path = save_path.with_name(f"{save_path.name}.{os.getpid()}.tmp")
            with open(tmp_path, "wb") as handle:
                np.savez_compressed(
                    handle,
                    motion_intensity_envelope=intensity_envelope.astype(np.float32),
                    motion_beatness_envelope=beatness_envelope.astype(np.float32),
                    weighted_fk_speed=weighted_speed.astype(np.float32),
                    smoothed_weighted_fk_speed=smoothed.astype(np.float32),
                    audio_beat_frames=beat_frames.astype(np.int64),
                    intensity_peaks=intensity_peaks.astype(np.float32),
                    intensity_peaks_raw=raw_intensity.astype(np.float32),
                    beatness_peaks=beatness_peaks.astype(np.float32),
                    beatness_peaks_raw=raw_beatness.astype(np.float32),
                )
            tmp_path.replace(save_path)
            written += 1
    return {"written": written, "skipped": skipped, "total": len(split_paths)}


def _write_metadata(data_path, metadata):
    path = Path(data_path) / METADATA_NAME
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def extract_motion_control_v2_features(args):
    data_path = Path(args.data_path)
    device = torch.device(args.device)
    kinematics = G1TorchKinematics(args.g1_fk_model_path, root_quat_order=args.g1_root_quat_order).to(device)
    keypoint_indices = _resolve_keypoint_indices(kinematics)
    normalization = collect_train_stats(
        data_path,
        kinematics,
        keypoint_indices,
        args.batch_size,
        device,
        args.window_radius,
        args.beatness_speed_sigma,
        args.shoulder_inner,
        args.shoulder_outer,
    )
    split_stats = {}
    for split in ("train", "test"):
        split_stats[split] = write_split_features(
            data_path,
            split,
            kinematics,
            keypoint_indices,
            args.batch_size,
            device,
            args.window_radius,
            args.intensity_sigma,
            args.beatness_speed_sigma,
            args.beatness_envelope_sigma,
            args.shoulder_inner,
            args.shoulder_outer,
            normalization,
            skip_completed=not args.force,
        )
    metadata = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_dir": FEATURE_DIR_NAME,
        "metadata_name": METADATA_NAME,
        "source_beat_feature": "gaussian_beat_feats",
        "target_frames": TARGET_FRAMES,
        "fps": FPS,
        "window_radius": args.window_radius,
        "intensity_sigma": args.intensity_sigma,
        "beatness_speed_sigma": args.beatness_speed_sigma,
        "beatness_envelope_sigma": args.beatness_envelope_sigma,
        "shoulder_inner": args.shoulder_inner,
        "shoulder_outer": args.shoulder_outer,
        "keypoint_weights": KEYPOINT_WEIGHTS,
        "normalization": normalization,
        "splits": split_stats,
    }
    _write_metadata(data_path, metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--g1_fk_model_path",
        type=str,
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_radius", type=int, default=DEFAULT_WINDOW_RADIUS)
    parser.add_argument("--intensity_sigma", type=float, default=DEFAULT_INTENSITY_SIGMA)
    parser.add_argument("--beatness_speed_sigma", type=float, default=DEFAULT_BEATNESS_SPEED_SIGMA)
    parser.add_argument("--beatness_envelope_sigma", type=float, default=DEFAULT_BEATNESS_ENVELOPE_SIGMA)
    parser.add_argument("--shoulder_inner", type=int, default=DEFAULT_SHOULDER_INNER)
    parser.add_argument("--shoulder_outer", type=int, default=DEFAULT_SHOULDER_OUTER)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.window_radius < 1:
        raise ValueError("--window_radius must be positive")
    if args.shoulder_inner < 1 or args.shoulder_outer < args.shoulder_inner:
        raise ValueError("--shoulder_outer must be >= --shoulder_inner >= 1")
    return args


if __name__ == "__main__":
    extract_motion_control_v2_features(parse_args())
