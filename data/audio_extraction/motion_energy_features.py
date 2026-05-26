import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from data.audio_extraction.wav2clip_stft_beat_features import TARGET_FRAMES
from model.g1_torch_kinematics import G1TorchKinematics


CACHE_VERSION = "motion_energy_v1"
FPS = 30
DEFAULT_WINDOW_RADIUS = 6
DEFAULT_ENVELOPE_SIGMA = 5.0
FEATURE_DIR_NAME = "motion_energy_feats"
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
    return (
        np.stack(roots, axis=0),
        np.stack(rots, axis=0),
        np.stack(dofs, axis=0),
    )


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


def _peak_values_for_frames(weighted_speed, beat_frames, window_radius, peak_mode):
    peaks = []
    for beat_frame in beat_frames:
        start = max(int(beat_frame) - window_radius, 0)
        end = min(int(beat_frame) + window_radius + 1, weighted_speed.shape[0])
        window = weighted_speed[start:end]
        if peak_mode == "max":
            peak = float(np.max(window))
        elif peak_mode == "p75":
            peak = float(np.percentile(window, 75))
        else:
            raise ValueError(f"Unsupported energy_peak_mode: {peak_mode}")
        peaks.append(peak)
    return np.asarray(peaks, dtype=np.float32)


def _normalize_peaks(peaks, p05, p95):
    if peaks.size == 0:
        return peaks.astype(np.float32)
    return np.clip((peaks - p05) / (p95 - p05), 0.0, 1.0).astype(np.float32)


def _build_envelope(beat_frames, normalized_peaks, sigma):
    frames = np.arange(TARGET_FRAMES, dtype=np.float32)
    envelope = np.zeros((TARGET_FRAMES,), dtype=np.float32)
    for beat_frame, peak in zip(beat_frames, normalized_peaks):
        values = float(peak) * np.exp(-0.5 * ((frames - float(beat_frame)) / sigma) ** 2)
        envelope = np.maximum(envelope, values.astype(np.float32))
    return envelope[:, None].astype(np.float32)


def _feature_file_is_valid(path):
    try:
        with np.load(path) as data:
            return (
                data["beat_energy_envelope"].shape == (TARGET_FRAMES, 1)
                and data["weighted_fk_speed"].shape == (TARGET_FRAMES,)
                and data["audio_beat_frames"].ndim == 1
                and data["beat_energy_peaks"].ndim == 1
                and np.isfinite(data["beat_energy_envelope"]).all()
                and np.isfinite(data["weighted_fk_speed"]).all()
                and np.isfinite(data["beat_energy_peaks"]).all()
            )
    except Exception:
        return False


def _iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collect_train_peak_stats(
    data_path,
    kinematics,
    keypoint_indices,
    batch_size,
    device,
    window_radius,
    peak_mode,
):
    train_paths = _motion_paths(data_path, "train")
    if not train_paths:
        raise ValueError(f"No train motions found under {data_path}")
    all_peaks = []
    for batch_paths in tqdm(
        list(_iter_batches(train_paths, batch_size)),
        desc="Collect train motion-energy peaks",
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
            peaks = _peak_values_for_frames(weighted_speed, beat_frames, window_radius, peak_mode)
            if peaks.size:
                all_peaks.append(peaks)
    if not all_peaks:
        raise ValueError("No audio beat frames found in the train GaussianBeat cache.")
    all_peaks = np.concatenate(all_peaks, axis=0)
    p05, p95 = np.percentile(all_peaks, [5, 95]).astype(np.float32)
    if not np.isfinite([p05, p95]).all() or p95 <= p05:
        raise ValueError(f"Invalid motion-energy normalization stats: p05={p05}, p95={p95}")
    return float(p05), float(p95), int(all_peaks.size)


def write_split_features(
    data_path,
    split,
    kinematics,
    keypoint_indices,
    batch_size,
    device,
    window_radius,
    peak_mode,
    envelope_sigma,
    p05,
    p95,
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
        desc=f"Write {split} motion-energy features",
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
            raw_peaks = _peak_values_for_frames(weighted_speed, beat_frames, window_radius, peak_mode)
            normalized_peaks = _normalize_peaks(raw_peaks, p05, p95)
            envelope = _build_envelope(beat_frames, normalized_peaks, envelope_sigma)
            save_path = _feature_path(data_path, split, path.stem)
            tmp_path = save_path.with_name(f"{save_path.name}.{os.getpid()}.tmp")
            with open(tmp_path, "wb") as handle:
                np.savez_compressed(
                    handle,
                    beat_energy_envelope=envelope.astype(np.float32),
                    weighted_fk_speed=weighted_speed.astype(np.float32),
                    audio_beat_frames=beat_frames.astype(np.int64),
                    beat_energy_peaks=normalized_peaks.astype(np.float32),
                    beat_energy_peaks_raw=raw_peaks.astype(np.float32),
                )
            tmp_path.replace(save_path)
            written += 1
    return {"total": len(split_paths), "written": written, "skipped": skipped}


def write_metadata(data_path, metadata):
    path = Path(data_path) / "motion_energy_metadata.json"
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def extract_motion_energy_features(args):
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    device = torch.device(requested_device)
    kinematics = G1TorchKinematics(
        args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
    ).to(device)
    kinematics.eval()
    keypoint_indices = _resolve_keypoint_indices(kinematics)

    p05, p95, train_peak_count = collect_train_peak_stats(
        args.data_path,
        kinematics,
        keypoint_indices,
        args.batch_size,
        device,
        args.window_radius,
        args.energy_peak_mode,
    )
    split_results = {}
    for split in args.splits:
        split_results[split] = write_split_features(
            args.data_path,
            split,
            kinematics,
            keypoint_indices,
            args.batch_size,
            device,
            args.window_radius,
            args.energy_peak_mode,
            args.envelope_sigma,
            p05,
            p95,
            skip_completed=not args.force,
        )
    metadata = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(Path(args.data_path).resolve()),
        "feature_dir": FEATURE_DIR_NAME,
        "fps": FPS,
        "target_frames": TARGET_FRAMES,
        "source_beat_feature": "gaussian_beat_feats",
        "energy_peak_mode": args.energy_peak_mode,
        "window_radius": args.window_radius,
        "envelope_sigma": args.envelope_sigma,
        "normalization": {
            "source_split": "train",
            "p05": p05,
            "p95": p95,
            "train_peak_count": train_peak_count,
        },
        "keypoint_weights": KEYPOINT_WEIGHTS,
        "splits": split_results,
    }
    write_metadata(args.data_path, metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--window_radius", type=int, default=DEFAULT_WINDOW_RADIUS)
    parser.add_argument("--envelope_sigma", type=float, default=DEFAULT_ENVELOPE_SIGMA)
    parser.add_argument("--energy_peak_mode", choices=("max", "p75"), default="max")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.window_radius < 0:
        raise ValueError("--window_radius must be non-negative")
    if args.envelope_sigma <= 0:
        raise ValueError("--envelope_sigma must be positive")
    return args


if __name__ == "__main__":
    extract_motion_energy_features(parse_args())
