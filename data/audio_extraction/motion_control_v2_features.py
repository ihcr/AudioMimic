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
from rotation_transforms import quaternion_to_matrix


CACHE_VERSION = "motion_control_v2"
LOCAL_CACHE_VERSION = "motion_control_v3_root_local"
SUPPORT_CACHE_VERSION = "motion_control_v4_support"
FPS = 30
DEFAULT_WINDOW_RADIUS = 6
DEFAULT_INTENSITY_SIGMA = 5.0
DEFAULT_BEATNESS_SPEED_SIGMA = 5.0
DEFAULT_BEATNESS_ENVELOPE_SIGMA = 3.0
DEFAULT_SHOULDER_INNER = 3
DEFAULT_SHOULDER_OUTER = 6
DEFAULT_CONTACT_HEIGHT_MARGIN = 0.03
DEFAULT_CONTACT_VERTICAL_SPEED_THRESHOLD = 0.2
DEFAULT_NEAR_SUPPORT_MARGIN = 0.08
FEATURE_DIR_NAME = "motion_control_v2_feats"
METADATA_NAME = "motion_control_v2_metadata.json"
LOCAL_FEATURE_DIR_NAME = "motion_control_v3_local_feats"
LOCAL_METADATA_NAME = "motion_control_v3_local_metadata.json"
SUPPORT_FEATURE_DIR_NAME = "motion_control_v4_support_feats"
SUPPORT_METADATA_NAME = "motion_control_v4_support_metadata.json"
WORLD_FRAME = "world"
ROOT_LOCAL_FRAME = "root_local"
KEYPOINT_WEIGHTS = {
    "left_wrist_yaw_link": 0.35,
    "right_wrist_yaw_link": 0.35,
    "left_ankle_roll_link": 0.10,
    "right_ankle_roll_link": 0.10,
    "torso_link": 0.10,
}
BODY_SUPPORT_KEYPOINT_WEIGHTS = {
    "torso_link": 0.40,
    "left_ankle_roll_link": 0.30,
    "right_ankle_roll_link": 0.30,
}
UPPER_KEYPOINT_WEIGHTS = {
    "left_wrist_yaw_link": 0.50,
    "right_wrist_yaw_link": 0.50,
}


def _split_dir(data_path, split):
    return Path(data_path) / split


def _motion_paths(data_path, split):
    return sorted((_split_dir(data_path, split) / "motions_sliced").glob("*.pkl"))


def _gaussian_beat_path(data_path, split, stem):
    return _split_dir(data_path, split) / "gaussian_beat_feats" / f"{stem}.npy"


def _feature_path(data_path, split, stem, feature_dir_name=FEATURE_DIR_NAME):
    return _split_dir(data_path, split) / feature_dir_name / f"{stem}.npz"


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


def _resolve_keypoint_indices(kinematics, keypoint_weights=None):
    keypoint_weights = KEYPOINT_WEIGHTS if keypoint_weights is None else keypoint_weights
    missing = [name for name in keypoint_weights if name not in kinematics.keypoint_names]
    if missing:
        raise ValueError(f"G1 keypoints missing from kinematics model: {missing}")
    return [kinematics.keypoint_names.index(name) for name in keypoint_weights]


@torch.inference_mode()
def _weighted_fk_speed_batch(
    kinematics,
    root_pos,
    root_rot,
    dof_pos,
    keypoint_indices,
    device,
    coordinate_frame=WORLD_FRAME,
    keypoint_weights=None,
):
    keypoint_weights = KEYPOINT_WEIGHTS if keypoint_weights is None else keypoint_weights
    root_pos = torch.from_numpy(root_pos).to(device=device, dtype=torch.float32)
    root_rot = torch.from_numpy(root_rot).to(device=device, dtype=torch.float32)
    dof_pos = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32)
    result = kinematics(root_pos, root_rot, dof_pos)
    keypoints = result["keypoints"].index_select(
        -2,
        torch.tensor(keypoint_indices, device=device, dtype=torch.long),
    )
    if coordinate_frame == ROOT_LOCAL_FRAME:
        root_rot_wxyz = kinematics._root_quaternion_wxyz(root_rot)
        root_rotation = quaternion_to_matrix(root_rot_wxyz)
        centered = keypoints - root_pos.unsqueeze(-2)
        keypoints = torch.matmul(
            root_rotation.transpose(-1, -2).unsqueeze(-3),
            centered.unsqueeze(-1),
        ).squeeze(-1)
    elif coordinate_frame != WORLD_FRAME:
        raise ValueError(f"Unsupported coordinate_frame: {coordinate_frame}")
    weights = torch.tensor(
        [keypoint_weights[name] for name in keypoint_weights],
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
    coordinate_frame=WORLD_FRAME,
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


def _support_feature_file_is_valid(path):
    try:
        with np.load(path) as data:
            required = {
                "body_intensity_envelope": (TARGET_FRAMES, 1),
                "support_beatness_envelope": (TARGET_FRAMES, 1),
                "upper_beatness_envelope": (TARGET_FRAMES, 1),
                "support_contact": (TARGET_FRAMES, 2),
                "body_weighted_fk_speed": (TARGET_FRAMES,),
                "support_weighted_fk_speed": (TARGET_FRAMES,),
                "upper_weighted_fk_speed": (TARGET_FRAMES,),
                "lowest_foot_heights": (TARGET_FRAMES, 2),
            }
            for key, shape in required.items():
                if data[key].shape != shape or not np.isfinite(data[key]).all():
                    return False
            return (
                data["audio_beat_frames"].ndim == 1
                and data["body_intensity_peaks"].ndim == 1
                and data["support_beatness_peaks"].ndim == 1
                and data["upper_beatness_peaks"].ndim == 1
                and np.isfinite(data["body_intensity_peaks"]).all()
                and np.isfinite(data["support_beatness_peaks"]).all()
                and np.isfinite(data["upper_beatness_peaks"]).all()
            )
    except Exception:
        return False


def _support_contact_from_feet(
    feet,
    contact_height_margin=DEFAULT_CONTACT_HEIGHT_MARGIN,
    contact_vertical_speed_threshold=DEFAULT_CONTACT_VERTICAL_SPEED_THRESHOLD,
    near_support_margin=DEFAULT_NEAR_SUPPORT_MARGIN,
):
    feet = np.asarray(feet, dtype=np.float32)
    heights = feet[:, :, 2]
    ground = float(np.percentile(heights, 1.0))
    velocity = np.zeros_like(feet, dtype=np.float32)
    if feet.shape[0] > 1:
        velocity[1:] = (feet[1:] - feet[:-1]) * float(FPS)
    vertical_speed = np.abs(velocity[:, :, 2])
    contact = (
        (heights <= ground + float(contact_height_margin))
        & (vertical_speed < float(contact_vertical_speed_threshold))
    ).astype(np.float32)
    near_support = (heights <= ground + float(near_support_margin)).astype(np.float32)
    return {
        "support_contact": contact.astype(np.float32),
        "near_support": near_support.astype(np.float32),
        "lowest_foot_heights": heights.astype(np.float32),
        "ground": np.asarray(ground, dtype=np.float32),
    }


def _beat_support_gate(near_support, beat_frames, window_radius):
    near_support = np.asarray(near_support, dtype=np.float32)
    any_near = np.max(near_support, axis=1)
    gates = []
    for beat_frame in beat_frames:
        start = max(int(beat_frame) - window_radius, 0)
        end = min(int(beat_frame) + window_radius + 1, any_near.shape[0])
        gates.append(float(np.max(any_near[start:end])) if start < end else 0.0)
    return np.asarray(gates, dtype=np.float32)


@torch.inference_mode()
def _support_control_batch(
    kinematics,
    root_pos,
    root_rot,
    dof_pos,
    body_indices,
    upper_indices,
    device,
    coordinate_frame=ROOT_LOCAL_FRAME,
):
    root_pos_tensor = torch.from_numpy(root_pos).to(device=device, dtype=torch.float32)
    root_rot_tensor = torch.from_numpy(root_rot).to(device=device, dtype=torch.float32)
    dof_pos_tensor = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32)
    result = kinematics(root_pos_tensor, root_rot_tensor, dof_pos_tensor)

    def selected_keypoints(indices):
        keypoints = result["keypoints"].index_select(
            -2,
            torch.tensor(indices, device=device, dtype=torch.long),
        )
        if coordinate_frame == ROOT_LOCAL_FRAME:
            root_rot_wxyz = kinematics._root_quaternion_wxyz(root_rot_tensor)
            root_rotation = quaternion_to_matrix(root_rot_wxyz)
            centered = keypoints - root_pos_tensor.unsqueeze(-2)
            keypoints = torch.matmul(
                root_rotation.transpose(-1, -2).unsqueeze(-3),
                centered.unsqueeze(-1),
            ).squeeze(-1)
        elif coordinate_frame != WORLD_FRAME:
            raise ValueError(f"Unsupported coordinate_frame: {coordinate_frame}")
        return keypoints

    def weighted_speed(indices, weights):
        keypoints = selected_keypoints(indices)
        frame_speed = keypoints.new_zeros(keypoints.shape[:2] + (keypoints.shape[-2],))
        frame_speed[:, 1:] = torch.linalg.vector_norm(
            keypoints[:, 1:] - keypoints[:, :-1],
            dim=-1,
        ) * FPS
        weight_tensor = torch.tensor(
            [weights[name] for name in weights],
            device=device,
            dtype=keypoints.dtype,
        )
        return (frame_speed * weight_tensor.view(1, 1, -1)).sum(dim=-1)

    body_speed = weighted_speed(body_indices, BODY_SUPPORT_KEYPOINT_WEIGHTS)
    upper_speed = weighted_speed(upper_indices, UPPER_KEYPOINT_WEIGHTS)
    feet = result["feet"]
    return {
        "body_speed": body_speed.detach().cpu().numpy().astype(np.float32),
        "support_speed": body_speed.detach().cpu().numpy().astype(np.float32),
        "upper_speed": upper_speed.detach().cpu().numpy().astype(np.float32),
        "feet": feet.detach().cpu().numpy().astype(np.float32),
    }


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
    coordinate_frame=WORLD_FRAME,
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
            coordinate_frame=coordinate_frame,
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
    feature_dir_name=FEATURE_DIR_NAME,
    coordinate_frame=WORLD_FRAME,
    skip_completed=True,
):
    split_paths = _motion_paths(data_path, split)
    if not split_paths:
        raise ValueError(f"No {split} motions found under {data_path}")
    output_dir = _split_dir(data_path, split) / feature_dir_name
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
            if not skip_completed
            or not _feature_file_is_valid(
                _feature_path(data_path, split, path.stem, feature_dir_name)
            )
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
            coordinate_frame=coordinate_frame,
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
            save_path = _feature_path(data_path, split, path.stem, feature_dir_name)
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


def collect_support_train_stats(
    data_path,
    kinematics,
    body_indices,
    upper_indices,
    batch_size,
    device,
    window_radius,
    beatness_speed_sigma,
    shoulder_inner,
    shoulder_outer,
    contact_height_margin,
    contact_vertical_speed_threshold,
    near_support_margin,
    coordinate_frame=ROOT_LOCAL_FRAME,
):
    train_paths = _motion_paths(data_path, "train")
    if not train_paths:
        raise ValueError(f"No train motions found under {data_path}")
    all_body_intensity = []
    all_support_beatness = []
    all_upper_beatness = []
    for batch_paths in tqdm(
        list(_iter_batches(train_paths, batch_size)),
        desc="Collect train V6a body/support stats",
        unit="batch",
    ):
        root_pos, root_rot, dof_pos = _load_motion_batch(batch_paths)
        controls = _support_control_batch(
            kinematics,
            root_pos,
            root_rot,
            dof_pos,
            body_indices,
            upper_indices,
            device,
            coordinate_frame=coordinate_frame,
        )
        for path, body_speed, support_speed, upper_speed, feet in zip(
            batch_paths,
            controls["body_speed"],
            controls["support_speed"],
            controls["upper_speed"],
            controls["feet"],
        ):
            beat_frames = _load_audio_beat_frames(_gaussian_beat_path(data_path, "train", path.stem))
            contact_payload = _support_contact_from_feet(
                feet,
                contact_height_margin=contact_height_margin,
                contact_vertical_speed_threshold=contact_vertical_speed_threshold,
                near_support_margin=near_support_margin,
            )
            body_intensity = intensity_values_for_frames(
                body_speed,
                beat_frames,
                window_radius,
            )
            smoothed_support = gaussian_filter1d(
                support_speed,
                sigma=beatness_speed_sigma,
            ).astype(np.float32)
            raw_support_beatness = beatness_values_for_frames(
                smoothed_support,
                beat_frames,
                window_radius,
                shoulder_inner,
                shoulder_outer,
            )
            support_gate = _beat_support_gate(
                contact_payload["near_support"],
                beat_frames,
                window_radius,
            )
            support_beatness = raw_support_beatness * support_gate
            smoothed_upper = gaussian_filter1d(
                upper_speed,
                sigma=beatness_speed_sigma,
            ).astype(np.float32)
            upper_beatness = beatness_values_for_frames(
                smoothed_upper,
                beat_frames,
                window_radius,
                shoulder_inner,
                shoulder_outer,
            )
            if body_intensity.size:
                all_body_intensity.append(body_intensity)
            if support_beatness.size:
                all_support_beatness.append(support_beatness)
            if upper_beatness.size:
                all_upper_beatness.append(upper_beatness)
    if not all_body_intensity or not all_support_beatness or not all_upper_beatness:
        raise ValueError("No audio beat frames found in the train GaussianBeat cache.")

    def stats_for(name, values):
        values = np.concatenate(values, axis=0)
        p05, p95 = np.percentile(values, [5, 95]).astype(np.float32)
        if not np.isfinite([p05, p95]).all() or p95 <= p05:
            raise ValueError(f"Invalid {name} normalization stats: p05={p05}, p95={p95}")
        return {
            "p05": float(p05),
            "p95": float(p95),
            "train_peak_count": int(values.size),
        }

    return {
        "body_intensity": stats_for("body_intensity", all_body_intensity),
        "support_beatness": stats_for("support_beatness", all_support_beatness),
        "upper_beatness": stats_for("upper_beatness", all_upper_beatness),
    }


def write_support_split_features(
    data_path,
    split,
    kinematics,
    body_indices,
    upper_indices,
    batch_size,
    device,
    window_radius,
    intensity_sigma,
    beatness_speed_sigma,
    beatness_envelope_sigma,
    shoulder_inner,
    shoulder_outer,
    normalization,
    contact_height_margin,
    contact_vertical_speed_threshold,
    near_support_margin,
    feature_dir_name=SUPPORT_FEATURE_DIR_NAME,
    coordinate_frame=ROOT_LOCAL_FRAME,
    skip_completed=True,
):
    split_paths = _motion_paths(data_path, split)
    if not split_paths:
        raise ValueError(f"No {split} motions found under {data_path}")
    output_dir = _split_dir(data_path, split) / feature_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for batch_paths in tqdm(
        list(_iter_batches(split_paths, batch_size)),
        desc=f"Write {split} V6a body/support features",
        unit="batch",
    ):
        pending = [
            path
            for path in batch_paths
            if not skip_completed
            or not _support_feature_file_is_valid(
                _feature_path(data_path, split, path.stem, feature_dir_name)
            )
        ]
        skipped += len(batch_paths) - len(pending)
        if not pending:
            continue
        root_pos, root_rot, dof_pos = _load_motion_batch(pending)
        controls = _support_control_batch(
            kinematics,
            root_pos,
            root_rot,
            dof_pos,
            body_indices,
            upper_indices,
            device,
            coordinate_frame=coordinate_frame,
        )
        for path, body_speed, support_speed, upper_speed, feet in zip(
            pending,
            controls["body_speed"],
            controls["support_speed"],
            controls["upper_speed"],
            controls["feet"],
        ):
            beat_frames = _load_audio_beat_frames(_gaussian_beat_path(data_path, split, path.stem))
            contact_payload = _support_contact_from_feet(
                feet,
                contact_height_margin=contact_height_margin,
                contact_vertical_speed_threshold=contact_vertical_speed_threshold,
                near_support_margin=near_support_margin,
            )
            smoothed_support = gaussian_filter1d(
                support_speed,
                sigma=beatness_speed_sigma,
            ).astype(np.float32)
            smoothed_upper = gaussian_filter1d(
                upper_speed,
                sigma=beatness_speed_sigma,
            ).astype(np.float32)
            raw_body_intensity = intensity_values_for_frames(
                body_speed,
                beat_frames,
                window_radius,
            )
            raw_support_beatness_ungated = beatness_values_for_frames(
                smoothed_support,
                beat_frames,
                window_radius,
                shoulder_inner,
                shoulder_outer,
            )
            support_gate = _beat_support_gate(
                contact_payload["near_support"],
                beat_frames,
                window_radius,
            )
            raw_support_beatness = raw_support_beatness_ungated * support_gate
            raw_upper_beatness = beatness_values_for_frames(
                smoothed_upper,
                beat_frames,
                window_radius,
                shoulder_inner,
                shoulder_outer,
            )
            body_intensity_peaks = _normalize(
                raw_body_intensity,
                normalization["body_intensity"]["p05"],
                normalization["body_intensity"]["p95"],
            )
            support_beatness_peaks = _normalize(
                raw_support_beatness,
                normalization["support_beatness"]["p05"],
                normalization["support_beatness"]["p95"],
            )
            upper_beatness_peaks = _normalize(
                raw_upper_beatness,
                normalization["upper_beatness"]["p05"],
                normalization["upper_beatness"]["p95"],
            )
            body_intensity_envelope = _build_envelope(
                beat_frames,
                body_intensity_peaks,
                intensity_sigma,
            )
            support_beatness_envelope = _build_envelope(
                beat_frames,
                support_beatness_peaks,
                beatness_envelope_sigma,
            )
            upper_beatness_envelope = _build_envelope(
                beat_frames,
                upper_beatness_peaks,
                beatness_envelope_sigma,
            )
            save_path = _feature_path(data_path, split, path.stem, feature_dir_name)
            tmp_path = save_path.with_name(f"{save_path.name}.{os.getpid()}.tmp")
            with open(tmp_path, "wb") as handle:
                np.savez_compressed(
                    handle,
                    body_intensity_envelope=body_intensity_envelope.astype(np.float32),
                    support_beatness_envelope=support_beatness_envelope.astype(np.float32),
                    upper_beatness_envelope=upper_beatness_envelope.astype(np.float32),
                    support_contact=contact_payload["support_contact"].astype(np.float32),
                    body_weighted_fk_speed=body_speed.astype(np.float32),
                    support_weighted_fk_speed=support_speed.astype(np.float32),
                    upper_weighted_fk_speed=upper_speed.astype(np.float32),
                    smoothed_support_weighted_fk_speed=smoothed_support.astype(np.float32),
                    smoothed_upper_weighted_fk_speed=smoothed_upper.astype(np.float32),
                    audio_beat_frames=beat_frames.astype(np.int64),
                    body_intensity_peaks=body_intensity_peaks.astype(np.float32),
                    body_intensity_peaks_raw=raw_body_intensity.astype(np.float32),
                    support_beatness_peaks=support_beatness_peaks.astype(np.float32),
                    support_beatness_peaks_raw=raw_support_beatness.astype(np.float32),
                    support_beatness_peaks_raw_ungated=raw_support_beatness_ungated.astype(np.float32),
                    support_gate=support_gate.astype(np.float32),
                    upper_beatness_peaks=upper_beatness_peaks.astype(np.float32),
                    upper_beatness_peaks_raw=raw_upper_beatness.astype(np.float32),
                    lowest_foot_heights=contact_payload["lowest_foot_heights"].astype(np.float32),
                    near_support=contact_payload["near_support"].astype(np.float32),
                    ground=contact_payload["ground"].astype(np.float32),
                )
            tmp_path.replace(save_path)
            written += 1
    return {"written": written, "skipped": skipped, "total": len(split_paths)}


def _write_metadata(data_path, metadata, metadata_name=METADATA_NAME):
    path = Path(data_path) / metadata_name
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
        coordinate_frame=args.coordinate_frame,
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
            feature_dir_name=args.feature_dir_name,
            coordinate_frame=args.coordinate_frame,
            skip_completed=not args.force,
        )
    metadata = {
        "cache_version": args.cache_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_dir": args.feature_dir_name,
        "metadata_name": args.metadata_name,
        "coordinate_frame": args.coordinate_frame,
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
    _write_metadata(data_path, metadata, metadata_name=args.metadata_name)
    return metadata


def extract_motion_control_v4_support_features(args):
    data_path = Path(args.data_path)
    device = torch.device(args.device)
    kinematics = G1TorchKinematics(
        args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
    ).to(device)
    body_indices = _resolve_keypoint_indices(kinematics, BODY_SUPPORT_KEYPOINT_WEIGHTS)
    upper_indices = _resolve_keypoint_indices(kinematics, UPPER_KEYPOINT_WEIGHTS)
    normalization = collect_support_train_stats(
        data_path,
        kinematics,
        body_indices,
        upper_indices,
        args.batch_size,
        device,
        args.window_radius,
        args.beatness_speed_sigma,
        args.shoulder_inner,
        args.shoulder_outer,
        args.contact_height_margin,
        args.contact_vertical_speed_threshold,
        args.near_support_margin,
        coordinate_frame=args.coordinate_frame,
    )
    split_stats = {}
    for split in ("train", "test"):
        split_stats[split] = write_support_split_features(
            data_path,
            split,
            kinematics,
            body_indices,
            upper_indices,
            args.batch_size,
            device,
            args.window_radius,
            args.intensity_sigma,
            args.beatness_speed_sigma,
            args.beatness_envelope_sigma,
            args.shoulder_inner,
            args.shoulder_outer,
            normalization,
            args.contact_height_margin,
            args.contact_vertical_speed_threshold,
            args.near_support_margin,
            feature_dir_name=args.feature_dir_name,
            coordinate_frame=args.coordinate_frame,
            skip_completed=not args.force,
        )
    metadata = {
        "cache_version": args.cache_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_dir": args.feature_dir_name,
        "metadata_name": args.metadata_name,
        "coordinate_frame": args.coordinate_frame,
        "source_beat_feature": "gaussian_beat_feats",
        "target_frames": TARGET_FRAMES,
        "fps": FPS,
        "window_radius": args.window_radius,
        "intensity_sigma": args.intensity_sigma,
        "beatness_speed_sigma": args.beatness_speed_sigma,
        "beatness_envelope_sigma": args.beatness_envelope_sigma,
        "shoulder_inner": args.shoulder_inner,
        "shoulder_outer": args.shoulder_outer,
        "contact_height_margin": args.contact_height_margin,
        "contact_vertical_speed_threshold": args.contact_vertical_speed_threshold,
        "near_support_margin": args.near_support_margin,
        "body_support_keypoint_weights": BODY_SUPPORT_KEYPOINT_WEIGHTS,
        "upper_keypoint_weights": UPPER_KEYPOINT_WEIGHTS,
        "normalization": normalization,
        "splits": split_stats,
    }
    _write_metadata(data_path, metadata, metadata_name=args.metadata_name)
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
    parser.add_argument(
        "--coordinate_frame",
        choices=(WORLD_FRAME, ROOT_LOCAL_FRAME),
        default=WORLD_FRAME,
    )
    parser.add_argument("--support_v6a", action="store_true")
    parser.add_argument("--contact_height_margin", type=float, default=DEFAULT_CONTACT_HEIGHT_MARGIN)
    parser.add_argument(
        "--contact_vertical_speed_threshold",
        type=float,
        default=DEFAULT_CONTACT_VERTICAL_SPEED_THRESHOLD,
    )
    parser.add_argument("--near_support_margin", type=float, default=DEFAULT_NEAR_SUPPORT_MARGIN)
    parser.add_argument("--feature_dir_name", type=str, default=FEATURE_DIR_NAME)
    parser.add_argument("--metadata_name", type=str, default=METADATA_NAME)
    parser.add_argument("--cache_version", type=str, default=CACHE_VERSION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.support_v6a:
        if args.feature_dir_name == FEATURE_DIR_NAME:
            args.feature_dir_name = SUPPORT_FEATURE_DIR_NAME
        if args.metadata_name == METADATA_NAME:
            args.metadata_name = SUPPORT_METADATA_NAME
        if args.cache_version == CACHE_VERSION:
            args.cache_version = SUPPORT_CACHE_VERSION
        if args.coordinate_frame == WORLD_FRAME:
            args.coordinate_frame = ROOT_LOCAL_FRAME
    if args.window_radius < 1:
        raise ValueError("--window_radius must be positive")
    if args.shoulder_inner < 1 or args.shoulder_outer < args.shoulder_inner:
        raise ValueError("--shoulder_outer must be >= --shoulder_inner >= 1")
    if args.contact_height_margin <= 0 or args.near_support_margin <= 0:
        raise ValueError("contact and near-support margins must be positive")
    if args.contact_vertical_speed_threshold <= 0:
        raise ValueError("--contact_vertical_speed_threshold must be positive")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.support_v6a:
        extract_motion_control_v4_support_features(parsed_args)
    else:
        extract_motion_control_v2_features(parsed_args)
