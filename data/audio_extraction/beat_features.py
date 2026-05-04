import glob
import os
import pickle
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import torch
from tqdm import tqdm

from eval.g1_kinematics import forward_g1_kinematics

SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

SMPL_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]

_HAS_CHILDREN = [
    any(parent == joint_idx for parent in SMPL_PARENTS)
    for joint_idx in range(len(SMPL_PARENTS))
]

if not hasattr(scipy.signal, "hann") and hasattr(scipy.signal, "windows"):
    scipy.signal.hann = scipy.signal.windows.hann


def _sanitize_beat_frames(beat_frames, seq_len):
    beat_frames = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
    if seq_len <= 0:
        return np.zeros(0, dtype=np.int64)
    if beat_frames.size == 0:
        return beat_frames
    beat_frames = np.clip(beat_frames, 0, seq_len - 1)
    return np.unique(beat_frames)


def nearest_beat_distance(beat_frames, seq_len=150):
    beat_frames = _sanitize_beat_frames(beat_frames, seq_len)
    if seq_len <= 0:
        return np.zeros(0, dtype=np.int64)
    if beat_frames.size == 0:
        return np.full(seq_len, seq_len, dtype=np.int64)

    frame_ids = np.arange(seq_len, dtype=np.int64)[:, None]
    distances = np.abs(frame_ids - beat_frames[None, :])
    return distances.min(axis=1).astype(np.int64)


def local_beat_spacing(beat_frames, seq_len=150):
    beat_frames = _sanitize_beat_frames(beat_frames, seq_len)
    if seq_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if beat_frames.size < 2:
        return np.ones(seq_len, dtype=np.float32)

    spacing = np.empty(seq_len, dtype=np.float32)
    intervals = np.diff(beat_frames).astype(np.float32)
    spacing[:] = intervals[0]

    for idx in range(len(intervals)):
        start = beat_frames[idx]
        end = beat_frames[idx + 1]
        spacing[start:end] = max(intervals[idx], 1.0)

    spacing[beat_frames[-1] :] = max(intervals[-1], 1.0)
    return np.clip(spacing, 1.0, None).astype(np.float32)


def _beat_mask(beat_frames, seq_len):
    mask = np.zeros(seq_len, dtype=np.float32)
    if beat_frames.size:
        mask[beat_frames] = 1.0
    return mask


def _axis_angle_to_quaternion(axis_angle):
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angle = angle * 0.5
    scale = torch.where(
        angle > 1e-8,
        torch.sin(half_angle) / angle,
        0.5 - (angle * angle) / 48.0,
    )
    xyz = axis_angle * scale
    w = torch.cos(half_angle)
    return torch.cat((w, xyz), dim=-1)


def _quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quaternion_to_axis_angle(quaternion):
    quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    xyz = quaternion[..., 1:]
    xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    half_angle = torch.atan2(xyz_norm, quaternion[..., :1])
    angle = 2.0 * half_angle
    scale = torch.where(xyz_norm > 1e-8, angle / xyz_norm, torch.full_like(xyz_norm, 2.0))
    return xyz * scale


def _quaternion_apply(quaternion, points):
    q_xyz = quaternion[..., 1:]
    uv = torch.cross(q_xyz, points, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return points + 2.0 * (quaternion[..., :1] * uv + uuv)


def _rotate_root_to_z_up(local_q, root_pos):
    root_q = local_q[:, :1, :]
    root_q_quat = _axis_angle_to_quaternion(root_q)
    rotation = torch.tensor(
        [0.7071068, 0.7071068, 0.0, 0.0],
        dtype=local_q.dtype,
        device=local_q.device,
    ).view(1, 1, 4)
    root_q = _quaternion_to_axis_angle(_quaternion_multiply(rotation, root_q_quat))
    local_q = local_q.clone()
    local_q[:, :1, :] = root_q

    root_pos = root_pos.clone()
    root_pos = torch.stack(
        (root_pos[:, 0], -root_pos[:, 2], root_pos[:, 1]), dim=-1
    )
    return local_q, root_pos


def _forward_kinematics(local_q, root_pos):
    offsets = torch.tensor(
        SMPL_OFFSETS, dtype=local_q.dtype, device=local_q.device
    ).view(1, len(SMPL_OFFSETS), 3)
    rotations = _axis_angle_to_quaternion(local_q)

    positions_world = []
    rotations_world = []

    for joint_idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx == -1:
            positions_world.append(root_pos)
            rotations_world.append(rotations[:, 0])
            continue

        joint_offset = offsets[:, joint_idx].expand(local_q.shape[0], -1)
        joint_position = _quaternion_apply(rotations_world[parent_idx], joint_offset)
        joint_position = joint_position + positions_world[parent_idx]
        positions_world.append(joint_position)

        if _HAS_CHILDREN[joint_idx]:
            rotations_world.append(
                _quaternion_multiply(rotations_world[parent_idx], rotations[:, joint_idx])
            )
        else:
            rotations_world.append(None)

    return torch.stack(positions_world, dim=1)


def _smooth_curve(curve, sigma=1.0, radius=2):
    if curve.size == 0:
        return curve
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _detect_local_minima(curve, min_gap=5):
    if curve.size == 0:
        return np.zeros(0, dtype=np.int64)

    candidates = []
    for idx in range(1, len(curve) - 1):
        if curve[idx] <= curve[idx - 1] and curve[idx] < curve[idx + 1]:
            candidates.append(idx)

    if not candidates:
        return np.array([int(np.argmin(curve))], dtype=np.int64)

    selected = []
    for idx in candidates:
        if not selected or idx - selected[-1] >= min_gap:
            selected.append(idx)
            continue
        if curve[idx] < curve[selected[-1]]:
            selected[-1] = idx

    return np.asarray(selected, dtype=np.int64)


def load_audio_beat_frames(wav_path, fps=30, seq_len=None):
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    _, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    beat_frames = np.rint(np.asarray(beat_times) * fps).astype(np.int64)
    if seq_len is None:
        return np.unique(np.clip(beat_frames, 0, None))
    return _sanitize_beat_frames(beat_frames, seq_len)


def mean_joint_speed_curve(full_pose):
    full_pose = np.asarray(full_pose, dtype=np.float32)
    if full_pose.ndim != 3 or full_pose.shape[-1] != 3:
        raise ValueError("full_pose must have shape [T, J, 3]")
    if full_pose.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if full_pose.shape[0] == 1:
        return np.zeros(1, dtype=np.float32)

    velocity = np.linalg.norm(full_pose[1:] - full_pose[:-1], axis=-1).mean(axis=-1)
    return np.concatenate((velocity[:1], velocity), axis=0).astype(np.float32)


def detect_motion_beat_frames_from_pose(full_pose, min_gap=5):
    full_pose = np.asarray(full_pose, dtype=np.float32)
    if full_pose.ndim != 3 or full_pose.shape[-1] != 3:
        raise ValueError("full_pose must have shape [T, J, 3]")
    if full_pose.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    velocity = mean_joint_speed_curve(full_pose)
    smooth_velocity = _smooth_curve(velocity.astype(np.float32))
    beat_frames = _detect_local_minima(smooth_velocity, min_gap=min_gap)
    return _sanitize_beat_frames(beat_frames, full_pose.shape[0])


def _zscore_curve(curve):
    curve = np.asarray(curve, dtype=np.float32)
    if curve.size == 0:
        return curve
    std = float(curve.std())
    if std < 1e-8:
        return np.zeros_like(curve)
    return (curve - float(curve.mean())) / std


def detect_g1_motion_beat_frames(root_pos, root_rot, dof_pos, fps=30, min_gap=5):
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot = np.asarray(root_rot, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    if root_pos.shape[0] < 3:
        return np.zeros(0, dtype=np.int64)

    root_velocity = np.diff(root_pos, axis=0) * float(fps)
    root_linear = np.linalg.norm(root_velocity[:, (0, 1)], axis=-1)
    root_linear = np.concatenate((root_linear[:1], root_linear), axis=0)

    if root_rot.shape[0] < 2:
        root_angular = np.zeros(root_pos.shape[0], dtype=np.float32)
    else:
        quat_norm = np.linalg.norm(root_rot, axis=-1, keepdims=True)
        root_rot = root_rot / np.maximum(quat_norm, 1e-8)
        dots = np.sum(root_rot[1:] * root_rot[:-1], axis=-1)
        dots = np.clip(np.abs(dots), -1.0, 1.0)
        root_angular = (2.0 * np.arccos(dots) * float(fps)).astype(np.float32)
        root_angular = np.concatenate((root_angular[:1], root_angular), axis=0)

    joint_velocity = np.diff(dof_pos, axis=0) * float(fps)
    joint_speed = np.mean(np.abs(joint_velocity), axis=-1)
    joint_speed = np.concatenate((joint_speed[:1], joint_speed), axis=0)

    speed = _zscore_curve(root_linear) + _zscore_curve(root_angular) + _zscore_curve(joint_speed)
    smooth_speed = _smooth_curve(speed.astype(np.float32))
    beat_frames = _detect_local_minima(smooth_speed, min_gap=min_gap)
    return _sanitize_beat_frames(beat_frames, root_pos.shape[0])


def detect_g1_fk_motion_beat_frames(keypoints, fps=30, min_gap=5):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
        raise ValueError("G1 FK keypoints must have shape [T, K, 3]")
    if keypoints.shape[0] < 3:
        return np.zeros(0, dtype=np.int64)
    speed = mean_joint_speed_curve(keypoints) * float(fps)
    smooth_speed = _smooth_curve(speed.astype(np.float32))
    beat_frames = _detect_local_minima(smooth_speed, min_gap=min_gap)
    return _sanitize_beat_frames(beat_frames, keypoints.shape[0])


def extract_audio_beats_librosa(wav_path, fps=30, seq_len=150):
    beat_frames = load_audio_beat_frames(wav_path, fps=fps, seq_len=seq_len)
    beat_mask = _beat_mask(beat_frames, seq_len)
    beat_dist = nearest_beat_distance(beat_frames, seq_len)
    beat_spacing = local_beat_spacing(beat_frames, seq_len)
    return beat_frames, beat_mask, beat_dist, beat_spacing


def extract_motion_beats_from_motion_pkl(
    motion_pkl_path,
    fps=30,
    seq_len=150,
    g1_motion_beat_source="proxy",
    g1_fk_model_path=None,
    g1_root_quat_order="wxyz",
):
    with open(motion_pkl_path, "rb") as handle:
        motion = pickle.load(handle)

    if {"root_pos", "root_rot", "dof_pos"}.issubset(motion):
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)[:seq_len]
        root_rot = np.asarray(motion["root_rot"], dtype=np.float32)[:seq_len]
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)[:seq_len]
        if g1_motion_beat_source == "proxy":
            beat_frames = detect_g1_motion_beat_frames(
                root_pos,
                root_rot,
                dof_pos,
                fps=fps,
                min_gap=5,
            )
        elif g1_motion_beat_source == "fk":
            if g1_fk_model_path is None:
                raise ValueError("g1_fk_model_path is required for FK G1 beat extraction")
            fk_result = forward_g1_kinematics(
                {
                    "root_pos": root_pos,
                    "root_rot": root_rot,
                    "dof_pos": dof_pos,
                },
                model_path=g1_fk_model_path,
                root_quat_order=g1_root_quat_order,
            )
            beat_frames = detect_g1_fk_motion_beat_frames(
                fk_result["keypoints"],
                fps=fps,
                min_gap=5,
            )
        else:
            raise ValueError(f"Unsupported G1 motion beat source: {g1_motion_beat_source}")
        beat_mask = _beat_mask(beat_frames, len(root_pos))
        beat_dist = nearest_beat_distance(beat_frames, len(root_pos))
        beat_spacing = local_beat_spacing(beat_frames, len(root_pos))
        return beat_frames, beat_mask, beat_dist, beat_spacing

    pos = np.asarray(motion["pos"], dtype=np.float32)
    q = np.asarray(motion["q"], dtype=np.float32)

    raw_fps = 60
    stride = max(raw_fps // fps, 1)
    pos = pos[::stride]
    q = q[::stride]

    root_pos = torch.from_numpy(pos[:seq_len])
    local_q = torch.from_numpy(q[:seq_len]).reshape(-1, 24, 3)
    local_q, root_pos = _rotate_root_to_z_up(local_q, root_pos)
    joints = _forward_kinematics(local_q, root_pos).cpu().numpy()

    beat_frames = detect_motion_beat_frames_from_pose(joints, min_gap=5)

    beat_mask = _beat_mask(beat_frames, len(root_pos))
    beat_dist = nearest_beat_distance(beat_frames, len(root_pos))
    beat_spacing = local_beat_spacing(beat_frames, len(root_pos))
    return beat_frames, beat_mask, beat_dist, beat_spacing


def extract_folder(
    motion_dir,
    wav_dir,
    out_dir,
    fps=30,
    seq_len=150,
    g1_motion_beat_source="proxy",
    g1_fk_model_path=None,
    g1_root_quat_order="wxyz",
):
    motion_paths = sorted(glob.glob(os.path.join(motion_dir, "*.pkl")))
    wav_paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motion_map = {Path(path).stem: path for path in motion_paths}
    wav_map = {Path(path).stem: path for path in wav_paths}

    if set(motion_map) != set(wav_map):
        raise ValueError("motions_sliced and wavs_sliced must match by basename")

    clip_names = sorted(motion_map)
    for clip_name in tqdm(
        clip_names,
        total=len(clip_names),
        desc="Beat metadata",
        unit="clip",
    ):
        output_path = out_dir / f"{clip_name}.npz"
        if output_path.exists():
            continue
        motion_beats, motion_mask, motion_dist, motion_spacing = (
            extract_motion_beats_from_motion_pkl(
                motion_map[clip_name],
                fps=fps,
                seq_len=seq_len,
                g1_motion_beat_source=g1_motion_beat_source,
                g1_fk_model_path=g1_fk_model_path,
                g1_root_quat_order=g1_root_quat_order,
            )
        )
        audio_beats, audio_mask, audio_dist, audio_spacing = extract_audio_beats_librosa(
            wav_map[clip_name], fps=fps, seq_len=seq_len
        )

        np.savez(
            output_path,
            motion_beats=np.asarray(motion_beats, dtype=np.int64),
            motion_mask=np.asarray(motion_mask, dtype=np.float32),
            motion_dist=np.asarray(motion_dist, dtype=np.int64),
            motion_spacing=np.asarray(motion_spacing, dtype=np.float32),
            audio_beats=np.asarray(audio_beats, dtype=np.int64),
            audio_mask=np.asarray(audio_mask, dtype=np.float32),
            audio_dist=np.asarray(audio_dist, dtype=np.int64),
            audio_spacing=np.asarray(audio_spacing, dtype=np.float32),
        )
