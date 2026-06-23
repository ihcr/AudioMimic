import glob
import json
import pickle
import random
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from tqdm import tqdm

from eval.eval_bas_bap import (DEFAULT_BAP_TOLERANCE,
                               DEFAULT_BAS_SIGMA_SQUARED, BAS_DIRECTION,
                               ROBOPERFORM_BAS_DIRECTION, compute_bas_score,
                               compute_roboperform_bas_score,
                               greedy_match_count, load_audio_beat_frames)
from eval.g1_kinematics import forward_g1_kinematics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FPS = 30.0
ROOT_UP_AXIS = 2
ROOT_FLAT_AXES = (0, 1)
SMPL_ONLY_METRIC_NAMES = ("PFC", "Distg", "Distk", "Divk", "Divm")
BODY_RESPONSE_GROUPS = (
    ("Wrist", ("wrist",)),
    ("Foot", ("ankle", "foot")),
    ("Torso", ("pelvis", "torso")),
    ("FullBody", ()),
)
FAILURE_PANEL_SPECS = (
    ("wrist_jerk_highest", "G1WristJerkMean", True),
    ("foot_jerk_highest", "G1FootJerkMean", True),
    ("foot_high_lift_highest", "G1FootHighLiftRate", True),
    ("no_near_support_highest", "G1NoNearSupportRate", True),
    ("ground_penetration_highest", "G1GroundPenetration", True),
    ("beat_recall_lowest", "G1BeatRecall", False),
    ("unmatched_motion_beat_highest", "G1UnmatchedMotionBeatRate", True),
)


def _as_float_array(value, name, ndim=None):
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} expected {ndim} dimensions, got {array.ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize_quaternions(quat):
    quat = _as_float_array(quat, "root_rot", ndim=2)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return quat / norm


def load_g1_motion(path_or_payload):
    if isinstance(path_or_payload, (str, Path)):
        with open(path_or_payload, "rb") as handle:
            payload = pickle.load(handle)
        path = str(path_or_payload)
    else:
        payload = path_or_payload
        path = payload.get("path", "")

    root_pos = payload.get("root_pos", payload.get("pos"))
    root_rot = payload.get("root_rot")
    dof_pos = payload.get("dof_pos")
    if root_rot is None or dof_pos is None:
        q = _as_float_array(payload["q"], "q", ndim=2)
        if q.shape[-1] < 33:
            raise ValueError(f"G1 q expected at least 33 channels, got {q.shape[-1]}")
        root_rot = q[:, :4]
        dof_pos = q[:, 4:33]

    root_pos = _as_float_array(root_pos, "root_pos", ndim=2)
    root_rot = _normalize_quaternions(root_rot)
    dof_pos = _as_float_array(dof_pos, "dof_pos", ndim=2)
    if root_pos.shape[-1] != 3:
        raise ValueError(f"root_pos expected 3 channels, got {root_pos.shape[-1]}")
    if root_rot.shape[-1] != 4:
        raise ValueError(f"root_rot expected 4 channels, got {root_rot.shape[-1]}")
    if dof_pos.shape[-1] != 29:
        raise ValueError(f"dof_pos expected 29 channels, got {dof_pos.shape[-1]}")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError("G1 root_pos, root_rot, and dof_pos must have matching frames")

    return {
        "path": path,
        "fps": float(payload.get("fps", FPS)),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "audio_path": payload.get("audio_path", ""),
        "designated_beat_frames": payload.get("designated_beat_frames"),
        "beat_rep": payload.get("beat_rep", ""),
    }


def finite_mean(values, default=0.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(values.mean())


def finite_max(values, default=0.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(values.max())


def finite_min(values, default=0.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(values.min())


def finite_percentile(values, percentile, default=0.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, percentile))


def compute_root_derivatives(motion):
    root_pos = motion["root_pos"]
    fps = motion["fps"]
    if root_pos.shape[0] < 2:
        velocity = np.zeros((0, 3), dtype=np.float32)
    else:
        velocity = np.diff(root_pos, axis=0) * fps
    if velocity.shape[0] < 2:
        acceleration = np.zeros((0, 3), dtype=np.float32)
    else:
        acceleration = np.diff(velocity, axis=0) * fps
    if acceleration.shape[0] < 2:
        jerk = np.zeros((0, 3), dtype=np.float32)
    else:
        jerk = np.diff(acceleration, axis=0) * fps
    return velocity, acceleration, jerk


def compute_root_angular_speed(motion):
    root_rot = motion["root_rot"]
    if root_rot.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    dots = np.sum(root_rot[1:] * root_rot[:-1], axis=-1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return (2.0 * np.arccos(dots) * motion["fps"]).astype(np.float32)


def compute_root_up_z(motion, root_quat_order="xyzw"):
    root_rot = _normalize_quaternions(motion["root_rot"])
    if root_quat_order == "xyzw":
        x = root_rot[:, 0]
        y = root_rot[:, 1]
    elif root_quat_order == "wxyz":
        x = root_rot[:, 1]
        y = root_rot[:, 2]
    else:
        raise ValueError(f"Unsupported root_quat_order: {root_quat_order}")
    return (1.0 - 2.0 * (x * x + y * y)).astype(np.float32)


def compute_joint_derivatives(motion):
    dof_pos = motion["dof_pos"]
    fps = motion["fps"]
    if dof_pos.shape[0] < 2:
        velocity = np.zeros((0, dof_pos.shape[-1]), dtype=np.float32)
    else:
        velocity = np.diff(dof_pos, axis=0) * fps
    if velocity.shape[0] < 2:
        acceleration = np.zeros((0, dof_pos.shape[-1]), dtype=np.float32)
    else:
        acceleration = np.diff(velocity, axis=0) * fps
    if acceleration.shape[0] < 2:
        jerk = np.zeros((0, dof_pos.shape[-1]), dtype=np.float32)
    else:
        jerk = np.diff(acceleration, axis=0) * fps
    return velocity, acceleration, jerk


def compute_reference_bounds(reference_motions, lower_percentile=0.5, upper_percentile=99.5, margin_fraction=0.05):
    if not reference_motions:
        raise ValueError("reference_motions must not be empty")
    dof_values = np.concatenate([motion["dof_pos"] for motion in reference_motions], axis=0)
    heights = np.concatenate(
        [motion["root_pos"][:, ROOT_UP_AXIS] for motion in reference_motions],
        axis=0,
    )
    dof_lower = np.percentile(dof_values, lower_percentile, axis=0)
    dof_upper = np.percentile(dof_values, upper_percentile, axis=0)
    dof_margin = (dof_upper - dof_lower) * float(margin_fraction)
    height_lower = float(np.percentile(heights, lower_percentile))
    height_upper = float(np.percentile(heights, upper_percentile))
    height_margin = (height_upper - height_lower) * float(margin_fraction)
    return {
        "dof_lower": dof_lower - dof_margin,
        "dof_upper": dof_upper + dof_margin,
        "root_height_lower": height_lower - height_margin,
        "root_height_upper": height_upper + height_margin,
    }


def compute_joint_range_violation_rate(motion, bounds):
    dof_pos = motion["dof_pos"]
    violations = (dof_pos < bounds["dof_lower"]) | (dof_pos > bounds["dof_upper"])
    return float(np.mean(violations)) if violations.size else 0.0


def compute_root_height_violation_rate(motion, bounds):
    heights = motion["root_pos"][:, ROOT_UP_AXIS]
    violations = (
        (heights < bounds["root_height_lower"])
        | (heights > bounds["root_height_upper"])
    )
    return float(np.mean(violations)) if violations.size else 0.0


def g1_motion_speed_components(motion):
    root_velocity, _, _ = compute_root_derivatives(motion)
    joint_velocity, _, _ = compute_joint_derivatives(motion)
    root_linear = (
        np.linalg.norm(root_velocity[:, ROOT_FLAT_AXES], axis=-1)
        if root_velocity.size
        else np.zeros(0, dtype=np.float32)
    )

    root_angular = compute_root_angular_speed(motion)
    joint_speed = (
        np.mean(np.abs(joint_velocity), axis=-1)
        if joint_velocity.size
        else np.zeros(0, dtype=np.float32)
    )
    return root_linear, root_angular, joint_speed


def _zscore_curve(curve):
    curve = np.asarray(curve, dtype=np.float32)
    if curve.size == 0:
        return curve
    std = float(curve.std())
    if std < 1e-8:
        return np.zeros_like(curve)
    return (curve - float(curve.mean())) / std


def compute_g1_motion_speed_curve(motion):
    components = g1_motion_speed_components(motion)
    if not components[0].size:
        return np.zeros(0, dtype=np.float32)
    return sum(_zscore_curve(component) for component in components).astype(np.float32)


def compute_g1_joint_speed_curve(motion):
    joint_velocity, _, _ = compute_joint_derivatives(motion)
    if not joint_velocity.size:
        return np.zeros(0, dtype=np.float32)
    return np.mean(np.abs(joint_velocity), axis=-1).astype(np.float32)


def detect_g1_motion_beat_frames(motion, sigma=5):
    speed = compute_g1_motion_speed_curve(motion)
    if speed.size < 3:
        return np.zeros(0, dtype=np.int64)
    smoothed = gaussian_filter1d(speed, sigma=sigma)
    return np.asarray(argrelextrema(smoothed, np.less)[0], dtype=np.int64)


def detect_g1_roboperform_motion_beat_frames(motion, sigma=5):
    speed = compute_g1_joint_speed_curve(motion)
    if speed.size < 3:
        return np.zeros(0, dtype=np.int64)
    smoothed = gaussian_filter1d(speed, sigma=sigma)
    return np.asarray(argrelextrema(smoothed, np.less)[0], dtype=np.int64)


def evaluate_g1_beats(motion, bas_sigma_squared=DEFAULT_BAS_SIGMA_SQUARED, bap_tolerance=DEFAULT_BAP_TOLERANCE):
    audio_path = motion.get("audio_path")
    if not audio_path:
        return None
    motion_beats = detect_g1_motion_beat_frames(motion)
    roboperform_motion_beats = detect_g1_roboperform_motion_beat_frames(motion)
    audio_beats = load_audio_beat_frames(
        audio_path,
        fps=int(round(motion["fps"])),
        seq_len=motion["root_pos"].shape[0],
    )
    result = {
        "G1BAS": compute_bas_score(
            music_beats=audio_beats,
            motion_beats=motion_beats,
            sigma_squared=bas_sigma_squared,
        ),
        "G1RoboPerformBAS": compute_roboperform_bas_score(
            music_beats=audio_beats,
            motion_beats=roboperform_motion_beats,
            sigma_squared=bas_sigma_squared,
        ),
        "num_generated_beats": int(len(motion_beats)),
        "num_roboperform_generated_beats": int(len(roboperform_motion_beats)),
        "num_audio_beats": int(len(audio_beats)),
    }
    designated_beats = motion.get("designated_beat_frames")
    if designated_beats is not None:
        designated_beats = np.asarray(designated_beats, dtype=np.int64).reshape(-1)
        matched = greedy_match_count(
            motion_beats,
            designated_beats,
            tolerance=bap_tolerance,
        )
        result.update(
            {
                "matched_designated_beats": int(matched),
                "num_designated_beats": int(len(designated_beats)),
            }
        )
    return result


def compute_fk_keypoint_speed_curve(keypoints, fps):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
        raise ValueError("keypoints must have shape [T, K, 3]")
    if keypoints.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if keypoints.shape[0] == 1:
        return np.zeros(1, dtype=np.float32)
    velocity = np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=-1).mean(axis=-1)
    velocity = np.concatenate((velocity[:1], velocity), axis=0) * float(fps)
    return velocity.astype(np.float32)


def detect_fk_motion_beat_frames(keypoints, fps=FPS, sigma=5):
    speed = compute_fk_keypoint_speed_curve(keypoints, fps=fps)
    if speed.size < 3:
        return np.zeros(0, dtype=np.int64)
    smoothed = gaussian_filter1d(speed, sigma=sigma)
    return np.asarray(argrelextrema(smoothed, np.less)[0], dtype=np.int64)


def _keypoint_indices_by_tokens(keypoint_names, tokens):
    if not tokens:
        return list(range(len(keypoint_names)))
    lowered = [name.lower() for name in keypoint_names]
    return [
        idx
        for idx, name in enumerate(lowered)
        if any(token in name for token in tokens)
    ]


def _select_fk_keypoints(fk_result, tokens):
    keypoints = _as_float_array(fk_result["keypoints"], "keypoints", ndim=3)
    keypoint_names = list(fk_result.get("keypoint_names", []))
    if not keypoint_names:
        keypoint_names = [f"keypoint_{idx}" for idx in range(keypoints.shape[1])]
    indices = _keypoint_indices_by_tokens(keypoint_names, tokens)
    if not indices:
        return np.zeros((keypoints.shape[0], 0, 3), dtype=np.float32)
    return keypoints[:, indices, :]


def compute_keypoint_jerk_mean(keypoints, fps):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
        raise ValueError("keypoints must have shape [T, K, 3]")
    if keypoints.shape[0] < 4 or keypoints.shape[1] == 0:
        return 0.0
    velocity = np.diff(keypoints, axis=0) * float(fps)
    acceleration = np.diff(velocity, axis=0) * float(fps)
    jerk = np.diff(acceleration, axis=0) * float(fps)
    return finite_mean(np.linalg.norm(jerk, axis=-1))


def _greedy_match_offsets(generated_beats, target_beats, tolerance):
    generated_beats = np.sort(np.asarray(generated_beats, dtype=np.int64).reshape(-1))
    target_beats = np.sort(np.asarray(target_beats, dtype=np.int64).reshape(-1))
    generated_idx = 0
    target_idx = 0
    offsets = []
    while generated_idx < len(generated_beats) and target_idx < len(target_beats):
        offset = int(generated_beats[generated_idx] - target_beats[target_idx])
        if abs(offset) <= tolerance:
            offsets.append(offset)
            generated_idx += 1
            target_idx += 1
        elif generated_beats[generated_idx] < target_beats[target_idx] - tolerance:
            generated_idx += 1
        else:
            target_idx += 1
    return np.asarray(offsets, dtype=np.float32)


def compute_beat_timing_report(generated_beats, target_beats, tolerance=DEFAULT_BAP_TOLERANCE):
    generated_beats = np.asarray(generated_beats, dtype=np.int64).reshape(-1)
    target_beats = np.asarray(target_beats, dtype=np.int64).reshape(-1)
    offsets = _greedy_match_offsets(generated_beats, target_beats, tolerance)
    matched = int(offsets.size)
    unmatched_generated = max(int(generated_beats.size) - matched, 0)
    unmatched_target = max(int(target_beats.size) - matched, 0)
    precision = matched / max(int(generated_beats.size), 1)
    recall = matched / max(int(target_beats.size), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "matched": matched,
        "num_generated_beats": int(generated_beats.size),
        "num_target_beats": int(target_beats.size),
        "unmatched_generated_beats": unmatched_generated,
        "unmatched_target_beats": unmatched_target,
        "unmatched_generated_rate": unmatched_generated / max(int(generated_beats.size), 1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "timing_mean_frames": finite_mean(offsets),
        "timing_std_frames": float(np.std(offsets)) if offsets.size else 0.0,
    }


def _event_on_target_rate(event_mask, target_beats, tolerance):
    event_mask = np.asarray(event_mask, dtype=bool).reshape(-1)
    target_beats = np.asarray(target_beats, dtype=np.int64).reshape(-1)
    if target_beats.size == 0:
        return 0.0
    hits = 0
    for beat in target_beats:
        start = max(int(beat) - int(tolerance), 0)
        stop = min(int(beat) + int(tolerance) + 1, event_mask.size)
        if stop > start and np.any(event_mask[start:stop]):
            hits += 1
    return hits / max(int(target_beats.size), 1)


def _padded_point_velocity(points, fps):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return np.zeros_like(points, dtype=np.float32)
    velocity = np.diff(points, axis=0) * float(fps)
    return np.concatenate((np.zeros_like(velocity[:1]), velocity), axis=0)


def compute_fk_body_response_metrics(fk_result, target_beats, fps, tolerance=DEFAULT_BAP_TOLERANCE):
    metrics = {}
    full_body_f1 = 0.0
    for group_name, tokens in BODY_RESPONSE_GROUPS:
        group_keypoints = _select_fk_keypoints(fk_result, tokens)
        group_beats = detect_fk_motion_beat_frames(group_keypoints, fps=fps)
        timing = compute_beat_timing_report(
            generated_beats=group_beats,
            target_beats=target_beats,
            tolerance=tolerance,
        )
        metrics[f"G1{group_name}BeatPrecision"] = timing["precision"]
        metrics[f"G1{group_name}BeatRecall"] = timing["recall"]
        metrics[f"G1{group_name}BeatF1"] = timing["f1"]
        metrics[f"num_{group_name.lower()}_fk_generated_beats"] = timing["num_generated_beats"]
        if group_name == "FullBody":
            full_body_f1 = timing["f1"]

    wrist_f1 = metrics.get("G1WristBeatF1", 0.0)
    metrics["G1WristDominanceRatio"] = (
        0.0 if full_body_f1 < 1e-8 else wrist_f1 / full_body_f1
    )
    wrist_keypoints = _select_fk_keypoints(fk_result, ("wrist",))
    foot_keypoints = _select_fk_keypoints(fk_result, ("ankle", "foot"))
    metrics["G1WristJerkMean"] = compute_keypoint_jerk_mean(wrist_keypoints, fps=fps)
    metrics["G1FootJerkMean"] = compute_keypoint_jerk_mean(foot_keypoints, fps=fps)
    return metrics


def compute_fk_foot_diagnostics(
    fk_result,
    fps,
    target_beats=None,
    beat_tolerance=DEFAULT_BAP_TOLERANCE,
    contact_height_margin=0.03,
    near_support_margin=0.08,
    high_lift_margin=0.20,
):
    left = _as_float_array(fk_result["left_foot_points"], "left_foot_points", ndim=2)
    right = _as_float_array(fk_result["right_foot_points"], "right_foot_points", ndim=2)
    feet = np.stack([left, right], axis=1)
    if feet.shape[0] < 2:
        clearance = finite_mean(feet[:, :, ROOT_UP_AXIS])
        return {
            "G1FootSliding": 0.0,
            "G1GroundPenetration": 0.0,
            "G1FootClearanceMean": clearance,
            "G1FootContactOnBeatRate": 0.0,
            "G1NearSupportOnBeatRate": 0.0,
            "G1NoNearSupportRate": 0.0,
            "G1FootHighLiftRate": 0.0,
        }

    velocity = _padded_point_velocity(feet, fps=fps)
    horizontal_speed = np.linalg.norm(velocity[:, :, ROOT_FLAT_AXES], axis=-1)
    vertical_speed = np.abs(velocity[:, :, ROOT_UP_AXIS])
    heights = feet[:, :, ROOT_UP_AXIS]
    ground = float(np.percentile(feet[:, :, ROOT_UP_AXIS], 1.0))
    contact = (heights <= ground + contact_height_margin) & (vertical_speed < 0.2)
    sliding = float(np.mean(horizontal_speed[contact])) if np.any(contact) else 0.0
    penetration = np.maximum(ground - feet[:, :, ROOT_UP_AXIS], 0.0)
    near_support = heights <= ground + near_support_margin
    any_contact = np.any(contact, axis=1)
    any_near_support = np.any(near_support, axis=1)
    target_beats = [] if target_beats is None else target_beats
    return {
        "G1FootSliding": sliding,
        "G1GroundPenetration": finite_max(penetration),
        "G1FootClearanceMean": finite_mean(feet[:, :, ROOT_UP_AXIS] - ground),
        "G1FootContactOnBeatRate": _event_on_target_rate(
            any_contact,
            target_beats=target_beats,
            tolerance=beat_tolerance,
        ),
        "G1NearSupportOnBeatRate": _event_on_target_rate(
            any_near_support,
            target_beats=target_beats,
            tolerance=beat_tolerance,
        ),
        "G1NoNearSupportRate": finite_mean(~any_near_support),
        "G1FootHighLiftRate": finite_mean(heights > ground + high_lift_margin),
    }


def evaluate_g1_fk_metrics(motion, fk_model_path, root_quat_order="xyzw", bap_tolerance=DEFAULT_BAP_TOLERANCE):
    audio_path = motion.get("audio_path")
    if not audio_path:
        return None
    fk_result = forward_g1_kinematics(
        motion,
        fk_model_path,
        root_quat_order=root_quat_order,
    )
    motion_beats = detect_fk_motion_beat_frames(
        fk_result["keypoints"],
        fps=motion["fps"],
    )
    audio_beats = load_audio_beat_frames(
        audio_path,
        fps=int(round(motion["fps"])),
        seq_len=motion["root_pos"].shape[0],
    )
    timing = compute_beat_timing_report(
        generated_beats=motion_beats,
        target_beats=audio_beats,
        tolerance=bap_tolerance,
    )
    duration_seconds = motion["root_pos"].shape[0] / max(float(motion["fps"]), 1e-8)
    diagnostics = compute_fk_foot_diagnostics(
        fk_result,
        fps=motion["fps"],
        target_beats=audio_beats,
        beat_tolerance=bap_tolerance,
    )
    body_response = compute_fk_body_response_metrics(
        fk_result,
        target_beats=audio_beats,
        fps=motion["fps"],
        tolerance=bap_tolerance,
    )
    return {
        "G1FKBAS": compute_bas_score(
            music_beats=audio_beats,
            motion_beats=motion_beats,
            sigma_squared=DEFAULT_BAS_SIGMA_SQUARED,
        ),
        "G1FKRoboPerformBAS": compute_roboperform_bas_score(
            music_beats=audio_beats,
            motion_beats=motion_beats,
            sigma_squared=DEFAULT_BAS_SIGMA_SQUARED,
        ),
        "G1BeatPrecision": timing["precision"],
        "G1BeatRecall": timing["recall"],
        "G1BeatF1": timing["f1"],
        "G1BeatTimingMeanFrames": timing["timing_mean_frames"],
        "G1BeatTimingStdFrames": timing["timing_std_frames"],
        "G1UnmatchedMotionBeatRate": timing["unmatched_generated_rate"],
        "duration_seconds": duration_seconds,
        "num_fk_generated_beats": timing["num_generated_beats"],
        "num_fk_audio_beats": timing["num_target_beats"],
        "num_fk_matched_beats": timing["matched"],
        "num_fk_unmatched_motion_beats": timing["unmatched_generated_beats"],
        **diagnostics,
        **body_response,
        "fk_metadata": fk_result.get("metadata", {}),
    }


def summarize_g1_motion(motion, bounds=None, root_quat_order="xyzw"):
    root_pos = motion["root_pos"]
    dof_pos = motion["dof_pos"]
    root_velocity, root_acceleration, root_jerk = compute_root_derivatives(motion)
    root_angular = compute_root_angular_speed(motion)
    root_up_z = compute_root_up_z(motion, root_quat_order=root_quat_order)
    joint_velocity, joint_acceleration, joint_jerk = compute_joint_derivatives(motion)
    root_speed = np.linalg.norm(root_velocity, axis=-1) if root_velocity.size else []
    root_accel = (
        np.linalg.norm(root_acceleration, axis=-1)
        if root_acceleration.size
        else []
    )
    root_jerk_norm = np.linalg.norm(root_jerk, axis=-1) if root_jerk.size else []
    joint_speed = np.abs(joint_velocity).reshape(-1) if joint_velocity.size else []
    joint_accel = (
        np.abs(joint_acceleration).reshape(-1)
        if joint_acceleration.size
        else []
    )
    joint_jerk_abs = np.abs(joint_jerk).reshape(-1) if joint_jerk.size else []
    heights = root_pos[:, ROOT_UP_AXIS]
    flat_path = root_pos[:, ROOT_FLAT_AXES]
    drift = (
        float(np.linalg.norm(flat_path[-1] - flat_path[0]))
        if flat_path.shape[0] > 1
        else 0.0
    )
    path_length = (
        float(np.linalg.norm(np.diff(flat_path, axis=0), axis=-1).sum())
        if flat_path.shape[0] > 1
        else 0.0
    )
    flat_range = (
        float(np.linalg.norm(flat_path.max(axis=0) - flat_path.min(axis=0)))
        if flat_path.size
        else 0.0
    )

    summary = {
        "frames": int(root_pos.shape[0]),
        "fps": float(motion["fps"]),
        "root_height_min": float(np.min(heights)),
        "root_height_max": float(np.max(heights)),
        "root_height_mean": float(np.mean(heights)),
        "root_drift": drift,
        "root_flat_range": flat_range,
        "root_path_length": path_length,
        "root_velocity_mean": finite_mean(root_speed),
        "root_velocity_max": finite_max(root_speed),
        "root_angular_velocity_mean": finite_mean(root_angular),
        "root_angular_velocity_p95": finite_percentile(root_angular, 95),
        "root_angular_velocity_p99": finite_percentile(root_angular, 99),
        "root_angular_velocity_max": finite_max(root_angular),
        "root_angular_velocity_gt_pi_rate": finite_mean(root_angular > np.pi),
        "root_angular_velocity_gt_2pi_rate": finite_mean(root_angular > (2.0 * np.pi)),
        "root_up_z_min": finite_min(root_up_z),
        "root_up_z_p01": finite_percentile(root_up_z, 1),
        "root_up_z_mean": finite_mean(root_up_z),
        "root_tilt_gt_30deg_rate": finite_mean(root_up_z < np.cos(np.deg2rad(30.0))),
        "root_tilt_gt_60deg_rate": finite_mean(root_up_z < np.cos(np.deg2rad(60.0))),
        "root_inverted_rate": finite_mean(root_up_z < 0.0),
        "root_acceleration_mean": finite_mean(root_accel),
        "root_acceleration_max": finite_max(root_accel),
        "root_smoothness_jerk_mean": finite_mean(root_jerk_norm),
        "joint_velocity_mean": finite_mean(joint_speed),
        "joint_velocity_max": finite_max(joint_speed),
        "joint_acceleration_mean": finite_mean(joint_accel),
        "joint_acceleration_max": finite_max(joint_accel),
        "joint_smoothness_jerk_mean": finite_mean(joint_jerk_abs),
        "joint_position_std_mean": float(dof_pos.std(axis=0).mean()),
        "joint_position_range_mean": float(
            (dof_pos.max(axis=0) - dof_pos.min(axis=0)).mean()
        ),
    }
    if bounds is not None:
        summary["joint_range_violation_rate"] = compute_joint_range_violation_rate(
            motion,
            bounds,
        )
        summary["root_height_violation_rate"] = compute_root_height_violation_rate(
            motion,
            bounds,
        )
    return summary


def extract_g1_distribution_feature(motion):
    summary = summarize_g1_motion(motion)
    dof_pos = motion["dof_pos"]
    return np.concatenate(
        [
            np.asarray(
                [
                    summary["root_height_mean"],
                    summary["root_height_min"],
                    summary["root_height_max"],
                    summary["root_drift"],
                    summary["root_path_length"],
                    summary["root_velocity_mean"],
                    summary["root_velocity_max"],
                    summary["root_acceleration_mean"],
                    summary["root_acceleration_max"],
                    summary["joint_velocity_mean"],
                    summary["joint_velocity_max"],
                    summary["joint_acceleration_mean"],
                    summary["joint_acceleration_max"],
                ],
                dtype=np.float32,
            ),
            dof_pos.mean(axis=0).astype(np.float32),
            dof_pos.std(axis=0).astype(np.float32),
            dof_pos.min(axis=0).astype(np.float32),
            dof_pos.max(axis=0).astype(np.float32),
        ],
        axis=0,
    )


def average_pairwise_distance(features):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] < 2 or features.shape[1] == 0:
        return 0.0
    total = 0.0
    count = 0
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            total += float(np.linalg.norm(features[i] - features[j]))
            count += 1
    return total / max(count, 1)


def compute_g1_distribution_metrics(motions, reference_motions, std_epsilon=1e-6):
    generated_features = np.stack([extract_g1_distribution_feature(motion) for motion in motions])
    reference_features = np.stack(
        [extract_g1_distribution_feature(motion) for motion in reference_motions]
    )
    mean = reference_features.mean(axis=0)
    std = reference_features.std(axis=0)
    valid_dims = std > std_epsilon
    if not np.any(valid_dims):
        normalized_generated = np.zeros((generated_features.shape[0], 0), dtype=np.float32)
        normalized_reference = np.zeros((reference_features.shape[0], 0), dtype=np.float32)
    else:
        normalized_generated = (generated_features[:, valid_dims] - mean[valid_dims]) / std[valid_dims]
        normalized_reference = (reference_features[:, valid_dims] - mean[valid_dims]) / std[valid_dims]

    generated_center = normalized_generated.mean(axis=0) if normalized_generated.size else np.zeros(0)
    reference_center = normalized_reference.mean(axis=0) if normalized_reference.size else np.zeros(0)
    return {
        "G1Dist": float(np.linalg.norm(generated_center - reference_center)),
        "G1Div": average_pairwise_distance(normalized_generated),
        "G1ReferenceDiv": average_pairwise_distance(normalized_reference),
        "G1FeatureZeroVarianceDims": int((~valid_dims).sum()),
    }


def aggregate_summaries(summaries):
    keys = [
        "root_height_min",
        "root_height_max",
        "root_height_mean",
        "root_drift",
        "root_flat_range",
        "root_path_length",
        "root_velocity_mean",
        "root_velocity_max",
        "root_angular_velocity_mean",
        "root_angular_velocity_p95",
        "root_angular_velocity_p99",
        "root_angular_velocity_max",
        "root_angular_velocity_gt_pi_rate",
        "root_angular_velocity_gt_2pi_rate",
        "root_up_z_min",
        "root_up_z_p01",
        "root_up_z_mean",
        "root_tilt_gt_30deg_rate",
        "root_tilt_gt_60deg_rate",
        "root_inverted_rate",
        "root_acceleration_mean",
        "root_acceleration_max",
        "root_smoothness_jerk_mean",
        "joint_velocity_mean",
        "joint_velocity_max",
        "joint_acceleration_mean",
        "joint_acceleration_max",
        "joint_smoothness_jerk_mean",
        "joint_position_std_mean",
        "joint_position_range_mean",
        "joint_range_violation_rate",
        "root_height_violation_rate",
    ]
    aggregated = {}
    for key in keys:
        values = [summary[key] for summary in summaries if key in summary]
        aggregated[key] = finite_mean(values)
    return {
        "RootHeightMin": finite_mean([summary["root_height_min"] for summary in summaries]),
        "RootHeightMax": finite_mean([summary["root_height_max"] for summary in summaries]),
        "RootHeightMean": aggregated["root_height_mean"],
        "RootDriftMean": aggregated["root_drift"],
        "RootFlatRangeMean": aggregated["root_flat_range"],
        "RootPathLengthMean": aggregated["root_path_length"],
        "RootVelocityMean": aggregated["root_velocity_mean"],
        "RootVelocityMax": finite_max([summary["root_velocity_max"] for summary in summaries]),
        "RootAngularVelocityMean": aggregated["root_angular_velocity_mean"],
        "RootAngularVelocityP95": aggregated["root_angular_velocity_p95"],
        "RootAngularVelocityP99": aggregated["root_angular_velocity_p99"],
        "RootAngularVelocityMax": finite_max(
            [summary["root_angular_velocity_max"] for summary in summaries]
        ),
        "RootAngularVelocityGtPiRate": aggregated["root_angular_velocity_gt_pi_rate"],
        "RootAngularVelocityGt2PiRate": aggregated["root_angular_velocity_gt_2pi_rate"],
        "RootUpZMin": finite_min([summary["root_up_z_min"] for summary in summaries]),
        "RootUpZP01": aggregated["root_up_z_p01"],
        "RootUpZMean": aggregated["root_up_z_mean"],
        "RootTiltGt30DegRate": aggregated["root_tilt_gt_30deg_rate"],
        "RootTiltGt60DegRate": aggregated["root_tilt_gt_60deg_rate"],
        "RootInvertedRate": aggregated["root_inverted_rate"],
        "RootAccelerationMean": aggregated["root_acceleration_mean"],
        "RootAccelerationMax": finite_max([summary["root_acceleration_max"] for summary in summaries]),
        "RootSmoothnessJerkMean": aggregated["root_smoothness_jerk_mean"],
        "JointVelocityMean": aggregated["joint_velocity_mean"],
        "JointVelocityMax": finite_max([summary["joint_velocity_max"] for summary in summaries]),
        "JointAccelerationMean": aggregated["joint_acceleration_mean"],
        "JointAccelerationMax": finite_max([summary["joint_acceleration_max"] for summary in summaries]),
        "JointSmoothnessJerkMean": aggregated["joint_smoothness_jerk_mean"],
        "JointPositionStdMean": aggregated["joint_position_std_mean"],
        "JointPositionRangeMean": aggregated["joint_position_range_mean"],
        "ReferenceRangeViolationRate": aggregated["joint_range_violation_rate"],
        "RootHeightViolationRate": aggregated["root_height_violation_rate"],
    }


def aggregate_beat_metrics(beat_records):
    beat_records = [record for record in beat_records if record is not None]
    if not beat_records:
        return {
            "G1BAS": 0.0,
            "G1BAS_direction": BAS_DIRECTION,
            "G1RoboPerformBAS": 0.0,
            "G1RoboPerformBAS_direction": ROBOPERFORM_BAS_DIRECTION,
            "G1BAP": 0.0,
            "G1BAP_precision": 0.0,
            "G1BAP_recall": 0.0,
            "num_scored_files": 0,
            "num_generated_beats": 0,
            "num_roboperform_generated_beats": 0,
            "num_audio_beats": 0,
            "num_designated_beats": 0,
        }
    generated_beats = sum(record["num_generated_beats"] for record in beat_records)
    roboperform_generated_beats = sum(
        record["num_roboperform_generated_beats"] for record in beat_records
    )
    audio_beats = sum(record["num_audio_beats"] for record in beat_records)
    designated_beats = sum(record.get("num_designated_beats", 0) for record in beat_records)
    matched_designated = sum(record.get("matched_designated_beats", 0) for record in beat_records)
    return {
        "G1BAS": finite_mean([record["G1BAS"] for record in beat_records]),
        "G1BAS_direction": BAS_DIRECTION,
        "G1RoboPerformBAS": finite_mean(
            [record["G1RoboPerformBAS"] for record in beat_records]
        ),
        "G1RoboPerformBAS_direction": ROBOPERFORM_BAS_DIRECTION,
        "G1BAP": matched_designated / max(generated_beats, 1),
        "G1BAP_precision": matched_designated / max(generated_beats, 1),
        "G1BAP_recall": matched_designated / max(designated_beats, 1),
        "num_scored_files": len(beat_records),
        "num_generated_beats": int(generated_beats),
        "num_roboperform_generated_beats": int(roboperform_generated_beats),
        "num_audio_beats": int(audio_beats),
        "num_designated_beats": int(designated_beats),
    }


def aggregate_fk_metrics(fk_records):
    fk_records = [record for record in fk_records if record is not None]
    if not fk_records:
        return {}
    duration_seconds = sum(record.get("duration_seconds", 0.0) for record in fk_records)
    generated_beats = sum(record["num_fk_generated_beats"] for record in fk_records)
    audio_beats = sum(record["num_fk_audio_beats"] for record in fk_records)
    matched_beats = sum(record["num_fk_matched_beats"] for record in fk_records)
    unmatched_motion_beats = sum(
        record.get("num_fk_unmatched_motion_beats", 0) for record in fk_records
    )
    wrist_beat_f1 = finite_mean([record["G1WristBeatF1"] for record in fk_records])
    full_body_beat_f1 = finite_mean([record["G1FullBodyBeatF1"] for record in fk_records])
    return {
        "G1FKBAS": finite_mean([record["G1FKBAS"] for record in fk_records]),
        "G1FKBAS_direction": BAS_DIRECTION,
        "G1FKRoboPerformBAS": finite_mean(
            [record["G1FKRoboPerformBAS"] for record in fk_records]
        ),
        "G1FKRoboPerformBAS_direction": ROBOPERFORM_BAS_DIRECTION,
        "G1BeatPrecision": finite_mean([record["G1BeatPrecision"] for record in fk_records]),
        "G1BeatRecall": finite_mean([record["G1BeatRecall"] for record in fk_records]),
        "G1BeatF1": finite_mean([record["G1BeatF1"] for record in fk_records]),
        "G1BeatTimingMeanFrames": finite_mean(
            [record["G1BeatTimingMeanFrames"] for record in fk_records]
        ),
        "G1BeatTimingStdFrames": finite_mean(
            [record["G1BeatTimingStdFrames"] for record in fk_records]
        ),
        "G1MotionBeatDensity": generated_beats / max(duration_seconds, 1e-8),
        "G1AudioBeatDensity": audio_beats / max(duration_seconds, 1e-8),
        "G1BeatDensityRatio": (
            (generated_beats / max(duration_seconds, 1e-8))
            / max(audio_beats / max(duration_seconds, 1e-8), 1e-8)
        ),
        "G1UnmatchedMotionBeatRate": unmatched_motion_beats / max(generated_beats, 1),
        "G1WristBeatF1": wrist_beat_f1,
        "G1FootBeatF1": finite_mean([record["G1FootBeatF1"] for record in fk_records]),
        "G1TorsoBeatF1": finite_mean([record["G1TorsoBeatF1"] for record in fk_records]),
        "G1FullBodyBeatF1": full_body_beat_f1,
        "G1WristDominanceRatio": wrist_beat_f1 / max(full_body_beat_f1, 1e-8),
        "G1WristJerkMean": finite_mean([record["G1WristJerkMean"] for record in fk_records]),
        "G1FootJerkMean": finite_mean([record["G1FootJerkMean"] for record in fk_records]),
        "G1FootSliding": finite_mean([record["G1FootSliding"] for record in fk_records]),
        "G1GroundPenetration": finite_max(
            [record["G1GroundPenetration"] for record in fk_records]
        ),
        "G1FootClearanceMean": finite_mean(
            [record["G1FootClearanceMean"] for record in fk_records]
        ),
        "G1FootContactOnBeatRate": finite_mean(
            [record["G1FootContactOnBeatRate"] for record in fk_records]
        ),
        "G1NearSupportOnBeatRate": finite_mean(
            [record["G1NearSupportOnBeatRate"] for record in fk_records]
        ),
        "G1NoNearSupportRate": finite_mean(
            [record["G1NoNearSupportRate"] for record in fk_records]
        ),
        "G1FootHighLiftRate": finite_mean(
            [record["G1FootHighLiftRate"] for record in fk_records]
        ),
        "num_fk_scored_files": len(fk_records),
        "num_fk_generated_beats": int(generated_beats),
        "num_fk_audio_beats": int(audio_beats),
        "num_fk_matched_beats": int(matched_beats),
        "num_fk_unmatched_motion_beats": int(unmatched_motion_beats),
    }


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
    return payload


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)


def write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_g1_table(metrics, method_name):
    table = {
        "Method": method_name,
        "Files": metrics["num_motion_files"],
        "G1 Beat Align.": metrics["G1BAS"],
        "G1 RoboPerform BAS": metrics["G1RoboPerformBAS"],
        "G1 Beat Match": metrics["G1BAP_precision"],
        "Root Drift": metrics["RootDriftMean"],
        "Root Flat Range": metrics["RootFlatRangeMean"],
        "Root Ang. Vel. P99": metrics["RootAngularVelocityP99"],
        "Root Ang. Vel. Max": metrics["RootAngularVelocityMax"],
        "Root Up Z P01": metrics["RootUpZP01"],
        "Root Tilt >60 Rate": metrics["RootTiltGt60DegRate"],
        "Root Height Min": metrics["RootHeightMin"],
        "Root Height Max": metrics["RootHeightMax"],
        "Joint Range Viol.": metrics["ReferenceRangeViolationRate"],
        "Joint Pos. Std": metrics["JointPositionStdMean"],
        "Joint Pos. Range": metrics["JointPositionRangeMean"],
        "G1Dist": metrics["G1Dist"],
        "G1Div": metrics["G1Div"],
    }
    if "G1FKBAS" in metrics:
        table.update(
            {
                "G1 FK Beat Align.": metrics["G1FKBAS"],
                "G1 FK RoboPerform BAS": metrics["G1FKRoboPerformBAS"],
                "G1 Beat F1": metrics["G1BeatF1"],
                "G1 Beat Recall": metrics["G1BeatRecall"],
                "G1 Beat Density Ratio": metrics["G1BeatDensityRatio"],
                "G1 Unmatched Motion Beat Rate": metrics["G1UnmatchedMotionBeatRate"],
                "G1 Wrist Beat F1": metrics["G1WristBeatF1"],
                "G1 Foot Beat F1": metrics["G1FootBeatF1"],
                "G1 Torso Beat F1": metrics["G1TorsoBeatF1"],
                "G1 Wrist Dominance": metrics["G1WristDominanceRatio"],
                "G1 Foot Contact On Beat": metrics["G1FootContactOnBeatRate"],
                "G1 Near Support On Beat": metrics["G1NearSupportOnBeatRate"],
                "G1 No Near Support Rate": metrics["G1NoNearSupportRate"],
                "G1 Foot High Lift Rate": metrics["G1FootHighLiftRate"],
                "G1 Wrist Jerk Mean": metrics["G1WristJerkMean"],
                "G1 Foot Jerk Mean": metrics["G1FootJerkMean"],
                "G1 Foot Sliding": metrics["G1FootSliding"],
                "G1 Ground Penetration": metrics["G1GroundPenetration"],
            }
        )
    return table


def render_g1_paper_report(metrics, table):
    lines = [
        "# G1 Robot-Native Evaluation Report",
        "",
        "This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.",
        "",
        f"- Generated clips: {metrics['num_motion_files']}",
        f"- Finite motion rate: {metrics['FiniteMotionRate']}",
        f"- Beat alignment: {metrics['G1BAS']}",
        f"- RoboPerform BAS: {metrics['G1RoboPerformBAS']}",
        f"- Designated beat precision: {metrics['G1BAP_precision']}",
        f"- Designated beat recall: {metrics['G1BAP_recall']}",
        f"- Root drift mean: {metrics['RootDriftMean']}",
        f"- Root flat range mean: {metrics['RootFlatRangeMean']}",
        f"- Root angular velocity p99: {metrics['RootAngularVelocityP99']}",
        f"- Root angular velocity max: {metrics['RootAngularVelocityMax']}",
        f"- Root up-z p01: {metrics['RootUpZP01']}",
        f"- Root tilt >60deg rate: {metrics['RootTiltGt60DegRate']}",
        f"- Root height mean: {metrics['RootHeightMean']}",
        f"- Joint position std mean: {metrics['JointPositionStdMean']}",
        f"- Joint position range mean: {metrics['JointPositionRangeMean']}",
        f"- Joint range violation rate: {metrics['ReferenceRangeViolationRate']}",
        f"- G1 feature distance: {metrics['G1Dist']}",
        f"- G1 diversity: {metrics['G1Div']}",
        "",
    ]
    if "G1FKBAS" in metrics:
        lines.extend(
            [
                "## FK Metrics",
                "",
                f"- FK beat alignment: {metrics['G1FKBAS']}",
                f"- FK RoboPerform BAS: {metrics['G1FKRoboPerformBAS']}",
                f"- Beat F1: {metrics['G1BeatF1']}",
                f"- Beat recall: {metrics['G1BeatRecall']}",
                f"- Beat timing mean frames: {metrics['G1BeatTimingMeanFrames']}",
                f"- Beat timing std frames: {metrics['G1BeatTimingStdFrames']}",
                f"- Beat density ratio: {metrics['G1BeatDensityRatio']}",
                f"- Unmatched motion beat rate: {metrics['G1UnmatchedMotionBeatRate']}",
                f"- Wrist beat F1: {metrics['G1WristBeatF1']}",
                f"- Foot beat F1: {metrics['G1FootBeatF1']}",
                f"- Torso beat F1: {metrics['G1TorsoBeatF1']}",
                f"- Wrist dominance ratio: {metrics['G1WristDominanceRatio']}",
                f"- Foot contact on beat rate: {metrics['G1FootContactOnBeatRate']}",
                f"- Near support on beat rate: {metrics['G1NearSupportOnBeatRate']}",
                f"- No near support rate: {metrics['G1NoNearSupportRate']}",
                f"- Foot high-lift rate: {metrics['G1FootHighLiftRate']}",
                f"- Wrist jerk mean: {metrics['G1WristJerkMean']}",
                f"- Foot jerk mean: {metrics['G1FootJerkMean']}",
                f"- Foot sliding: {metrics['G1FootSliding']}",
                f"- Ground penetration: {metrics['G1GroundPenetration']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Table Row",
            "",
            "```json",
            json.dumps(json_safe(table), indent=2, sort_keys=True),
            "```",
            "",
            "## Deferred Metrics",
            "",
            "- Contact quality, link tracking error, and simulator success need a controller rollout.",
            "- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _plot_vertical_markers(axis, frames, color, label):
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    first = True
    for frame in frames:
        axis.axvline(
            int(frame),
            color=color,
            linewidth=0.8,
            alpha=0.5,
            label=label if first else None,
        )
        first = False


def render_g1_diagnostics(motion_files, render_dir, diagnostic_count=8, seed=1234):
    if diagnostic_count <= 0:
        return []
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    selected = list(motion_files)
    if len(selected) > diagnostic_count:
        rng = random.Random(seed)
        selected = sorted(rng.sample(selected, diagnostic_count))

    outputs = []
    for motion_file in selected:
        motion = load_g1_motion(motion_file)
        root_pos = motion["root_pos"]
        dof_pos = motion["dof_pos"]
        frames = np.arange(root_pos.shape[0])
        speed_curve = compute_g1_motion_speed_curve(motion)
        motion_beats = detect_g1_motion_beat_frames(motion)
        audio_beats = []
        if motion.get("audio_path"):
            try:
                audio_beats = load_audio_beat_frames(
                    motion["audio_path"],
                    fps=int(round(motion["fps"])),
                    seq_len=root_pos.shape[0],
                )
            except Exception:
                audio_beats = []
        designated_beats = motion.get("designated_beat_frames")
        if designated_beats is None:
            designated_beats = []

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        axes[0, 0].plot(root_pos[:, 0], root_pos[:, 1], linewidth=1.5)
        axes[0, 0].set_title("Root XY path")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        axes[0, 0].axis("equal")

        axes[0, 1].plot(frames, root_pos[:, ROOT_UP_AXIS], linewidth=1.5)
        axes[0, 1].set_title("Root height")
        axes[0, 1].set_xlabel("frame")

        heatmap = axes[1, 0].imshow(dof_pos.T, aspect="auto", interpolation="nearest")
        axes[1, 0].set_title("Joint angles")
        axes[1, 0].set_xlabel("frame")
        axes[1, 0].set_ylabel("dof")
        fig.colorbar(heatmap, ax=axes[1, 0], fraction=0.046, pad=0.04)

        speed_frames = np.arange(speed_curve.shape[0])
        axes[1, 1].plot(speed_frames, speed_curve, linewidth=1.2, label="G1 speed")
        _plot_vertical_markers(axes[1, 1], audio_beats, "tab:blue", "audio")
        _plot_vertical_markers(axes[1, 1], designated_beats, "tab:red", "designated")
        _plot_vertical_markers(axes[1, 1], motion_beats, "tab:green", "motion")
        axes[1, 1].set_title("Motion speed and beats")
        axes[1, 1].set_xlabel("frame")
        axes[1, 1].legend(loc="best", fontsize=8)

        output_path = render_dir / f"{Path(motion_file).stem}_diagnostic.png"
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        outputs.append(str(output_path))
    return outputs


def compact_metric_record(record):
    if record is None:
        return {}
    return {
        key: value
        for key, value in record.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def build_failure_panel(per_file_records, top_k=8):
    panel = {}
    for panel_name, metric_key, descending in FAILURE_PANEL_SPECS:
        candidates = [
            record
            for record in per_file_records
            if metric_key in record and np.isfinite(record[metric_key])
        ]
        candidates = sorted(
            candidates,
            key=lambda record: record[metric_key],
            reverse=descending,
        )
        panel[panel_name] = [
            {
                "path": record["path"],
                "metric": metric_key,
                "value": float(record[metric_key]),
                "metrics": compact_metric_record(record),
            }
            for record in candidates[:top_k]
        ]
    return panel


def select_failure_panel_paths(failure_panel, max_count):
    selected = []
    seen = set()
    for entries in failure_panel.values():
        for entry in entries[:1]:
            path = entry["path"]
            if path not in seen:
                selected.append(path)
                seen.add(path)
            if len(selected) >= max_count:
                return selected
    return selected


def load_motion_dir(motion_path, sample_limit=None, seed=1234):
    motion_files = sorted(glob.glob(str(Path(motion_path) / "*.pkl")))
    if sample_limit is not None and len(motion_files) > sample_limit:
        rng = random.Random(seed)
        motion_files = sorted(rng.sample(motion_files, sample_limit))
    motions = []
    bad_files = []
    for motion_file in tqdm(motion_files, desc="G1 load", unit="file"):
        try:
            motions.append(load_g1_motion(motion_file))
        except Exception as exc:
            bad_files.append({"path": motion_file, "error": str(exc)})
    return motion_files, motions, bad_files


def run_g1_motion_evaluation(
    motion_path,
    reference_motion_path,
    metrics_path,
    g1_table_path,
    motion_audit_path,
    paper_report_path,
    render_dir,
    diagnostic_count=8,
    checkpoint="",
    feature_type="jukebox",
    use_beats=True,
    beat_rep="distance",
    seed=1234,
    sample_limit=None,
    enable_fk_metrics=False,
    fk_model_path=None,
    root_quat_order="xyzw",
    failure_panel_path=None,
):
    motion_files, motions, bad_files = load_motion_dir(
        motion_path,
        sample_limit=sample_limit,
        seed=seed,
    )
    if not motions:
        raise FileNotFoundError(f"No valid G1 motion pickle files found in {motion_path}")
    _, reference_motions, reference_bad_files = load_motion_dir(
        reference_motion_path,
        sample_limit=sample_limit,
        seed=seed,
    )
    if not reference_motions:
        raise FileNotFoundError(
            f"No valid G1 reference motion pickle files found in {reference_motion_path}"
        )

    bounds = compute_reference_bounds(reference_motions)
    summaries = [
        summarize_g1_motion(
            motion,
            bounds=bounds,
            root_quat_order=root_quat_order,
        )
        for motion in motions
    ]
    beat_records = [
        evaluate_g1_beats(motion)
        for motion in tqdm(motions, desc="G1 beat metrics", unit="file")
    ]
    fk_records = []
    if enable_fk_metrics:
        if fk_model_path is None:
            raise ValueError("fk_model_path is required when enable_fk_metrics is true")
        fk_records = [
            evaluate_g1_fk_metrics(
                motion,
                fk_model_path=fk_model_path,
                root_quat_order=root_quat_order,
            )
            for motion in tqdm(motions, desc="G1 FK metrics", unit="file")
        ]
    metrics = {
        "checkpoint": checkpoint,
        "feature_type": feature_type,
        "motion_format": "g1",
        "use_beats": bool(use_beats),
        "beat_rep": beat_rep if use_beats else "none",
        "seed": int(seed),
        "num_motion_files": len(motion_files),
        "num_valid_motion_files": len(motions),
        "num_reference_files": len(reference_motions),
        "BadFileCount": len(bad_files),
        "FiniteMotionRate": len(motions) / max(len(motion_files), 1),
        "fk_metrics_enabled": bool(enable_fk_metrics),
    }
    if enable_fk_metrics:
        metrics.update(
            {
                "fk_model_path": str(fk_model_path),
                "root_quat_order": root_quat_order,
            }
        )
    metrics.update(aggregate_beat_metrics(beat_records))
    metrics.update(aggregate_fk_metrics(fk_records))
    metrics.update(aggregate_summaries(summaries))
    metrics.update(compute_g1_distribution_metrics(motions, reference_motions))
    for name in SMPL_ONLY_METRIC_NAMES:
        metrics.pop(name, None)

    method_name = f"G1 {Path(checkpoint).stem}" if checkpoint else "G1 evaluation"
    table = build_g1_table(metrics, method_name=method_name)
    diagnostics = render_g1_diagnostics(
        motion_files,
        render_dir=render_dir,
        diagnostic_count=diagnostic_count,
        seed=seed,
    )
    per_file_records = []
    for motion, summary, fk_record in zip(
        motions,
        summaries,
        fk_records if fk_records else [None] * len(motions),
    ):
        per_file_records.append(
            {
                "path": motion["path"],
                **summary,
                **compact_metric_record(fk_record),
            }
        )
    failure_panel = build_failure_panel(per_file_records)
    failure_diagnostics = []
    if diagnostic_count > 0:
        failure_paths = select_failure_panel_paths(
            failure_panel,
            max_count=diagnostic_count,
        )
        if failure_paths:
            failure_diagnostics = render_g1_diagnostics(
                failure_paths,
                render_dir=Path(render_dir) / "failure_panel",
                diagnostic_count=len(failure_paths),
                seed=seed,
            )
    failure_panel["diagnostic_renders"] = failure_diagnostics
    if failure_panel_path is None:
        failure_panel_path = Path(motion_audit_path).with_name("failure_panel.json")

    audit = {
        "num_files": len(motion_files),
        "num_valid_files": len(motions),
        "bad_files": bad_files,
        "reference_bad_files": reference_bad_files,
        "diagnostic_renders": diagnostics,
        "failure_panel_path": str(failure_panel_path),
        "failure_diagnostic_renders": failure_diagnostics,
        "fk_metrics_enabled": bool(enable_fk_metrics),
        "fk_model_path": str(fk_model_path) if fk_model_path else "",
        "root_quat_order": root_quat_order if enable_fk_metrics else "",
        "per_file": per_file_records,
    }

    write_json(metrics_path, metrics)
    write_json(g1_table_path, table)
    write_json(motion_audit_path, audit)
    write_json(failure_panel_path, failure_panel)
    write_text(paper_report_path, render_g1_paper_report(metrics, table))

    print(json.dumps(json_safe(metrics), indent=2, sort_keys=True))
    print(f"Saved G1 metrics to {metrics_path}")
    return metrics
