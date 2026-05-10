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
                               DEFAULT_BAS_SIGMA_SQUARED, compute_bas_score,
                               greedy_match_count, load_audio_beat_frames)
from eval.g1_kinematics import forward_g1_kinematics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FPS = 30.0
ROOT_UP_AXIS = 2
ROOT_FLAT_AXES = (0, 1)
SMPL_ONLY_METRIC_NAMES = ("PFC", "Distg", "Distk", "Divk", "Divm")


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

    root_rot = motion["root_rot"]
    if root_rot.shape[0] < 2:
        root_angular = np.zeros(0, dtype=np.float32)
    else:
        dots = np.sum(root_rot[1:] * root_rot[:-1], axis=-1)
        dots = np.clip(np.abs(dots), -1.0, 1.0)
        root_angular = (2.0 * np.arccos(dots) * motion["fps"]).astype(np.float32)

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


def detect_g1_motion_beat_frames(motion, sigma=5):
    speed = compute_g1_motion_speed_curve(motion)
    if speed.size < 3:
        return np.zeros(0, dtype=np.int64)
    smoothed = gaussian_filter1d(speed, sigma=sigma)
    return np.asarray(argrelextrema(smoothed, np.less)[0], dtype=np.int64)


def evaluate_g1_beats(motion, bas_sigma_squared=DEFAULT_BAS_SIGMA_SQUARED, bap_tolerance=DEFAULT_BAP_TOLERANCE):
    audio_path = motion.get("audio_path")
    if not audio_path:
        return None
    motion_beats = detect_g1_motion_beat_frames(motion)
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
        "num_generated_beats": int(len(motion_beats)),
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
    precision = matched / max(int(generated_beats.size), 1)
    recall = matched / max(int(target_beats.size), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "matched": matched,
        "num_generated_beats": int(generated_beats.size),
        "num_target_beats": int(target_beats.size),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "timing_mean_frames": finite_mean(offsets),
        "timing_std_frames": float(np.std(offsets)) if offsets.size else 0.0,
    }


def compute_fk_foot_diagnostics(fk_result, fps, contact_height_margin=0.03):
    left = _as_float_array(fk_result["left_foot_points"], "left_foot_points", ndim=2)
    right = _as_float_array(fk_result["right_foot_points"], "right_foot_points", ndim=2)
    feet = np.stack([left, right], axis=1)
    if feet.shape[0] < 2:
        return {
            "G1FootSliding": 0.0,
            "G1GroundPenetration": 0.0,
            "G1FootClearanceMean": finite_mean(feet[:, :, ROOT_UP_AXIS]),
        }

    velocity = np.diff(feet, axis=0) * float(fps)
    horizontal_speed = np.linalg.norm(velocity[:, :, ROOT_FLAT_AXES], axis=-1)
    vertical_speed = np.abs(velocity[:, :, ROOT_UP_AXIS])
    heights = feet[1:, :, ROOT_UP_AXIS]
    ground = float(np.percentile(feet[:, :, ROOT_UP_AXIS], 1.0))
    contact = (heights <= ground + contact_height_margin) & (vertical_speed < 0.2)
    sliding = float(np.mean(horizontal_speed[contact])) if np.any(contact) else 0.0
    penetration = np.maximum(ground - feet[:, :, ROOT_UP_AXIS], 0.0)
    return {
        "G1FootSliding": sliding,
        "G1GroundPenetration": finite_max(penetration),
        "G1FootClearanceMean": finite_mean(feet[:, :, ROOT_UP_AXIS] - ground),
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
    diagnostics = compute_fk_foot_diagnostics(fk_result, fps=motion["fps"])
    return {
        "G1FKBAS": compute_bas_score(
            music_beats=audio_beats,
            motion_beats=motion_beats,
            sigma_squared=DEFAULT_BAS_SIGMA_SQUARED,
        ),
        "G1BeatPrecision": timing["precision"],
        "G1BeatRecall": timing["recall"],
        "G1BeatF1": timing["f1"],
        "G1BeatTimingMeanFrames": timing["timing_mean_frames"],
        "G1BeatTimingStdFrames": timing["timing_std_frames"],
        "num_fk_generated_beats": timing["num_generated_beats"],
        "num_fk_audio_beats": timing["num_target_beats"],
        "num_fk_matched_beats": timing["matched"],
        **diagnostics,
        "fk_metadata": fk_result.get("metadata", {}),
    }


def summarize_g1_motion(motion, bounds=None):
    root_pos = motion["root_pos"]
    root_velocity, root_acceleration, root_jerk = compute_root_derivatives(motion)
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

    summary = {
        "frames": int(root_pos.shape[0]),
        "fps": float(motion["fps"]),
        "root_height_min": float(np.min(heights)),
        "root_height_max": float(np.max(heights)),
        "root_height_mean": float(np.mean(heights)),
        "root_drift": drift,
        "root_path_length": path_length,
        "root_velocity_mean": finite_mean(root_speed),
        "root_velocity_max": finite_max(root_speed),
        "root_acceleration_mean": finite_mean(root_accel),
        "root_acceleration_max": finite_max(root_accel),
        "root_smoothness_jerk_mean": finite_mean(root_jerk_norm),
        "joint_velocity_mean": finite_mean(joint_speed),
        "joint_velocity_max": finite_max(joint_speed),
        "joint_acceleration_mean": finite_mean(joint_accel),
        "joint_acceleration_max": finite_max(joint_accel),
        "joint_smoothness_jerk_mean": finite_mean(joint_jerk_abs),
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
        "root_path_length",
        "root_velocity_mean",
        "root_velocity_max",
        "root_acceleration_mean",
        "root_acceleration_max",
        "root_smoothness_jerk_mean",
        "joint_velocity_mean",
        "joint_velocity_max",
        "joint_acceleration_mean",
        "joint_acceleration_max",
        "joint_smoothness_jerk_mean",
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
        "RootPathLengthMean": aggregated["root_path_length"],
        "RootVelocityMean": aggregated["root_velocity_mean"],
        "RootVelocityMax": finite_max([summary["root_velocity_max"] for summary in summaries]),
        "RootAccelerationMean": aggregated["root_acceleration_mean"],
        "RootAccelerationMax": finite_max([summary["root_acceleration_max"] for summary in summaries]),
        "RootSmoothnessJerkMean": aggregated["root_smoothness_jerk_mean"],
        "JointVelocityMean": aggregated["joint_velocity_mean"],
        "JointVelocityMax": finite_max([summary["joint_velocity_max"] for summary in summaries]),
        "JointAccelerationMean": aggregated["joint_acceleration_mean"],
        "JointAccelerationMax": finite_max([summary["joint_acceleration_max"] for summary in summaries]),
        "JointSmoothnessJerkMean": aggregated["joint_smoothness_jerk_mean"],
        "ReferenceRangeViolationRate": aggregated["joint_range_violation_rate"],
        "RootHeightViolationRate": aggregated["root_height_violation_rate"],
    }


def aggregate_beat_metrics(beat_records):
    beat_records = [record for record in beat_records if record is not None]
    if not beat_records:
        return {
            "G1BAS": 0.0,
            "G1BAP": 0.0,
            "G1BAP_precision": 0.0,
            "G1BAP_recall": 0.0,
            "num_scored_files": 0,
            "num_generated_beats": 0,
            "num_audio_beats": 0,
            "num_designated_beats": 0,
        }
    generated_beats = sum(record["num_generated_beats"] for record in beat_records)
    audio_beats = sum(record["num_audio_beats"] for record in beat_records)
    designated_beats = sum(record.get("num_designated_beats", 0) for record in beat_records)
    matched_designated = sum(record.get("matched_designated_beats", 0) for record in beat_records)
    return {
        "G1BAS": finite_mean([record["G1BAS"] for record in beat_records]),
        "G1BAP": matched_designated / max(generated_beats, 1),
        "G1BAP_precision": matched_designated / max(generated_beats, 1),
        "G1BAP_recall": matched_designated / max(designated_beats, 1),
        "num_scored_files": len(beat_records),
        "num_generated_beats": int(generated_beats),
        "num_audio_beats": int(audio_beats),
        "num_designated_beats": int(designated_beats),
    }


def aggregate_fk_metrics(fk_records):
    fk_records = [record for record in fk_records if record is not None]
    if not fk_records:
        return {}
    return {
        "G1FKBAS": finite_mean([record["G1FKBAS"] for record in fk_records]),
        "G1BeatPrecision": finite_mean([record["G1BeatPrecision"] for record in fk_records]),
        "G1BeatRecall": finite_mean([record["G1BeatRecall"] for record in fk_records]),
        "G1BeatF1": finite_mean([record["G1BeatF1"] for record in fk_records]),
        "G1BeatTimingMeanFrames": finite_mean(
            [record["G1BeatTimingMeanFrames"] for record in fk_records]
        ),
        "G1BeatTimingStdFrames": finite_mean(
            [record["G1BeatTimingStdFrames"] for record in fk_records]
        ),
        "G1FootSliding": finite_mean([record["G1FootSliding"] for record in fk_records]),
        "G1GroundPenetration": finite_max(
            [record["G1GroundPenetration"] for record in fk_records]
        ),
        "G1FootClearanceMean": finite_mean(
            [record["G1FootClearanceMean"] for record in fk_records]
        ),
        "num_fk_scored_files": len(fk_records),
        "num_fk_generated_beats": int(
            sum(record["num_fk_generated_beats"] for record in fk_records)
        ),
        "num_fk_audio_beats": int(
            sum(record["num_fk_audio_beats"] for record in fk_records)
        ),
        "num_fk_matched_beats": int(
            sum(record["num_fk_matched_beats"] for record in fk_records)
        ),
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
        "G1 Beat Match": metrics["G1BAP_precision"],
        "Root Drift": metrics["RootDriftMean"],
        "Root Height Min": metrics["RootHeightMin"],
        "Root Height Max": metrics["RootHeightMax"],
        "Joint Range Viol.": metrics["ReferenceRangeViolationRate"],
        "G1Dist": metrics["G1Dist"],
        "G1Div": metrics["G1Div"],
    }
    if "G1FKBAS" in metrics:
        table.update(
            {
                "G1 FK Beat Align.": metrics["G1FKBAS"],
                "G1 Beat F1": metrics["G1BeatF1"],
                "G1 Foot Sliding": metrics["G1FootSliding"],
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
        f"- Designated beat precision: {metrics['G1BAP_precision']}",
        f"- Designated beat recall: {metrics['G1BAP_recall']}",
        f"- Root drift mean: {metrics['RootDriftMean']}",
        f"- Root height mean: {metrics['RootHeightMean']}",
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
                f"- Beat F1: {metrics['G1BeatF1']}",
                f"- Beat timing mean frames: {metrics['G1BeatTimingMeanFrames']}",
                f"- Beat timing std frames: {metrics['G1BeatTimingStdFrames']}",
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
    summaries = [summarize_g1_motion(motion, bounds=bounds) for motion in motions]
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

    audit = {
        "num_files": len(motion_files),
        "num_valid_files": len(motions),
        "bad_files": bad_files,
        "reference_bad_files": reference_bad_files,
        "diagnostic_renders": diagnostics,
        "fk_metrics_enabled": bool(enable_fk_metrics),
        "fk_model_path": str(fk_model_path) if fk_model_path else "",
        "root_quat_order": root_quat_order if enable_fk_metrics else "",
        "per_file": [
            {
                "path": motion["path"],
                **summary,
            }
            for motion, summary in zip(motions, summaries)
        ],
    }

    write_json(metrics_path, metrics)
    write_json(g1_table_path, table)
    write_json(motion_audit_path, audit)
    write_text(paper_report_path, render_g1_paper_report(metrics, table))

    print(json.dumps(json_safe(metrics), indent=2, sort_keys=True))
    print(f"Saved G1 metrics to {metrics_path}")
    return metrics
