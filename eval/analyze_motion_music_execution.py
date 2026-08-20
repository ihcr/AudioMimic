"""First-round audit of generated dance, music response, and SONIC retention.

This analysis deliberately reports a profile instead of a single "beauty"
score. Automatic metrics can describe smoothness, activity, repetition,
physical plausibility, and music response; aesthetic quality still requires a
blinded human study.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_generation_execution_gap import analyze_run
from eval.g1_kinematics import forward_g1_kinematics
from sonic_bridge import SONIC_REFERENCE_FROM_MUJOCO


DEFAULT_MODEL_PATH = REPO_ROOT / "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"
STATIC_SPEED_THRESHOLD_RAD_S = 0.08
REPEATED_POSE_THRESHOLD_RAD = 0.08
REPETITION_EXCLUSION_SECONDS = 2.0
FOOT_CONTACT_HEIGHT_M = 0.05
FOOT_SLIDE_SPEED_M_S = 0.20
BAS_SIGMA_SECONDS = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion_root",
        default="~/Musics2Dance-prior-dev/onlinegeneratedmotion",
        help="directory containing M0/M2/M4 PKLs and audio/",
    )
    parser.add_argument(
        "--tracking_root",
        default="eval/generation_to_execution_gap",
        help="directory containing standard aligned SONIC runs",
    )
    parser.add_argument(
        "--tracking_glob",
        default="m[024]_song098_seed1234_full_rate100_aligned_r0*",
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--output_dir",
        default="eval/motion_music_execution/first_round_20260819",
    )
    parser.add_argument("--analysis_fps", default=50.0, type=float)
    parser.add_argument("--max_lag_seconds", default=0.5, type=float)
    parser.add_argument("--fall_height", default=0.45, type=float)
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict]:
    payload = _load_json(path)
    return payload.get("records", payload) if isinstance(payload, dict) else payload


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def _interp_columns(source_t: np.ndarray, source_x: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_t, source_t, source_x[:, index]) for index in range(source_x.shape[1])],
        axis=1,
    )


def _monotonic_unique(t: np.ndarray, *values: np.ndarray):
    order = np.argsort(t, kind="stable")
    t = t[order]
    values = tuple(value[order] for value in values)
    keep = np.concatenate(([True], np.diff(t) > 1e-8))
    return (t[keep], *(value[keep] for value in values))


def _mujoco_from_sonic(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    output[:, SONIC_REFERENCE_FROM_MUJOCO] = values
    return output


def _percentiles(values: np.ndarray) -> dict:
    values = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _boundary_indices(frames: int, fps: float) -> np.ndarray:
    commit_seconds = 8.0 / 30.0
    indices = np.rint(np.arange(commit_seconds, frames / fps, commit_seconds) * fps).astype(int)
    return indices[(indices > 1) & (indices < frames)]


def _repeated_pose_ratio(
    dof: np.ndarray,
    fps: float,
    *,
    threshold_rad: float = REPEATED_POSE_THRESHOLD_RAD,
    exclusion_seconds: float = REPETITION_EXCLUSION_SECONDS,
) -> tuple[float, float]:
    frames = len(dof)
    exclusion = int(round(exclusion_seconds * fps))
    nearest = np.full(frames, np.inf, dtype=np.float64)
    indices = np.arange(frames)
    for start in range(0, frames, 96):
        stop = min(start + 96, frames)
        distance = np.sqrt(np.mean((dof[start:stop, None] - dof[None]) ** 2, axis=2))
        valid = np.abs(indices[None] - np.arange(start, stop)[:, None]) >= exclusion
        distance[~valid] = np.inf
        nearest[start:stop] = np.min(distance, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if not len(finite):
        return 0.0, float("nan")
    return float(np.mean(finite < threshold_rad)), float(np.median(finite))


def _angular_speed(quaternion: np.ndarray, fps: float) -> np.ndarray:
    quaternion = _normalize_quaternions(quaternion)
    dots = np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0)) * fps


def _fk_metrics(motion: dict, model_path: Path, quat_order: str) -> tuple[dict, np.ndarray]:
    fk = forward_g1_kinematics(motion, model_path, root_quat_order=quat_order)
    keypoints = np.asarray(fk["keypoints"], dtype=np.float64)
    left = np.asarray(fk["left_foot_points"], dtype=np.float64)
    right = np.asarray(fk["right_foot_points"], dtype=np.float64)
    fps = float(motion["fps"])

    keypoint_velocity = np.zeros_like(keypoints)
    keypoint_velocity[1:] = np.diff(keypoints, axis=0) * fps
    speed_curve = np.sqrt(np.mean(np.sum(keypoint_velocity**2, axis=2), axis=1))

    left_speed = np.linalg.norm(np.diff(left[:, :2], axis=0), axis=1) * fps
    right_speed = np.linalg.norm(np.diff(right[:, :2], axis=0), axis=1) * fps
    left_world_contact = left[1:, 2] < FOOT_CONTACT_HEIGHT_M
    right_world_contact = right[1:, 2] < FOOT_CONTACT_HEIGHT_M
    world_skating = (left_world_contact & (left_speed > FOOT_SLIDE_SPEED_M_S)) | (
        right_world_contact & (right_speed > FOOT_SLIDE_SPEED_M_S)
    )
    ground_height = float(np.percentile(np.concatenate((left[:, 2], right[:, 2])), 1.0))
    left_height = left[:, 2] - ground_height
    right_height = right[:, 2] - ground_height
    left_contact = left_height[1:] < FOOT_CONTACT_HEIGHT_M
    right_contact = right_height[1:] < FOOT_CONTACT_HEIGHT_M
    skating = (left_contact & (left_speed > FOOT_SLIDE_SPEED_M_S)) | (
        right_contact & (right_speed > FOOT_SLIDE_SPEED_M_S)
    )
    penetration = (left_height < -0.01) | (right_height < -0.01)

    pelvis = keypoints[:, 0]
    pelvis_velocity = np.gradient(pelvis, 1.0 / fps, axis=0)
    pelvis_acceleration = np.gradient(pelvis_velocity, 1.0 / fps, axis=0)
    planar_acceleration = np.linalg.norm(pelvis_acceleration[:, :2], axis=1)
    acceleration_scale = float(np.max(planar_acceleration))
    normalized_acceleration = planar_acceleration / max(acceleration_scale, 1e-12)
    foot_speed_product = np.zeros(len(pelvis), dtype=np.float64)
    foot_speed_product[1:] = left_speed * right_speed
    pfc_proxy = float(np.mean(foot_speed_product * normalized_acceleration))

    return (
        {
            "fsr_world_z0_contact5cm_speed20cms": float(np.mean(world_skating)),
            "fsr_ground_calibrated_proxy": float(np.mean(skating)),
            "left_contact_ratio": float(np.mean(left_contact)),
            "right_contact_ratio": float(np.mean(right_contact)),
            "ground_penetration_ratio": float(np.mean(penetration)),
            "estimated_ground_height_m": ground_height,
            "minimum_foot_height_relative_m": float(
                min(np.min(left_height), np.min(right_height))
            ),
            "pfc_proxy": pfc_proxy,
            "keypoint_speed": _percentiles(speed_curve),
        },
        speed_curve,
    )


def compute_motion_quality(
    motion: dict,
    *,
    model_path: Path,
    quat_order: str,
) -> tuple[dict, np.ndarray]:
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    fps = float(motion["fps"])
    velocity = np.gradient(dof, 1.0 / fps, axis=0)
    acceleration = np.gradient(velocity, 1.0 / fps, axis=0)
    jerk = np.gradient(acceleration, 1.0 / fps, axis=0)
    speed_curve = np.sqrt(np.mean(velocity**2, axis=1))
    smoothed_speed = gaussian_filter1d(speed_curve, sigma=max(1.0, 0.10 * fps))
    repeated_ratio, nearest_repeat = _repeated_pose_ratio(dof, fps)

    boundaries = _boundary_indices(len(dof), fps)
    position_jump = np.sqrt(np.mean(np.diff(dof, axis=0) ** 2, axis=1))
    velocity_jump = np.sqrt(np.mean(np.diff(velocity, axis=0) ** 2, axis=1))
    boundary_position = position_jump[boundaries - 1]
    boundary_velocity = velocity_jump[boundaries - 1]
    non_boundary = np.ones(len(position_jump), dtype=bool)
    non_boundary[boundaries - 1] = False

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    root_velocity = np.gradient(root_pos, 1.0 / fps, axis=0)
    root_planar_speed = np.linalg.norm(root_velocity[:, :2], axis=1)
    yaw_speed = _angular_speed(root_rot, fps)
    physical, fk_speed_curve = _fk_metrics(motion, model_path, quat_order)

    metrics = {
        "frames": int(len(dof)),
        "fps": fps,
        "duration_seconds": float(len(dof) / fps),
        "joint_amplitude_median_rad": float(
            np.median(np.percentile(dof, 95, axis=0) - np.percentile(dof, 5, axis=0))
        ),
        "motion_energy_rad2_s2": float(np.mean(velocity**2)),
        "joint_velocity_abs_rad_s": _percentiles(velocity),
        "joint_acceleration_abs_rad_s2": _percentiles(acceleration),
        "joint_jerk_abs_rad_s3": _percentiles(jerk),
        "static_ratio_speed_below_008": float(np.mean(smoothed_speed < STATIC_SPEED_THRESHOLD_RAD_S)),
        "repeated_pose_ratio_rms008_after2s": repeated_ratio,
        "median_nearest_nonlocal_pose_rms_rad": nearest_repeat,
        "c4_boundary_position_jump_p95_rad": float(np.percentile(boundary_position, 95)),
        "c4_boundary_velocity_jump_p95_rad_s": float(np.percentile(boundary_velocity, 95)),
        "c4_position_jump_ratio_to_nonboundary": _safe_ratio(
            float(np.median(boundary_position)), float(np.median(position_jump[non_boundary]))
        ),
        "c4_velocity_jump_ratio_to_nonboundary": _safe_ratio(
            float(np.median(boundary_velocity)), float(np.median(velocity_jump[non_boundary]))
        ),
        "root_planar_displacement_m": float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2])),
        "root_path_length_m": float(np.sum(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1))),
        "root_planar_speed_p95_m_s": float(np.percentile(root_planar_speed, 95)),
        "root_height_min_m": float(np.min(root_pos[:, 2])),
        "root_height_median_m": float(np.median(root_pos[:, 2])),
        "root_angular_speed_p95_rad_s": float(np.percentile(yaw_speed, 95)),
        "physical": physical,
    }
    return metrics, fk_speed_curve


def load_audio_analysis(audio_path: Path, fps: float, frames: int) -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/audiomimic-numba-cache")
    import librosa
    import scipy.signal

    # librosa 0.9 uses the pre-SciPy-1.13 alias removed from scipy.signal.
    if not hasattr(scipy.signal, "hann"):
        scipy.signal.hann = scipy.signal.windows.hann

    y, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = 512
    onset = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    onset_times = librosa.times_like(onset, sr=sample_rate, hop_length=hop_length)
    target_times = np.arange(frames, dtype=np.float64) / fps
    onset_curve = np.interp(target_times, onset_times, onset, left=0.0, right=0.0)
    _, beat_times = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        units="time",
    )
    beat_times = np.asarray(beat_times, dtype=np.float64)
    beat_times = beat_times[beat_times < frames / fps]
    return onset_curve, beat_times


def best_correlation(audio: np.ndarray, motion: np.ndarray, fps: float, max_lag_seconds: float = 1.0):
    audio = _zscore(audio)
    motion = _zscore(motion)
    max_frames = int(round(max_lag_seconds * fps))
    lags = np.arange(-max_frames, max_frames + 1)
    correlations = []
    for lag in lags:
        if lag >= 0:
            left, right = audio[: len(audio) - lag or None], motion[lag:]
        else:
            left, right = audio[-lag:], motion[: len(motion) + lag]
        correlations.append(float(np.mean(left * right)) if len(left) else float("nan"))
    correlations = np.asarray(correlations)
    best_index = int(np.nanargmax(correlations))
    zero_index = int(np.flatnonzero(lags == 0)[0])
    return {
        "zero_lag_correlation": float(correlations[zero_index]),
        "best_correlation": float(correlations[best_index]),
        "best_lag_seconds": float(lags[best_index] / fps),
    }


def _bas(audio_beats: np.ndarray, motion_beats: np.ndarray, direction: str) -> float:
    if not len(audio_beats) or not len(motion_beats):
        return 0.0
    source, target = (audio_beats, motion_beats) if direction == "music_to_motion" else (motion_beats, audio_beats)
    scores = [
        np.exp(-np.min((target - event) ** 2) / (2.0 * BAS_SIGMA_SECONDS**2))
        for event in source
    ]
    return float(np.mean(scores))


def compute_music_metrics(
    speed_curve: np.ndarray,
    *,
    fps: float,
    audio_path: Path,
    audio_cache: dict,
) -> dict:
    key = (str(audio_path), float(fps), int(len(speed_curve)))
    if key not in audio_cache:
        audio_cache[key] = load_audio_analysis(audio_path, fps, len(speed_curve))
    onset_curve, audio_beats = audio_cache[key]
    smooth_speed = gaussian_filter1d(speed_curve, sigma=max(1.0, fps * 5.0 / 30.0))
    impact_curve = gaussian_filter1d(
        np.abs(np.gradient(smooth_speed, 1.0 / fps)),
        sigma=max(1.0, 0.05 * fps),
    )
    motion_beat_frames = np.asarray(argrelextrema(smooth_speed, np.less)[0], dtype=np.int64)
    motion_beats = motion_beat_frames / fps
    speed_correlation = best_correlation(onset_curve, smooth_speed, fps)
    impact_correlation = best_correlation(onset_curve, impact_curve, fps)
    return {
        "speed_zero_lag_correlation": speed_correlation["zero_lag_correlation"],
        "speed_best_correlation": speed_correlation["best_correlation"],
        "speed_best_lag_seconds": speed_correlation["best_lag_seconds"],
        "impact_zero_lag_correlation": impact_correlation["zero_lag_correlation"],
        "impact_best_correlation": impact_correlation["best_correlation"],
        "impact_best_lag_seconds": impact_correlation["best_lag_seconds"],
        "impact_lag_reliable_at_corr010": bool(impact_correlation["best_correlation"] >= 0.10),
        "bas_music_to_motion": _bas(audio_beats, motion_beats, "music_to_motion"),
        "bas_motion_to_music": _bas(audio_beats, motion_beats, "motion_to_music"),
        "audio_beats": int(len(audio_beats)),
        "motion_beats": int(len(motion_beats)),
        "motion_beats_per_minute": float(len(motion_beats) / (len(speed_curve) / fps) * 60.0),
    }


def load_reference_motion(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    motion = {
        "fps": float(payload.get("fps", 30.0)),
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
    }
    metadata = {
        "route": str(payload["route"]),
        "sequence_id": str(payload["sequence_id"]),
        "training_seed": int(payload["training_seed"]),
        "sampling_seed": int(payload["sampling_seed"]),
        "checkpoint_update": int(payload["checkpoint_update"]),
        "motion_mse": float(payload["motion_mse"]),
        "audio_start_seconds": float(payload["audio_start_seconds"]),
    }
    return motion, metadata


def load_execution_pair(run_dir: Path, fps: float) -> tuple[dict, dict, dict]:
    metadata = _load_json(run_dir / "offline_sonic_playback.json")
    started = float(metadata["playback_started_monotonic_seconds"])
    finished = float(metadata["playback_finished_monotonic_seconds"])
    feedback = [
        record
        for record in _records(run_dir / "sonic_feedback.json")
        if started <= float(record.get("_received_monotonic_seconds", -np.inf)) <= finished
    ]
    t = np.asarray([record["_received_monotonic_seconds"] for record in feedback]) - started
    fields = {
        name: np.asarray([record[name] for record in feedback], dtype=np.float64)
        for name in (
            "body_q_target",
            "body_q_measured",
            "base_trans_target",
            "base_trans_measured",
            "base_quat_target",
            "base_quat_measured",
        )
    }
    ordered = _monotonic_unique(t, *(fields[name] for name in fields))
    t = ordered[0]
    fields = {name: value for name, value in zip(fields, ordered[1:])}
    alignment = float(metadata.get("alignment_seconds", 0.0))
    duration = float(metadata.get("selected_source_duration_seconds", 60.0))
    grid = np.arange(alignment, min(alignment + duration, float(t[-1])), 1.0 / fps)
    fields = {name: _interp_columns(t, value, grid) for name, value in fields.items()}
    for name in ("base_quat_target", "base_quat_measured"):
        fields[name] = _normalize_quaternions(fields[name])

    target = {
        "fps": fps,
        "dof_pos": _mujoco_from_sonic(fields["body_q_target"]),
        "root_pos": fields["base_trans_target"],
        "root_rot": fields["base_quat_target"],
    }
    sim = [
        record
        for record in _records(run_dir / "sim_state.json")
        if started <= float(record.get("_received_monotonic_seconds", -np.inf)) <= finished
        and "base_position" in record
        and "base_quat" in record
    ]
    sim_t = np.asarray([record["_received_monotonic_seconds"] for record in sim]) - started
    sim_pos = np.asarray([record["base_position"] for record in sim], dtype=np.float64)
    sim_quat = np.asarray([record["base_quat"] for record in sim], dtype=np.float64)
    sim_t, sim_pos, sim_quat = _monotonic_unique(sim_t, sim_pos, sim_quat)
    measured = {
        "fps": fps,
        "dof_pos": _mujoco_from_sonic(fields["body_q_measured"]),
        "root_pos": _interp_columns(sim_t, sim_pos, grid),
        "root_rot": _normalize_quaternions(_interp_columns(sim_t, sim_quat, grid)),
    }
    return target, measured, metadata


def resample_motion(motion: dict, fps: float, frames: int) -> dict:
    source_fps = float(motion["fps"])
    source_t = np.arange(len(motion["dof_pos"]), dtype=np.float64) / source_fps
    target_t = np.arange(frames, dtype=np.float64) / fps
    target_t = np.minimum(target_t, source_t[-1])
    return {
        "fps": fps,
        "dof_pos": _interp_columns(source_t, np.asarray(motion["dof_pos"]), target_t),
        "root_pos": _interp_columns(source_t, np.asarray(motion["root_pos"]), target_t),
        "root_rot": _normalize_quaternions(
            _interp_columns(source_t, np.asarray(motion["root_rot"]), target_t)
        ),
    }


def _flatten(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, child, output)
    elif isinstance(value, (int, float, np.integer, np.floating)) or value is None:
        output[prefix] = value


def _write_csv(path: Path, records: list[dict]) -> None:
    flattened = []
    fields = []
    for record in records:
        row = {}
        _flatten("", record, row)
        flattened.append(row)
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _numeric_path(record: dict, path: str):
    value = record
    for part in path.split("."):
        value = value[part]
    return value


def summarize_routes(records: list[dict], metric_paths: tuple[str, ...]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["route"]].append(record)
    output = {}
    for route, route_records in sorted(grouped.items()):
        output[route] = {"samples": len(route_records)}
        for path in metric_paths:
            values = np.asarray([_numeric_path(record, path) for record in route_records], dtype=float)
            output[route][path] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return output


def _lag_aligned_arrays(
    target: np.ndarray,
    measured: np.ndarray,
    lag_seconds: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    frames = int(round(lag_seconds * fps))
    if frames >= 0:
        return target[: len(target) - frames or None], measured[frames:]
    return target[-frames:], measured[: len(measured) + frames]


def _binary_metrics(target: np.ndarray, measured: np.ndarray) -> dict:
    target = np.asarray(target, dtype=bool)
    measured = np.asarray(measured, dtype=bool)
    true_positive = int(np.sum(target & measured))
    false_positive = int(np.sum(~target & measured))
    false_negative = int(np.sum(target & ~measured))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else None
    recall = true_positive / recall_denominator if recall_denominator else None
    if precision is None or recall is None or precision + recall == 0.0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_contact_ratio": float(np.mean(target)),
        "measured_contact_ratio": float(np.mean(measured)),
    }


def _transition_timing_metrics(
    target: np.ndarray,
    measured: np.ndarray,
    fps: float,
    tolerance_seconds: float = 0.25,
) -> dict:
    target_transitions = np.flatnonzero(np.diff(np.asarray(target, dtype=np.int8)) != 0) + 1
    measured_transitions = np.flatnonzero(np.diff(np.asarray(measured, dtype=np.int8)) != 0) + 1
    if not len(target_transitions):
        return {
            "target_transitions": 0,
            "measured_transitions": int(len(measured_transitions)),
            "matched_ratio": 1.0 if not len(measured_transitions) else 0.0,
            "median_absolute_error_ms": 0.0 if not len(measured_transitions) else None,
            "p95_absolute_error_ms": 0.0 if not len(measured_transitions) else None,
        }
    if not len(measured_transitions):
        return {
            "target_transitions": int(len(target_transitions)),
            "measured_transitions": 0,
            "matched_ratio": 0.0,
            "median_absolute_error_ms": None,
            "p95_absolute_error_ms": None,
        }
    nearest_frames = np.min(
        np.abs(target_transitions[:, None] - measured_transitions[None, :]), axis=1
    )
    matched = nearest_frames <= int(round(tolerance_seconds * fps))
    matched_errors_ms = nearest_frames[matched] * 1000.0 / fps
    return {
        "target_transitions": int(len(target_transitions)),
        "measured_transitions": int(len(measured_transitions)),
        "matched_ratio": float(np.mean(matched)),
        "median_absolute_error_ms": (
            float(np.median(matched_errors_ms)) if len(matched_errors_ms) else None
        ),
        "p95_absolute_error_ms": (
            float(np.percentile(matched_errors_ms, 95)) if len(matched_errors_ms) else None
        ),
    }


def compute_fk_tracking_metrics(
    target_motion: dict,
    measured_motion: dict,
    *,
    model_path: Path,
    lag_seconds: float,
) -> dict:
    fps = float(target_motion["fps"])
    target_fk = forward_g1_kinematics(target_motion, model_path, root_quat_order="wxyz")
    measured_fk = forward_g1_kinematics(measured_motion, model_path, root_quat_order="wxyz")

    target_keypoints = np.asarray(target_fk["keypoints"], dtype=np.float64)
    measured_keypoints = np.asarray(measured_fk["keypoints"], dtype=np.float64)
    target_local = target_keypoints - target_keypoints[:, :1]
    measured_local = measured_keypoints - measured_keypoints[:, :1]
    target_local, measured_local = _lag_aligned_arrays(
        target_local, measured_local, lag_seconds, fps
    )
    keypoint_error = np.linalg.norm(measured_local[:, 1:] - target_local[:, 1:], axis=2)
    keypoint_names = list(target_fk["keypoint_names"])[1:]

    contact = {}
    for side in ("left", "right"):
        target_foot = np.asarray(target_fk[f"{side}_foot_points"], dtype=np.float64)
        measured_foot = np.asarray(measured_fk[f"{side}_foot_points"], dtype=np.float64)
        target_ground = float(np.percentile(target_foot[:, 2], 1.0))
        measured_ground = float(np.percentile(measured_foot[:, 2], 1.0))
        target_mask = target_foot[:, 2] - target_ground < FOOT_CONTACT_HEIGHT_M
        measured_mask = measured_foot[:, 2] - measured_ground < FOOT_CONTACT_HEIGHT_M
        target_mask, measured_mask = _lag_aligned_arrays(
            target_mask, measured_mask, lag_seconds, fps
        )
        contact[side] = {
            **_binary_metrics(target_mask, measured_mask),
            "transition_timing": _transition_timing_metrics(target_mask, measured_mask, fps),
        }

    return {
        "root_relative_empkpe_m": float(np.mean(keypoint_error)),
        "root_relative_empkpe_p95_m": float(np.percentile(keypoint_error, 95)),
        "per_keypoint_mean_error_m": {
            name: float(np.mean(keypoint_error[:, index]))
            for index, name in enumerate(keypoint_names)
        },
        "contact": contact,
    }


def _execution_loss(target: dict, measured: dict, target_music: dict, measured_music: dict) -> dict:
    return {
        "motion_energy_retention": _safe_ratio(
            measured["motion_energy_rad2_s2"], target["motion_energy_rad2_s2"]
        ),
        "median_amplitude_statistic_ratio": _safe_ratio(
            measured["joint_amplitude_median_rad"], target["joint_amplitude_median_rad"]
        ),
        "jerk_p95_ratio": _safe_ratio(
            measured["joint_jerk_abs_rad_s3"]["p95"], target["joint_jerk_abs_rad_s3"]["p95"]
        ),
        "fsr_ground_calibrated_proxy_change": measured["physical"]["fsr_ground_calibrated_proxy"]
        - target["physical"]["fsr_ground_calibrated_proxy"],
        "static_ratio_change": measured["static_ratio_speed_below_008"]
        - target["static_ratio_speed_below_008"],
        "repeated_pose_ratio_change": measured["repeated_pose_ratio_rms008_after2s"]
        - target["repeated_pose_ratio_rms008_after2s"],
        "onset_impact_correlation_retention": _safe_ratio(
            measured_music["impact_best_correlation"], target_music["impact_best_correlation"]
        ),
        "onset_impact_correlation_change": measured_music["impact_best_correlation"]
        - target_music["impact_best_correlation"],
        "bas_music_to_motion_retention": _safe_ratio(
            measured_music["bas_music_to_motion"], target_music["bas_music_to_motion"]
        ),
        "bas_music_to_motion_change": measured_music["bas_music_to_motion"]
        - target_music["bas_music_to_motion"],
        "additional_music_response_lag_seconds": measured_music["impact_best_lag_seconds"]
        - target_music["impact_best_lag_seconds"],
    }


def main(args: argparse.Namespace) -> None:
    motion_root = Path(args.motion_root).expanduser().resolve()
    tracking_root = Path(args.tracking_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_cache = {}

    reference_records = []
    for path in sorted(motion_root.rglob("*.pkl")):
        motion, metadata = load_reference_motion(path)
        audio_path = motion_root / "audio" / f"{metadata['sequence_id']}_t000128_60s.wav"
        quality, speed_curve = compute_motion_quality(
            motion,
            model_path=model_path,
            quat_order="xyzw",
        )
        music = compute_music_metrics(
            speed_curve,
            fps=motion["fps"],
            audio_path=audio_path,
            audio_cache=audio_cache,
        )
        record = {
            "motion_id": path.stem,
            "motion_path": str(path),
            "audio_path": str(audio_path),
            **metadata,
            "quality": quality,
            "music": music,
        }
        reference_records.append(record)
        print(
            f"reference {metadata['route']} {path.stem}: "
            f"energy={quality['motion_energy_rad2_s2']:.3f} "
            f"jerk95={quality['joint_jerk_abs_rad_s3']['p95']:.1f} "
            f"impact_corr={music['impact_best_correlation']:.3f} "
            f"BAS={music['bas_music_to_motion']:.3f}"
        )

    execution_records = []
    for run_dir in sorted(tracking_root.glob(args.tracking_glob)):
        sonic_target_motion, measured_motion, playback = load_execution_pair(
            run_dir, args.analysis_fps
        )
        source_motion, source = load_reference_motion(Path(playback["motion_path"]))
        target_motion = resample_motion(source_motion, args.analysis_fps, len(measured_motion["dof_pos"]))
        route = str(source["route"])
        sequence_id = str(source["sequence_id"])
        audio_path = motion_root / "audio" / f"{sequence_id}_t000128_60s.wav"
        target_quality, target_speed = compute_motion_quality(
            target_motion,
            model_path=model_path,
            quat_order="xyzw",
        )
        measured_quality, measured_speed = compute_motion_quality(
            measured_motion,
            model_path=model_path,
            quat_order="wxyz",
        )
        target_music = compute_music_metrics(
            target_speed,
            fps=args.analysis_fps,
            audio_path=audio_path,
            audio_cache=audio_cache,
        )
        measured_music = compute_music_metrics(
            measured_speed,
            fps=args.analysis_fps,
            audio_path=audio_path,
            audio_cache=audio_cache,
        )
        tracking = analyze_run(
            run_dir,
            fps=args.analysis_fps,
            max_lag_seconds=args.max_lag_seconds,
            fall_height=args.fall_height,
            trim_start=0.0,
            trim_end=0.0,
            include_post_fall=False,
        )
        fk_tracking = compute_fk_tracking_metrics(
            sonic_target_motion,
            measured_motion,
            model_path=model_path,
            lag_seconds=tracking["global_lag_seconds"],
        )
        losses = _execution_loss(target_quality, measured_quality, target_music, measured_music)
        record = {
            "run_id": run_dir.name,
            "route": route,
            "sequence_id": sequence_id,
            "motion_path": playback["motion_path"],
            "target_quality": target_quality,
            "execution_quality": measured_quality,
            "target_music": target_music,
            "execution_music": measured_music,
            "loss": losses,
            "tracking": {
                "raw_position_rmse_rad": tracking["raw_position_rmse_rad"],
                "lag_compensated_position_rmse_rad": tracking[
                    "lag_compensated_position_rmse_rad"
                ],
                "global_lag_seconds": tracking["global_lag_seconds"],
                "minimum_base_height_m": tracking["minimum_base_height_m"],
                "survival_seconds": tracking["survival_seconds"],
                "fell": tracking["fell_below_height_threshold"],
                "body_groups": tracking["groups"],
                "fk": fk_tracking,
            },
        }
        execution_records.append(record)
        print(
            f"execution {route} {run_dir.name}: "
            f"energy_ret={losses['motion_energy_retention']:.3f} "
            f"impact_corr={target_music['impact_best_correlation']:.3f}->"
            f"{measured_music['impact_best_correlation']:.3f} "
            f"BAS={target_music['bas_music_to_motion']:.3f}->{measured_music['bas_music_to_motion']:.3f}"
        )

    reference_metric_paths = (
        "motion_mse",
        "quality.joint_amplitude_median_rad",
        "quality.motion_energy_rad2_s2",
        "quality.joint_velocity_abs_rad_s.p95",
        "quality.joint_acceleration_abs_rad_s2.p95",
        "quality.joint_jerk_abs_rad_s3.p95",
        "quality.static_ratio_speed_below_008",
        "quality.repeated_pose_ratio_rms008_after2s",
        "quality.c4_position_jump_ratio_to_nonboundary",
        "quality.c4_velocity_jump_ratio_to_nonboundary",
        "quality.root_planar_displacement_m",
        "quality.root_path_length_m",
        "quality.root_height_min_m",
        "quality.physical.fsr_world_z0_contact5cm_speed20cms",
        "quality.physical.pfc_proxy",
        "music.speed_zero_lag_correlation",
        "music.speed_best_correlation",
        "music.impact_zero_lag_correlation",
        "music.impact_best_correlation",
        "music.impact_best_lag_seconds",
        "music.bas_music_to_motion",
        "music.bas_motion_to_music",
    )
    execution_metric_paths = (
        "tracking.raw_position_rmse_rad",
        "tracking.lag_compensated_position_rmse_rad",
        "tracking.global_lag_seconds",
        "tracking.fk.root_relative_empkpe_m",
        "tracking.fk.root_relative_empkpe_p95_m",
        "tracking.fk.contact.left.f1",
        "tracking.fk.contact.right.f1",
        "tracking.fk.contact.left.transition_timing.matched_ratio",
        "tracking.fk.contact.right.transition_timing.matched_ratio",
        "tracking.body_groups.full_body.band_power_retention.low_0_1.median",
        "tracking.body_groups.full_body.band_power_retention.mid_1_3.median",
        "tracking.body_groups.full_body.band_power_retention.high_3_8.median",
        "tracking.body_groups.legs.band_power_retention.high_3_8.median",
        "tracking.body_groups.waist.band_power_retention.high_3_8.median",
        "tracking.body_groups.arms.band_power_retention.high_3_8.median",
        "loss.motion_energy_retention",
        "loss.median_amplitude_statistic_ratio",
        "loss.jerk_p95_ratio",
        "loss.fsr_ground_calibrated_proxy_change",
        "loss.static_ratio_change",
        "loss.repeated_pose_ratio_change",
        "loss.onset_impact_correlation_change",
        "loss.bas_music_to_motion_change",
        "loss.additional_music_response_lag_seconds",
    )
    matched_song098 = [record for record in reference_records if record["sequence_id"] == "098"]
    summary = {
        "scope": {
            "reference_files": len(reference_records),
            "matched_song098_reference_files": len(matched_song098),
            "execution_runs": len(execution_records),
            "notes": [
                "Reference route summaries use all available exported PKLs.",
                "Matched song098 summaries remove the M0 song065 music confound.",
                "Execution summaries are three SONIC repeats of one fixed seed1234 trajectory per route.",
                "Automatic metrics do not establish aesthetic beauty without a human study.",
            ],
        },
        "reference_all_available": summarize_routes(reference_records, reference_metric_paths),
        "reference_matched_song098": summarize_routes(matched_song098, reference_metric_paths),
        "execution_fixed_reference_repeats": summarize_routes(execution_records, execution_metric_paths),
    }
    (output_dir / "reference_metrics.json").write_text(
        json.dumps(reference_records, indent=2), encoding="utf-8"
    )
    (output_dir / "execution_metrics.json").write_text(
        json.dumps(execution_records, indent=2), encoding="utf-8"
    )
    (output_dir / "route_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output_dir / "reference_metrics.csv", reference_records)
    _write_csv(output_dir / "execution_metrics.csv", execution_records)
    print(f"wrote first-round audit to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
