"""Compute reference-to-execution metrics for SONIC future-window ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FPS = 30.0
COMMIT_FRAMES = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_lag_seconds", type=float, default=0.5)
    parser.add_argument("--fall_height", type=float, default=0.45)
    return parser.parse_args()


def _records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", payload) if isinstance(payload, dict) else payload


def _interp_columns(source_t, source_x, target_t):
    return np.stack(
        [np.interp(target_t, source_t, source_x[:, index]) for index in range(source_x.shape[1])],
        axis=1,
    )


def _normalize_quat(quat):
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-8)


def _quat_conjugate_wxyz(quat):
    result = np.asarray(quat, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply_wxyz(left, right):
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _relative_quat(sequence):
    sequence = _normalize_quat(sequence)
    inverse_initial = _quat_conjugate_wxyz(sequence[0])
    return _normalize_quat(_quat_multiply_wxyz(np.broadcast_to(inverse_initial, sequence.shape), sequence))


def _reference_quat(raw):
    yaw = 0.0
    quats = []
    for frame, value in enumerate(raw):
        if frame % COMMIT_FRAMES:
            yaw += float(np.arctan2(value[3], value[4]))
        quats.append((np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)))
    return np.asarray(quats, dtype=np.float64)


def _orientation_error_deg(reference, actual):
    dot = np.abs(np.sum(_normalize_quat(reference) * _normalize_quat(actual), axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _start_sim_time(runtime, sim_records):
    target = float(runtime["online_started_monotonic"])
    candidates = [record for record in sim_records if "_received_monotonic_seconds" in record]
    if not candidates:
        raise ValueError("sim_state records do not carry receive timestamps")
    nearest = min(candidates, key=lambda record: abs(float(record["_received_monotonic_seconds"]) - target))
    return float(nearest["sim_time"]), abs(float(nearest["_received_monotonic_seconds"]) - target)


def analyze_run(run_dir: Path, max_lag: float, fall_height: float):
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    raw = np.asarray(np.load(run_dir / "generated_motion.npy"), dtype=np.float64)
    feedback_records = _records(run_dir / "sonic_feedback.json")
    s66_records = _records(run_dir / "s66_exec.json")
    sim_records = _records(run_dir / "sim_state.json")
    start_sim, start_alignment_error = _start_sim_time(runtime, sim_records)

    s66_records = [record for record in s66_records if record.get("sim_time") is not None]
    execution_t = np.asarray([float(record["sim_time"]) - start_sim for record in s66_records])
    execution = np.asarray([record["s66"] for record in s66_records], dtype=np.float64)
    duration = raw.shape[0] / FPS
    online_start = float(runtime["online_started_monotonic"])
    feedback_records = [
        record
        for record in feedback_records
        if online_start <= float(record.get("_received_monotonic_seconds", -np.inf)) < online_start + duration
    ]
    if len(feedback_records) < 20:
        raise ValueError(f"{run_dir}: insufficient SONIC feedback records")
    measured_q = np.asarray([record["body_q_measured"] for record in feedback_records], dtype=np.float64)
    target_q = np.asarray([record["body_q_target"] for record in feedback_records], dtype=np.float64)
    controller_q_error = measured_q - target_q
    measured_base_quat = np.asarray(
        [record["base_quat_measured"] for record in feedback_records], dtype=np.float64
    )
    target_base_quat = np.asarray(
        [record["base_quat_target"] for record in feedback_records], dtype=np.float64
    )
    controller_orientation_error = _orientation_error_deg(target_base_quat, measured_base_quat)
    measured_base_trans = np.asarray(
        [record["base_trans_measured"] for record in feedback_records], dtype=np.float64
    )
    target_base_trans = np.asarray(
        [record["base_trans_target"] for record in feedback_records], dtype=np.float64
    )
    translation_delta = measured_base_trans - target_base_trans
    controller_translation_error = np.linalg.norm(translation_delta - translation_delta[0], axis=1)
    keep = (execution_t >= 0.0) & (execution_t < duration)
    execution_t, execution = execution_t[keep], execution[keep]
    if execution.shape[0] < 20:
        raise ValueError(f"{run_dir}: insufficient synchronized execution records")

    order = np.argsort(execution_t)
    execution_t, execution = execution_t[order], execution[order]
    unique = np.concatenate(([True], np.diff(execution_t) > 1e-8))
    execution_t, execution = execution_t[unique], execution[unique]
    reference_t = np.arange(raw.shape[0], dtype=np.float64) / FPS
    reference_q = raw[:, 5:34]
    reference_dq = np.zeros_like(reference_q)
    reference_dq[1:] = np.diff(reference_q, axis=0) * FPS
    actual_q = execution[:, 5:34]
    actual_dq = execution[:, 34:63]

    reference_q_at_actual = _interp_columns(reference_t, reference_q, execution_t)
    reference_dq_at_actual = _interp_columns(reference_t, reference_dq, execution_t)
    position_error = actual_q - reference_q_at_actual
    velocity_error = actual_dq - reference_dq_at_actual

    lag_candidates = np.arange(-max_lag, max_lag + 0.0001, 0.01)
    lag_rmse = []
    for lag in lag_candidates:
        valid = (execution_t - lag >= reference_t[0]) & (execution_t - lag <= reference_t[-1])
        if valid.sum() < 20:
            lag_rmse.append(np.inf)
            continue
        shifted_reference = _interp_columns(reference_t, reference_q, execution_t[valid] - lag)
        lag_rmse.append(float(np.sqrt(np.mean((actual_q[valid] - shifted_reference) ** 2))))
    best_index = int(np.argmin(lag_rmse))
    best_lag = float(lag_candidates[best_index])

    valid_lag = (execution_t - best_lag >= 0.0) & (execution_t - best_lag <= reference_t[-1])
    lag_reference = _interp_columns(reference_t, reference_q, execution_t[valid_lag] - best_lag)
    lag_error = actual_q[valid_lag] - lag_reference
    ref_std = np.std(reference_q_at_actual, axis=0)
    actual_std = np.std(actual_q, axis=0)
    active = ref_std > 0.02
    amplitude_ratio = actual_std[active] / ref_std[active]
    correlations = []
    for index in np.flatnonzero(active):
        if actual_std[index] > 1e-6:
            correlations.append(np.corrcoef(reference_q_at_actual[:, index], actual_q[:, index])[0, 1])

    sim = [record for record in sim_records if record.get("sim_time") is not None]
    sim_t = np.asarray([float(record["sim_time"]) - start_sim for record in sim])
    sim_keep = (sim_t >= 0.0) & (sim_t < duration)
    sim_t = sim_t[sim_keep]
    sim = [record for record, selected in zip(sim, sim_keep) if selected]
    base_height = np.asarray([record["base_position"][2] for record in sim], dtype=np.float64)
    actual_quat = np.asarray([record["base_quat"] for record in sim], dtype=np.float64)
    reference_quat = _reference_quat(raw)
    reference_quat_at_sim = np.stack(
        [reference_quat[np.clip(np.rint(sim_t * FPS).astype(int), 0, len(reference_quat) - 1)]],
        axis=0,
    )[0]
    orientation_error = _orientation_error_deg(
        _relative_quat(reference_quat_at_sim),
        _relative_quat(actual_quat),
    )
    fallen = base_height < fall_height
    survival = float(sim_t[np.argmax(fallen)]) if fallen.any() else duration

    per_joint_rmse = np.sqrt(np.mean(position_error**2, axis=0))
    return {
        "preview_mode": runtime["preview_mode"],
        "feedback_source": runtime.get("feedback_source", "fixed_rollout"),
        "duration_seconds": duration,
        "execution_samples": int(len(execution_t)),
        "sim_samples": int(len(sim_t)),
        "sonic_feedback_samples": int(len(feedback_records)),
        "start_alignment_error_seconds": start_alignment_error,
        "controller_joint_target_rmse_rad": float(np.sqrt(np.mean(controller_q_error**2))),
        "controller_joint_target_mae_rad": float(np.mean(np.abs(controller_q_error))),
        "controller_base_target_orientation_mean_error_deg": float(
            np.mean(controller_orientation_error)
        ),
        "controller_base_target_orientation_p95_error_deg": float(
            np.percentile(controller_orientation_error, 95)
        ),
        "controller_base_target_relative_translation_mean_error_m": float(
            np.mean(controller_translation_error)
        ),
        "controller_base_target_relative_translation_p95_error_m": float(
            np.percentile(controller_translation_error, 95)
        ),
        "joint_position_rmse_rad": float(np.sqrt(np.mean(position_error**2))),
        "joint_position_mae_rad": float(np.mean(np.abs(position_error))),
        "joint_position_max_abs_rad": float(np.max(np.abs(position_error))),
        "joint_velocity_rmse_rad_s": float(np.sqrt(np.mean(velocity_error**2))),
        "best_lag_seconds": best_lag,
        "lag_compensated_joint_rmse_rad": float(np.sqrt(np.mean(lag_error**2))),
        "median_active_joint_amplitude_ratio": float(np.median(amplitude_ratio)),
        "mean_active_joint_correlation": float(np.nanmean(correlations)),
        "root_orientation_mean_error_deg": float(np.mean(orientation_error)),
        "root_orientation_p95_error_deg": float(np.percentile(orientation_error, 95)),
        "minimum_base_height_m": float(np.min(base_height)),
        "survival_seconds": survival,
        "fell_below_height_threshold": bool(fallen.any()),
        "per_joint_position_rmse_rad": per_joint_rmse.tolist(),
    }


def main(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [analyze_run(Path(run).expanduser().resolve(), args.max_lag_seconds, args.fall_height) for run in args.runs]
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    scalar_keys = [key for key, value in results[0].items() if not isinstance(value, list)]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in scalar_keys} for row in results])
    for result in results:
        print(
            f"{result['feedback_source']}+{result['preview_mode']}",
            f"q_rmse={result['joint_position_rmse_rad']:.4f}",
            f"lag={result['best_lag_seconds']:.3f}s",
            f"lag_rmse={result['lag_compensated_joint_rmse_rad']:.4f}",
            f"ori={result['root_orientation_mean_error_deg']:.2f}deg",
            f"survival={result['survival_seconds']:.2f}s",
        )


if __name__ == "__main__":
    main(parse_args())
