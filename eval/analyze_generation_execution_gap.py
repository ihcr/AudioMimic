"""Quantify the gap between SONIC joint targets and measured execution."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.g1_kinematics import EXPECTED_G1_29DOF_JOINTS
from sonic_bridge import SONIC_REFERENCE_FROM_MUJOCO


JOINT_NAMES = tuple(EXPECTED_G1_29DOF_JOINTS[index] for index in SONIC_REFERENCE_FROM_MUJOCO)
GROUPS = {
    "legs": tuple(index for index, name in enumerate(JOINT_NAMES) if any(part in name for part in ("hip", "knee", "ankle"))),
    "waist": tuple(index for index, name in enumerate(JOINT_NAMES) if "waist" in name),
    "arms": tuple(index for index, name in enumerate(JOINT_NAMES) if any(part in name for part in ("shoulder", "elbow", "wrist"))),
    "full_body": tuple(range(29)),
}
BANDS_HZ = {"low_0_1": (0.0, 1.0), "mid_1_3": (1.0, 3.0), "high_3_8": (3.0, 8.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="stream_to_sonic output directories")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--analysis_fps", type=float, default=50.0)
    parser.add_argument("--max_lag_seconds", type=float, default=0.5)
    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--trim_start_seconds", type=float, default=0.0)
    parser.add_argument("--trim_end_seconds", type=float, default=0.0)
    parser.add_argument(
        "--include_post_fall_tracking",
        action="store_true",
        help="include feedback after base height first falls below the threshold",
    )
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict]:
    payload = _load_json(path)
    return payload.get("records", payload) if isinstance(payload, dict) else payload


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


def _common_grid(t: np.ndarray, fps: float, trim_start: float, trim_end: float) -> np.ndarray:
    start = float(t[0]) + trim_start
    stop = float(t[-1]) - trim_end
    if stop - start < 2.0:
        raise ValueError("less than two seconds remain after trimming")
    return np.arange(start, stop, 1.0 / fps, dtype=np.float64)


def estimate_lag_seconds(
    target: np.ndarray,
    measured: np.ndarray,
    fps: float,
    max_lag_seconds: float,
) -> tuple[float, np.ndarray]:
    """Return positive lag when measured(t) best matches target(t - lag)."""
    max_frames = int(round(max_lag_seconds * fps))
    candidates = np.arange(-max_frames, max_frames + 1, dtype=np.int64)
    errors = np.full(candidates.shape, np.inf, dtype=np.float64)
    for offset, lag_frames in enumerate(candidates):
        if lag_frames >= 0:
            left, right = target[: len(target) - lag_frames or None], measured[lag_frames:]
        else:
            left, right = target[-lag_frames:], measured[: len(measured) + lag_frames]
        if len(left) >= int(fps):
            errors[offset] = np.sqrt(np.mean((right - left) ** 2))
    best = int(candidates[int(np.argmin(errors))])
    return best / fps, errors


def _lag_aligned(target: np.ndarray, measured: np.ndarray, lag_seconds: float, fps: float):
    frames = int(round(lag_seconds * fps))
    if frames >= 0:
        return target[: len(target) - frames or None], measured[frames:]
    return target[-frames:], measured[: len(measured) + frames]


def _welch_band_power(values: np.ndarray, fps: float, low: float, high: float) -> np.ndarray:
    segment = min(256, len(values))
    if segment < 16:
        return np.full(values.shape[1], np.nan)
    step = max(1, segment // 2)
    window = np.hanning(segment)
    scale = fps * np.sum(window**2)
    spectra = []
    for start in range(0, len(values) - segment + 1, step):
        centered = values[start : start + segment] - np.mean(values[start : start + segment], axis=0)
        spectra.append(np.abs(np.fft.rfft(centered * window[:, None], axis=0)) ** 2 / scale)
    psd = np.mean(spectra, axis=0)
    frequencies = np.fft.rfftfreq(segment, d=1.0 / fps)
    selected = (frequencies >= low) & (frequencies < high)
    return np.trapezoid(psd[selected], frequencies[selected], axis=0)


def _safe_ratios(numerator: np.ndarray, denominator: np.ndarray, active: np.ndarray) -> np.ndarray:
    ratios = np.full_like(denominator, np.nan, dtype=np.float64)
    keep = active & (denominator > 1e-10)
    ratios[keep] = numerator[keep] / denominator[keep]
    return ratios


def _summarize(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"median": None, "p95": None, "mean": None}
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "mean": float(np.mean(finite)),
    }


def analyze_run(
    run_dir: Path,
    *,
    fps: float,
    max_lag_seconds: float,
    fall_height: float,
    trim_start: float,
    trim_end: float,
    include_post_fall: bool,
) -> dict:
    metadata = _load_json(run_dir / "offline_sonic_playback.json")
    started = float(metadata["playback_started_monotonic_seconds"])
    finished = float(metadata["playback_finished_monotonic_seconds"])
    feedback = [
        record
        for record in _records(run_dir / "sonic_feedback.json")
        if started <= float(record.get("_received_monotonic_seconds", -np.inf)) <= finished
    ]
    if len(feedback) < int(2 * fps):
        raise ValueError(f"{run_dir}: insufficient feedback inside playback interval")
    t = np.asarray([record["_received_monotonic_seconds"] for record in feedback], dtype=np.float64) - started
    target = np.asarray([record["body_q_target"] for record in feedback], dtype=np.float64)
    measured = np.asarray([record["body_q_measured"] for record in feedback], dtype=np.float64)
    t, target, measured = _monotonic_unique(t, target, measured)
    alignment_seconds = float(metadata.get("alignment_seconds", 0.0))
    grid = _common_grid(t, fps, alignment_seconds + trim_start, trim_end)

    sim_records = _records(run_dir / "sim_state.json")
    sim = [
        record
        for record in sim_records
        if started <= float(record.get("_received_monotonic_seconds", -np.inf)) <= finished
        and "base_position" in record
    ]
    heights = np.asarray([record["base_position"][2] for record in sim], dtype=np.float64)
    sim_t = np.asarray([record["_received_monotonic_seconds"] for record in sim], dtype=np.float64) - started
    fallen = heights < fall_height
    first_fall_time = float(sim_t[np.flatnonzero(fallen)[0]]) if fallen.any() else None
    if first_fall_time is not None and not include_post_fall:
        grid = grid[grid < first_fall_time]
        if len(grid) < int(2 * fps):
            raise ValueError(f"{run_dir}: less than two seconds of pre-fall feedback")

    target = _interp_columns(t, target, grid)
    measured = _interp_columns(t, measured, grid)
    target_velocity = np.gradient(target, 1.0 / fps, axis=0)
    measured_velocity = np.gradient(measured, 1.0 / fps, axis=0)
    raw_error = measured - target
    raw_velocity_error = measured_velocity - target_velocity
    active = np.ptp(target, axis=0) > 0.02

    per_joint_lag = np.empty(29, dtype=np.float64)
    per_joint_aligned_rmse = np.empty(29, dtype=np.float64)
    for joint in range(29):
        lag, _ = estimate_lag_seconds(
            target[:, joint : joint + 1], measured[:, joint : joint + 1], fps, max_lag_seconds
        )
        per_joint_lag[joint] = lag
        aligned_target, aligned_measured = _lag_aligned(
            target[:, joint], measured[:, joint], lag, fps
        )
        per_joint_aligned_rmse[joint] = np.sqrt(np.mean((aligned_measured - aligned_target) ** 2))

    global_lag, _ = estimate_lag_seconds(target[:, active], measured[:, active], fps, max_lag_seconds)
    aligned_target, aligned_measured = _lag_aligned(target, measured, global_lag, fps)
    target_amplitude = np.percentile(target, 95, axis=0) - np.percentile(target, 5, axis=0)
    measured_amplitude = np.percentile(measured, 95, axis=0) - np.percentile(measured, 5, axis=0)
    amplitude_retention = _safe_ratios(measured_amplitude, target_amplitude, active)
    target_energy = np.mean(target_velocity**2, axis=0)
    measured_energy = np.mean(measured_velocity**2, axis=0)
    energy_retention = _safe_ratios(measured_energy, target_energy, active)
    band_retention = {}
    for name, (low, high) in BANDS_HZ.items():
        target_power = _welch_band_power(target, fps, low, high)
        measured_power = _welch_band_power(measured, fps, low, high)
        band_retention[name] = _safe_ratios(measured_power, target_power, active)

    group_metrics = {}
    for name, indices_tuple in GROUPS.items():
        indices = np.asarray(indices_tuple, dtype=np.int64)
        group_active = indices[active[indices]]
        group_metrics[name] = {
            "raw_position_rmse_rad": float(np.sqrt(np.mean(raw_error[:, indices] ** 2))),
            "raw_velocity_rmse_rad_s": float(np.sqrt(np.mean(raw_velocity_error[:, indices] ** 2))),
            "lag_ms": _summarize(per_joint_lag[group_active] * 1000.0),
            "lag_compensated_position_rmse_rad": float(np.sqrt(np.mean((aligned_measured[:, indices] - aligned_target[:, indices]) ** 2))),
            "amplitude_retention": _summarize(amplitude_retention[group_active]),
            "energy_retention": _summarize(energy_retention[group_active]),
            "band_power_retention": {
                band: _summarize(ratios[group_active]) for band, ratios in band_retention.items()
            },
        }

    total_survival = first_fall_time if first_fall_time is not None else finished - started
    survival = max(0.0, total_survival - alignment_seconds)

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "motion_sha256": metadata["motion_sha256"],
        "playback_duration_seconds": finished - started,
        "tracking_analysis_duration_seconds": float(grid[-1] - grid[0]),
        "tracking_truncated_at_fall": bool(first_fall_time is not None and not include_post_fall),
        "excluded_alignment_seconds": alignment_seconds,
        "analysis_fps": fps,
        "feedback_samples": len(feedback),
        "active_joints": int(np.sum(active)),
        "global_lag_seconds": global_lag,
        "raw_position_rmse_rad": float(np.sqrt(np.mean(raw_error**2))),
        "raw_velocity_rmse_rad_s": float(np.sqrt(np.mean(raw_velocity_error**2))),
        "lag_compensated_position_rmse_rad": float(np.sqrt(np.mean((aligned_measured - aligned_target) ** 2))),
        "minimum_base_height_m": float(np.min(heights)) if len(heights) else None,
        "survival_seconds": survival,
        "total_survival_including_alignment_seconds": total_survival,
        "fell_below_height_threshold": bool(fallen.any()),
        "groups": group_metrics,
        "per_joint": {
            name: {
                "active": bool(active[index]),
                "raw_rmse_rad": float(np.sqrt(np.mean(raw_error[:, index] ** 2))),
                "aligned_rmse_rad": float(per_joint_aligned_rmse[index]),
                "lag_ms": float(per_joint_lag[index] * 1000.0),
                "amplitude_retention": None if not np.isfinite(amplitude_retention[index]) else float(amplitude_retention[index]),
                "energy_retention": None if not np.isfinite(energy_retention[index]) else float(energy_retention[index]),
                "band_power_retention": {
                    band: None if not np.isfinite(values[index]) else float(values[index])
                    for band, values in band_retention.items()
                },
            }
            for index, name in enumerate(JOINT_NAMES)
        },
    }


def _summary_row(result: dict) -> dict:
    full = result["groups"]["full_body"]
    return {
        "run_id": result["run_id"],
        "playback_duration_seconds": result["playback_duration_seconds"],
        "tracking_analysis_duration_seconds": result["tracking_analysis_duration_seconds"],
        "raw_position_rmse_rad": result["raw_position_rmse_rad"],
        "aligned_position_rmse_rad": result["lag_compensated_position_rmse_rad"],
        "global_lag_ms": result["global_lag_seconds"] * 1000.0,
        "amplitude_retention_median": full["amplitude_retention"]["median"],
        "energy_retention_median": full["energy_retention"]["median"],
        "high_frequency_retention_median": full["band_power_retention"]["high_3_8"]["median"],
        "minimum_base_height_m": result["minimum_base_height_m"],
        "survival_seconds": result["survival_seconds"],
        "fell": result["fell_below_height_threshold"],
    }


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        analyze_run(
            Path(run).expanduser().resolve(),
            fps=args.analysis_fps,
            max_lag_seconds=args.max_lag_seconds,
            fall_height=args.fall_height,
            trim_start=args.trim_start_seconds,
            trim_end=args.trim_end_seconds,
            include_post_fall=args.include_post_fall_tracking,
        )
        for run in args.runs
    ]
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = [_summary_row(result) for result in results]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            row["run_id"],
            f"raw={row['raw_position_rmse_rad']:.4f}rad",
            f"aligned={row['aligned_position_rmse_rad']:.4f}rad",
            f"lag={row['global_lag_ms']:.0f}ms",
            f"amp={row['amplitude_retention_median']:.3f}",
            f"energy={row['energy_retention_median']:.3f}",
            f"survival={row['survival_seconds']:.1f}s",
        )


if __name__ == "__main__":
    main(parse_args())
