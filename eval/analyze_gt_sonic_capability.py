"""Analyze the fixed low/medium/high GT reference calibration runs through SONIC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_generation_execution_gap import analyze_run
from eval.analyze_motion_music_execution import (
    DEFAULT_MODEL_PATH,
    compute_fk_tracking_metrics,
    load_execution_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--analysis_fps", type=float, default=50.0)
    parser.add_argument("--max_lag_seconds", type=float, default=0.5)
    parser.add_argument("--fall_height", type=float, default=0.45)
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "std": None, "median": None}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
    }


def _summary(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["level"]].append(row)
    output = {}
    metric_names = (
        "global_lag_ms",
        "raw_joint_rmse_rad",
        "aligned_joint_rmse_rad",
        "root_relative_empkpe_m",
        "amplitude_retention_median",
        "energy_retention_median",
        "low_band_retention_median",
        "mid_band_retention_median",
        "high_band_retention_median",
        "arms_high_band_retention_median",
        "left_contact_f1",
        "right_contact_f1",
    )
    for level, level_rows in grouped.items():
        output[level] = {
            "runs": len(level_rows),
            "successes": sum(row["success"] for row in level_rows),
            "success_rate": float(np.mean([row["success"] for row in level_rows])),
            **{
                metric: _mean_std([row[metric] for row in level_rows if row[metric] is not None])
                for metric in metric_names
            },
        }
    return output


def _empirical_gates(rows: list[dict], *, calibration_source: str) -> dict:
    successful = [row for row in rows if row["success"]]
    if len(successful) < 3:
        return {
            "status": "insufficient_successful_gt_runs",
            "successful_runs": len(successful),
            "required_successful_runs": 3,
        }

    def percentile(name: str, quantile: float) -> float:
        return float(np.percentile([row[name] for row in successful], quantile))

    native = calibration_source == "sonic_native"
    return {
        "status": (
            "provisional_sonic_native_calibration"
            if native
            else "provisional_gt_calibration"
        ),
        "successful_runs": len(successful),
        "rule": (
            "Error ceilings use successful SONIC-native P95 and retention floors use P05. "
            "These are interface/tracker known-distribution baselines, not independent GT gates."
            if native
            else "Error ceilings use successful-GT P95; retention floors use successful-GT P05. Freeze only after reviewing all three dynamic tiers."
        ),
        "tracking_lag_p95_ms": percentile("global_lag_ms", 95),
        "aligned_joint_rmse_p95_rad": percentile("aligned_joint_rmse_rad", 95),
        "root_relative_empkpe_p95_m": percentile("root_relative_empkpe_m", 95),
        "amplitude_retention_p05": percentile("amplitude_retention_median", 5),
        "energy_retention_p05": percentile("energy_retention_median", 5),
        "low_band_retention_p05": percentile("low_band_retention_median", 5),
        "mid_band_retention_p05": percentile("mid_band_retention_median", 5),
        "high_band_retention_p05": percentile("high_band_retention_median", 5),
    }


def _planned_runs(manifest: dict) -> dict[str, dict]:
    if "planned_runs" in manifest:
        return {run["run_id"]: run for run in manifest["planned_runs"]}
    if "selected" not in manifest:
        raise ValueError("capability manifest requires planned_runs or selected")
    planned = {}
    for level, reference in manifest["selected"].items():
        motion_id = reference["name"]
        for repeat in (1, 2, 3):
            run_id = f"sonic_native_{level}_{motion_id}_r0{repeat}"
            planned[run_id] = {
                "run_id": run_id,
                "level": level,
                "repeat": repeat,
                "motion_id": motion_id,
            }
    return planned


def main(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_source = (
        "sonic_native"
        if manifest.get("schema_version") == "sonic_known_trackable_capability_v1"
        else "retargeted_gt"
    )
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    planned = _planned_runs(manifest)
    for run_id, plan in planned.items():
        run_dir = runs_root / run_id
        required = ("offline_sonic_playback.json", "sonic_feedback.json", "sim_state.json")
        if not run_dir.is_dir() or any(not (run_dir / name).is_file() for name in required):
            missing.append(run_id)
            continue

        tracking = analyze_run(
            run_dir,
            fps=args.analysis_fps,
            max_lag_seconds=args.max_lag_seconds,
            fall_height=args.fall_height,
            trim_start=0.0,
            trim_end=0.0,
            include_post_fall=False,
        )
        target_motion, measured_motion, playback = load_execution_pair(
            run_dir, args.analysis_fps
        )
        fk = compute_fk_tracking_metrics(
            target_motion,
            measured_motion,
            model_path=model_path,
            lag_seconds=tracking["global_lag_seconds"],
        )
        full = tracking["groups"]["full_body"]
        arms = tracking["groups"]["arms"]
        expected_duration = float(playback["selected_source_duration_seconds"])
        success = bool(
            not tracking["fell_below_height_threshold"]
            and tracking["survival_seconds"] >= expected_duration - 0.1
        )
        rows.append(
            {
                "run_id": run_id,
                "level": plan["level"],
                "repeat": plan["repeat"],
                "motion_id": plan["motion_id"],
                "expected_duration_seconds": expected_duration,
                "success": success,
                "survival_seconds": tracking["survival_seconds"],
                "minimum_base_height_m": tracking["minimum_base_height_m"],
                "global_lag_ms": tracking["global_lag_seconds"] * 1000.0,
                "raw_joint_rmse_rad": tracking["raw_position_rmse_rad"],
                "aligned_joint_rmse_rad": tracking["lag_compensated_position_rmse_rad"],
                "root_relative_empkpe_m": fk["root_relative_empkpe_m"],
                "root_relative_empkpe_p95_m": fk["root_relative_empkpe_p95_m"],
                "amplitude_retention_median": full["amplitude_retention"]["median"],
                "energy_retention_median": full["energy_retention"]["median"],
                "low_band_retention_median": full["band_power_retention"]["low_0_1"]["median"],
                "mid_band_retention_median": full["band_power_retention"]["mid_1_3"]["median"],
                "high_band_retention_median": full["band_power_retention"]["high_3_8"]["median"],
                "arms_high_band_retention_median": arms["band_power_retention"]["high_3_8"]["median"],
                "left_contact_f1": fk["contact"]["left"]["f1"],
                "right_contact_f1": fk["contact"]["right"]["f1"],
            }
        )

    payload = {
        "schema_version": "sonic_capability_analysis_v2",
        "calibration_source": calibration_source,
        "manifest": str(manifest_path),
        "planned_runs": len(planned),
        "analyzed_runs": len(rows),
        "missing_runs": missing,
        "tier_summary": _summary(rows),
        "empirical_gates": _empirical_gates(
            rows, calibration_source=calibration_source
        ),
        "runs": rows,
    }
    (output_dir / "capability_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if rows:
        with (output_dir / "capability_runs.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"analyzed={len(rows)}/{len(planned)} missing={len(missing)}")
    for level, summary in payload["tier_summary"].items():
        print(
            level,
            f"success={summary['successes']}/{summary['runs']}",
            f"rmse={summary['aligned_joint_rmse_rad']['mean']:.4f}",
            f"empkpe={summary['root_relative_empkpe_m']['mean']:.4f}",
            f"energy={summary['energy_retention_median']['mean']:.3f}",
            f"high={summary['high_band_retention_median']['mean']:.3f}",
        )


if __name__ == "__main__":
    main(parse_args())
