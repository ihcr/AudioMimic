"""Compare M0/M2/M4 generation and execution with two SONIC capability baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROUTES = ("M0", "M2", "M4")
DYNAMIC_KEYS = {
    "energy": "quality.motion_energy_rad2_s2",
    "velocity_p95": "quality.joint_velocity_abs_rad_s.p95",
    "acceleration_p95": "quality.joint_acceleration_abs_rad_s2.p95",
    "jerk_p95": "quality.joint_jerk_abs_rad_s3.p95",
}
EXECUTION_PATHS = {
    "aligned_rmse_rad": ("tracking", "lag_compensated_position_rmse_rad"),
    "empkpe_m": ("tracking", "fk", "root_relative_empkpe_m"),
    "lag_ms": ("tracking", "global_lag_seconds"),
    "minimum_base_height_m": ("tracking", "minimum_base_height_m"),
    "amplitude_retention": (
        "tracking",
        "body_groups",
        "full_body",
        "amplitude_retention",
        "median",
    ),
    "energy_retention": (
        "tracking",
        "body_groups",
        "full_body",
        "energy_retention",
        "median",
    ),
    "low_band_retention": (
        "tracking",
        "body_groups",
        "full_body",
        "band_power_retention",
        "low_0_1",
        "median",
    ),
    "mid_band_retention": (
        "tracking",
        "body_groups",
        "full_body",
        "band_power_retention",
        "mid_1_3",
        "median",
    ),
    "high_band_retention": (
        "tracking",
        "body_groups",
        "full_body",
        "band_power_retention",
        "high_3_8",
        "median",
    ),
    "arms_high_band_retention": (
        "tracking",
        "body_groups",
        "arms",
        "band_power_retention",
        "high_3_8",
        "median",
    ),
}
BASELINE_KEYS = {
    "aligned_rmse_rad": "aligned_joint_rmse_rad",
    "empkpe_m": "root_relative_empkpe_m",
    "lag_ms": "global_lag_ms",
    "amplitude_retention": "amplitude_retention_median",
    "energy_retention": "energy_retention_median",
    "low_band_retention": "low_band_retention_median",
    "mid_band_retention": "mid_band_retention_median",
    "high_band_retention": "high_band_retention_median",
    "arms_high_band_retention": "arms_high_band_retention_median",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route_summary",
        type=Path,
        default=Path("eval/motion_music_execution/first_round_20260819/route_summary.json"),
    )
    parser.add_argument(
        "--execution_metrics",
        type=Path,
        default=Path("eval/motion_music_execution/first_round_20260819/execution_metrics.json"),
    )
    parser.add_argument(
        "--native_capability",
        type=Path,
        default=Path(
            "eval/gt_sonic_capability/known_trackable_analysis/"
            "capability_9x_corrected/capability_analysis.json"
        ),
    )
    parser.add_argument(
        "--gt_capability",
        type=Path,
        default=Path("eval/gt_sonic_capability/retargeted_gt_analysis_v2/capability_analysis.json"),
    )
    parser.add_argument(
        "--gt_manifest",
        type=Path,
        default=Path("eval/gt_sonic_capability/selection_20260819/selection_manifest.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("eval/generator_vs_tracker_baselines/20260820"),
    )
    return parser.parse_args()


def _load(path: Path):
    return json.loads(path.expanduser().resolve().read_text())


def _nested(record: dict, path: tuple[str, ...]):
    value = record
    for key in path:
        value = value[key]
    return value


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def _nearest_gt_tier(dynamic: dict[str, float], primary: dict) -> tuple[str, dict]:
    distances = {}
    field_names = {
        "energy": "motion_energy_rad2_s2",
        "velocity_p95": "velocity_p95_rad_s",
        "acceleration_p95": "acceleration_p95_rad_s2",
        "jerk_p95": "jerk_p95_rad_s3",
    }
    for tier, reference in primary.items():
        log_ratios = {
            name: math.log(dynamic[name] / float(reference[field]))
            for name, field in field_names.items()
        }
        distances[tier] = {
            "log_euclidean_distance": float(
                np.sqrt(np.mean(np.square(list(log_ratios.values()))))
            ),
            "ratios": {name: float(math.exp(value)) for name, value in log_ratios.items()},
        }
    nearest = min(distances, key=lambda tier: distances[tier]["log_euclidean_distance"])
    return nearest, distances


def _execution_summary(records: list[dict]) -> dict:
    output = {"runs": len(records), "successes": sum(not item["tracking"]["fell"] for item in records)}
    for name, path in EXECUTION_PATHS.items():
        values = [float(_nested(item, path)) for item in records]
        if name == "lag_ms":
            values = [value * 1000.0 for value in values]
        output[name] = _mean_std(values)
    return output


def _tier_means(capability: dict, tier: str) -> dict:
    summary = capability["tier_summary"][tier]
    return {
        name: float(summary[key]["mean"])
        for name, key in BASELINE_KEYS.items()
    }


def _relative(execution: dict, baseline: dict) -> dict:
    return {
        name: float(execution[name]["mean"] / baseline[name])
        for name in BASELINE_KEYS
    }


def _gate_diagnostics(execution: dict, capability: dict) -> dict:
    gate = capability["empirical_gates"]
    checks = {
        "aligned_rmse": execution["aligned_rmse_rad"]["mean"]
        <= gate["aligned_joint_rmse_p95_rad"],
        "empkpe": execution["empkpe_m"]["mean"] <= gate["root_relative_empkpe_p95_m"],
        "amplitude": execution["amplitude_retention"]["mean"]
        >= gate["amplitude_retention_p05"],
        "energy": execution["energy_retention"]["mean"] >= gate["energy_retention_p05"],
        "low_band": execution["low_band_retention"]["mean"]
        >= gate["low_band_retention_p05"],
        "mid_band": execution["mid_band_retention"]["mean"]
        >= gate["mid_band_retention_p05"],
        "high_band": execution["high_band_retention"]["mean"]
        >= gate["high_band_retention_p05"],
    }
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "interpretation": "diagnostic mean-vs-envelope comparison; not a statistical acceptance test",
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _write_report(payload: dict, path: Path) -> None:
    routes = payload["routes"]
    lines = [
        "# M0/M2/M4 vs SONIC Capability Baselines",
        "",
        "Date: 2026-08-20",
        "",
        "## Scope",
        "",
        "Each route uses one fixed 60 s seed1234 execution reference repeated three times. "
        "Reference-quality summaries use the available song098 exports. Capability baselines use "
        "three repeats of one low/medium/high sequence and therefore remain diagnostic.",
        "",
        "## Generator Reference",
        "",
        "| Route | Nearest GT tier | Energy | Velocity P95 | Acceleration P95 | Jerk P95 | BAS music->motion | Impact corr. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ROUTES:
        item = routes[route]
        dynamic = item["reference_dynamic"]
        music = item["reference_music"]
        lines.append(
            f"| {route} | {item['nearest_gt_dynamic_tier']} | "
            f"{_fmt(dynamic['energy'])} | {_fmt(dynamic['velocity_p95'])} | "
            f"{_fmt(dynamic['acceleration_p95'], 1)} | {_fmt(dynamic['jerk_p95'], 1)} | "
            f"{_fmt(music['bas_music_to_motion'])} | {_fmt(music['impact_best_correlation'])} |"
        )
    lines.extend(
        [
            "",
            "All routes are closest to the selected low-dynamics GT reference. M0 is the most "
            "dynamic; M4 is the most conservative. M0 is unconditional, so its BAS is an incidental "
            "baseline rather than evidence of music response.",
            "",
            "## SONIC Execution",
            "",
            "| Route | Success | Aligned RMSE | EMPKPE | Lag | Amplitude | Energy | 0--1 Hz | 1--3 Hz | 3--8 Hz | Arms 3--8 Hz |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for route in ROUTES:
        execution = routes[route]["execution"]
        lines.append(
            f"| {route} | {execution['successes']}/{execution['runs']} | "
            f"{_fmt(execution['aligned_rmse_rad']['mean'], 4)} | "
            f"{_fmt(execution['empkpe_m']['mean'], 4)} m | "
            f"{_fmt(execution['lag_ms']['mean'], 0)} ms | "
            f"{_fmt(execution['amplitude_retention']['mean'])} | "
            f"{_fmt(execution['energy_retention']['mean'])} | "
            f"{_fmt(execution['low_band_retention']['mean'])} | "
            f"{_fmt(execution['mid_band_retention']['mean'])} | "
            f"{_fmt(execution['high_band_retention']['mean'])} | "
            f"{_fmt(execution['arms_high_band_retention']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Baseline-Normalized Findings",
            "",
            "| Route | RMSE / native-low | RMSE / GT-low | EMPKPE / native-low | Energy / native-low | Mid-band / native-low | Native gate checks | GT gate checks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for route in ROUTES:
        item = routes[route]
        native = item["relative_to_nearest_native_tier"]
        gt = item["relative_to_nearest_gt_tier"]
        lines.append(
            f"| {route} | {_fmt(native['aligned_rmse_rad'], 2)}x | "
            f"{_fmt(gt['aligned_rmse_rad'], 2)}x | {_fmt(native['empkpe_m'], 2)}x | "
            f"{_fmt(native['energy_retention'], 2)}x | {_fmt(native['mid_band_retention'], 2)}x | "
            f"{item['native_gate_diagnostics']['passed']}/7 | "
            f"{item['gt_gate_diagnostics']['passed']}/7 |"
        )
    lines.extend(
        [
            "",
            "M4 has the lowest execution error, followed by M0 and M2. All three preserve pose "
            "amplitude but retain only about half of median per-joint dynamic energy. Mid/high-band "
            "loss is stronger than either low-tier baseline. M2 high-band mean is inflated by one "
            "oscillatory run and must not be interpreted as superior detail retention.",
            "",
            "The automatic music metrics do not establish an M2/M4 advantage over unconditional M0. "
            "No aesthetic or dance-beauty claim follows from these metrics; blinded human evaluation "
            "and broader song/seed coverage remain required.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(args: argparse.Namespace) -> None:
    route_summary = _load(args.route_summary)
    execution_records = _load(args.execution_metrics)
    native = _load(args.native_capability)
    gt = _load(args.gt_capability)
    gt_manifest = _load(args.gt_manifest)
    grouped = defaultdict(list)
    for record in execution_records:
        grouped[record["route"]].append(record)

    routes = {}
    reference = route_summary["reference_matched_song098"]
    for route in ROUTES:
        dynamic = {
            name: float(reference[route][key]["mean"])
            for name, key in DYNAMIC_KEYS.items()
        }
        nearest, distances = _nearest_gt_tier(dynamic, gt_manifest["primary"])
        execution = _execution_summary(grouped[route])
        native_tier = _tier_means(native, nearest)
        gt_tier = _tier_means(gt, nearest)
        routes[route] = {
            "reference_samples": int(reference[route]["samples"]),
            "reference_dynamic": dynamic,
            "reference_music": {
                "bas_music_to_motion": float(
                    reference[route]["music.bas_music_to_motion"]["mean"]
                ),
                "bas_motion_to_music": float(
                    reference[route]["music.bas_motion_to_music"]["mean"]
                ),
                "impact_best_correlation": float(
                    reference[route]["music.impact_best_correlation"]["mean"]
                ),
                "speed_best_correlation": float(
                    reference[route]["music.speed_best_correlation"]["mean"]
                ),
            },
            "nearest_gt_dynamic_tier": nearest,
            "dynamic_tier_distances": distances,
            "execution": execution,
            "nearest_native_tier_baseline": native_tier,
            "nearest_gt_tier_baseline": gt_tier,
            "relative_to_nearest_native_tier": _relative(execution, native_tier),
            "relative_to_nearest_gt_tier": _relative(execution, gt_tier),
            "native_gate_diagnostics": _gate_diagnostics(execution, native),
            "gt_gate_diagnostics": _gate_diagnostics(execution, gt),
        }

    payload = {
        "schema_version": "generator_vs_sonic_capability_v1",
        "scope": {
            "routes": list(ROUTES),
            "execution_repeats_per_route": 3,
            "reference_scope": "song098 exports; M0 has two matched files, M2/M4 have three",
            "aesthetic_evidence": False,
            "gate_status": "diagnostic_only",
        },
        "routes": routes,
        "cross_route": {
            "lowest_aligned_rmse": min(
                ROUTES, key=lambda route: routes[route]["execution"]["aligned_rmse_rad"]["mean"]
            ),
            "highest_energy_retention": max(
                ROUTES, key=lambda route: routes[route]["execution"]["energy_retention"]["mean"]
            ),
            "music_condition_advantage_supported": False,
            "reason": (
                "M2/M4 do not improve the preregistered automatic music metrics over unconditional M0; "
                "the present single-song, small-sample evidence is insufficient."
            ),
        },
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(payload, output_dir / "COMPARISON.md")
    print(f"Wrote {output_dir / 'comparison.json'}")
    for route in ROUTES:
        item = routes[route]
        print(
            route,
            f"tier={item['nearest_gt_dynamic_tier']}",
            f"rmse={item['execution']['aligned_rmse_rad']['mean']:.4f}",
            f"empkpe={item['execution']['empkpe_m']['mean']:.4f}",
            f"energy={item['execution']['energy_retention']['mean']:.3f}",
            f"native_checks={item['native_gate_diagnostics']['passed']}/7",
        )


if __name__ == "__main__":
    main(parse_args())
