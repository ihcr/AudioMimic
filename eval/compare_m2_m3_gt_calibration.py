"""Compare available M2/M3 generator references with the sealed GT ranges.

This is a reference-level comparison only. It does not rank routes by one
aggregate score and it does not replace the paired counterfactual protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = {
    "motion_energy": (("quality", "motion_energy_rad2_s2"), "motion_energy"),
    "joint_jerk_p95": (("quality", "joint_jerk_abs_rad_s3", "p95"), "joint_jerk_p95"),
    "static_ratio": (("quality", "static_ratio_speed_below_008"), "static_ratio"),
    "repeated_pose_ratio": (
        ("quality", "repeated_pose_ratio_rms008_after2s"),
        "repeated_pose_ratio",
    ),
    "fsr_proxy": (("quality", "physical", "fsr_ground_calibrated_proxy"), "fsr_proxy"),
    "pfc_proxy": (("quality", "physical", "pfc_proxy"), "pfc_proxy"),
    "root_height_min": (("quality", "root_height_min_m"), "root_height_min"),
    "impact_corr": (("music", "impact_best_correlation"), "impact_corr"),
    "impact_abs_lag": (("music", "impact_best_lag_seconds"), "impact_abs_lag"),
    "bas": (("music", "bas_music_to_motion"), "bas"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m2-reference",
        type=Path,
        default=Path(
            "eval/motion_music_execution/gt_calibrated_m0_m2_m4_song098_v2/reference_metrics.json"
        ),
    )
    parser.add_argument(
        "--m3-pair",
        type=Path,
        nargs="+",
        default=[
            Path("eval/mrt2_metrics/m3_012_pair_metrics.json"),
            Path("eval/mrt2_metrics/m3_065_pair_metrics.json"),
        ],
    )
    parser.add_argument(
        "--gt-suite",
        type=Path,
        default=Path("eval/benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/motion_music_execution/m2_m3_gt_comparison_v2"),
    )
    return parser.parse_args()


def _get(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    for key in path[:-1]:
        value = value.get(key, {})
        if not isinstance(value, dict):
            return None
    result = value.get(path[-1])
    return float(result) if isinstance(result, (int, float)) else None


def _row(route: str, source: str, record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "route": route,
        "source": source,
        "sequence_id": record.get("sequence_id", payload.get("label", "")).replace("M3-", ""),
        "seed": record.get("sampling_seed", record.get("seed", "")),
        "label": payload.get("label", record.get("motion_id", "")),
    }
    for name, (path, _) in METRICS.items():
        row[name] = _get(payload, path)
    return row


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    m2_records = json.loads(args.m2_reference.read_text(encoding="utf-8"))
    rows = []
    for record in m2_records:
        if record.get("route") == "M2":
            rows.append(_row("M2", str(args.m2_reference), record, record))
    for path in args.m3_pair:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generator = dict(payload["generator"])
        generator["label"] = payload.get("label", path.stem)
        rows.append(_row("M3", str(path), {}, generator))
    return rows


def _add_gt_flags(rows: list[dict[str, Any]], gt_path: Path) -> dict[str, Any]:
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    summary = payload["summary"]["metrics"]
    for row in rows:
        for name, (_, gt_name) in METRICS.items():
            value = row[name]
            distribution = summary.get(gt_name, {})
            if value is None or "q10" not in distribution or "q90" not in distribution:
                row[f"{name}_in_gt_q10_q90"] = "NA"
            else:
                row[f"{name}_in_gt_q10_q90"] = bool(
                    distribution["q10"] <= value <= distribution["q90"]
                )
    return summary


def main() -> None:
    args = parse_args()
    rows = _load_rows(args)
    gt_summary = _add_gt_flags(rows, args.gt_suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "m2_m3_gt_comparison_v1",
        "gt_suite": str(args.gt_suite),
        "rows": rows,
        "gt_quantiles": {
            name: {key: gt_summary[gt_name].get(key) for key in ("q10", "median", "q90")}
            for name, (_, gt_name) in METRICS.items()
            if gt_name in gt_summary
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    gt_scope = (
        "the full 1,611-sequence AIST++/FineDance GT distribution"
        if "all_v1" in str(args.gt_suite)
        else "the sealed 38-sequence GT calibration suite"
    )
    lines = [
        "# M2/M3 Generator Reference vs GT Calibration",
        "",
        f"This table compares available generator references with {gt_scope} quantiles.",
        "It is descriptive; it is not an aggregate model ranking.",
        "",
        "| route | sequence | energy | jerk P95 | impact corr | abs lag | BAS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fmt = lambda key: "NA" if row[key] is None else f"{row[key]:.4f}"
        lines.append(
            f"| {row['route']} | {row['sequence_id']} | {fmt('motion_energy')} | "
            f"{fmt('joint_jerk_p95')} | {fmt('impact_corr')} | {fmt('impact_abs_lag')} | {fmt('bas')} |"
        )
    lines.extend(
        [
            "",
            "M3 currently has two 60-second paired reference/execution artifacts (012 and 065);",
            "it is not yet a multi-song, multi-seed final comparison.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
