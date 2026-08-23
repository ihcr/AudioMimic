"""Analyze motion-quality metric sensitivity on controlled GT corruptions."""

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

from eval.eval_bas_bap import load_audio_beat_frames
from eval.g1_metrics import (
    compute_beat_timing_report,
    detect_g1_motion_beat_frames,
    evaluate_g1_beats,
    load_g1_motion,
    summarize_g1_motion,
)


DEFAULT_MANIFEST = Path("eval/benchmark_v1/gt/motion_corruptions_v1/corruption_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.percentile(np.abs(values), percentile)) if values.size else 0.0


def _motion_metrics(path: Path) -> dict:
    motion = load_g1_motion(path)
    summary = summarize_g1_motion(motion)
    fps = float(motion["fps"])
    dof = motion["dof_pos"]
    velocity = np.diff(dof, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    speed = np.mean(np.abs(velocity), axis=1) if velocity.size else np.zeros(0)
    audio_path = motion.get("audio_path", "")
    beat_metrics = evaluate_g1_beats(motion) or {}
    timing = {}
    if audio_path:
        target = load_audio_beat_frames(
            audio_path, fps=int(round(fps)), seq_len=len(dof)
        )
        timing = compute_beat_timing_report(
            detect_g1_motion_beat_frames(motion), target
        )
    return {
        "frames": int(len(dof)),
        "duration_seconds": float(len(dof) / max(fps, 1e-6)),
        "motion_energy_rad2_s2": float(np.mean(velocity**2)) if velocity.size else 0.0,
        "velocity_p95_rad_s": _percentile(velocity, 95),
        "acceleration_p95_rad_s2": _percentile(acceleration, 95),
        "jerk_p95_rad_s3": _percentile(jerk, 95),
        "static_ratio_below_0p05_rad_s": float(np.mean(speed < 0.05)) if speed.size else 1.0,
        "joint_position_std_mean": summary["joint_position_std_mean"],
        "joint_position_range_mean": summary["joint_position_range_mean"],
        "joint_smoothness_jerk_mean": summary["joint_smoothness_jerk_mean"],
        "root_drift_m": summary["root_drift"],
        "root_path_length_m": summary["root_path_length"],
        "G1BAS": beat_metrics.get("G1BAS"),
        "G1RoboPerformBAS": beat_metrics.get("G1RoboPerformBAS"),
        "G1BeatPrecision": timing.get("precision"),
        "G1BeatRecall": timing.get("recall"),
        "G1BeatF1": timing.get("f1"),
    }


def _repeat_similarity(path: Path) -> float:
    motion = load_g1_motion(path)
    speed = np.mean(np.abs(np.diff(motion["dof_pos"], axis=0)), axis=1)
    if len(speed) < 45:
        return 0.0
    speed = (speed - speed.mean()) / max(float(speed.std()), 1e-8)
    max_lag = min(len(speed) // 2, 180)
    correlations = []
    for lag in range(15, max_lag + 1):
        correlations.append(float(np.mean(speed[:-lag] * speed[lag:])))
    return max(correlations) if correlations else 0.0


def _markdown(summary: list[dict], checks: list[dict], manifest_path: Path) -> str:
    lines = [
        "# GT Motion Corruption Calibration",
        "",
        f"- Source manifest: `{manifest_path}`",
        "- Dataset: AIST++ retargeted G1, declared crossmodal test split.",
        "- Purpose: verify metric sensitivity; this is not a model ranking result.",
        "",
        "## Aggregate Response",
        "",
        "| variant | severity | n | energy | jerk p95 | static ratio | G1BAS | beat F1 | repeat similarity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(key):
            value = row.get(key)
            return "n/a" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['variant']} | {row['severity']} | {row['n']} | "
            f"{fmt('motion_energy_rad2_s2')} | {fmt('jerk_p95_rad_s3')} | "
            f"{fmt('static_ratio_below_0p05_rad_s')} | {fmt('G1BAS')} | "
            f"{fmt('G1BeatF1')} | {fmt('repeat_similarity')} |"
        )
    lines += ["", "## Directional Checks", "", "| check | result | observed |", "|---|---|---|"]
    for check in checks:
        lines.append(
            f"| {check['check']} | **{check['result']}** | {check['observed']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Jitter should primarily increase jerk and perturbation magnitude.",
        "- Low-pass corruption should reduce high-frequency jerk and usually reduce energy.",
        "- Freeze corruption should increase the static ratio and reduce local energy.",
        "- Repeat corruption is reported with a self-similarity diagnostic; it is not treated as a single universal quality score.",
        "- Beat metrics are diagnostic only here. They should not be used as the sole dance-quality criterion.",
    ]
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else manifest_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in payload["records"]:
        metrics = _motion_metrics(Path(record["path"]))
        metrics["repeat_similarity"] = _repeat_similarity(Path(record["path"]))
        rows.append({**record, **metrics})

    grouped = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["severity"]), []).append(row)
    clean_by_sequence = {
        row["sequence_id"]: row
        for row in rows
        if row["variant"] == "clean"
    }
    summary = []
    for (variant, severity), group in sorted(grouped.items()):
        row = {"variant": variant, "severity": severity, "n": len(group)}
        keys = (
            "motion_energy_rad2_s2",
            "jerk_p95_rad_s3",
            "static_ratio_below_0p05_rad_s",
            "G1BAS",
            "G1BeatF1",
            "repeat_similarity",
        )
        for key in keys:
            values = [item[key] for item in group if item.get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        summary.append(row)

    def delta(variant: str, severity: str, key: str) -> float:
        values = []
        for row in grouped[(variant, severity)]:
            base = clean_by_sequence[row["sequence_id"]][key]
            if row.get(key) is not None and base is not None:
                values.append(row[key] - base)
        return float(np.mean(values)) if values else 0.0

    checks = []
    for variant, key, label in (
        ("jitter", "jerk_p95_rad_s3", "jitter increases jerk p95"),
        ("lowpass", "jerk_p95_rad_s3", "lowpass decreases jerk p95"),
        ("freeze", "static_ratio_below_0p05_rad_s", "freeze increases static ratio"),
    ):
        observed = delta(variant, "high", key)
        passed = observed > 0.0 if variant in ("jitter", "freeze") else observed < 0.0
        checks.append(
            {
                "check": label,
                "result": "PASS" if passed else "WARN",
                "observed": f"mean delta {observed:+.6f}",
            }
        )

    fieldnames = sorted({key for row in rows for key in row if key != "details"})
    with (output_dir / "per_sequence_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": "gt_motion_corruption_calibration_metrics_v1",
        "source_manifest": str(manifest_path),
        "rows": rows,
        "aggregate": summary,
        "directional_checks": checks,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        _markdown(summary, checks, manifest_path), encoding="utf-8"
    )
    print(f"Analyzed {len(rows)} motions")
    for check in checks:
        print(f"{check['result']}: {check['check']} ({check['observed']})")
    print(f"Report: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main(parse_args())
