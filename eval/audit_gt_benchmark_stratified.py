"""Summarize GT benchmark metrics by dataset, tempo and dance-style strata."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_SUITE = Path("eval/benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_metrics.json")
DEFAULT_CORRUPTIONS = Path("eval/benchmark_v1/gt/stratified_corruptions_v1/per_sequence_metrics.csv")
DEFAULT_OUTPUT = Path("eval/benchmark_v1/gt/stratified_audit_v1")

METRICS = {
    "bas_music_to_motion": ("BAS", "higher"),
    "bas_motion_to_music": ("BAS_reverse", "higher"),
    "event_f1": ("BeatF1", "higher"),
    "impact_best_correlation": ("ImpactCorr", "higher"),
    "impact_abs_lag": ("AbsLag", "lower"),
    "tempo_error": ("TempoErr", "lower"),
    "phase_error": ("PhaseErr", "lower"),
    "motion_energy": ("Energy", "reference"),
    "joint_jerk_p95": ("JerkP95", "lower"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--corruption-metrics", type=Path, default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-corruption-checks",
        action="store_true",
        help="Only summarize the GT oracle; do not attach the 38-sequence corruption audit.",
    )
    return parser.parse_args()


def _tempo_bin(bpm: float) -> str:
    if bpm < 90.0:
        return "slow_<90"
    if bpm < 120.0:
        return "medium_90-120"
    return "fast_>=120"


def _aist_style(sequence_id: str) -> str:
    match = re.match(r"g([A-Za-z]{2})_", sequence_id)
    return f"AIST_genre_{match.group(1)}" if match else "AIST_genre_unknown"


def _summary(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "q10": None, "q90": None, "mean": None}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "q10": float(np.percentile(values, 10)),
        "q90": float(np.percentile(values, 90)),
        "mean": float(np.mean(values)),
    }


def _gt_rows(suite: dict) -> list[dict]:
    rows = []
    for record in suite["records"]:
        music = record["music"]
        dataset = record["dataset"]
        style = record.get("style") or []
        style_name = "+".join(style) if style else _aist_style(record["sequence_id"])
        row = {
            "dataset": dataset,
            "sequence_id": record["sequence_id"],
            "tempo_bin": _tempo_bin(float(music["audio_bpm"])),
            "style": style_name,
            "audio_bpm": float(music["audio_bpm"]),
        }
        quality = record["quality"]
        for source, (output, _) in METRICS.items():
            if source == "motion_energy":
                row[output] = float(quality["motion_energy_rad2_s2"])
            elif source == "joint_jerk_p95":
                row[output] = float(quality["joint_jerk_abs_rad_s3"]["p95"])
            elif source == "event_f1":
                row[output] = float(music["event_f1"]["f1"])
            elif source == "impact_abs_lag":
                row[output] = abs(float(music["impact_best_lag_seconds"]))
            elif source == "tempo_error":
                row[output] = float(music["tempo_abs_error_bpm"])
            elif source == "phase_error":
                row[output] = float(music["phase"]["mean_phase_error_cycles"])
            else:
                row[output] = float(music[source])
        rows.append(row)
    return rows


def _corruption_checks(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    checks = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["variant"], row["severity"])].append(row)
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        clean = groups[(dataset, "clean", "clean")]
        for variant, key, expected in (
            ("jitter", "jerk_p95_rad_s3", "increase"),
            ("lowpass", "jerk_p95_rad_s3", "decrease"),
            ("freeze", "static_ratio_below_0p05_rad_s", "increase"),
            ("freeze", "G1BeatF1", "decrease"),
        ):
            base = float(np.mean([float(r[key]) for r in clean]))
            high = groups[(dataset, variant, "high")]
            value = float(np.mean([float(r[key]) for r in high]))
            delta = value - base
            passed = delta > 0 if expected == "increase" else delta < 0
            checks.append(
                {
                    "dataset": dataset,
                    "check": f"{variant}_{key}",
                    "expected": expected,
                    "delta_high_minus_clean": delta,
                    "result": "PASS" if passed else "WARN",
                }
            )
    return checks


def main(args: argparse.Namespace) -> None:
    suite = json.loads(args.gt_suite.expanduser().resolve().read_text(encoding="utf-8"))
    rows = _gt_rows(suite)
    strata = defaultdict(list)
    for row in rows:
        strata[("dataset", row["dataset"])].append(row)
        strata[("tempo", row["tempo_bin"])].append(row)
        strata[("style", row["style"])].append(row)

    summaries = []
    for (axis, value), items in sorted(strata.items()):
        summary = {"axis": axis, "stratum": value, "n": len(items)}
        for _, (output, _) in METRICS.items():
            summary[output] = _summary([item[output] for item in items])
        summaries.append(summary)

    checks = [] if args.skip_corruption_checks else _corruption_checks(args.corruption_metrics.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gt_stratified_audit_v1",
        "gt_suite": str(args.gt_suite.expanduser().resolve()),
        "corruption_metrics": None if args.skip_corruption_checks else str(args.corruption_metrics.expanduser().resolve()),
        "sequence_count": len(rows),
        "strata": summaries,
        "corruption_checks": checks,
        "tempo_bins": {"slow_<90": "BPM < 90", "medium_90-120": "90 <= BPM < 120", "fast_>=120": "BPM >= 120"},
        "interpretation": "GT strata are empirical reference distributions. Small style strata are descriptive and must not be treated as significant rankings.",
    }
    (output_dir / "stratified_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "stratified_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["axis", "stratum", "n"] + [output for output, _ in METRICS.values()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {"axis": summary["axis"], "stratum": summary["stratum"], "n": summary["n"]}
            for output, _ in METRICS.values():
                row[output] = summary[output]["median"]
            writer.writerow(row)

    lines = [
        "# Stratified GT Benchmark Audit v1",
        "",
        "The same music-motion metrics are reported separately for AIST++, FineDance,",
        "tempo bands and available style/genre strata. Values are GT reference distributions,",
        "not universal ideal scores.",
        "",
        "## GT Strata (median)",
        "",
        "| axis | stratum | n | BPM | BAS | reverse BAS | Beat F1 | Impact corr. | abs lag | tempo err | phase err | energy | jerk P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def value(key):
            item = row[key]
            return "n/a" if item["median"] is None else f"{item['median']:.3f}"
        bpm = _summary([item["audio_bpm"] for item in rows if (item["dataset"] if row["axis"] == "dataset" else item["tempo_bin"] if row["axis"] == "tempo" else item["style"]) == row["stratum"]])
        bpm_text = "n/a" if bpm["median"] is None else f"{bpm['median']:.1f}"
        lines.append(
            f"| {row['axis']} | {row['stratum']} | {row['n']} | {bpm_text} | "
            f"{value('BAS')} | {value('BAS_reverse')} | {value('BeatF1')} | {value('ImpactCorr')} | "
            f"{value('AbsLag')} | {value('TempoErr')} | {value('PhaseErr')} | {value('Energy')} | {value('JerkP95')} |"
        )
    lines.extend([
        "",
        "## Corruption Direction Checks",
        "",
        "| dataset | check | expected | result | high-clean delta |",
        "|---|---|---|---|---:|",
    ])
    if checks:
        for check in checks:
            lines.append(
                f"| {check['dataset']} | {check['check']} | {check['expected']} | **{check['result']}** | "
                f"{check['delta_high_minus_clean']:+.4f} |"
            )
    else:
        lines.append("| n/a | GT-only full-dataset summary | n/a | **not run** | n/a |")
    lines.extend([
        "",
        "## Decision Rules",
        "",
        "1. A metric is retained only when its intended response is stable within both datasets.",
        "2. Tempo/style strata define calibration ranges; they are not pooled into one ideal score.",
        "3. A style with fewer than three clips is descriptive only and cannot support a significance claim.",
        "4. BAS remains a core beat-alignment metric and must be interpreted with Beat F1/coverage, onset response, lag, tempo and phase.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} GT records, {len(summaries)} strata and {len(checks)} checks to {output_dir}")
    for check in checks:
        print(f"{check['result']}: {check['dataset']} {check['check']}")


if __name__ == "__main__":
    main(parse_args())
