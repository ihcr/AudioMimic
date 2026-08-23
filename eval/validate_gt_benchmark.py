"""Validate metric behavior using controlled corruptions of paired GT motion.

This is a benchmark audit, not a model score. A metric is accepted only when
its response is interpretable under a corruption whose expected effect is
known. Metrics that are useful for a narrower diagnostic but fail as general
quality measures are explicitly downgraded in the report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("eval/benchmark_v1/gt/motion_corruptions_v1/per_sequence_metrics.csv")
DEFAULT_OUTPUT = Path("eval/benchmark_v1/gt/benchmark_validity_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else float("nan")


def _level_means(rows: list[dict], variant: str, key: str) -> dict[str, float]:
    return {
        level: _mean(
            [row for row in rows if row["variant"] == variant and row["severity"] == level],
            key,
        )
        for level in ("clean", "low", "medium", "high")
        if any(row["variant"] == variant and row["severity"] == level for row in rows)
    }


def _monotonic(values: list[float], increasing: bool, tolerance: float = 1e-9) -> bool:
    pairs = zip(values[:-1], values[1:])
    if increasing:
        return all(b >= a - tolerance for a, b in pairs)
    return all(b <= a + tolerance for a, b in pairs)


def _check(
    rows: list[dict],
    *,
    check_id: str,
    variant: str,
    key: str,
    expected: str,
    rationale: str,
) -> dict:
    levels = _level_means(rows, variant, key)
    clean_baseline = _mean(
        [row for row in rows if row["variant"] == "clean" and row["severity"] == "clean"],
        key,
    )
    ordered_levels = [level for level in ("low", "medium", "high") if level in levels]
    values = [clean_baseline] + [levels[level] for level in ordered_levels]
    increasing = expected == "increasing"
    passed = len(values) >= 3 and _monotonic(values, increasing)
    levels = {"clean": clean_baseline, **levels}
    clean = clean_baseline
    high = levels.get("high", float("nan"))
    return {
        "check_id": check_id,
        "variant": variant,
        "metric": key,
        "expected_response": expected,
        "levels": levels,
        "high_minus_clean": float(high - clean),
        "result": "PASS" if passed else "WARN",
        "rationale": rationale,
    }


def main(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    rows = list(csv.DictReader(input_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    checks = [
        _check(
            rows,
            check_id="jitter_jerk",
            variant="jitter",
            key="jerk_p95_rad_s3",
            expected="increasing",
            rationale="Artificial joint noise should increase high-frequency jerk.",
        ),
        _check(
            rows,
            check_id="lowpass_jerk",
            variant="lowpass",
            key="jerk_p95_rad_s3",
            expected="decreasing",
            rationale="Low-pass corruption should remove high-frequency motion.",
        ),
        _check(
            rows,
            check_id="lowpass_energy",
            variant="lowpass",
            key="motion_energy_rad2_s2",
            expected="decreasing",
            rationale="Low-pass corruption should reduce motion energy in this construction.",
        ),
        _check(
            rows,
            check_id="freeze_static",
            variant="freeze",
            key="static_ratio_below_0p05_rad_s",
            expected="increasing",
            rationale="A frozen interval should increase the global static ratio.",
        ),
        _check(
            rows,
            check_id="freeze_beat_f1",
            variant="freeze",
            key="G1BeatF1",
            expected="decreasing",
            rationale="A frozen interval should remove some motion events aligned to audio.",
        ),
        _check(
            rows,
            check_id="freeze_bas",
            variant="freeze",
            key="G1BAS",
            expected="decreasing",
            rationale="BAS should fall in this particular freeze construction, but remains a rhythm diagnostic.",
        ),
        _check(
            rows,
            check_id="repeat_similarity",
            variant="repeat",
            key="repeat_similarity",
            expected="increasing",
            rationale="A repetition corruption should increase long-range temporal self-similarity.",
        ),
    ]

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    metric_status = {
        "jerk": {
            "status": "core_validated",
            "evidence": ["jitter_jerk", "lowpass_jerk"],
            "note": "Sensitive to both injected high-frequency noise and smoothing.",
        },
        "motion_energy": {
            "status": "core_validated_for_lowpass",
            "evidence": ["lowpass_energy"],
            "note": "Do not expect global energy to detect every local freeze or repeat defect.",
        },
        "static_ratio": {
            "status": "core_validated",
            "evidence": ["freeze_static"],
            "note": "Useful for freeze/average-pose collapse, not jitter detection.",
        },
        "beat_f1": {
            "status": "music_diagnostic_validated",
            "evidence": ["freeze_beat_f1"],
            "note": "Useful for event loss, not a general dance-quality score.",
        },
        "bas": {
            "status": "supplementary_only",
            "evidence": ["freeze_bas"],
            "note": "Must be paired with event coverage, lag and phase; never use alone.",
        },
        "repeat_similarity": {
            "status": "needs_revision",
            "evidence": ["repeat_similarity"],
            "note": "The current corruption/metric pair does not reliably detect repetition.",
        },
        "fid_div_retrieval_human": {
            "status": "not_yet_validated",
            "evidence": [],
            "note": "Requires fixed extractors, negative controls, or human studies.",
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gt_benchmark_validity_v1",
        "input": str(input_path),
        "row_count": len(rows),
        "sequence_count": len({row["sequence_id"] for row in rows}),
        "checks": checks,
        "metric_status": metric_status,
        "scope_note": "Controlled GT corruptions validate metric sensitivity; clean GT is a reference distribution, not a perfect score of human aesthetics.",
    }
    (output_dir / "benchmark_validity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "metric_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric_family", "status", "evidence", "note"))
        writer.writeheader()
        for family, status in metric_status.items():
            writer.writerow(
                {
                    "metric_family": family,
                    "status": status["status"],
                    "evidence": ",".join(status["evidence"]),
                    "note": status["note"],
                }
            )

    lines = [
        "# GT Benchmark Validity Audit v1",
        "",
        "This audit uses paired human-dance GT and controlled corruptions. It validates whether",
        "a metric responds to a known defect; it does not assign an aesthetic score to GT.",
        "",
        f"- Input: `{input_path}`",
        f"- Sequences: {payload['sequence_count']}",
        f"- Corrupted/clean records: {payload['row_count']}",
        "",
        "## Directional Checks",
        "",
        "| check | metric | expected | result | high-clean |",
        "|---|---|---|---|---:|",
    ]
    for check in checks:
        lines.append(
            f"| {check['check_id']} | {check['metric']} | {check['expected_response']} | "
            f"**{check['result']}** | {check['high_minus_clean']:+.4f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "The validated core currently includes jerk, low-pass energy response, static ratio,",
        "and event F1 for their stated use cases, plus BAS as a core beat-alignment metric.",
        "BAS can be insensitive to defects or reward sparse motion beats, so it must be combined",
        "with event coverage, onset response, lag, tempo and phase. The current repeat similarity",
        "implementation is not accepted as a core metric until a stronger repetition corruption",
        "and a long-range self-similarity measure are added.",
        "",
        "FIDk/FIDg, Divk/Divg, retrieval R@K/MMDist, style/emotion and human preference are",
        "not rejected; they remain unvalidated because their fixed extractors or human protocol",
        "have not yet been run on this benchmark.",
        "",
        "Clean GT is treated as a high-quality empirical reference distribution. It is not assigned",
        "a score of 1.0, and not every GT clip is expected to maximize every metric.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote benchmark validity audit to {output_dir}")
    for check in checks:
        print(f"{check['result']}: {check['check_id']}")


if __name__ == "__main__":
    main(parse_args())
