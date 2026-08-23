"""Aggregate the formal M3 paired/wrong/shifted/null generator evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = {
    "paired_joint_rmse_rad": ("paired_rmse_rad", "lower"),
    "paired_normalized_effect": ("normalized_effect", "lower"),
    "motion_energy_rad2_s2": ("energy", "neutral"),
    "joint_jerk_p95_rad_s3": ("jerk_p95", "lower"),
    "bas_music_to_motion": ("bas", "higher"),
    "impact_best_correlation": ("impact_corr", "higher"),
    "impact_best_lag_seconds": ("impact_abs_lag", "lower_abs"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        nargs="+",
        default=[
            Path("eval/m3_music_ablation/formal_30s/song012/analysis_v2"),
            Path("eval/m3_music_ablation/formal_30s/song065/analysis_v2"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/m3_music_ablation/formal_30s/aggregate_v2"),
    )
    return parser.parse_args()


def _metric(record: dict, name: str) -> float:
    if name == "paired_joint_rmse_rad":
        return float(record["paired_joint_rmse_rad"])
    if name == "paired_normalized_effect":
        return float(record["paired_normalized_effect"])
    if name == "motion_energy_rad2_s2":
        return float(record["quality"]["motion_energy_rad2_s2"])
    if name == "joint_jerk_p95_rad_s3":
        return float(record["quality"]["joint_jerk_abs_rad_s3"]["p95"])
    value = float(record["music"][name])
    return abs(value) if name == "impact_best_lag_seconds" else value


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median,
    }


def _wins(paired: list[float], control: list[float], direction: str) -> int:
    if direction == "higher":
        return sum(a > b for a, b in zip(paired, control))
    if direction == "lower":
        return sum(a < b for a, b in zip(paired, control))
    if direction == "lower_abs":
        return sum(abs(a) < abs(b) for a, b in zip(paired, control))
    return None


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    for analysis_dir in args.analysis_dir:
        payload = json.loads((analysis_dir / "metrics.json").read_text(encoding="utf-8"))
        records.extend(payload["records"])
    if len(records) != 24:
        raise ValueError(f"expected 24 formal records, found {len(records)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        label = record["label"]
        if label.startswith("paired"):
            condition = "paired"
        elif label.startswith("wrong"):
            condition = "wrong"
        elif label.startswith("shifted"):
            condition = "shifted_4s"
        else:
            condition = "null"
        row = {
            "song": record["motion_sequence_id"],
            "seed": record["seed"],
            "label": label,
            "condition": condition,
            "condition_variant": record["condition_variant"],
            "condition_shift_frames": record["condition_shift_frames"],
        }
        for source_name, (output_name, _) in METRICS.items():
            row[output_name] = _metric(record, source_name)
        rows.append(row)
    rows.sort(key=lambda row: (row["song"], row["seed"], row["label"]))
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    condition_summary = {}
    for condition, condition_rows in sorted(grouped.items()):
        condition_summary[condition] = {
            output_name: _summary([row[output_name] for row in condition_rows])
            for output_name, _ in METRICS.values()
        }

    blocks: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        song = row["song"]
        seed = int(row["seed"])
        condition = row["condition"]
        blocks[(song, seed)][condition] = row

    contrasts = {}
    for control in ("wrong", "shifted_4s", "null"):
        contrasts[control] = {}
        for source_name, (output_name, direction) in METRICS.items():
            paired_values = [block["paired"][output_name] for block in blocks.values()]
            control_values = [block[control][output_name] for block in blocks.values()]
            deltas = [a - b for a, b in zip(paired_values, control_values)]
            contrasts[control][output_name] = {
                "paired_minus_control_mean": mean(deltas),
                "paired_wins": _wins(paired_values, control_values, direction),
                "n": len(deltas),
            }

    report = {
        "schema_version": "m3_music_ablation_formal_v2",
        "n_records": len(records),
        "n_blocks": len(blocks),
        "analysis_dirs": [str(path) for path in args.analysis_dir],
        "condition_summary": condition_summary,
        "paired_contrasts": contrasts,
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# M3 Music-Condition Causal Ablation: Aggregate v2",
        "",
        "Formal generator-level evaluation: songs 012/065, seeds 1234/2345/3456,",
        "conditions paired, wrong-song, +4 s shifted, and null. Total: 24 trajectories.",
        "",
        "## Condition Summary",
        "",
        "| condition | energy | jerk P95 | BAS | impact corr. | impact lag |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"normal": "paired/wrong/shifted", "null": "null"}
    for condition, summary in condition_summary.items():
        def fmt(key: str) -> str:
            item = summary[key]
            return f"{item['mean']:.3f} +/- {item['std']:.3f}"
        lines.append(
            f"| {condition} | {fmt('energy')} | {fmt('jerk_p95')} | "
            f"{fmt('bas')} | {fmt('impact_corr')} | {fmt('impact_abs_lag')} |"
        )
    lines.extend([
        "",
        "## Paired Contrasts",
        "",
        "A win uses the metric direction defined in the frozen evaluation map; `impact_abs_lag` uses absolute lag.",
        "",
        "| control | metric | paired-control mean | paired wins / 6 |",
        "|---|---|---:|---:|",
    ])
    for control, metrics in contrasts.items():
        for metric in (
            "energy",
            "jerk_p95",
            "bas",
            "impact_corr",
            "impact_abs_lag",
        ):
            item = metrics[metric]
            wins = item['paired_wins']
            wins_text = "not ranked" if wins is None else f"{wins} / {item['n']}"
            lines.append(
                f"| {control} | {metric} | {item['paired_minus_control_mean']:.4f} | "
                f"{wins_text} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The sidecar is causally active because changing the music condition changes the generated trajectory.",
        "This table does not yet establish correct music alignment: paired must win on event/phase metrics",
        "against wrong, shifted, and null controls before making that claim.",
        "",
        "Next: separate RMS-only and predicted-FMS-only controls, record sidecar magnitude, then run the",
        "paired reference through the fixed SONIC execution protocol.",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} records and {len(blocks)} paired blocks to {args.output_dir}")


if __name__ == "__main__":
    main()
