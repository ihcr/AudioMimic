"""Stratify FineDance music-motion metrics by dance style and audio tempo."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"
WINDOWED = ROOT / "eval/benchmark_v1/gt/finedance_windowed_v1/windowed_metrics.csv"
ORACLE = ROOT / "eval/benchmark_v1/gt/gt_oracle_suite_all_v1/gt_oracle_suite_metrics.json"
OUTPUT = ROOT / "eval/benchmark_v1/gt/finedance_stratified_v1"


MUSIC_FIELDS = (
    "bas_music_to_motion",
    "event_f1",
    "impact_corr",
    "tempo_error_bpm",
    "phase_error_cycles",
)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _load_metadata() -> dict[str, dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [r for r in payload["datasets"]["finedance"]["records"] if r.get("paired_valid")]
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    oracle_rows = {
        str(r["sequence_id"]): r
        for r in oracle["records"]
        if r["dataset"] == "finedance"
    }
    metadata = {}
    for row in rows:
        sequence_id = str(row["sequence_id"])
        oracle_row = oracle_rows.get(sequence_id, {})
        audio_bpm = oracle_row.get("music", {}).get("audio_bpm")
        if audio_bpm is None:
            tempo = "unknown"
        elif float(audio_bpm) < 90.0:
            tempo = "slow"
        elif float(audio_bpm) <= 130.0:
            tempo = "medium"
        else:
            tempo = "fast"
        styles = [str(s) for s in row.get("style", [])] or ["Unknown"]
        metadata[sequence_id] = {
            "styles": styles,
            "tempo_bin": tempo,
            "audio_bpm": audio_bpm,
        }
    return metadata


def _load_window_rows(metadata: dict[str, dict]) -> list[dict]:
    rows = []
    with WINDOWED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sequence_id = str(row["sequence_id"])
            if sequence_id not in metadata:
                continue
            row["sequence_id"] = sequence_id
            row["stage"] = row["stage"]
            row["window"] = row["window"]
            for field in MUSIC_FIELDS + ("activity_corr_source_to_g1",):
                value = row.get(field, "")
                row[field] = float(value) if value not in ("", None) else None
            rows.append(row)
    return rows


def _summarize(rows: list[dict], label: str, value: str) -> dict:
    summary = {"group": label, "value": value, "n_sequences": len({r["sequence_id"] for r in rows}), "n_windows": len(rows)}
    for field in MUSIC_FIELDS:
        values = [float(r[field]) for r in rows if r.get(field) is not None and np.isfinite(float(r[field]))]
        summary[f"{field}_median"] = _median(values)
        summary[f"{field}_mean"] = _mean(values)
    return summary


def main() -> None:
    metadata = _load_metadata()
    rows = _load_window_rows(metadata)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []

    styles = sorted({style for item in metadata.values() for style in item["styles"]})
    for style in styles:
        sequence_ids = {sid for sid, item in metadata.items() if style in item["styles"]}
        for window in ("5s", "16s", "full"):
            for stage in ("source", "g1_target"):
                group = [
                    r for r in rows
                    if r["sequence_id"] in sequence_ids
                    and r["window"] == window
                    and r["stage"] == stage
                ]
                if group:
                    summaries.append(_summarize(group, "style", style) | {"window": window, "stage": stage})

    for tempo in ("slow", "medium", "fast", "unknown"):
        sequence_ids = {sid for sid, item in metadata.items() if item["tempo_bin"] == tempo}
        for window in ("5s", "16s", "full"):
            for stage in ("source", "g1_target"):
                group = [
                    r for r in rows
                    if r["sequence_id"] in sequence_ids
                    and r["window"] == window
                    and r["stage"] == stage
                ]
                if group:
                    summaries.append(_summarize(group, "tempo", tempo) | {"window": window, "stage": stage})

    fields = ["group", "value", "window", "stage", "n_sequences", "n_windows"]
    for field in MUSIC_FIELDS:
        fields.extend([f"{field}_median", f"{field}_mean"])
    with (OUTPUT / "stratified_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    (OUTPUT / "metadata.json").write_text(
        json.dumps({"style_counts": {s: sum(s in x["styles"] for x in metadata.values()) for s in styles},
                    "tempo_counts": {t: sum(x["tempo_bin"] == t for x in metadata.values()) for t in ("slow", "medium", "fast", "unknown")},
                    "tempo_definition_bpm": {"slow": "<90", "medium": "90-130", "fast": ">130"}},
                   indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# FineDance Dance-Style and Tempo-Stratified Metrics",
        "",
        "This report stratifies the same music-motion evaluator by FineDance dance-style labels",
        "and audio tempo. A sequence with multiple labels contributes to each applicable style;",
        "style counts therefore overlap. All values are diagnostic distributions, not style rankings.",
        "",
        "Tempo bins: slow `<90 BPM`, medium `90--130 BPM`, fast `>130 BPM`.",
        "",
        "## Dance-style coverage",
        "",
        "| style | sequence count |",
        "|---|---:|",
    ]
    for style in styles:
        report.append(f"| {style} | {sum(style in x['styles'] for x in metadata.values())} |")
    report += [
        "",
        "## Style summary (full sequence)",
        "",
        "| style | stage | n seq | BAS | Event F1 | Impact corr | Tempo error | Phase error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        if item["group"] != "style" or item["window"] != "full":
            continue
        report.append(
            f"| {item['value']} | {item['stage']} | {item['n_sequences']} "
            f"| {item['bas_music_to_motion_median']:.4f} | {item['event_f1_median']:.4f} "
            f"| {item['impact_corr_median']:.4f} | {item['tempo_error_bpm_median']:.4f} "
            f"| {item['phase_error_cycles_median']:.4f} |"
        )
    report += [
        "",
        "## Tempo summary (full sequence)",
        "",
        "| tempo bin | stage | n seq | BAS | Event F1 | Impact corr | Tempo error | Phase error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        if item["group"] != "tempo" or item["window"] != "full":
            continue
        report.append(
            f"| {item['value']} | {item['stage']} | {item['n_sequences']} "
            f"| {item['bas_music_to_motion_median']:.4f} | {item['event_f1_median']:.4f} "
            f"| {item['impact_corr_median']:.4f} | {item['tempo_error_bpm_median']:.4f} "
            f"| {item['phase_error_cycles_median']:.4f} |"
        )
    report += [
        "",
        "Use the common core metrics for every style. Style-specific claims require adequate sample",
        "counts and should be supplemented with style/emotion retrieval or blinded human evaluation.",
        "A style with very few sequences is retained for transparency but must not be used for a strong claim.",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {len(summaries)} stratified summaries to {OUTPUT}")


if __name__ == "__main__":
    main()
