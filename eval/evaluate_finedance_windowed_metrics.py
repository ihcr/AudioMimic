"""Evaluate FineDance source/G1 music metrics at fixed temporal windows.

This is a diagnostic complement to the full-sequence oracle report. It uses the
same audio beat detector and motion activity definition as the retargeting audit,
but evaluates complete 5-second and 16-second windows separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, find_peaks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_motion_music_execution import (  # noqa: E402
    _bas,
    best_correlation,
    load_audio_analysis,
)
from eval.evaluate_finedance_gt_oracle import (  # noqa: E402
    _bpm,
    _event_f1,
    _phase_error,
)
from eval.evaluate_retargeting_loss import (  # noqa: E402
    _load_manifest_records,
    _load_pair,
    _normalized_correlation,
    _resample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"),
    )
    parser.add_argument("--scope", choices=("test", "all"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/benchmark_v1/gt/finedance_windowed_v1"),
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--event-tolerance-seconds", type=float, default=0.20)
    parser.add_argument("--max-lag-seconds", type=float, default=1.0)
    parser.add_argument("--windows", type=float, nargs="+", default=[5.0, 16.0])
    return parser.parse_args()


def _window_music(
    speed: np.ndarray,
    onset: np.ndarray,
    audio_beats: np.ndarray,
    fps: float,
    tolerance: float,
    max_lag: float,
) -> dict:
    # Match the formal evaluator: the BAS/correlation branch uses a 5-frame
    # smooth at 30 FPS, while the event branch uses the shorter 0.10 s smooth.
    smooth_speed_base = gaussian_filter1d(
        speed, sigma=max(1.0, fps * 5.0 / 30.0)
    )
    impact_base = gaussian_filter1d(
        np.abs(np.gradient(smooth_speed_base, 1.0 / fps)),
        sigma=max(1.0, 0.05 * fps),
    )
    smooth_speed_event = gaussian_filter1d(speed, sigma=max(1.0, 0.10 * fps))
    impact_event = gaussian_filter1d(
        np.abs(np.gradient(smooth_speed_event, 1.0 / fps)),
        sigma=max(1.0, 0.05 * fps),
    )
    prominence = max(float(np.std(impact_event) * 0.25), 1e-8)
    event_frames, _ = find_peaks(
        impact_event,
        distance=max(1, int(round(0.25 * fps))),
        prominence=prominence,
    )
    motion_events = event_frames / fps
    beat_frames = np.asarray(argrelextrema(smooth_speed_base, np.less)[0], dtype=np.int64)
    motion_beats = beat_frames / fps
    speed_corr = best_correlation(onset, smooth_speed_base, fps, max_lag)
    impact_corr = best_correlation(onset, impact_base, fps, max_lag)
    audio_bpm = _bpm(audio_beats)
    motion_bpm = _bpm(motion_events)
    return {
        "speed_corr": speed_corr["best_correlation"],
        "speed_lag_seconds": speed_corr["best_lag_seconds"],
        "impact_corr": impact_corr["best_correlation"],
        "impact_lag_seconds": impact_corr["best_lag_seconds"],
        "bas_music_to_motion": _bas(audio_beats, motion_beats, "music_to_motion"),
        "bas_motion_to_music": _bas(audio_beats, motion_beats, "motion_to_music"),
        "audio_beats": int(len(audio_beats)),
        "motion_beats": int(len(motion_beats)),
        "event_f1": _event_f1(audio_beats, motion_events, tolerance),
        "audio_bpm": audio_bpm,
        "motion_bpm": motion_bpm,
        "tempo_error_bpm": (
            abs(float(audio_bpm) - float(motion_bpm))
            if audio_bpm is not None and motion_bpm is not None
            else None
        ),
        "phase_error_cycles": _phase_error(
            audio_beats, motion_events, impact_corr["best_lag_seconds"]
        )["mean_phase_error_cycles"],
    }


def _finite_mean(rows: list[dict], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    return float(np.mean(values)) if values else None


def _finite_median(rows: list[dict], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    return float(np.median(values)) if values else None


def main() -> None:
    args = parse_args()
    records = _load_manifest_records(args.manifest, "finedance", args.scope)
    if not records:
        raise RuntimeError("No paired FineDance records found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_cache: dict = {}
    rows: list[dict] = []

    for index, record in enumerate(records, start=1):
        source, target, audio_path, _, duration = _load_pair("finedance", record, args)
        target_frames = min(len(target["activity"]), int(round(duration * args.target_fps)))
        source_activity = _resample(
            source["activity"], source["fps"], target_frames, args.target_fps
        )
        target_activity = target["activity"][:target_frames]
        audio_key = (str(audio_path), float(args.target_fps), int(target_frames))
        if audio_key not in audio_cache:
            audio_cache[audio_key] = load_audio_analysis(
                audio_path, args.target_fps, target_frames
            )
        onset, full_audio_beats = audio_cache[audio_key]
        for window_seconds in [None, *args.windows]:
            if window_seconds is None:
                starts = [0]
                window_label = "full"
                window_frames = target_frames
            else:
                window_frames = int(round(window_seconds * args.target_fps))
                starts = list(range(0, target_frames - window_frames + 1, window_frames))
                window_label = f"{window_seconds:g}s"
            for start in starts:
                end = start + window_frames
                audio_beats = full_audio_beats[
                    (full_audio_beats >= start / args.target_fps)
                    & (full_audio_beats < end / args.target_fps)
                ] - start / args.target_fps
                source_slice = source_activity[start:end]
                target_slice = target_activity[start:end]
                onset_slice = onset[start:end]
                if len(source_slice) < int(2.0 * args.target_fps):
                    continue
                source_music = _window_music(
                    source_slice, onset_slice, audio_beats, args.target_fps,
                    args.event_tolerance_seconds, args.max_lag_seconds,
                )
                target_music = _window_music(
                    target_slice, onset_slice, audio_beats, args.target_fps,
                    args.event_tolerance_seconds, args.max_lag_seconds,
                )
                activity_corr = _normalized_correlation(
                    source_slice, target_slice, args.target_fps, args.max_lag_seconds
                )
                for stage, music in (("source", source_music), ("g1_target", target_music)):
                    rows.append({
                        "sequence_id": str(record["sequence_id"]),
                        "stage": stage,
                        "window": window_label,
                        "start_seconds": start / args.target_fps,
                        "duration_seconds": len(source_slice) / args.target_fps,
                        "activity_corr_source_to_g1": activity_corr["best"] if stage == "source" else "",
                        "bas_music_to_motion": music["bas_music_to_motion"],
                        "bas_motion_to_music": music["bas_motion_to_music"],
                        "event_precision": music["event_f1"]["precision"],
                        "event_recall": music["event_f1"]["recall"],
                        "event_f1": music["event_f1"]["f1"],
                        "impact_corr": music["impact_corr"],
                        "impact_lag_seconds": music["impact_lag_seconds"],
                        "tempo_error_bpm": music["tempo_error_bpm"],
                        "phase_error_cycles": music["phase_error_cycles"],
                        "audio_beats": music["audio_beats"],
                        "motion_beats": music["motion_beats"],
                    })
        if index % 10 == 0 or index == len(records):
            print(f"processed {index}/{len(records)} FineDance sequences", flush=True)

    fields = list(rows[0])
    with (args.output_dir / "windowed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "windowed_metrics.json").write_text(
        json.dumps({
            "schema_version": "finedance_windowed_metrics_v1",
            "scope": args.scope,
            "sequence_count": len(records),
            "windows_seconds": args.windows,
            "rows": rows,
        }, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    summary_rows = []
    for window in ["5s", "16s", "full"]:
        for stage in ("source", "g1_target"):
            group = [r for r in rows if r["window"] == window and r["stage"] == stage]
            if not group:
                continue
            summary_rows.append({
                "window": window,
                "stage": stage,
                "n_windows": len(group),
                "bas_median": _finite_median(group, "bas_music_to_motion"),
                "event_f1_median": _finite_median(group, "event_f1"),
                "impact_corr_median": _finite_median(group, "impact_corr"),
                "tempo_error_median_bpm": _finite_median(group, "tempo_error_bpm"),
                "phase_error_median_cycles": _finite_median(group, "phase_error_cycles"),
                "bas_mean": _finite_mean(group, "bas_music_to_motion"),
                "event_f1_mean": _finite_mean(group, "event_f1"),
                "impact_corr_mean": _finite_mean(group, "impact_corr"),
            })
    summary_fields = list(summary_rows[0])
    with (args.output_dir / "windowed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    report = [
        "# FineDance Windowed Source/G1 Music Metrics",
        "",
        f"Scope: **{args.scope}**, paired sequences: **{len(records)}**.",
        "Fixed windows are non-overlapping complete windows; an incomplete tail is omitted.",
        "The detector, tolerance, FPS and audio clock are identical to the formal retargeting audit.",
        "",
        "| window | stage | n windows | BAS median | Event F1 median | Impact corr median | Tempo error median | Phase error median |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        report.append(
            f"| {item['window']} | {item['stage']} | {item['n_windows']} "
            f"| {item['bas_median']:.4f} | {item['event_f1_median']:.4f} "
            f"| {item['impact_corr_median']:.4f} | {item['tempo_error_median_bpm']:.4f} "
            f"| {item['phase_error_median_cycles']:.4f} |"
        )
    report += [
        "",
        "Interpretation: compare source vs G1 within each window first. If the source is already low",
        "at 5 s and 16 s, the difference is not caused by long-sequence averaging alone. If source is",
        "reasonable but G1 drops, the dominant issue is retargeting. Full-sequence values are reported",
        "separately because they mix local rhythm with phrase and transition structure.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} window rows to {args.output_dir}")


if __name__ == "__main__":
    main()
