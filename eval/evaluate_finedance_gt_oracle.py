"""Evaluate the sealed FineDance cross-genre test as a G1 GT oracle.

This evaluator is model-independent.  It reads paired retargeted G1 motion and
audio from the frozen manifest and reports a profile for motion quality and
music alignment.  It does not produce a single aesthetic score and it does
not use a generator checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_motion_music_execution import (
    DEFAULT_MODEL_PATH,
    compute_motion_quality,
    compute_music_metrics,
    load_audio_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/benchmark_v1/gt/finedance_gt_oracle_v1"),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--event-tolerance-seconds", type=float, default=0.20)
    parser.add_argument("--max-lag-seconds", type=float, default=1.0)
    return parser.parse_args()


def _load_motion(path: Path) -> dict[str, np.ndarray | float]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "fps": float(payload.get("fps", 30.0)),
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
    }


def _event_f1(audio_events: np.ndarray, motion_events: np.ndarray, tolerance: float) -> dict:
    audio_events = np.asarray(audio_events, dtype=np.float64)
    motion_events = np.asarray(motion_events, dtype=np.float64)
    used: set[int] = set()
    true_positive = 0
    errors = []
    for event in audio_events:
        candidates = [
            (abs(float(candidate - event)), index)
            for index, candidate in enumerate(motion_events)
            if index not in used and abs(float(candidate - event)) <= tolerance
        ]
        if candidates:
            error, index = min(candidates)
            used.add(index)
            true_positive += 1
            errors.append(error)
    precision = true_positive / max(len(motion_events), 1)
    recall = true_positive / max(len(audio_events), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_events": int(true_positive),
        "median_abs_timing_error_seconds": float(np.median(errors)) if errors else None,
    }


def _bpm(events: np.ndarray) -> float | None:
    intervals = np.diff(np.asarray(events, dtype=np.float64))
    intervals = intervals[(intervals >= 0.25) & (intervals <= 2.0)]
    return float(60.0 / np.median(intervals)) if len(intervals) else None


def _phase_error(
    audio_events: np.ndarray,
    motion_events: np.ndarray,
    lag_seconds: float,
) -> dict:
    audio_events = np.asarray(audio_events, dtype=np.float64)
    shifted_motion = np.asarray(motion_events, dtype=np.float64) + float(lag_seconds)
    if not len(audio_events) or not len(shifted_motion):
        return {"mean_phase_error_cycles": None, "median_phase_error_cycles": None}
    period = np.median(np.diff(audio_events)) if len(audio_events) > 1 else np.nan
    if not np.isfinite(period) or period <= 0:
        return {"mean_phase_error_cycles": None, "median_phase_error_cycles": None}
    errors = np.min(np.abs(shifted_motion[:, None] - audio_events[None]), axis=1)
    cycles = np.minimum(errors / period, 0.5)
    return {
        "mean_phase_error_cycles": float(np.mean(cycles)),
        "median_phase_error_cycles": float(np.median(cycles)),
    }


def _music_extension(
    speed_curve: np.ndarray,
    fps: float,
    audio_path: Path,
    audio_cache: dict,
    tolerance: float,
    max_lag_seconds: float,
) -> dict:
    onset_curve, audio_beats = audio_cache.setdefault(
        (str(audio_path), float(fps), int(len(speed_curve))),
        load_audio_analysis(audio_path, fps, len(speed_curve)),
    )
    smooth_speed = gaussian_filter1d(speed_curve, sigma=max(1.0, 0.10 * fps))
    impact = gaussian_filter1d(np.abs(np.gradient(smooth_speed, 1.0 / fps)), sigma=max(1.0, 0.05 * fps))
    prominence = max(float(np.std(impact) * 0.25), 1e-8)
    motion_frames, _ = find_peaks(
        impact,
        distance=max(1, int(round(0.25 * fps))),
        prominence=prominence,
    )
    motion_beats = motion_frames / fps
    base = compute_music_metrics(
        speed_curve,
        fps=fps,
        audio_path=audio_path,
        audio_cache=audio_cache,
    )
    f1 = _event_f1(audio_beats, motion_beats, tolerance)
    phase = _phase_error(audio_beats, motion_beats, base["impact_best_lag_seconds"])
    audio_bpm = _bpm(audio_beats)
    motion_bpm = _bpm(motion_beats)
    return {
        **base,
        "audio_bpm": audio_bpm,
        "motion_impact_bpm": motion_bpm,
        "tempo_abs_error_bpm": (
            abs(float(audio_bpm) - float(motion_bpm))
            if audio_bpm is not None and motion_bpm is not None
            else None
        ),
        "motion_impact_events": int(len(motion_beats)),
        "event_f1": f1,
        "phase": phase,
        "event_tolerance_seconds": tolerance,
        "max_lag_seconds": max_lag_seconds,
    }


def _mean_std(rows: list[dict], path: tuple[str, ...]) -> dict:
    values = []
    for row in rows:
        value: object = row
        for key in path:
            value = value[key]  # type: ignore[index]
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset = manifest["datasets"]["finedance"]
    test_ids = set(dataset["split"]["test_assets_available"])
    records = [
        record
        for record in dataset["records"]
        if record["sequence_id"] in test_ids and record["paired_valid"]
    ]
    if len(records) != len(test_ids):
        raise ValueError(f"Expected {len(test_ids)} paired test records, found {len(records)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_cache: dict = {}
    results = []
    for index, record in enumerate(sorted(records, key=lambda item: item["sequence_id"]), start=1):
        motion = _load_motion(Path(record["g1_motion"]["path"]))
        fps = float(motion["fps"])
        frames = min(
            len(motion["dof_pos"]),
            int(round(float(record["common_duration_seconds"]) * fps)),
        )
        for key in ("root_pos", "root_rot", "dof_pos"):
            motion[key] = np.asarray(motion[key])[:frames]
        quality, speed_curve = compute_motion_quality(
            motion,
            model_path=args.model_path,
            quat_order=record["g1_motion"]["root_quat_order"],
        )
        music = _music_extension(
            speed_curve,
            fps,
            Path(record["audio"]["path"]),
            audio_cache,
            args.event_tolerance_seconds,
            args.max_lag_seconds,
        )
        results.append(
            {
                "sequence_id": record["sequence_id"],
                "style": record.get("style", []),
                "label_name": record.get("label_name"),
                "frames": frames,
                "duration_seconds": float(frames / fps),
                "motion_path": record["g1_motion"]["path"],
                "audio_path": record["audio"]["path"],
                "quality": quality,
                "music": music,
            }
        )
        print(
            f"[{index}/{len(records)}] {record['sequence_id']} "
            f"energy={quality['motion_energy_rad2_s2']:.3f} "
            f"event_f1={music['event_f1']['f1']:.3f} "
            f"tempo_err={music['tempo_abs_error_bpm']}",
            flush=True,
        )

    summary_paths = {
        "quality.motion_energy_rad2_s2": ("quality", "motion_energy_rad2_s2"),
        "quality.joint_jerk_abs_rad_s3.p95": ("quality", "joint_jerk_abs_rad_s3", "p95"),
        "quality.static_ratio_speed_below_008": ("quality", "static_ratio_speed_below_008"),
        "quality.repeated_pose_ratio_rms008_after2s": ("quality", "repeated_pose_ratio_rms008_after2s"),
        "quality.physical.fsr_ground_calibrated_proxy": ("quality", "physical", "fsr_ground_calibrated_proxy"),
        "quality.physical.pfc_proxy": ("quality", "physical", "pfc_proxy"),
        "quality.root_height_min_m": ("quality", "root_height_min_m"),
        "music.speed_best_correlation": ("music", "speed_best_correlation"),
        "music.impact_best_correlation": ("music", "impact_best_correlation"),
        "music.impact_best_lag_seconds": ("music", "impact_best_lag_seconds"),
        "music.bas_music_to_motion": ("music", "bas_music_to_motion"),
        "music.event_f1.f1": ("music", "event_f1", "f1"),
        "music.event_f1.median_abs_timing_error_seconds": (
            "music", "event_f1", "median_abs_timing_error_seconds"
        ),
        "music.audio_bpm": ("music", "audio_bpm"),
        "music.motion_impact_bpm": ("music", "motion_impact_bpm"),
        "music.tempo_abs_error_bpm": ("music", "tempo_abs_error_bpm"),
        "music.phase.mean_phase_error_cycles": (
            "music", "phase", "mean_phase_error_cycles"
        ),
    }
    summary = {
        "test_count": len(results),
        "test_ids": [row["sequence_id"] for row in results],
        "quality_policy": "unfiltered sealed FineDance-G1 test; common motion/G1/audio duration",
        "execution_status": "not included; run SONIC separately and report retention against these GT rows",
        "metrics": {name: _mean_std(results, path) for name, path in summary_paths.items()},
    }
    payload = {
        "schema_version": "finedance_gt_oracle_v1",
        "manifest": str(args.manifest.resolve()),
        "model_independent": True,
        "event_tolerance_seconds": args.event_tolerance_seconds,
        "max_lag_seconds": args.max_lag_seconds,
        "summary": summary,
        "records": results,
    }
    (args.output_dir / "gt_oracle_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    fields = ["sequence_id", "label_name", "style", "frames", "duration_seconds"]
    for name in summary_paths:
        fields.append(name)
    with (args.output_dir / "gt_oracle_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {
                "sequence_id": row["sequence_id"],
                "label_name": row["label_name"],
                "style": ";".join(row["style"]),
                "frames": row["frames"],
                "duration_seconds": row["duration_seconds"],
            }
            for name, path in summary_paths.items():
                value: object = row
                for key in path:
                    value = value[key]  # type: ignore[index]
                flat[name] = value
            writer.writerow(flat)
    report = [
        "# FineDance-G1 GT Oracle Evaluation",
        "",
        "This report evaluates the frozen 18-sequence FineDance cross-genre test "
        "using paired retargeted G1 motion and audio. It is model-independent and "
        "does not use a generator checkpoint.",
        "",
        f"Test sequences: **{len(results)}**.",
        "",
        "The report separates motion quality from music correspondence. Event F1 "
        "matches audio beat events to motion impact events within the configured "
        "tolerance; tempo error is absolute BPM difference; phase error is nearest "
        "event offset normalized by the audio beat period after the measured lag.",
        "",
        "## Aggregate metrics",
        "",
        "| metric | mean | std | count |",
        "|---|---:|---:|---:|",
    ]
    for name, values in summary["metrics"].items():
        report.append(
            f"| `{name}` | {values['mean'] if values['mean'] is not None else 'n/a'} | "
            f"{values['std'] if values['std'] is not None else 'n/a'} | {values['count']} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "These values define the empirical GT reference range; they are not a "
            "claim that the dataset is perfectly musical or physically executable "
            "on the robot. Generated references must be compared against this same "
            "test set, and SONIC execution must be evaluated separately through "
            "reference-to-execution retention.",
            "",
            "Existing paired-vs-wrong-song retrieval results remain in "
            "`eval/benchmark_v1/gt/finedance_music_pairing_v1/`; this report adds the "
            "full motion-quality and event/tempo/phase profile.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
