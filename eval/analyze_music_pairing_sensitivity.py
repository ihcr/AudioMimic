"""Audit fixed-trajectory music pairing without claiming causal conditioning.

This analysis compares the correct song098 clock against deterministic circular
audio shifts and a wrong-song control.  It can reveal temporal pairing evidence
in existing reference/execution trajectories, but only regenerated
shuffle/silence conditions can establish that a generator uses its music input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_motion_music_execution import (
    BAS_SIGMA_SECONDS,
    DEFAULT_MODEL_PATH,
    _bas,
    best_correlation,
    compute_motion_quality,
    load_audio_analysis,
    load_execution_pair,
    load_reference_motion,
)


METRICS = (
    "impact_zero_lag_correlation",
    "impact_best_correlation",
    "speed_zero_lag_correlation",
    "bas_music_to_motion",
    "bas_motion_to_music",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion_root",
        type=Path,
        default=Path("~/Musics2Dance-prior-dev/onlinegeneratedmotion"),
    )
    parser.add_argument(
        "--tracking_root",
        type=Path,
        default=Path("eval/generation_to_execution_gap"),
    )
    parser.add_argument(
        "--tracking_glob",
        default="m[024]_song098_seed1234_full_rate100_aligned_r0*",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--shift_step_seconds", type=float, default=2.0)
    parser.add_argument("--analysis_fps", type=float, default=50.0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("eval/music_pairing_sensitivity/20260820"),
    )
    return parser.parse_args()


def _motion_signals(speed_curve: np.ndarray, fps: float) -> dict:
    speed_curve = np.asarray(speed_curve, dtype=np.float64)
    smooth_speed = gaussian_filter1d(speed_curve, sigma=max(1.0, fps * 5.0 / 30.0))
    impact_curve = gaussian_filter1d(
        np.abs(np.gradient(smooth_speed, 1.0 / fps)),
        sigma=max(1.0, 0.05 * fps),
    )
    beat_frames = np.asarray(argrelextrema(smooth_speed, np.less)[0], dtype=np.int64)
    return {
        "smooth_speed": smooth_speed,
        "impact_curve": impact_curve,
        "motion_beats": beat_frames / fps,
    }


def _pair_metrics(
    onset_curve: np.ndarray,
    audio_beats: np.ndarray,
    motion: dict,
    fps: float,
) -> dict[str, float]:
    speed = best_correlation(onset_curve, motion["smooth_speed"], fps)
    impact = best_correlation(onset_curve, motion["impact_curve"], fps)
    return {
        "impact_zero_lag_correlation": impact["zero_lag_correlation"],
        "impact_best_correlation": impact["best_correlation"],
        "impact_best_lag_seconds": impact["best_lag_seconds"],
        "speed_zero_lag_correlation": speed["zero_lag_correlation"],
        "speed_best_correlation": speed["best_correlation"],
        "speed_best_lag_seconds": speed["best_lag_seconds"],
        "bas_music_to_motion": _bas(
            audio_beats, motion["motion_beats"], "music_to_motion"
        ),
        "bas_motion_to_music": _bas(
            audio_beats, motion["motion_beats"], "motion_to_music"
        ),
    }


def _circular_audio_shift(
    onset_curve: np.ndarray,
    audio_beats: np.ndarray,
    shift_seconds: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    frames = len(onset_curve)
    duration = frames / fps
    shifted_onset = np.roll(onset_curve, int(round(shift_seconds * fps)))
    shifted_beats = np.sort(np.mod(audio_beats + shift_seconds, duration))
    return shifted_onset, shifted_beats


def _metric_contrast(paired: float, null_values: list[float], wrong: float) -> dict:
    values = np.asarray(null_values, dtype=np.float64)
    std = float(np.std(values))
    return {
        "paired": float(paired),
        "shift_null_mean": float(np.mean(values)),
        "shift_null_std": std,
        "shift_null_p95": float(np.percentile(values, 95)),
        "paired_minus_null_mean": float(paired - np.mean(values)),
        "paired_rank_percentile": float(100.0 * np.mean(paired > values)),
        "paired_z_vs_shift_null": (
            float((paired - np.mean(values)) / std) if std > 1e-12 else None
        ),
        "wrong_song": float(wrong),
        "paired_minus_wrong_song": float(paired - wrong),
    }


def analyze_pairing(
    speed_curve: np.ndarray,
    *,
    fps: float,
    paired_audio: Path,
    wrong_audio: Path,
    shift_step_seconds: float,
    audio_cache: dict,
) -> dict:
    frames = len(speed_curve)
    duration = frames / fps
    motion = _motion_signals(speed_curve, fps)
    paired_onset, paired_beats = load_audio_analysis(
        paired_audio, fps, frames
    ) if (str(paired_audio), fps, frames) not in audio_cache else audio_cache[
        (str(paired_audio), fps, frames)
    ]
    audio_cache[(str(paired_audio), fps, frames)] = paired_onset, paired_beats
    wrong_onset, wrong_beats = load_audio_analysis(
        wrong_audio, fps, frames
    ) if (str(wrong_audio), fps, frames) not in audio_cache else audio_cache[
        (str(wrong_audio), fps, frames)
    ]
    audio_cache[(str(wrong_audio), fps, frames)] = wrong_onset, wrong_beats

    paired = _pair_metrics(paired_onset, paired_beats, motion, fps)
    wrong = _pair_metrics(wrong_onset, wrong_beats, motion, fps)
    shifts = np.arange(shift_step_seconds, duration, shift_step_seconds)
    shifted_metrics = []
    for shift in shifts:
        shifted_onset, shifted_beats = _circular_audio_shift(
            paired_onset, paired_beats, float(shift), fps
        )
        shifted_metrics.append(_pair_metrics(shifted_onset, shifted_beats, motion, fps))

    return {
        "duration_seconds": float(duration),
        "shift_step_seconds": float(shift_step_seconds),
        "shift_count": int(len(shifts)),
        "motion_beats": int(len(motion["motion_beats"])),
        "audio_beats": int(len(paired_beats)),
        "metrics": {
            metric: _metric_contrast(
                paired[metric],
                [record[metric] for record in shifted_metrics],
                wrong[metric],
            )
            for metric in METRICS
        },
    }


def _route_from_run_id(run_id: str) -> str:
    return run_id.split("_", 1)[0].upper()


def _summarize(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["source"], record["route"])].append(record)
    summary = {}
    for (source, route), items in sorted(grouped.items()):
        key = f"{source}:{route}"
        summary[key] = {"source": source, "route": route, "samples": len(items)}
        for metric in METRICS:
            contrasts = [item["pairing"]["metrics"][metric] for item in items]
            summary[key][metric] = {
                name: float(np.mean([record[name] for record in contrasts]))
                for name in (
                    "paired",
                    "shift_null_mean",
                    "paired_minus_null_mean",
                    "paired_rank_percentile",
                    "wrong_song",
                    "paired_minus_wrong_song",
                )
            }
    return summary


def _write_report(payload: dict, path: Path) -> None:
    lines = [
        "# Music Pairing Sensitivity: Existing M0/M2/M4 Exports",
        "",
        "Date: 2026-08-20",
        "",
        "This is a fixed-trajectory pairing audit. It does not regenerate motion under shuffled "
        "or silent conditions and therefore cannot establish causal use of music by the generator.",
        "",
        "| Source | Route | N | Impact corr. paired/null | Impact rank | BAS M2M paired/null | BAS rank | BAS paired-wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(payload["summary"]):
        item = payload["summary"][key]
        impact = item["impact_zero_lag_correlation"]
        bas = item["bas_music_to_motion"]
        lines.append(
            f"| {item['source']} | {item['route']} | {item['samples']} | "
            f"{impact['paired']:.3f}/{impact['shift_null_mean']:.3f} | "
            f"{impact['paired_rank_percentile']:.1f}% | "
            f"{bas['paired']:.3f}/{bas['shift_null_mean']:.3f} | "
            f"{bas['paired_rank_percentile']:.1f}% | "
            f"{bas['paired_minus_wrong_song']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "A high rank means the correct audio clock scores above most circular shifts. It is a "
            "temporal-alignment diagnostic, not a calibrated p-value. M0 is the unconditional negative "
            "control; similar M0 and M2/M4 ranks weaken a music-conditioning claim.",
            "",
            "The next causal experiment must regenerate each seed with paired, time-shifted, shuffled, "
            "and silent music using the same M2 checkpoint and sampling noise.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    motion_root = args.motion_root.expanduser().resolve()
    tracking_root = args.tracking_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    paired_audio = motion_root / "audio/098_t000128_60s.wav"
    wrong_audio = motion_root / "audio/065_t000128_60s.wav"
    audio_cache = {}
    records = []

    reference_paths = sorted(
        path
        for directory in ("uncond_m0", "m2_predicted_fms", "m4_oracle_fms")
        for path in (motion_root / directory).glob("*song098.pkl")
    )
    for path in reference_paths:
        motion, metadata = load_reference_motion(path)
        _, speed_curve = compute_motion_quality(
            motion, model_path=model_path, quat_order="xyzw"
        )
        records.append(
            {
                "source": "reference",
                "route": metadata["route"].upper(),
                "id": path.stem,
                "path": str(path),
                "pairing": analyze_pairing(
                    speed_curve,
                    fps=float(motion["fps"]),
                    paired_audio=paired_audio,
                    wrong_audio=wrong_audio,
                    shift_step_seconds=args.shift_step_seconds,
                    audio_cache=audio_cache,
                ),
            }
        )

    for run_dir in sorted(tracking_root.glob(args.tracking_glob)):
        _, measured, _ = load_execution_pair(run_dir, args.analysis_fps)
        _, speed_curve = compute_motion_quality(
            measured, model_path=model_path, quat_order="wxyz"
        )
        records.append(
            {
                "source": "execution",
                "route": _route_from_run_id(run_dir.name),
                "id": run_dir.name,
                "path": str(run_dir),
                "pairing": analyze_pairing(
                    speed_curve,
                    fps=float(measured["fps"]),
                    paired_audio=paired_audio,
                    wrong_audio=wrong_audio,
                    shift_step_seconds=args.shift_step_seconds,
                    audio_cache=audio_cache,
                ),
            }
        )

    payload = {
        "schema_version": "music_pairing_sensitivity_v1",
        "evidence_scope": "fixed_trajectory_pairing_only",
        "causal_music_conditioning_evidence": False,
        "paired_song": "098",
        "wrong_song": "065",
        "null": "deterministic circular shifts of paired audio clock",
        "bas_sigma_seconds": BAS_SIGMA_SECONDS,
        "records": records,
        "summary": _summarize(records),
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairing_sensitivity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(payload, output_dir / "PAIRING_SENSITIVITY.md")
    print(f"Wrote {output_dir / 'pairing_sensitivity.json'}")
    for key, item in payload["summary"].items():
        impact = item["impact_zero_lag_correlation"]
        bas = item["bas_music_to_motion"]
        print(
            key,
            f"impact_rank={impact['paired_rank_percentile']:.1f}%",
            f"bas_rank={bas['paired_rank_percentile']:.1f}%",
            f"bas_wrong_delta={bas['paired_minus_wrong_song']:+.3f}",
        )


if __name__ == "__main__":
    main(parse_args())
