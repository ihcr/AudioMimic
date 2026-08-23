"""Audit FineDance audio-motion correspondence with deterministic wrong-song controls.

The audit is deliberately sequence-level. It does not claim that every valid
dance accent must occur at an audio onset; it tests whether the paired WAV is
more compatible with its G1 motion than a fixed cyclic permutation of the test
WAVs. This makes the result useful for validating the benchmark before model
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import wave
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, resample_poly, stft


TEST_IDS = (
    "063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
    "179", "065", "137", "161", "092", "120", "037", "109", "204", "144",
)
IGNORE_IDS = {"116", "117", "118", "119", "120", "121", "122", "123", "202", "130"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", type=Path, default=Path("data/finedance-g1-retargeted"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/finedance/music_wav"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("eval/benchmark_v1/gt/finedance_music_pairing_v1"),
    )
    parser.add_argument("--max-lag-seconds", type=float, default=2.0)
    parser.add_argument("--event-tolerance-seconds", type=float, default=0.20)
    return parser.parse_args()


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            values[:, 0].astype(np.int32)
            | (values[:, 1].astype(np.int32) << 8)
            | (values[:, 2].astype(np.int32) << 16)
        )
        signed[signed & 0x800000] -= 1 << 24
        audio = signed.astype(np.float32) / float(1 << 23)
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported PCM sample width {width} in {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    dof = np.asarray(payload["dof_pos"], dtype=np.float64)
    fps = float(payload.get("fps", 30.0))
    if dof.ndim != 2 or dof.shape[1] != 29 or fps <= 0:
        raise ValueError(f"Invalid G1 payload: {path}")
    velocity = np.diff(dof, axis=0, prepend=dof[:1]) * fps
    # Joint energy is used instead of root translation so drift cannot create
    # a false music match.
    energy = np.sqrt(np.mean(velocity * velocity, axis=1))
    energy -= np.median(energy)
    return energy, fps


def audio_onset_envelope(audio: np.ndarray, rate: int, target_fps: float) -> np.ndarray:
    audio = audio.astype(np.float64, copy=False)
    if not len(audio):
        return np.zeros(1, dtype=np.float64)
    _, _, spectrum = stft(
        audio,
        fs=rate,
        window="hann",
        nperseg=min(2048, len(audio)),
        noverlap=min(1536, max(0, len(audio) - 1)),
        boundary=None,
    )
    magnitude = np.abs(spectrum)
    flux = np.maximum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]), 0.0).sum(axis=0)
    if len(flux) < 2:
        return np.zeros(max(1, int(round(len(audio) / rate * target_fps))))
    source_times = np.linspace(0.0, len(audio) / rate, len(flux), endpoint=False)
    target_times = np.arange(max(1, int(np.floor(len(audio) / rate * target_fps)))) / target_fps
    envelope = np.interp(target_times, source_times, flux, left=flux[0], right=flux[-1])
    return envelope


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.mean(values)
    scale = np.std(values)
    return values / scale if scale > 1e-8 else np.zeros_like(values)


def correlation_at_lag(motion: np.ndarray, audio: np.ndarray, lag_frames: int) -> float:
    if lag_frames >= 0:
        left = motion[lag_frames:]
        right = audio[: len(left)]
    else:
        right = audio[-lag_frames:]
        left = motion[: len(right)]
    if len(left) < 8:
        return 0.0
    left = standardize(left)
    right = standardize(right)
    return float(np.mean(left * right))


def best_lag_correlation(
    motion: np.ndarray,
    audio: np.ndarray,
    fps: float,
    max_lag_seconds: float,
) -> tuple[float, float, float]:
    limit = int(round(max_lag_seconds * fps))
    values = [(correlation_at_lag(motion, audio, lag), lag) for lag in range(-limit, limit + 1)]
    best, lag = max(values, key=lambda item: item[0])
    return float(correlation_at_lag(motion, audio, 0)), float(best), float(lag / fps)


def peak_f1(motion: np.ndarray, audio: np.ndarray, fps: float, tolerance: float) -> float:
    motion_peaks, _ = find_peaks(standardize(motion), distance=max(1, int(0.20 * fps)), prominence=0.35)
    audio_peaks, _ = find_peaks(standardize(audio), distance=max(1, int(0.20 * fps)), prominence=0.35)
    if not len(motion_peaks) or not len(audio_peaks):
        return 0.0
    max_distance = tolerance * fps
    used = set()
    true_positive = 0
    for peak in audio_peaks:
        candidates = [
            (abs(int(candidate) - int(peak)), int(candidate))
            for candidate in motion_peaks
            if int(candidate) not in used and abs(int(candidate) - int(peak)) <= max_distance
        ]
        if candidates:
            _, match = min(candidates)
            used.add(match)
            true_positive += 1
    precision = true_positive / len(audio_peaks)
    recall = true_positive / len(motion_peaks)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def analyse_pair(
    motion: np.ndarray,
    audio: np.ndarray,
    fps: float,
    audio_rate: int,
    max_lag_seconds: float,
    event_tolerance_seconds: float,
) -> dict[str, float]:
    envelope = audio_onset_envelope(audio, audio_rate, fps)
    frames = min(len(motion), len(envelope))
    motion = motion[:frames]
    envelope = envelope[:frames]
    zero, best, lag = best_lag_correlation(motion, envelope, fps, max_lag_seconds)
    motion_spectrum = np.abs(np.fft.rfft(standardize(motion)))
    audio_spectrum = np.abs(np.fft.rfft(standardize(envelope)))
    frequencies = np.fft.rfftfreq(frames, d=1.0 / fps)
    band = (frequencies >= 0.5) & (frequencies <= 4.0)
    motion_rate = float(frequencies[band][np.argmax(motion_spectrum[band])]) if np.any(band) else 0.0
    audio_rate_hz = float(frequencies[band][np.argmax(audio_spectrum[band])]) if np.any(band) else 0.0
    return {
        "frames": int(frames),
        "duration_seconds": float(frames / fps),
        "motion_energy_mean": float(np.mean(np.abs(motion))),
        "audio_onset_mean": float(np.mean(envelope)),
        "zero_lag_corr": zero,
        "best_lag_corr": best,
        "best_lag_seconds": lag,
        "motion_dominant_rate_hz": motion_rate,
        "audio_dominant_rate_hz": audio_rate_hz,
        "dominant_rate_error_hz": abs(motion_rate - audio_rate_hz),
        "event_f1": peak_f1(motion, envelope, fps, event_tolerance_seconds),
    }


def main() -> None:
    args = parse_args()
    g1_root = args.g1_root.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()
    ids = [sequence_id for sequence_id in TEST_IDS if sequence_id not in IGNORE_IDS]
    pairs = []
    motions = {}
    audios = {}
    for sequence_id in ids:
        motion_path = g1_root / f"{sequence_id}.pkl"
        audio_path = audio_root / f"{sequence_id}.wav"
        if not motion_path.is_file() or not audio_path.is_file():
            raise FileNotFoundError(f"Missing FineDance pair for {sequence_id}: {motion_path}, {audio_path}")
        motion, fps = load_motion_energy(motion_path)
        audio, rate = load_wav_mono(audio_path)
        motions[sequence_id] = (motion, fps)
        audios[sequence_id] = (audio, rate)

    # The cyclic shift is deterministic and preserves the test-set marginals.
    wrong_audio_id = {sequence_id: ids[(index + 1) % len(ids)] for index, sequence_id in enumerate(ids)}
    for sequence_id in ids:
        motion, fps = motions[sequence_id]
        audio, rate = audios[sequence_id]
        paired = analyse_pair(motion, audio, fps, rate, args.max_lag_seconds, args.event_tolerance_seconds)
        wrong_audio, wrong_rate = audios[wrong_audio_id[sequence_id]]
        wrong = analyse_pair(motion, wrong_audio, fps, wrong_rate, args.max_lag_seconds, args.event_tolerance_seconds)
        all_wrong = []
        for other_id in ids:
            if other_id == sequence_id:
                continue
            other_audio, other_rate = audios[other_id]
            all_wrong.append(
                analyse_pair(
                    motion,
                    other_audio,
                    fps,
                    other_rate,
                    args.max_lag_seconds,
                    args.event_tolerance_seconds,
                )
            )
        wrong_best_values = np.asarray([item["best_lag_corr"] for item in all_wrong])
        wrong_zero_values = np.asarray([item["zero_lag_corr"] for item in all_wrong])
        paired_rank = float(np.mean(paired["best_lag_corr"] > wrong_best_values))
        pairs.append({
            "sequence_id": sequence_id,
            "wrong_song_id": wrong_audio_id[sequence_id],
            "paired": paired,
            "wrong_song": wrong,
            "wrong_song_all_mean": {
                "zero_lag_corr": float(np.mean(wrong_zero_values)),
                "best_lag_corr": float(np.mean(wrong_best_values)),
                "event_f1": float(np.mean([item["event_f1"] for item in all_wrong])),
            },
            "paired_best_corr_rank": paired_rank,
            "paired_is_top1": bool(paired_rank == 1.0),
            "best_corr_margin": paired["best_lag_corr"] - wrong["best_lag_corr"],
            "zero_lag_corr_margin": paired["zero_lag_corr"] - wrong["zero_lag_corr"],
            "event_f1_margin": paired["event_f1"] - wrong["event_f1"],
        })

    def mean(path: str, rows: list[dict]) -> float:
        values = []
        for row in rows:
            value = row
            for part in path.split("."):
                value = value[part]
            values.append(float(value))
        return float(np.mean(values))

    summary = {
        "test_ids": ids,
        "pair_count": len(pairs),
        "wrong_song_control": "cyclic shift by one test sequence ID",
        "max_lag_seconds": args.max_lag_seconds,
        "event_tolerance_seconds": args.event_tolerance_seconds,
        "paired_mean": {
            "zero_lag_corr": mean("paired.zero_lag_corr", pairs),
            "best_lag_corr": mean("paired.best_lag_corr", pairs),
            "event_f1": mean("paired.event_f1", pairs),
            "dominant_rate_error_hz": mean("paired.dominant_rate_error_hz", pairs),
        },
        "wrong_song_mean": {
            "zero_lag_corr": mean("wrong_song.zero_lag_corr", pairs),
            "best_lag_corr": mean("wrong_song.best_lag_corr", pairs),
            "event_f1": mean("wrong_song.event_f1", pairs),
            "dominant_rate_error_hz": mean("wrong_song.dominant_rate_error_hz", pairs),
        },
        "mean_margin": {
            "zero_lag_corr": mean("zero_lag_corr_margin", pairs),
            "best_lag_corr": mean("best_corr_margin", pairs),
            "event_f1": mean("event_f1_margin", pairs),
        },
        "all_wrong_song_control": {
            "mean_zero_lag_corr": mean("wrong_song_all_mean.zero_lag_corr", pairs),
            "mean_best_lag_corr": mean("wrong_song_all_mean.best_lag_corr", pairs),
            "mean_event_f1": mean("wrong_song_all_mean.event_f1", pairs),
            "mean_best_corr_margin": mean(
                "paired.best_lag_corr", pairs
            ) - mean("wrong_song_all_mean.best_lag_corr", pairs),
            "mean_paired_rank_percentile": mean("paired_best_corr_rank", pairs),
            "top1_rate": float(np.mean([row["paired_is_top1"] for row in pairs])),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pairing_metrics.json").write_text(
        json.dumps({"summary": summary, "pairs": pairs}, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "pairing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sequence_id", "wrong_song_id", "paired_zero_lag_corr", "wrong_zero_lag_corr",
            "paired_best_lag_corr", "wrong_best_lag_corr", "best_corr_margin",
            "paired_event_f1", "wrong_event_f1", "event_f1_margin",
            "paired_best_lag_seconds", "paired_dominant_rate_error_hz",
            "wrong_all_mean_best_lag_corr", "paired_best_corr_rank", "paired_is_top1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pairs:
            writer.writerow({
                "sequence_id": row["sequence_id"],
                "wrong_song_id": row["wrong_song_id"],
                "paired_zero_lag_corr": row["paired"]["zero_lag_corr"],
                "wrong_zero_lag_corr": row["wrong_song"]["zero_lag_corr"],
                "paired_best_lag_corr": row["paired"]["best_lag_corr"],
                "wrong_best_lag_corr": row["wrong_song"]["best_lag_corr"],
                "best_corr_margin": row["best_corr_margin"],
                "paired_event_f1": row["paired"]["event_f1"],
                "wrong_event_f1": row["wrong_song"]["event_f1"],
                "event_f1_margin": row["event_f1_margin"],
                "paired_best_lag_seconds": row["paired"]["best_lag_seconds"],
                "paired_dominant_rate_error_hz": row["paired"]["dominant_rate_error_hz"],
                "wrong_all_mean_best_lag_corr": row["wrong_song_all_mean"]["best_lag_corr"],
                "paired_best_corr_rank": row["paired_best_corr_rank"],
                "paired_is_top1": row["paired_is_top1"],
            })
    report = [
        "# FineDance Music-Motion Pairing Audit",
        "",
        "This is a benchmark integrity check, not a model score. Each official",
        "cross-genre test motion is compared with its same-ID WAV and a deterministic",
        "cyclic wrong-song control. A positive margin means the paired WAV is more",
        "compatible under that diagnostic; it is not proof of perfect beat alignment.",
        "",
        f"- Test pairs: **{summary['pair_count']}**",
        f"- Paired zero-lag correlation: **{summary['paired_mean']['zero_lag_corr']:.4f}**",
        f"- Wrong-song zero-lag correlation: **{summary['wrong_song_mean']['zero_lag_corr']:.4f}**",
        f"- Zero-lag margin: **{summary['mean_margin']['zero_lag_corr']:.4f}**",
        f"- Paired best-lag correlation: **{summary['paired_mean']['best_lag_corr']:.4f}**",
        f"- Wrong-song best-lag correlation: **{summary['wrong_song_mean']['best_lag_corr']:.4f}**",
        f"- Best-lag margin: **{summary['mean_margin']['best_lag_corr']:.4f}**",
        f"- Paired event F1: **{summary['paired_mean']['event_f1']:.4f}**",
        f"- Wrong-song event F1: **{summary['wrong_song_mean']['event_f1']:.4f}**",
        f"- Event-F1 margin: **{summary['mean_margin']['event_f1']:.4f}**",
        f"- All-wrong-song mean best-correlation margin: **{summary['all_wrong_song_control']['mean_best_corr_margin']:.4f}**",
        f"- Paired correlation mean rank percentile among all wrong songs: **{summary['all_wrong_song_control']['mean_paired_rank_percentile']:.4f}**",
        f"- Paired song retrieval top-1 rate: **{summary['all_wrong_song_control']['top1_rate']:.4f}**",
        "",
        "Interpretation: use positive paired-vs-wrong margins as evidence that the",
        "music and motion pairing contains signal. Use per-sequence results and a",
        "stronger permutation test before making a paper-level musicality claim.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
