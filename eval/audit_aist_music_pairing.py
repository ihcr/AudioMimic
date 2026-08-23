"""Audit AIST++ paired music-motion integrity with wrong-song controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_finedance_music_pairing import (
    analyse_pair,
    load_motion_energy,
    load_wav_mono,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "eval/benchmark_v1/gt/manifest_v1/gt_benchmark_manifest.json"
DEFAULT_OUTPUT = ROOT / "eval/benchmark_v1/gt/aist_music_pairing_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-lag-seconds", type=float, default=2.0)
    parser.add_argument("--event-tolerance-seconds", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
    dataset = payload["datasets"]["aistpp"]
    test_ids = set(dataset["split"]["test_assets_available"])
    records = [r for r in dataset["records"] if r.get("paired_valid") and r["sequence_id"] in test_ids]
    records.sort(key=lambda r: r["sequence_id"])
    if not records:
        raise RuntimeError("No paired AIST++ test records found")

    motions = {}
    audios = {}
    for record in records:
        motion, fps = load_motion_energy(Path(record["motion"]["path"]))
        audio, rate = load_wav_mono(Path(record["audio"]["path"]))
        motions[record["sequence_id"]] = (motion, fps)
        audios[record["sequence_id"]] = (audio, rate)

    ids = [r["sequence_id"] for r in records]
    pairs = []
    for index, sequence_id in enumerate(ids):
        wrong_id = ids[(index + 1) % len(ids)]
        motion, fps = motions[sequence_id]
        audio, rate = audios[sequence_id]
        wrong_audio, wrong_rate = audios[wrong_id]
        paired = analyse_pair(
            motion, audio, fps, rate, args.max_lag_seconds, args.event_tolerance_seconds
        )
        wrong = analyse_pair(
            motion, wrong_audio, fps, wrong_rate, args.max_lag_seconds, args.event_tolerance_seconds
        )
        all_wrong = []
        for other_id in ids:
            if other_id == sequence_id:
                continue
            other_audio, other_rate = audios[other_id]
            all_wrong.append(
                analyse_pair(
                    motion, other_audio, fps, other_rate,
                    args.max_lag_seconds, args.event_tolerance_seconds,
                )
            )
        wrong_best = np.asarray([item["best_lag_corr"] for item in all_wrong])
        pairs.append({
            "sequence_id": sequence_id,
            "wrong_song_id": wrong_id,
            "paired": paired,
            "wrong_song": wrong,
            "wrong_song_all_mean": {
                "zero_lag_corr": float(np.mean([x["zero_lag_corr"] for x in all_wrong])),
                "best_lag_corr": float(np.mean(wrong_best)),
                "event_f1": float(np.mean([x["event_f1"] for x in all_wrong])),
            },
            "best_corr_margin": paired["best_lag_corr"] - wrong["best_lag_corr"],
            "event_f1_margin": paired["event_f1"] - wrong["event_f1"],
            "paired_best_corr_rank": float(np.mean(paired["best_lag_corr"] > wrong_best)),
            "paired_is_top1": bool(np.all(paired["best_lag_corr"] >= wrong_best)),
        })

    def mean(path: str) -> float:
        values = []
        for row in pairs:
            value = row
            for part in path.split("."):
                value = value[part]
            values.append(float(value))
        return float(np.mean(values))

    summary = {
        "pair_count": len(pairs),
        "wrong_song_control": "cyclic shift by one AIST++ test sequence ID",
        "paired_mean": {
            "zero_lag_corr": mean("paired.zero_lag_corr"),
            "best_lag_corr": mean("paired.best_lag_corr"),
            "event_f1": mean("paired.event_f1"),
        },
        "wrong_song_mean": {
            "zero_lag_corr": mean("wrong_song.zero_lag_corr"),
            "best_lag_corr": mean("wrong_song.best_lag_corr"),
            "event_f1": mean("wrong_song.event_f1"),
        },
        "mean_margin": {
            "best_lag_corr": mean("best_corr_margin"),
            "event_f1": mean("event_f1_margin"),
        },
        "all_wrong_song_control": {
            "mean_best_corr_margin": mean("paired.best_lag_corr") - mean("wrong_song_all_mean.best_lag_corr"),
            "mean_paired_rank_percentile": mean("paired_best_corr_rank"),
            "top1_rate": float(np.mean([row["paired_is_top1"] for row in pairs])),
        },
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairing_metrics.json").write_text(
        json.dumps({"summary": summary, "pairs": pairs}, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# AIST++ Music-Motion Pairing Audit",
        "",
        "This is a benchmark integrity check, not a model score. Same-ID audio is",
        "compared with a cyclic wrong-song control and all other AIST++ test songs.",
        "",
        f"- Test pairs: **{summary['pair_count']}**",
        f"- Paired-vs-wrong best-lag correlation margin: **{summary['mean_margin']['best_lag_corr']:.4f}**",
        f"- Paired-vs-wrong event-F1 margin: **{summary['mean_margin']['event_f1']:.4f}**",
        f"- Paired correlation top-1 rate among wrong songs: **{summary['all_wrong_song_control']['top1_rate']:.4f}**",
        "",
        "Use the paired distribution as the positive reference and the wrong-song",
        "distribution as a negative control. This calibration is separate from GMR",
        "and generator evaluation.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
