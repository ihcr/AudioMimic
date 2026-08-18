"""Expand online C4 motion with the measured between-commit inference gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated_motion", required=True)
    parser.add_argument("--runtime_json", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    motion = np.asarray(np.load(args.generated_motion), dtype=np.float32)
    runtime = json.loads(Path(args.runtime_json).read_text(encoding="utf-8"))
    records = runtime["commit_records"]
    commit_frames = int(runtime["commit_frames"])
    if motion.shape != (len(records) * commit_frames, 34):
        raise ValueError("generated motion and commit records disagree")

    timeline: list[np.ndarray] = []
    inserted_frames = 0
    for index, record in enumerate(records):
        commit = motion[index * commit_frames : (index + 1) * commit_frames]
        timeline.append(commit)
        if index + 1 < len(records):
            next_latency_ms = float(records[index + 1]["latency_ms"]["total"])
            hold_frames = max(0, int(round(next_latency_ms * float(args.fps) / 1000.0)))
            if hold_frames:
                timeline.append(np.repeat(commit[-1:], hold_frames, axis=0))
                inserted_frames += hold_frames

    expanded = np.concatenate(timeline, axis=0)
    output = Path(args.output_npy).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, expanded)
    print(
        f"Wrote {expanded.shape[0]} frames ({expanded.shape[0] / args.fps:.3f}s); "
        f"inserted {inserted_frames} inference-gap frames"
    )


if __name__ == "__main__":
    main(parse_args())
