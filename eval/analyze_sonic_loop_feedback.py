"""Diagnose feedback-induced commit seams and pre-fall behavior in SONIC loops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FPS = 30.0
COMMIT_FRAMES = 8
FALL_HEIGHT = 0.45


def _records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", payload) if isinstance(payload, dict) else payload


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def analyze(run_dir: Path):
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    commits = runtime["commit_records"]
    motion = np.asarray(np.load(run_dir / "generated_motion.npy"), dtype=np.float64)
    joint = motion[:, 5:34]

    boundary_frame = np.arange(COMMIT_FRAMES, len(joint), COMMIT_FRAMES)
    boundary_velocity = np.max(
        np.abs(joint[boundary_frame] - joint[boundary_frame - 1]), axis=1
    ) * FPS
    correction = np.asarray(
        [record["measured_synthetic_joint_rmse"] for record in commits[:-1]],
        dtype=np.float64,
    )
    latency = np.asarray(
        [record["latency_ms"]["total"] for record in commits], dtype=np.float64
    )
    deadline_ms = COMMIT_FRAMES / FPS * 1000.0

    sim = [record for record in _records(run_dir / "sim_state.json") if record.get("sim_time") is not None]
    online_start = float(runtime["online_started_monotonic"])
    nearest = min(
        sim,
        key=lambda record: abs(float(record["_received_monotonic_seconds"]) - online_start),
    )
    start_sim_time = float(nearest["sim_time"])
    sim_time = np.asarray([float(record["sim_time"]) - start_sim_time for record in sim])
    height = np.asarray([record["base_position"][2] for record in sim], dtype=np.float64)
    valid = sim_time >= 0.0
    sim_time, height = sim_time[valid], height[valid]
    fallen = height < FALL_HEIGHT
    fall_time = float(sim_time[np.argmax(fallen)]) if fallen.any() else None

    prefall = {}
    if fall_time is not None:
        first_commit = max(0, int(np.floor((fall_time - 2.0) * FPS / COMMIT_FRAMES)))
        last_commit = min(len(commits), int(np.ceil(fall_time * FPS / COMMIT_FRAMES)))
        seam_slice = boundary_velocity[max(0, first_commit - 1) : max(0, last_commit - 1)]
        correction_slice = np.asarray(
            [record["measured_synthetic_joint_rmse"] for record in commits[first_commit:last_commit]]
        )
        latency_slice = latency[first_commit:last_commit]
        prefall = {
            "window_start_seconds": max(0.0, fall_time - 2.0),
            "window_stop_seconds": fall_time,
            "commit_start": first_commit,
            "commit_stop": last_commit,
            "boundary_velocity_max_rad_s": float(np.max(seam_slice)) if len(seam_slice) else None,
            "s66_correction_mean_rad": float(np.mean(correction_slice)),
            "latency_mean_ms": float(np.mean(latency_slice)),
            "deadline_miss_rate": float(np.mean(latency_slice > deadline_ms)),
        }

    return {
        "run_dir": str(run_dir),
        "feedback_source": runtime.get("feedback_source"),
        "commits": len(commits),
        "fall_time_seconds": fall_time,
        "minimum_base_height_m": float(np.min(height)),
        "s66_correction_to_next_boundary_velocity_correlation": _correlation(
            correction, boundary_velocity
        ),
        "s66_correction_mean_rad": float(np.mean(correction)),
        "next_boundary_velocity_p95_rad_s": float(np.percentile(boundary_velocity, 95)),
        "latency_deadline_miss_rate": float(np.mean(latency > deadline_ms)),
        "prefall_2s": prefall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [analyze(Path(run).expanduser().resolve()) for run in args.runs]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(
            result["feedback_source"],
            f"fall={result['fall_time_seconds']}",
            f"corr={result['s66_correction_to_next_boundary_velocity_correlation']:.3f}",
            f"seam_p95={result['next_boundary_velocity_p95_rad_s']:.2f}",
        )


if __name__ == "__main__":
    main()
