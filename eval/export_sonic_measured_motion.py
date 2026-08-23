"""Export recorded SONIC/MuJoCo state as a portable motion pickle."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_pkl", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def _load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)["records"]


def _unique_time(time: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(time, kind="stable")
    time = time[order]
    values = tuple(value[order] for value in values)
    keep = np.r_[np.diff(time) > 0.0, True]
    return (time[keep], *(value[keep] for value in values))


def _interp(time: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(grid, time, values[:, column]) for column in range(values.shape[1])]
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    with (run_dir / "offline_sonic_playback.json").open() as handle:
        metadata = json.load(handle)
    started = float(metadata["playback_started_monotonic_seconds"])
    alignment = float(metadata.get("alignment_seconds", 0.0))
    duration = float(metadata["selected_source_duration_seconds"])
    grid = alignment + np.arange(round(duration * args.fps)) / args.fps

    feedback = _load_records(run_dir / "sonic_feedback.json")
    feedback = [
        record
        for record in feedback
        if "body_q_measured" in record
        and started <= float(record.get("_received_monotonic_seconds", -np.inf))
    ]
    feedback_time = np.asarray(
        [record["_received_monotonic_seconds"] for record in feedback], dtype=np.float64
    ) - started
    body_q = np.asarray([record["body_q_measured"] for record in feedback], dtype=np.float64)
    feedback_time, body_q = _unique_time(feedback_time, body_q)

    sim = _load_records(run_dir / "sim_state.json")
    sim = [
        record
        for record in sim
        if "base_position" in record
        and started <= float(record.get("_received_monotonic_seconds", -np.inf))
    ]
    sim_time = np.asarray(
        [record["_received_monotonic_seconds"] for record in sim], dtype=np.float64
    ) - started
    root_pos = np.asarray([record["base_position"] for record in sim], dtype=np.float64)
    root_quat_wxyz = np.asarray([record["base_quat"] for record in sim], dtype=np.float64)
    sim_time, root_pos, root_quat_wxyz = _unique_time(sim_time, root_pos, root_quat_wxyz)

    coverage_end = min(feedback_time[-1], sim_time[-1])
    if grid[-1] - coverage_end > 1.0 / args.fps:
        raise RuntimeError("recorded feedback does not cover the requested motion duration")

    root_quat_wxyz = _interp(sim_time, root_quat_wxyz, grid)
    root_quat_wxyz /= np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True).clip(min=1e-8)
    payload = {
        "root_pos": _interp(sim_time, root_pos, grid).astype(np.float32).tolist(),
        "root_rot": root_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32).tolist(),
        # SONIC feedback already uses the MuJoCo model's 29-DoF joint order.
        "dof_pos": _interp(feedback_time, body_q, grid).astype(np.float32).tolist(),
        "fps": float(args.fps),
    }
    output = Path(args.output_pkl).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    print(f"Exported {len(grid)} frames at {args.fps:g} FPS to {output}")


if __name__ == "__main__":
    main()
