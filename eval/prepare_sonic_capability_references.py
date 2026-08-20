"""Convert known-trackable SONIC references into the AudioMimic G1 contract.

SONIC's bundled CSV references store joints in SONIC/IsaacLab order at 50 Hz.
The AudioMimic bridge accepts MuJoCo-order pickles at 30 Hz and applies the
SONIC permutation at the wire boundary.  This utility performs the inverse
permutation once, retimes the reference, and freezes low/medium/high examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic_bridge import SONIC_REFERENCE_FROM_MUJOCO


SOURCE_FPS = 50.0
TARGET_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"empty SONIC reference CSV: {path}")
    return np.asarray(rows[1:], dtype=np.float32)


def _sample_positions(values: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, values.shape[0] - 1)
    fraction = (source_positions - left).astype(np.float32)
    return (
        (1.0 - fraction[:, None]) * values[left]
        + fraction[:, None] * values[right]
    ).astype(np.float32)


def _sample_quaternions_wxyz(
    values: np.ndarray, source_positions: np.ndarray
) -> np.ndarray:
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, values.shape[0] - 1)
    fraction = (source_positions - left).astype(np.float32)
    output = np.empty((source_positions.size, 4), dtype=np.float32)
    for index, (left_index, right_index, amount) in enumerate(
        zip(left, right, fraction)
    ):
        first = values[left_index]
        second = values[right_index]
        if float(np.dot(first, second)) < 0.0:
            second = -second
        quaternion = (1.0 - amount) * first + amount * second
        output[index] = quaternion / np.linalg.norm(quaternion)
    return output


def convert_reference(reference_dir: Path) -> tuple[dict, dict]:
    joint_sonic = _read_csv(reference_dir / "joint_pos.csv")
    body_pos = _read_csv(reference_dir / "body_pos.csv").reshape(-1, 14, 3)
    body_quat_wxyz = _read_csv(reference_dir / "body_quat.csv").reshape(-1, 14, 4)
    if joint_sonic.shape[1] != 29:
        raise ValueError(f"{reference_dir}: expected 29 joints, got {joint_sonic.shape}")
    frames = min(joint_sonic.shape[0], body_pos.shape[0], body_quat_wxyz.shape[0])
    source_positions = np.arange(
        max(1, int(np.floor(frames * TARGET_FPS / SOURCE_FPS))), dtype=np.float64
    ) * (SOURCE_FPS / TARGET_FPS)
    source_positions = np.minimum(source_positions, frames - 1)

    joint_sonic_30 = _sample_positions(joint_sonic[:frames], source_positions)
    joint_mujoco_30 = np.empty_like(joint_sonic_30)
    joint_mujoco_30[:, SONIC_REFERENCE_FROM_MUJOCO] = joint_sonic_30
    root_pos_30 = _sample_positions(body_pos[:frames, 0], source_positions)
    root_wxyz_30 = _sample_quaternions_wxyz(
        body_quat_wxyz[:frames, 0], source_positions
    )
    root_xyzw_30 = root_wxyz_30[:, [1, 2, 3, 0]]

    velocity = np.diff(joint_mujoco_30, axis=0) * TARGET_FPS
    acceleration = np.diff(velocity, axis=0) * TARGET_FPS
    metrics = {
        "name": reference_dir.name,
        "source_dir": str(reference_dir.resolve()),
        "source_fps": SOURCE_FPS,
        "output_fps": TARGET_FPS,
        "source_frames": int(frames),
        "output_frames": int(joint_mujoco_30.shape[0]),
        "duration_seconds": float(joint_mujoco_30.shape[0] / TARGET_FPS),
        "root_height_min_m": float(np.min(root_pos_30[:, 2])),
        "root_height_median_m": float(np.median(root_pos_30[:, 2])),
        "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(velocity**2))),
        "joint_velocity_p95_rad_s": float(np.quantile(np.abs(velocity), 0.95)),
        "joint_acceleration_p95_rad_s2": float(
            np.quantile(np.abs(acceleration), 0.95)
        ),
    }
    payload = {
        "fps": TARGET_FPS,
        "root_pos": root_pos_30,
        "root_rot": root_xyzw_30,
        "dof_pos": joint_mujoco_30,
        "source": "SONIC bundled known-trackable reference",
        "sequence_id": reference_dir.name,
    }
    return payload, metrics


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for reference_dir in sorted(path for path in args.reference_root.iterdir() if path.is_dir()):
        required = ("joint_pos.csv", "body_pos.csv", "body_quat.csv")
        if not all((reference_dir / name).is_file() for name in required):
            continue
        payload, metrics = convert_reference(reference_dir)
        output_path = args.output_dir / f"{reference_dir.name}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metrics["output_path"] = str(output_path.resolve())
        records.append(metrics)

    if len(records) < 3:
        raise RuntimeError("at least three SONIC references are required")
    eligible = [
        record
        for record in records
        if record["duration_seconds"] >= 8.0 and record["root_height_min_m"] >= 0.6
    ]
    ranked = sorted(eligible, key=lambda item: item["joint_velocity_rms_rad_s"])
    if len(ranked) < 3:
        raise RuntimeError("fewer than three upright, eight-second SONIC references")
    selected = {
        "low": ranked[0],
        "medium": ranked[(len(ranked) - 1) // 2],
        "high": ranked[-1],
    }
    manifest = {
        "schema_version": "sonic_known_trackable_capability_v1",
        "joint_contract": (
            "CSV SONIC order -> inverse SONIC_REFERENCE_FROM_MUJOCO -> "
            "30 Hz MuJoCo-order pickle -> standard wire adapter"
        ),
        "records": records,
        "selection_constraints": {
            "minimum_duration_seconds": 8.0,
            "minimum_reference_root_height_m": 0.6,
            "duplicate_policy": (
                "Duplicates are retained as provenance; selected levels may share dynamics."
            ),
        },
        "selected": selected,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value["name"] for key, value in selected.items()}, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main(parse_args())
