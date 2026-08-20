"""Select reproducible low/medium/high-dynamics G1 GT references for SONIC calibration."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--minimum_seconds", type=float, default=10.0)
    parser.add_argument("--candidates_per_level", type=int, default=8)
    return parser.parse_args()


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def _load_metrics(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    fps = float(payload.get("fps", 30.0))
    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float64)
    dof = np.asarray(payload["dof_pos"], dtype=np.float64)
    if root_pos.shape != (len(dof), 3):
        raise ValueError(f"{path}: invalid root_pos shape {root_pos.shape}")
    if root_rot.shape != (len(dof), 4):
        raise ValueError(f"{path}: invalid root_rot shape {root_rot.shape}")
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError(f"{path}: invalid dof_pos shape {dof.shape}")
    if not all(np.isfinite(value).all() for value in (root_pos, root_rot, dof)):
        raise ValueError(f"{path}: non-finite values")

    velocity = np.gradient(dof, 1.0 / fps, axis=0)
    acceleration = np.gradient(velocity, 1.0 / fps, axis=0)
    jerk = np.gradient(acceleration, 1.0 / fps, axis=0)
    absolute_velocity = np.abs(velocity).reshape(-1)
    absolute_acceleration = np.abs(acceleration).reshape(-1)
    absolute_jerk = np.abs(jerk).reshape(-1)
    return {
        "motion_id": path.stem,
        "path": str(path.resolve()),
        "genre": path.stem.split("_")[0],
        "frames": int(len(dof)),
        "fps": fps,
        "duration_seconds": float(len(dof) / fps),
        "motion_energy_rad2_s2": float(np.mean(velocity**2)),
        "velocity_p95_rad_s": float(np.percentile(absolute_velocity, 95)),
        "velocity_max_rad_s": float(np.max(absolute_velocity)),
        "acceleration_p95_rad_s2": float(np.percentile(absolute_acceleration, 95)),
        "jerk_p95_rad_s3": float(np.percentile(absolute_jerk, 95)),
        "root_height_min_m": float(np.min(root_pos[:, 2])),
        "root_height_median_m": float(np.median(root_pos[:, 2])),
        "root_path_length_m": float(
            np.sum(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1))
        ),
        "root_quat_order": "xyzw",
    }


def _select_candidates(records: list[dict], count: int) -> dict[str, list[dict]]:
    metrics = (
        "motion_energy_rad2_s2",
        "velocity_p95_rad_s",
        "acceleration_p95_rad_s2",
        "jerk_p95_rad_s3",
    )
    component_ranks = np.stack(
        [_percentile_rank(np.asarray([record[key] for record in records])) for key in metrics],
        axis=1,
    )
    composite = np.mean(component_ranks, axis=1)
    for record, score in zip(records, composite):
        record["dynamic_percentile"] = float(score)

    targets = {"low": 0.20, "medium": 0.50, "high": 0.80}
    selected = {}
    used_genres = set()
    for level, target in targets.items():
        ranked = sorted(records, key=lambda record: abs(record["dynamic_percentile"] - target))
        diverse = [record for record in ranked if record["genre"] not in used_genres]
        candidates = (diverse + [record for record in ranked if record not in diverse])[:count]
        selected[level] = candidates
        used_genres.add(candidates[0]["genre"])
    return selected


def main(args: argparse.Namespace) -> None:
    gt_root = Path(args.gt_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    rejected = []
    for path in sorted(gt_root.glob("*.pkl")):
        try:
            record = _load_metrics(path)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
            continue
        if record["duration_seconds"] >= args.minimum_seconds:
            records.append(record)

    selected = _select_candidates(records, args.candidates_per_level)
    primary = {level: candidates[0] for level, candidates in selected.items()}
    planned_runs = [
        {
            "run_id": f"gt_cap_{level}_{record['motion_id']}_r{repeat:02d}",
            "level": level,
            "repeat": repeat,
            "motion_id": record["motion_id"],
            "motion_path": record["path"],
            "output_dir": f"eval/gt_sonic_capability/runs/gt_cap_{level}_{record['motion_id']}_r{repeat:02d}",
        }
        for level, record in primary.items()
        for repeat in range(1, 4)
    ]
    manifest = {
        "schema_version": "gt_sonic_capability_selection_v1",
        "gt_root": str(gt_root),
        "selection_policy": {
            "minimum_seconds": args.minimum_seconds,
            "dynamic_score": "mean percentile rank of energy, velocity P95, acceleration P95 and jerk P95",
            "target_percentiles": {"low": 0.20, "medium": 0.50, "high": 0.80},
            "genre_rule": "Prefer a different genre for each primary reference.",
            "root_quat_order": "xyzw",
        },
        "eligible_files": len(records),
        "rejected_files": rejected,
        "primary": primary,
        "candidates": selected,
        "planned_runs": planned_runs,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    fields = list(records[0])
    with (output_dir / "all_gt_dynamics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda record: record["dynamic_percentile"]))

    print(f"eligible={len(records)} rejected={len(rejected)}")
    for level, record in primary.items():
        print(
            level,
            record["motion_id"],
            f"duration={record['duration_seconds']:.2f}s",
            f"dynamic_percentile={record['dynamic_percentile']:.3f}",
            f"energy={record['motion_energy_rad2_s2']:.3f}",
            f"velocity_p95={record['velocity_p95_rad_s']:.3f}",
            f"jerk_p95={record['jerk_p95_rad_s3']:.1f}",
        )


if __name__ == "__main__":
    main(parse_args())
