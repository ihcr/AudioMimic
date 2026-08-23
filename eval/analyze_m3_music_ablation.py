"""Analyze a controlled M3 paired/wrong/shifted/null rollout set."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_motion_music_execution import (
    DEFAULT_MODEL_PATH,
    compute_motion_quality,
    compute_music_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target_audio", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def load_motion(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "fps": float(payload.get("fps", 30.0)),
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
    }


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = [
        (item, load_motion(Path(item["motion_path"])))
        for item in manifest["variants"]
    ]
    paired_by_seed = {
        int(item["seed"]): motion["dof_pos"]
        for item, motion in entries
        if item["label"].startswith("paired")
    }
    audio_cache = {}
    records = []
    for item, motion in entries:
        label = item["label"]
        paired = paired_by_seed[int(item["seed"])]
        paired_scale = float(np.sqrt(np.mean((paired - paired.mean(axis=0)) ** 2)))
        quality, speed = compute_motion_quality(
            motion,
            model_path=Path(args.model_path).expanduser().resolve(),
            quat_order="xyzw",
        )
        music = compute_music_metrics(
            speed,
            fps=motion["fps"],
            audio_path=Path(args.target_audio).expanduser().resolve(),
            audio_cache=audio_cache,
        )
        delta = motion["dof_pos"] - paired
        records.append(
            {
                **item,
                "paired_joint_rmse_rad": float(np.sqrt(np.mean(delta**2))),
                "paired_joint_mae_rad": float(np.mean(np.abs(delta))),
                "paired_normalized_effect": (
                    float(np.sqrt(np.mean(delta**2)) / paired_scale)
                    if paired_scale > 1e-12
                    else None
                ),
                "quality": quality,
                "music": music,
            }
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for record in records:
        rows.append(
            {
                "label": record["label"],
                "seed": record["seed"],
                "paired_rmse_rad": record["paired_joint_rmse_rad"],
                "normalized_effect": record["paired_normalized_effect"],
                "energy": record["quality"]["motion_energy_rad2_s2"],
                "amplitude": record["quality"]["joint_amplitude_median_rad"],
                "jerk_p95": record["quality"]["joint_jerk_abs_rad_s3"]["p95"],
                "bas_music_to_motion": record["music"]["bas_music_to_motion"],
                "impact_best_correlation": record["music"]["impact_best_correlation"],
                "impact_best_lag_seconds": record["music"]["impact_best_lag_seconds"],
            }
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['label']}: delta={row['paired_rmse_rad']:.3f}rad "
            f"BAS={row['bas_music_to_motion']:.3f} "
            f"impact={row['impact_best_correlation']:.3f} "
            f"energy={row['energy']:.3f}"
        )


if __name__ == "__main__":
    main()
