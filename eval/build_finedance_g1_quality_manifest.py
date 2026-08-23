"""Build a per-clip FineDance-G1 retargeting quality manifest.

The manifest is deliberately diagnostic.  It records candidate training
filters, but never removes clips from the prepared tree and never filters the
sealed test split.  This keeps data-quality decisions auditable and prevents
the GT benchmark from being made artificially easier.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/finedance_g1_fkbeats"),
        help="Prepared tree containing train/test/motions_sliced/*.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/benchmark_v1/gt/finedance_g1_quality_v1"),
    )
    parser.add_argument("--low-height-threshold", type=float, default=0.2)
    return parser.parse_args()


def _load_root_height(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    root_pos = np.asarray(payload["root_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: unexpected root_pos shape {root_pos.shape}")
    values = root_pos[:, 2]
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite root height")
    return values


def _sequence_id(stem: str) -> str:
    return stem.split("_slice", 1)[0]


def _record(path: Path, split: str, low_height_threshold: float) -> dict[str, Any]:
    height = _load_root_height(path)
    negative_count = int(np.count_nonzero(height < 0.0))
    low_count = int(np.count_nonzero(height < low_height_threshold))
    return {
        "split": split,
        "clip_id": path.stem,
        "sequence_id": _sequence_id(path.stem),
        "motion_path": str(path.resolve()),
        "frames": int(height.size),
        "root_height_min_m": float(height.min()),
        "root_height_max_m": float(height.max()),
        "negative_root_frames": negative_count,
        "negative_root_fraction": float(negative_count / max(height.size, 1)),
        "low_root_frames": low_count,
        "low_root_fraction": float(low_count / max(height.size, 1)),
        "candidate_train_keep_nonnegative": bool(height.min() >= 0.0),
        "candidate_train_keep_root_ge_threshold": bool(
            height.min() >= low_height_threshold
        ),
        "sealed_test_keep": True,
    }


def _summary(records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    rows = [record for record in records if record["split"] == split]
    negative = [row for row in rows if row["negative_root_frames"] > 0]
    low = [row for row in rows if row["low_root_frames"] > 0]
    return {
        "clips": len(rows),
        "clips_with_any_negative_root": len(negative),
        "clips_with_any_root_below_threshold": len(low),
        "negative_root_frame_fraction_mean": float(
            np.mean([row["negative_root_fraction"] for row in rows])
        )
        if rows
        else 0.0,
        "root_height_min_m": min((row["root_height_min_m"] for row in rows), default=None),
        "root_height_max_m": max((row["root_height_max_m"] for row in rows), default=None),
        "candidate_keep_nonnegative": sum(
            row["candidate_train_keep_nonnegative"] for row in rows
        ),
        "candidate_keep_root_ge_threshold": sum(
            row["candidate_train_keep_root_ge_threshold"] for row in rows
        ),
        "test_policy": "keep_all" if split == "test" else "not_applicable",
    }


def _write_report(
    path: Path,
    records: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    threshold: float,
) -> None:
    train = summaries["train"]
    test = summaries["test"]
    path.write_text(
        "\n".join(
            [
                "# FineDance-G1 Quality Manifest",
                "",
                "This is a per-clip diagnostic for the retargeted FineDance-G1 "
                "cache. It does not delete or rewrite motion files.",
                "",
                f"Low-height diagnostic threshold: `{threshold:.3f} m`.",
                "",
                "| split | clips | any negative root | any root below threshold | "
                "candidate keep (z >= 0) | candidate keep (z >= threshold) |",
                "|---|---:|---:|---:|---:|---:|",
                f"| train | {train['clips']} | {train['clips_with_any_negative_root']} | "
                f"{train['clips_with_any_root_below_threshold']} | "
                f"{train['candidate_keep_nonnegative']} | "
                f"{train['candidate_keep_root_ge_threshold']} |",
                f"| test | {test['clips']} | {test['clips_with_any_negative_root']} | "
                f"{test['clips_with_any_root_below_threshold']} | kept | kept |",
                "",
                "## Policy",
                "",
                "- The sealed test split is always retained, including unusual "
                "retargeting clips; report its full distribution for GT calibration.",
                "- The two train columns are candidate policies for an ablation, "
                "not an automatic deletion rule.",
                "- Before training, compare unfiltered training, `z >= 0`, and "
                "retargeting correction if the correction is available.",
                "- Any reported generator score must state which training policy "
                "was used and must use the same test manifest.",
                "",
                "## Files",
                "",
                "- `quality_manifest.json`: complete per-clip records and summaries.",
                "- `quality_manifest.csv`: flat table for analysis and plotting.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for split in ("train", "test"):
        motion_dir = args.prepared_root / split / "motions_sliced"
        paths = sorted(motion_dir.glob("*.pkl"))
        if not paths:
            raise FileNotFoundError(f"No motion clips found in {motion_dir}")
        for path in paths:
            records.append(_record(path, split, args.low_height_threshold))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {split: _summary(records, split) for split in ("train", "test")}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_root": str(args.prepared_root.resolve()),
        "low_height_threshold_m": args.low_height_threshold,
        "policy": {
            "test": "keep_all",
            "train": "candidate_filters_only; no files removed",
        },
        "summaries": summaries,
        "sequence_counts": {
            split: dict(
                sorted(
                    Counter(
                        record["sequence_id"]
                        for record in records
                        if record["split"] == split
                    ).items()
                )
            )
            for split in ("train", "test")
        },
        "records": records,
    }
    (args.output_dir / "quality_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    fields = list(records[0])
    with (args.output_dir / "quality_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    _write_report(args.output_dir / "REPORT.md", records, summaries, args.low_height_threshold)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
