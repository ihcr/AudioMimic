"""Audit information loss from source human motion to retargeted G1 motion.

This is intentionally a cross-skeleton audit, not a fabricated joint-to-joint
error.  AIST++ source motion is SMPL and FineDance source motion is SMPLH,
while the target is G1.  The audit therefore compares representation-agnostic
signals: root dynamics, whole-body activity, event timing, and music alignment.

The same paired audio is evaluated against source and retargeted motion.  A
change in event F1/BAS/tempo/phase is a direct measure of music-response loss
introduced by retargeting.  Pose-space and root-space values are reported as
proxies because the source and target skeletons do not share joint semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
from scipy.signal import correlate, find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.evaluate_finedance_gt_oracle import _music_extension  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aist-manifest",
        type=Path,
        default=Path("eval/benchmark_v1/gt/manifest_v1/gt_benchmark_manifest.json"),
    )
    parser.add_argument(
        "--finedance-manifest",
        type=Path,
        default=Path("eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"),
    )
    parser.add_argument(
        "--aist-source-motion-root",
        type=Path,
        default=Path("data/edge_aistpp/motions"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/benchmark_v1/gt/retargeting_loss_v1"),
    )
    parser.add_argument("--scope", choices=("test", "all"), default="test")
    parser.add_argument("--event-tolerance-seconds", type=float, default=0.20)
    parser.add_argument("--max-lag-seconds", type=float, default=1.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    return parser.parse_args()


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_manifest_records(path: Path, dataset_name: str, scope: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload["datasets"][dataset_name]
    if scope == "all":
        records = dataset["records"]
    else:
        test_ids = set(dataset["split"]["test_assets_available"])
        records = [record for record in dataset["records"] if record["sequence_id"] in test_ids]
    return [record for record in records if record.get("paired_valid")]


def _axis_angle_speed(poses: np.ndarray, fps: float) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses.reshape(len(poses), -1, 3)
    delta = np.linalg.norm(poses[1:] - poses[:-1], axis=-1)
    speed = np.sqrt(np.mean((delta * fps) ** 2, axis=1))
    return np.concatenate(([speed[0] if len(speed) else 0.0], speed))


def _joint_difference_speed(values: np.ndarray, fps: float) -> np.ndarray:
    """Return a skeleton-agnostic activity curve for arbitrary joint width."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [T, D] joint values, got {values.shape}")
    delta = np.diff(values, axis=0)
    speed = np.sqrt(np.mean((delta * fps) ** 2, axis=1))
    return np.concatenate(([speed[0] if len(speed) else 0.0], speed))


def _rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1, 6)
    a1 = values[..., :3]
    a2 = values[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def _rot6d_speed(values: np.ndarray, fps: float) -> np.ndarray:
    matrices = _rot6d_to_matrix(values)
    relative = np.einsum("...ji,...jk->...ik", matrices[:-1], matrices[1:])
    trace = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    angles = np.arccos(trace)
    speed = np.sqrt(np.mean((angles * fps) ** 2, axis=1))
    return np.concatenate(([speed[0] if len(speed) else 0.0], speed))


def _resample(values: np.ndarray, source_fps: float, target_frames: int, target_fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == target_frames and abs(source_fps - target_fps) < 1e-9:
        return values.copy()
    source_t = np.arange(len(values), dtype=np.float64) / source_fps
    target_t = np.arange(target_frames, dtype=np.float64) / target_fps
    target_t = np.minimum(target_t, source_t[-1] if len(source_t) else 0.0)
    return np.interp(target_t, source_t, values)


def _root_speed(root_pos: np.ndarray, fps: float, horizontal_axes: tuple[int, int]) -> np.ndarray:
    root_pos = np.asarray(root_pos, dtype=np.float64)
    planar = root_pos[:, list(horizontal_axes)]
    velocity = np.gradient(planar, 1.0 / fps, axis=0)
    return np.linalg.norm(velocity, axis=1)


def _normalized_correlation(first: np.ndarray, second: np.ndarray, fps: float, max_lag: float) -> dict[str, float | None]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    count = min(len(first), len(second))
    if count < 4:
        return {"zero_lag": None, "best": None, "best_lag_seconds": None}
    first = first[:count]
    second = second[:count]
    first = (first - np.mean(first)) / max(np.std(first), 1e-12)
    second = (second - np.mean(second)) / max(np.std(second), 1e-12)
    zero = float(np.mean(first * second))
    corr = correlate(first, second, mode="full") / count
    lags = np.arange(-count + 1, count) / fps
    valid = np.abs(lags) <= max_lag
    index = int(np.argmax(corr[valid]))
    valid_lags = lags[valid]
    valid_corr = corr[valid]
    return {
        "zero_lag": zero,
        "best": float(valid_corr[index]),
        "best_lag_seconds": float(valid_lags[index]),
    }


def _event_retention(source: np.ndarray, target: np.ndarray, fps: float) -> dict[str, float | None]:
    def peaks(values: np.ndarray) -> np.ndarray:
        prominence = max(float(np.std(values) * 0.25), 1e-8)
        indices, _ = find_peaks(values, distance=max(1, int(round(0.25 * fps))), prominence=prominence)
        return indices / fps

    source_events = peaks(source)
    target_events = peaks(target)
    if not len(source_events) or not len(target_events):
        return {
            "source_events": int(len(source_events)),
            "target_events": int(len(target_events)),
            "event_count_ratio": None,
            "nearest_event_error_seconds": None,
        }
    errors = np.min(np.abs(target_events[:, None] - source_events[None]), axis=1)
    return {
        "source_events": int(len(source_events)),
        "target_events": int(len(target_events)),
        "event_count_ratio": float(len(target_events) / max(len(source_events), 1)),
        "nearest_event_error_seconds": float(np.median(errors)),
    }


def _load_pair(dataset_name: str, record: dict[str, Any], args: argparse.Namespace) -> tuple[dict, dict, Path, str, float]:
    if dataset_name == "aistpp":
        target = record["motion"]
        source_path = args.aist_source_motion_root / f"{record['sequence_id']}.pkl"
        source = _load_pickle(source_path)
        audio_path = Path(record["audio"]["path"])
        source_scale = float(np.asarray(source.get("smpl_scaling", [1.0])).reshape(-1)[0])
        if abs(source_scale) < 1e-12:
            raise ValueError(f"{source_path}: smpl_scaling must be non-zero")
        # Match the repository's SMPL preprocessing: normalize translation by
        # the fitted scale before comparing it with the metric G1 root in m.
        source_root_pos = np.asarray(source["smpl_trans"], dtype=np.float64) / source_scale
        source_motion = {
            "fps": 60.0,
            "root_pos": source_root_pos,
            "activity": _axis_angle_speed(np.asarray(source["smpl_poses"]), 60.0),
            # The subsequent [x, -z, y] convention preserves planar speed,
            # so the raw horizontal axes are equivalent here.
            "root_speed": _root_speed(source_root_pos, 60.0, (0, 2)),
        }
        quat_order = target["root_quat_order"]
        common_duration = min(float(record["motion"]["duration_seconds"]), float(record["audio"]["duration_seconds"]))
    else:
        target = record["g1_motion"]
        source_path = Path(record["source_motion"]["path"])
        source = np.load(source_path, mmap_mode="r")
        audio_path = Path(record["audio"]["path"])
        source = np.asarray(source, dtype=np.float64)
        source_motion = {
            "fps": 60.0,
            "root_pos": source[:, :3],
            "activity": _rot6d_speed(source[:, 3:], 60.0),
            "root_speed": _root_speed(source[:, :3], 60.0, (0, 2)),
        }
        quat_order = target["root_quat_order"]
        common_duration = float(record["common_duration_seconds"])
    target_path = Path(target["path"])
    target_payload = _load_pickle(target_path)
    target_motion = {
        "fps": float(target_payload.get("fps", 30.0)),
        "root_pos": np.asarray(target_payload["root_pos"], dtype=np.float64),
        "activity": _joint_difference_speed(np.asarray(target_payload["dof_pos"], dtype=np.float64), float(target_payload.get("fps", 30.0))),
        "root_speed": _root_speed(np.asarray(target_payload["root_pos"], dtype=np.float64), float(target_payload.get("fps", 30.0)), (0, 1)),
    }
    return source_motion, target_motion, audio_path, quat_order, common_duration


def _evaluate_pair(dataset_name: str, record: dict[str, Any], args: argparse.Namespace, audio_cache: dict) -> dict[str, Any]:
    source, target, audio_path, quat_order, duration = _load_pair(dataset_name, record, args)
    target_frames = min(len(target["activity"]), int(round(duration * args.target_fps)))
    target_activity = target["activity"][:target_frames]
    target_root_speed = target["root_speed"][:target_frames]
    source_activity = _resample(source["activity"], source["fps"], target_frames, args.target_fps)
    source_root_speed = _resample(source["root_speed"], source["fps"], target_frames, args.target_fps)
    source_music = _music_extension(source_activity, args.target_fps, audio_path, audio_cache, args.event_tolerance_seconds, args.max_lag_seconds)
    target_music = _music_extension(target_activity, args.target_fps, audio_path, audio_cache, args.event_tolerance_seconds, args.max_lag_seconds)
    source_root = _normalized_correlation(source_root_speed, target_root_speed, args.target_fps, args.max_lag_seconds)
    source_activity_corr = _normalized_correlation(source_activity, target_activity, args.target_fps, args.max_lag_seconds)
    source_path = (
        args.aist_source_motion_root / f"{record['sequence_id']}.pkl"
        if dataset_name == "aistpp"
        else Path(record["source_motion"]["path"])
    )
    target_path = Path(record["motion"]["path"] if dataset_name == "aistpp" else record["g1_motion"]["path"])
    return {
        "dataset": dataset_name,
        "sequence_id": record["sequence_id"],
        "source_motion_path": str(source_path),
        "target_motion_path": str(target_path),
        "audio_path": str(audio_path),
        "duration_seconds": float(target_frames / args.target_fps),
        "source_fps": float(source["fps"]),
        "target_fps": float(target["fps"]),
        "source_music": source_music,
        "target_music": target_music,
        "retarget_loss": {
            "activity_curve": source_activity_corr,
            "root_speed_curve": source_root,
            "activity_rms_ratio_target_over_source": float(np.sqrt(np.mean(target_activity**2)) / max(np.sqrt(np.mean(source_activity**2)), 1e-12)),
            "root_speed_rms_ratio_target_over_source": float(np.sqrt(np.mean(target_root_speed**2)) / max(np.sqrt(np.mean(source_root_speed**2)), 1e-12)),
            "activity_events": _event_retention(source_activity, target_activity, args.target_fps),
            "music_delta": {
                "bas": float(target_music["bas_music_to_motion"] - source_music["bas_music_to_motion"]),
                "event_f1": float(target_music["event_f1"]["f1"] - source_music["event_f1"]["f1"]),
                "impact_correlation": float(target_music["impact_best_correlation"] - source_music["impact_best_correlation"]),
                "absolute_impact_lag": float(abs(target_music["impact_best_lag_seconds"]) - abs(source_music["impact_best_lag_seconds"])),
                "tempo_error_bpm": float(target_music["tempo_abs_error_bpm"] - source_music["tempo_abs_error_bpm"]) if target_music["tempo_abs_error_bpm"] is not None and source_music["tempo_abs_error_bpm"] is not None else None,
                "phase_error_cycles": float(target_music["phase"]["mean_phase_error_cycles"] - source_music["phase"]["mean_phase_error_cycles"]),
            },
        },
        "quat_order": quat_order,
    }


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    loss = row["retarget_loss"]
    music_delta = loss["music_delta"]
    return {
        "dataset": row["dataset"],
        "sequence_id": row["sequence_id"],
        "duration_seconds": row["duration_seconds"],
        "source_fps": row["source_fps"],
        "target_fps": row["target_fps"],
        "activity_curve_best_corr": loss["activity_curve"]["best"],
        "activity_curve_best_lag_seconds": loss["activity_curve"]["best_lag_seconds"],
        "root_speed_curve_best_corr": loss["root_speed_curve"]["best"],
        "activity_rms_ratio_target_over_source": loss["activity_rms_ratio_target_over_source"],
        "root_speed_rms_ratio_target_over_source": loss["root_speed_rms_ratio_target_over_source"],
        "activity_event_count_ratio": loss["activity_events"]["event_count_ratio"],
        "activity_event_median_error_seconds": loss["activity_events"]["nearest_event_error_seconds"],
        "delta_bas": music_delta["bas"],
        "delta_event_f1": music_delta["event_f1"],
        "delta_impact_correlation": music_delta["impact_correlation"],
        "delta_absolute_impact_lag": music_delta["absolute_impact_lag"],
        "delta_tempo_error_bpm": music_delta["tempo_error_bpm"],
        "delta_phase_error_cycles": music_delta["phase_error_cycles"],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [key for key in _flatten(rows[0]) if key not in ("dataset", "sequence_id")] if rows else []
    result = {"count": len(rows), "datasets": {name: sum(row["dataset"] == name for row in rows) for name in ("aistpp", "finedance")}, "metrics": {}}
    for field in fields:
        values = [float(_flatten(row)[field]) for row in rows if _flatten(row)[field] is not None and np.isfinite(float(_flatten(row)[field]))]
        if values:
            result["metrics"][field] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "q10": float(np.percentile(values, 10)),
                "q90": float(np.percentile(values, 90)),
            }
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for name, manifest in (("aistpp", args.aist_manifest), ("finedance", args.finedance_manifest)):
        records = _load_manifest_records(manifest, name, args.scope)
        print(f"{name}: {len(records)} paired {args.scope} records", flush=True)
        pairs.extend((name, record) for record in records)
    audio_cache: dict = {}
    rows = []
    for index, (name, record) in enumerate(sorted(pairs, key=lambda value: (value[0], value[1]["sequence_id"])), start=1):
        row = _evaluate_pair(name, record, args, audio_cache)
        rows.append(row)
        print(f"[{index}/{len(pairs)}] {name}:{record['sequence_id']} delta_f1={row['retarget_loss']['music_delta']['event_f1']:.4f} activity_corr={row['retarget_loss']['activity_curve']['best']:.4f}", flush=True)
    payload = {
        "schema_version": "retargeting_loss_v1",
        "scope": args.scope,
        "source_motion_limitations": "Cross-skeleton proxy audit; source and G1 joint positions are not directly comparable.",
        "summary": _summary(rows),
        "per_dataset": {name: _summary([row for row in rows if row["dataset"] == name]) for name in ("aistpp", "finedance")},
        "records": rows,
    }
    (args.output_dir / "retargeting_loss_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flat_rows = [_flatten(row) for row in rows]
    if flat_rows:
        with (args.output_dir / "retargeting_loss_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    report = [
        "# GMR Retargeting Loss Audit",
        "",
        "This audit compares paired source human motion with retargeted G1 motion using the same audio.",
        "Because SMPL/SMPLH and G1 do not share joint semantics, activity/root/music proxies are used; this is not a direct joint position error.",
        "",
        f"Scope: **{args.scope}**, sequences: **{len(rows)}**.",
        "",
        "## Interpretation",
        "",
        "Positive `delta_*` means the target G1 result is larger/worse for that metric; negative means the target is lower/better. For BAS and event F1, negative is a loss. For absolute lag, positive is a loss.",
        "",
        "## Pooled summary",
        "",
        "| metric | mean | std | median | q10 | q90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in payload["summary"]["metrics"].items():
        report.append(f"| `{name}` | {values['mean']:.6f} | {values['std']:.6f} | {values['median']:.6f} | {values['q10']:.6f} | {values['q90']:.6f} |")
    report.extend([
        "",
        "The most important outputs for the music-to-G1 story are `delta_bas`, `delta_event_f1`, `delta_absolute_impact_lag`, `delta_phase_error_cycles`, and the activity/root curve correlations. These quantify whether retargeting itself removes musical response before SONIC execution.",
        "",
        "Raw source and target motion quality should still be reported separately. A retargeted G1 motion can have a different absolute jerk or energy scale while preserving the temporal structure of the source dance.",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved retargeting audit to {args.output_dir}")


if __name__ == "__main__":
    main()
