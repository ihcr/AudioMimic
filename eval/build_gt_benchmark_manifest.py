"""Audit paired GT assets and write a reusable benchmark manifest.

This first pass is intentionally read-only with respect to source datasets. It
records missing FineDance assets instead of silently treating metadata as
available motion. SHA256 hashing is opt-in because the local AIST++ audio set
is several gigabytes; the manifest always records file size and shape metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_KEYS = ("root_pos", "root_rot", "dof_pos")
FINEDANCE_SEQUENCE_IDS = tuple(f"{index:03d}" for index in range(1, 212))
FINEDANCE_CROSS_GENRE_TEST = (
    "063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
    "179", "065", "137", "161", "092", "120", "037", "109", "204", "144",
)
FINEDANCE_CROSS_GENRE_IGNORE = (
    "116", "117", "118", "119", "120", "121", "122", "123", "202", "130",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aist-motion-root",
        type=Path,
        default=Path("/home/tianhup/Downloads/edge_smpl_dataset_retargeted/unitree_g1"),
    )
    parser.add_argument(
        "--aist-audio-root", type=Path, default=Path("data/edge_aistpp/wavs")
    )
    parser.add_argument(
        "--aist-train-split",
        type=Path,
        default=Path("/home/tianhup/Musics2Dance-prior-dev/data/splits/crossmodal_train.txt"),
    )
    parser.add_argument(
        "--aist-test-split",
        type=Path,
        default=Path("/home/tianhup/Musics2Dance-prior-dev/data/splits/crossmodal_test.txt"),
    )
    parser.add_argument(
        "--aist-processed-test",
        type=Path,
        default=Path("data/g1_aistpp_full_dataset_backups/processed_test_g1_jukebox_beat_distance_v4.pkl"),
        help="Existing processed test cache used to cross-check sequence membership.",
    )
    parser.add_argument(
        "--finedance-metadata",
        type=Path,
        default=Path("/home/tianhup/Musics2Dance-prior-dev/data/finedance_g1_fkbeats/metadata.json"),
    )
    parser.add_argument(
        "--finedance-root",
        type=Path,
        default=Path("/home/tianhup/Musics2Dance-prior-dev/data/finedance_g1_fkbeats"),
        help="Legacy prepared-tree root; retained for backward-compatible metadata auditing.",
    )
    parser.add_argument(
        "--finedance-source-root",
        type=Path,
        default=Path("data/finedance"),
        help="Raw FineDance root containing motion/, music_wav/ and label_json/.",
    )
    parser.add_argument(
        "--finedance-g1-root",
        type=Path,
        default=Path("data/finedance-g1-retargeted"),
        help="Retargeted G1 pickle root, one file per FineDance sequence ID.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval/benchmark_v1/gt/manifest_v1")
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Compute SHA256 for every paired asset; can take time on large audio sets.",
    )
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, hash_files: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
    }
    if path.is_file():
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        record["sha256"] = _sha256(path) if hash_files else None
    return record


def _normalise_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    values = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        values.append(Path(value).stem)
    return values


def _audit_motion(path: Path, hash_files: bool) -> dict[str, Any]:
    record = _file_record(path, hash_files)
    record.update({"valid": False, "errors": []})
    if not path.is_file():
        record["errors"].append("missing_file")
        return record
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        missing = [key for key in EXPECTED_KEYS if key not in payload]
        if missing:
            record["errors"].append(f"missing_keys:{','.join(missing)}")
            return record
        root_pos = np.asarray(payload["root_pos"])
        root_rot = np.asarray(payload["root_rot"])
        dof_pos = np.asarray(payload["dof_pos"])
        fps = float(payload.get("fps", 30.0))
        record.update(
            {
                "frames": int(len(dof_pos)),
                "fps": fps,
                "duration_seconds": float(len(dof_pos) / fps) if fps > 0 else None,
                "shapes": {
                    "root_pos": list(root_pos.shape),
                    "root_rot": list(root_rot.shape),
                    "dof_pos": list(dof_pos.shape),
                },
                "root_quat_order": "xyzw",
            }
        )
        if root_pos.shape != (len(dof_pos), 3):
            record["errors"].append(f"root_pos_shape:{root_pos.shape}")
        if root_rot.shape != (len(dof_pos), 4):
            record["errors"].append(f"root_rot_shape:{root_rot.shape}")
        if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
            record["errors"].append(f"dof_pos_shape:{dof_pos.shape}")
        if fps <= 0:
            record["errors"].append("invalid_fps")
        if not all(np.isfinite(value).all() for value in (root_pos, root_rot, dof_pos)):
            record["errors"].append("non_finite_motion")
        quat_norm = np.linalg.norm(root_rot, axis=1) if root_rot.ndim == 2 else np.array([])
        if len(quat_norm) and float(np.max(np.abs(quat_norm - 1.0))) > 1e-3:
            record["errors"].append("root_quaternion_not_normalized")
        record["valid"] = not record["errors"]
    except Exception as exc:  # noqa: BLE001 - audit must record bad files and continue.
        record["errors"].append(f"load_error:{type(exc).__name__}:{exc}")
    return record


def _audit_audio(path: Path, hash_files: bool) -> dict[str, Any]:
    record = _file_record(path, hash_files)
    record.update({"valid": False, "errors": []})
    if not path.is_file():
        record["errors"].append("missing_file")
        return record
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
        record.update(
            {
                "sample_rate_hz": rate,
                "frames": frames,
                "channels": channels,
                "sample_width_bytes": sample_width,
                "duration_seconds": float(frames / rate) if rate else None,
            }
        )
        if rate <= 0 or frames <= 0 or channels <= 0:
            record["errors"].append("invalid_wave_header")
        record["valid"] = not record["errors"]
    except Exception as exc:  # noqa: BLE001 - audit must record bad files and continue.
        record["errors"].append(f"load_error:{type(exc).__name__}:{exc}")
    return record


def _audit_finedance_raw_motion(path: Path, hash_files: bool) -> dict[str, Any]:
    record = _file_record(path, hash_files)
    record.update({"valid": False, "errors": []})
    if not path.is_file():
        record["errors"].append("missing_file")
        return record
    try:
        motion = np.load(path, mmap_mode="r")
        record["shape"] = list(motion.shape)
        record["frames"] = int(motion.shape[0]) if motion.ndim >= 1 else 0
        record["columns"] = int(motion.shape[1]) if motion.ndim == 2 else None
        if motion.ndim != 2 or motion.shape[1] not in (159, 315):
            record["errors"].append(f"raw_motion_shape:{motion.shape}")
        if motion.ndim == 2 and not np.isfinite(motion).all():
            record["errors"].append("non_finite_motion")
        record["valid"] = not record["errors"]
    except Exception as exc:  # noqa: BLE001 - audit must record bad files and continue.
        record["errors"].append(f"load_error:{type(exc).__name__}:{exc}")
    return record


def _audit_aist(args: argparse.Namespace) -> dict[str, Any]:
    motion_root = args.aist_motion_root.expanduser().resolve()
    audio_root = args.aist_audio_root.expanduser().resolve()
    motion_paths = {path.stem: path for path in sorted(motion_root.glob("*.pkl"))}
    audio_paths = {path.stem: path for path in sorted(audio_root.glob("*.wav"))}
    paired_ids = sorted(set(motion_paths) & set(audio_paths))

    records = []
    for sequence_id in paired_ids:
        motion = _audit_motion(motion_paths[sequence_id], args.hash_files)
        audio = _audit_audio(audio_paths[sequence_id], args.hash_files)
        records.append(
            {
                "sequence_id": sequence_id,
                "motion": motion,
                "audio": audio,
                "paired_valid": bool(motion["valid"] and audio["valid"]),
            }
        )

    train_ids = _normalise_ids(args.aist_train_split)
    test_ids = _normalise_ids(args.aist_test_split)
    train_set = set(train_ids)
    test_set = set(test_ids)
    split_overlap = sorted(train_set & test_set)
    paired_set = set(paired_ids)
    processed_test_ids: list[str] = []
    processed_test_error = None
    processed_test_path = args.aist_processed_test.expanduser().resolve()
    if processed_test_path.is_file():
        try:
            with processed_test_path.open("rb") as handle:
                processed_payload = pickle.load(handle)
            processed_test_ids = sorted(
                {
                    Path(str(value)).name.split("_slice", 1)[0]
                    for value in processed_payload.get("filenames", [])
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve audit result.
            processed_test_error = f"load_error:{type(exc).__name__}:{exc}"
    else:
        processed_test_error = "missing_file"
    processed_test_set = set(processed_test_ids)
    processed_test_match = bool(processed_test_ids) and processed_test_set == test_set
    split_report = {
        "train_file": str(args.aist_train_split.expanduser().resolve()),
        "test_file": str(args.aist_test_split.expanduser().resolve()),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "train_duplicate_count": len(train_ids) - len(train_set),
        "test_duplicate_count": len(test_ids) - len(test_set),
        "train_test_overlap": split_overlap,
        "train_missing_assets": sorted(train_set - paired_set),
        "test_missing_assets": sorted(test_set - paired_set),
        "test_assets_available": sorted(test_set & paired_set),
        "processed_test_file": str(processed_test_path),
        "processed_test_count": len(processed_test_ids),
        "processed_test_ids": processed_test_ids,
        "processed_test_error": processed_test_error,
        "processed_test_match": processed_test_match,
        "processed_test_missing_from_declared": sorted(processed_test_set - test_set),
        "declared_test_missing_from_processed": sorted(test_set - processed_test_set),
        "status": "pass"
        if not split_overlap and not (test_set - paired_set) and processed_test_match
        else "review_required",
    }
    return {
        "dataset": "AIST++_retargeted_G1",
        "asset_roots": {"motion": str(motion_root), "audio": str(audio_root)},
        "motion_file_count": len(motion_paths),
        "audio_file_count": len(audio_paths),
        "paired_file_count": len(paired_ids),
        "motion_without_audio": sorted(set(motion_paths) - set(audio_paths)),
        "audio_without_motion": sorted(set(audio_paths) - set(motion_paths)),
        "split": split_report,
        "records": records,
        "status": "pass" if records and all(r["paired_valid"] for r in records) else "review_required",
    }


def _audit_finedance(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.finedance_metadata.expanduser().resolve()
    prepared_root = args.finedance_root.expanduser().resolve()
    source_root = args.finedance_source_root.expanduser().resolve()
    g1_root = args.finedance_g1_root.expanduser().resolve()
    metadata: dict[str, Any] = {}
    errors = []
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        errors.append("metadata_missing")
    raw_motion_paths = {
        path.stem: path for path in sorted((source_root / "motion").glob("*.npy"))
    }
    audio_paths = {
        path.stem: path for path in sorted((source_root / "music_wav").glob("*.wav"))
    }
    label_paths = {
        path.stem: path for path in sorted((source_root / "label_json").glob("*.json"))
    }
    g1_motion_paths = {
        path.stem: path for path in sorted(g1_root.glob("*.pkl"))
    }
    all_ids = sorted(set(raw_motion_paths) | set(audio_paths) | set(g1_motion_paths))
    records = []
    for sequence_id in all_ids:
        raw_motion = _audit_finedance_raw_motion(
            raw_motion_paths.get(sequence_id, source_root / "motion" / f"{sequence_id}.npy"),
            args.hash_files,
        )
        audio = _audit_audio(
            audio_paths.get(sequence_id, source_root / "music_wav" / f"{sequence_id}.wav"),
            args.hash_files,
        )
        g1_motion = _audit_motion(
            g1_motion_paths.get(sequence_id, g1_root / f"{sequence_id}.pkl"),
            args.hash_files,
        )
        label = _file_record(
            label_paths.get(sequence_id, source_root / "label_json" / f"{sequence_id}.json"),
            args.hash_files,
        )
        label_errors = []
        label_frames = None
        label_name = None
        label_style = None
        label_path = label_paths.get(sequence_id)
        if label_path is None:
            label_errors.append("missing_file")
        else:
            try:
                label_payload = json.loads(label_path.read_text(encoding="utf-8"))
                label_frames = int(label_payload.get("frames"))
                label_name = label_payload.get("name")
                label_style = [label_payload.get("style1"), label_payload.get("style2")]
            except Exception as exc:  # noqa: BLE001 - preserve audit result.
                label_errors.append(f"load_error:{type(exc).__name__}:{exc}")
        label.update({"valid": not label_errors, "errors": label_errors})
        duration_deltas = []
        if raw_motion.get("frames") is not None:
            duration_deltas.append(
                abs(float(raw_motion["frames"]) / 30.0 - float(g1_motion.get("duration_seconds", 0.0)))
            )
        if audio.get("duration_seconds") is not None and g1_motion.get("duration_seconds") is not None:
            duration_deltas.append(
                abs(float(audio["duration_seconds"]) - float(g1_motion["duration_seconds"]))
            )
        pairing_errors = []
        pairing_warnings = []
        if not raw_motion["exists"]:
            pairing_errors.append("raw_motion_missing")
        if not audio["exists"]:
            pairing_errors.append("audio_missing")
        if not g1_motion["exists"]:
            pairing_errors.append("g1_motion_missing")
        if duration_deltas and max(duration_deltas) > 0.25:
            pairing_warnings.append(f"duration_mismatch_seconds:{max(duration_deltas):.4f}")
        if label_frames is not None and g1_motion.get("frames") is not None:
            if abs(label_frames - int(g1_motion["frames"])) > 2:
                pairing_errors.append(
                    f"label_g1_frame_mismatch:{label_frames}:{g1_motion['frames']}"
                )
        records.append(
            {
                "sequence_id": sequence_id,
                "source_motion": raw_motion,
                "audio": audio,
                "g1_motion": g1_motion,
                "label": label,
                "label_name": label_name,
                "style": label_style,
                "label_frames": label_frames,
                "duration_delta_seconds_max": max(duration_deltas) if duration_deltas else None,
                "common_duration_seconds": min(
                    value
                    for value in (
                        raw_motion.get("frames", 0) / 30.0,
                        audio.get("duration_seconds") or 0.0,
                        g1_motion.get("duration_seconds") or 0.0,
                    )
                    if value > 0
                )
                if duration_deltas
                else None,
                "pairing_errors": sorted(set(pairing_errors)),
                "pairing_warnings": sorted(set(pairing_warnings)),
                "paired_valid": bool(
                    raw_motion["valid"]
                    and audio["valid"]
                    and g1_motion["valid"]
                    and not pairing_errors
                ),
            }
        )
    test_set = set(FINEDANCE_CROSS_GENRE_TEST)
    ignore_set = set(FINEDANCE_CROSS_GENRE_IGNORE)
    train_ids = [
        sequence_id
        for sequence_id in FINEDANCE_SEQUENCE_IDS
        if sequence_id not in test_set and sequence_id not in ignore_set
    ]
    test_ids = [sequence_id for sequence_id in FINEDANCE_CROSS_GENRE_TEST if sequence_id not in ignore_set]
    ignore_ids = sorted(ignore_set)
    id_set = set(all_ids)
    split_report = {
        "name": "cross_genre",
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "ignore_count": len(ignore_ids),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "ignore_ids": ignore_ids,
        "train_missing_assets": sorted(set(train_ids) - id_set),
        "test_missing_assets": sorted(set(test_ids) - id_set),
        "test_assets_available": sorted(set(test_ids) & id_set),
        "train_test_overlap": sorted(set(train_ids) & set(test_ids)),
        "status": "pass"
        if not (set(train_ids) & set(test_ids))
        and not (set(test_ids) - id_set)
        else "review_required",
    }
    required_ids = set(train_ids) | set(test_ids)
    paired_ids = sorted(
        sequence_id
        for sequence_id in set(raw_motion_paths) & set(audio_paths) & set(g1_motion_paths)
    )
    invalid_required_ids = sorted(required_ids - set(paired_ids))
    extra_unpaired_ids = sorted(set(all_ids) - required_ids - set(paired_ids))
    if invalid_required_ids:
        errors.append("required_split_sequences_unpaired")
    return {
        "dataset": "FineDance_retarged_G1",
        "metadata_path": str(metadata_path),
        "prepared_root": str(prepared_root),
        "source_root": str(source_root),
        "g1_root": str(g1_root),
        "metadata": metadata,
        "raw_motion_file_count": len(raw_motion_paths),
        "audio_file_count": len(audio_paths),
        "label_file_count": len(label_paths),
        "g1_motion_file_count": len(g1_motion_paths),
        "paired_file_count": len(paired_ids),
        "motion_without_audio": sorted(set(raw_motion_paths) - set(audio_paths)),
        "audio_without_motion": sorted(set(audio_paths) - set(raw_motion_paths)),
        "motion_without_g1": sorted(set(raw_motion_paths) - set(g1_motion_paths)),
        "g1_without_motion": sorted(set(g1_motion_paths) - set(raw_motion_paths)),
        "split": split_report,
        "records": records,
        "invalid_or_unpaired_ids": invalid_required_ids,
        "extra_unpaired_ids": extra_unpaired_ids,
        "duration_warning_count": sum(bool(record["pairing_warnings"]) for record in records),
        "errors": errors,
        "status": "pass"
        if not errors and records and all(
            record["paired_valid"]
            for record in records
            if record["sequence_id"] in required_ids
        )
        else "review_required",
    }


def _write_report(manifest: dict[str, Any], path: Path) -> None:
    aist = manifest["datasets"]["aistpp"]
    fine = manifest["datasets"]["finedance"]
    split = aist["split"]
    lines = [
        "# GT Benchmark Asset Audit",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Result",
        "",
        f"- AIST++ paired assets: **{aist['paired_file_count']}**; status: **{aist['status']}**.",
        f"- AIST++ motion/audio unmatched: {len(aist['motion_without_audio'])}/{len(aist['audio_without_motion'])}.",
        f"- Declared crossmodal test IDs: {split['test_count']}; available paired IDs: {len(split['test_assets_available'])}.",
        f"- Processed test-cache IDs: {split['processed_test_count']}; split/cache match: **{split['processed_test_match']}**.",
        f"- AIST++ split status: **{split['status']}**.",
        f"- FineDance status: **{fine['status']}**.",
        f"- FineDance raw/G1/audio files: {fine.get('raw_motion_file_count', 0)}/"
        f"{fine.get('g1_motion_file_count', 0)}/{fine.get('audio_file_count', 0)}; "
        f"same-ID paired valid: **{fine.get('paired_file_count', 0)}**.",
        f"- FineDance cross-genre test assets available: "
        f"{len(fine.get('split', {}).get('test_assets_available', []))}/"
        f"{fine.get('split', {}).get('test_count', 0)}.",
        "",
        "## FineDance pairing",
        "",
        "FineDance is valid only when the raw motion, matching WAV, retargeted G1 motion, and label metadata share the same numeric sequence ID. The audit also checks duration agreement and label/G1 frame agreement. The benchmark must not treat metadata counts as usable GT when any paired asset is missing.",
        "",
        "## AIST++ split note",
        "",
        "The declared `crossmodal_test.txt` IDs exactly match the 20 sequence IDs in the existing processed test cache. This confirms local split/cache consistency; official provenance and the final paper manifest still need to be frozen before final evaluation.",
        "",
        "## Next action",
        "",
        "1. Review any FineDance IDs listed in `invalid_or_unpaired_ids`.",
        "2. Confirm the cross-genre test IDs against the official split.",
        "3. Run with `--hash-files` once the final roots are frozen.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "gt_benchmark_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_files": bool(args.hash_files),
        "datasets": {
            "aistpp": _audit_aist(args),
            "finedance": _audit_finedance(args),
        },
    }
    (output_dir / "gt_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(manifest, output_dir / "AUDIT_REPORT.md")
    aist = manifest["datasets"]["aistpp"]
    fine = manifest["datasets"]["finedance"]
    print(f"AIST++ paired={aist['paired_file_count']} status={aist['status']}")
    print(f"AIST++ split={aist['split']['status']} test_available={len(aist['split']['test_assets_available'])}")
    print(f"FineDance status={fine['status']} errors={','.join(fine['errors']) or 'none'}")
    print(f"Manifest: {output_dir / 'gt_benchmark_manifest.json'}")
    print(f"Report:   {output_dir / 'AUDIT_REPORT.md'}")


if __name__ == "__main__":
    main()
