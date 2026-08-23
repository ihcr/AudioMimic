"""Build deterministic motion-quality calibration corruptions from AIST++ GT.

This is a metric-calibration tool, not a model benchmark.  The generated
motions keep the original music pairing and are used to check whether the
quality metrics respond in the expected direction to controlled defects.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


DEFAULT_MANIFEST = Path("eval/benchmark_v1/gt/manifest_v1/gt_benchmark_manifest.json")
DEFAULT_OUTPUT = Path("eval/benchmark_v1/gt/motion_corruptions_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _stable_seed(base: int, sequence_id: str, variant: str, level: str) -> int:
    digest = hashlib.sha256(
        f"{base}:{sequence_id}:{variant}:{level}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _copy_payload(path: Path, audio_path: str, max_seconds: float) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = ("root_pos", "root_rot", "dof_pos")
    if any(key not in payload for key in required):
        raise ValueError(f"{path}: missing one of {required}")
    result = copy.deepcopy(payload)
    result["root_pos"] = np.asarray(payload["root_pos"], dtype=np.float32).copy()
    result["root_rot"] = np.asarray(payload["root_rot"], dtype=np.float32).copy()
    result["dof_pos"] = np.asarray(payload["dof_pos"], dtype=np.float32).copy()
    frames = min(
        len(result["root_pos"]), len(result["root_rot"]), len(result["dof_pos"])
    )
    fps = float(result.get("fps", 30.0))
    if max_seconds > 0:
        frames = min(frames, max(2, int(round(max_seconds * fps))))
    for key in required:
        result[key] = result[key][:frames]
    result["fps"] = fps
    result["audio_path"] = audio_path
    return result


def _slice_window(length: int, size: int, center_fraction: float = 0.55) -> tuple[int, int]:
    size = min(max(int(size), 1), max(length - 1, 1))
    start = int(round(length * center_fraction - size / 2.0))
    start = min(max(start, 1), max(length - size - 1, 1))
    return start, start + size


def _corrupt(payload: dict, variant: str, level: str, seed: int) -> tuple[dict, dict]:
    result = copy.deepcopy(payload)
    rng = np.random.default_rng(seed)
    fps = float(result["fps"])
    dof = result["dof_pos"]
    root = result["root_pos"]
    frames = len(dof)
    severity = {"clean": -1, "low": 0, "medium": 1, "high": 2}[level]
    details = {"variant": variant, "severity": level, "seed": int(seed)}

    if variant == "clean":
        return result, details

    if variant == "jitter":
        sigma = (0.002, 0.005, 0.010)[severity]
        dof += rng.normal(0.0, sigma, size=dof.shape).astype(np.float32)
        root[:, :2] += rng.normal(0.0, sigma * 0.04, size=root[:, :2].shape).astype(
            np.float32
        )
        details.update({"joint_noise_std_rad": sigma, "root_xy_noise_std_m": sigma * 0.04})
    elif variant == "lowpass":
        sigma = (0.75, 1.5, 3.0)[severity]
        result["dof_pos"] = gaussian_filter1d(dof, sigma=sigma, axis=0).astype(np.float32)
        result["root_pos"] = gaussian_filter1d(root, sigma=sigma, axis=0).astype(np.float32)
        details["gaussian_sigma_frames"] = sigma
    elif variant == "freeze":
        seconds = (0.25, 0.5, 1.0)[severity]
        start, stop = _slice_window(frames, round(seconds * fps))
        result["dof_pos"][start:stop] = result["dof_pos"][start]
        result["root_pos"][start:stop] = result["root_pos"][start]
        result["root_rot"][start:stop] = result["root_rot"][start]
        details.update({"freeze_seconds": seconds, "start_frame": start, "stop_frame": stop})
    elif variant == "repeat":
        seconds = (0.5, 1.0, 2.0)[severity]
        size = min(round(seconds * fps), max(frames // 4, 1))
        start, stop = _slice_window(frames, size)
        source_start = max(1, start - size)
        source_stop = source_start + (stop - start)
        result["dof_pos"][start:stop] = dof[source_start:source_stop]
        result["root_pos"][start:stop] = root[source_start:source_stop]
        result["root_rot"][start:stop] = payload["root_rot"][source_start:source_stop]
        details.update(
            {
                "repeat_seconds": seconds,
                "start_frame": start,
                "stop_frame": stop,
                "source_start_frame": source_start,
                "source_stop_frame": source_stop,
            }
        )
    else:
        raise ValueError(f"Unknown corruption variant: {variant}")

    result["corruption"] = details
    return result, details


def main(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest["datasets"]["aistpp"]
    split = dataset["split"]
    test_ids = list(split["test_assets_available"])
    if args.max_sequences > 0:
        test_ids = test_ids[: args.max_sequences]
    records = {record["sequence_id"]: record for record in dataset["records"]}

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = ("clean", "jitter", "lowpass", "freeze", "repeat")
    levels = ("low", "medium", "high")
    output_records = []

    for sequence_id in test_ids:
        if sequence_id not in records:
            raise KeyError(f"Test sequence missing from manifest records: {sequence_id}")
        record = records[sequence_id]
        motion_path = Path(record["motion"]["path"])
        audio_path = record["audio"]["path"]
        clean_payload = _copy_payload(motion_path, audio_path, args.max_seconds)
        for variant in variants:
            current_levels = ("clean",) if variant == "clean" else levels
            for level in current_levels:
                payload, details = _corrupt(
                    clean_payload,
                    variant,
                    level,
                    _stable_seed(args.seed, sequence_id, variant, level),
                )
                destination = output_dir / variant / level / f"{sequence_id}.pkl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                output_records.append(
                    {
                        "sequence_id": sequence_id,
                        "variant": variant,
                        "severity": level,
                        "path": str(destination),
                        "audio_path": audio_path,
                        "frames": int(len(payload["dof_pos"])),
                        "fps": float(payload["fps"]),
                        "details": details,
                    }
                )

    result = {
        "schema_version": "gt_motion_corruption_calibration_v1",
        "source_manifest": str(manifest_path),
        "dataset": "AIST++_retargeted_G1",
        "seed": args.seed,
        "sequence_count": len(test_ids),
        "sequences": test_ids,
        "variants": variants,
        "levels": levels,
        "records": output_records,
    }
    (output_dir / "corruption_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Generated {len(output_records)} calibration motions from {len(test_ids)} sequences")
    print(f"Manifest: {output_dir / 'corruption_manifest.json'}")


if __name__ == "__main__":
    main(parse_args())
