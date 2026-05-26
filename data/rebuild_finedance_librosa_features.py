import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.audio_extraction.baseline_features import (  # noqa: E402
    BASELINE34_FEATURE_DIM,
    BASELINE_FEATURE_DIM,
    FPS,
    make_audio_feature,
    trim_feature_dim,
)

FEATURE_DIMS = {
    "baseline": BASELINE_FEATURE_DIM,
    "baseline34": BASELINE34_FEATURE_DIM,
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_finedance_root(data_root, explicit_root=None):
    if explicit_root:
        return Path(explicit_root)

    metadata_path = Path(data_root) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} is missing; pass --finedance_root explicitly."
        )
    metadata = read_json(metadata_path)
    if metadata.get("source_root"):
        return Path(metadata["source_root"])

    source_prepared = metadata.get("source_prepared_root")
    if source_prepared:
        source_metadata_path = Path(source_prepared) / "metadata.json"
        if not source_metadata_path.is_file():
            raise FileNotFoundError(
                f"{source_metadata_path} is missing; pass --finedance_root explicitly."
            )
        source_metadata = read_json(source_metadata_path)
        if source_metadata.get("source_root"):
            return Path(source_metadata["source_root"])

    raise ValueError(f"Could not resolve FineDance source root from {metadata_path}.")


def parse_slice_stem(stem):
    sequence_id, slice_part = stem.rsplit("_slice", 1)
    return sequence_id, int(slice_part)


def group_slice_stems(motion_dir):
    grouped = defaultdict(list)
    for path in sorted(Path(motion_dir).glob("*.pkl")):
        sequence_id, slice_idx = parse_slice_stem(path.stem)
        grouped[sequence_id].append((slice_idx, path.stem))
    if not grouped:
        raise ValueError(f"{motion_dir}: no sliced motions found.")
    return grouped


def rebuild_split(
    finedance_root,
    data_root,
    split_name,
    feature_type,
    fps=FPS,
    length_seconds=5.0,
    stride_seconds=0.5,
    clean=False,
):
    feature_dim = FEATURE_DIMS[feature_type]
    split_root = Path(data_root) / split_name
    motion_dir = split_root / "motions_sliced"
    feature_dir = split_root / f"{feature_type}_feats"
    if clean and feature_dir.exists():
        shutil.rmtree(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    window_frames = int(round(length_seconds * fps))
    stride_frames = int(round(stride_seconds * fps))
    grouped = group_slice_stems(motion_dir)

    written = 0
    for sequence_id, items in tqdm(
        sorted(grouped.items()),
        desc=f"{split_name} {feature_type}",
        unit="song",
    ):
        wav_path = Path(finedance_root) / "music_wav" / f"{sequence_id}.wav"
        if not wav_path.is_file():
            raise FileNotFoundError(f"Missing FineDance music wav: {wav_path}")
        full_feature = make_audio_feature(wav_path)
        full_feature = trim_feature_dim(full_feature, feature_dim=feature_dim)
        for slice_idx, stem in items:
            start = slice_idx * stride_frames
            end = start + window_frames
            if end > full_feature.shape[0]:
                raise ValueError(
                    f"{wav_path}: feature sequence is too short for {stem}; "
                    f"need frame {end}, have {full_feature.shape[0]}"
                )
            np.save(feature_dir / f"{stem}.npy", full_feature[start:end])
            written += 1
    return {
        "split": split_name,
        "feature_type": feature_type,
        "feature_dim": feature_dim,
        "songs": len(grouped),
        "clips": written,
        "feature_dir": str(feature_dir),
    }


def rebuild_features(
    data_root,
    finedance_root=None,
    feature_types=("baseline",),
    splits=("train", "test"),
    fps=FPS,
    length_seconds=5.0,
    stride_seconds=0.5,
    clean=False,
):
    data_root = Path(data_root)
    finedance_root = resolve_finedance_root(data_root, finedance_root)
    if not (finedance_root / "music_wav").is_dir():
        raise FileNotFoundError(f"{finedance_root / 'music_wav'} is missing.")

    results = []
    for feature_type in feature_types:
        if feature_type not in FEATURE_DIMS:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
        for split_name in splits:
            results.append(
                rebuild_split(
                    finedance_root=finedance_root,
                    data_root=data_root,
                    split_name=split_name,
                    feature_type=feature_type,
                    fps=fps,
                    length_seconds=length_seconds,
                    stride_seconds=stride_seconds,
                    clean=clean,
                )
            )
    return {
        "data_root": str(data_root),
        "finedance_root": str(finedance_root),
        "results": results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebuild FineDance Librosa features from full-song context and slice by prepared clip stems."
    )
    parser.add_argument("--data_root", default="data/finedance_g1_fkbeats")
    parser.add_argument("--finedance_root", default=None)
    parser.add_argument(
        "--feature_types",
        nargs="+",
        choices=tuple(FEATURE_DIMS),
        default=["baseline", "baseline34"],
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--length", type=float, default=5.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = rebuild_features(
        data_root=args.data_root,
        finedance_root=args.finedance_root,
        feature_types=args.feature_types,
        splits=args.splits,
        fps=args.fps,
        length_seconds=args.length,
        stride_seconds=args.stride,
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
