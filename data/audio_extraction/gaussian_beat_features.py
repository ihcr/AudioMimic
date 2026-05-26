import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from feature_config import GAUSSIAN_BEAT_DIM, WAV2CLIP_STFT_BEAT_DIM

from data.audio_extraction.wav2clip_stft_beat_features import (
    TARGET_FRAMES,
    extract_gaussian_beat,
)


def _feature_file_is_valid(path):
    try:
        features = np.load(path)
    except Exception:
        return False
    return features.shape == (TARGET_FRAMES, GAUSSIAN_BEAT_DIM) and np.isfinite(features).all()


def _load_from_combined_feature(path):
    features = np.load(path, mmap_mode="r")
    if features.shape != (TARGET_FRAMES, WAV2CLIP_STFT_BEAT_DIM):
        raise ValueError(
            f"Expected wav2clip_stft_beat feature shape "
            f"{(TARGET_FRAMES, WAV2CLIP_STFT_BEAT_DIM)} for {path}, got {features.shape}"
        )
    return np.asarray(features[:, -GAUSSIAN_BEAT_DIM:], dtype=np.float32)


def extract(fpath, skip_completed=True, dest_dir="gaussian_beat_feats", source_feature_dir=None):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")
    if os.path.exists(save_path) and skip_completed and _feature_file_is_valid(save_path):
        return None

    if source_feature_dir is None:
        features = extract_gaussian_beat(fpath)
    else:
        features = _load_from_combined_feature(Path(source_feature_dir) / f"{audio_name}.npy")
    assert features.shape == (TARGET_FRAMES, GAUSSIAN_BEAT_DIM), features.shape
    return features.astype(np.float32), save_path


def extract_folder(src, dest, skip_completed=True, shard_index=0, num_shards=1, source_feature_dir=None):
    fpaths = sorted(Path(src).glob("*.wav"))
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    fpaths = fpaths[shard_index::num_shards]
    for fpath in tqdm(fpaths):
        result = extract(
            fpath,
            skip_completed=skip_completed,
            dest_dir=dest,
            source_feature_dir=source_feature_dir,
        )
        if result is None:
            continue
        rep, path = result
        tmp_path = f"{path}.tmp.{os.getpid()}.npy"
        np.save(tmp_path, rep)
        os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dest")
    parser.add_argument("--source_feature_dir", default=None)
    parser.add_argument("--no_skip_completed", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_folder(
        args.src,
        args.dest,
        skip_completed=not args.no_skip_completed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        source_feature_dir=args.source_feature_dir,
    )
