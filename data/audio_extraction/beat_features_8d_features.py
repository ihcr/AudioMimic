import argparse
import os
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

SR = 22050
FPS = 30
TARGET_FRAMES = 150
FEATURE_DIM = 8
HOP_LENGTH = 512

FEATURE_NAMES = (
    "beat_pulse",
    "gaussian_beat",
    "dist_to_prev_beat_norm",
    "dist_to_next_beat_norm",
    "beat_phase_sin",
    "beat_phase_cos",
    "beat_interval_norm",
    "onset_strength_norm",
)


def _sanitize_beat_indices(beat_idxs, num_frames):
    beat_idxs = np.asarray(beat_idxs, dtype=np.int64).reshape(-1)
    if beat_idxs.size == 0:
        return beat_idxs
    beat_idxs = beat_idxs[(0 <= beat_idxs) & (beat_idxs < num_frames)]
    return np.unique(beat_idxs)


def _crop_or_pad(values, num_frames):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] >= num_frames:
        return values[:num_frames]
    padded = np.zeros((num_frames,), dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded


def gaussian_beat_from_indices(beat_idxs, num_frames=TARGET_FRAMES, sigma=1.5):
    pulse = np.zeros((num_frames,), dtype=np.float32)
    beat_idxs = _sanitize_beat_indices(beat_idxs, num_frames)
    if beat_idxs.size:
        pulse[beat_idxs] = 1.0
    if sigma > 0:
        pulse = gaussian_filter1d(pulse, sigma=sigma, mode="constant")
    peak = float(pulse.max()) if pulse.size else 0.0
    if peak > 1e-8:
        pulse = pulse / peak
    return pulse.astype(np.float32)


def _normalize_onset_strength(onset_strength, num_frames):
    values = _crop_or_pad(onset_strength, num_frames)
    min_value = float(values.min()) if values.size else 0.0
    max_value = float(values.max()) if values.size else 0.0
    denom = max_value - min_value
    if denom < 1e-8:
        return np.zeros((num_frames,), dtype=np.float32)
    return ((values - min_value) / denom).astype(np.float32)


def beat_features_8d_from_indices(beat_idxs, onset_strength, num_frames=TARGET_FRAMES):
    beat_idxs = _sanitize_beat_indices(beat_idxs, num_frames)
    frames = np.arange(num_frames, dtype=np.float32)

    beat_pulse = np.zeros((num_frames,), dtype=np.float32)
    if beat_idxs.size:
        beat_pulse[beat_idxs] = 1.0

    gaussian_beat = gaussian_beat_from_indices(beat_idxs, num_frames=num_frames)
    onset_strength_norm = _normalize_onset_strength(onset_strength, num_frames)

    if beat_idxs.size == 0:
        dist_to_prev = np.ones((num_frames,), dtype=np.float32)
        dist_to_next = np.ones((num_frames,), dtype=np.float32)
        phase = np.zeros((num_frames,), dtype=np.float32)
        interval = np.ones((num_frames,), dtype=np.float32)
    else:
        prev_indices = np.searchsorted(beat_idxs, np.arange(num_frames), side="right") - 1
        next_indices = np.searchsorted(beat_idxs, np.arange(num_frames), side="left")
        prev_beats = np.where(prev_indices >= 0, beat_idxs[np.maximum(prev_indices, 0)], 0)
        next_beats = np.where(
            next_indices < beat_idxs.size,
            beat_idxs[np.minimum(next_indices, beat_idxs.size - 1)],
            num_frames - 1,
        )
        interval = np.maximum(next_beats - prev_beats, 1).astype(np.float32)
        dist_to_prev = np.clip((frames - prev_beats.astype(np.float32)) / interval, 0.0, 1.0)
        dist_to_next = np.clip((next_beats.astype(np.float32) - frames) / interval, 0.0, 1.0)
        phase = np.clip((frames - prev_beats.astype(np.float32)) / interval, 0.0, 1.0)
        interval = np.clip(interval / max(num_frames - 1, 1), 0.0, 1.0)

    phase_angle = 2.0 * np.pi * phase
    features = np.stack(
        (
            beat_pulse,
            gaussian_beat,
            dist_to_prev,
            dist_to_next,
            np.sin(phase_angle).astype(np.float32),
            np.cos(phase_angle).astype(np.float32),
            interval.astype(np.float32),
            onset_strength_norm,
        ),
        axis=-1,
    ).astype(np.float32)
    assert features.shape == (num_frames, FEATURE_DIM), features.shape
    return features


def extract_beat_indices_and_onset(fpath):
    audio, sr = librosa.load(fpath, sr=SR, mono=True)
    envelope = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_LENGTH)
    tempo = librosa.beat.tempo(onset_envelope=envelope, sr=sr, hop_length=HOP_LENGTH)[0]
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sr,
        hop_length=HOP_LENGTH,
        start_bpm=tempo,
        tightness=100,
        units="frames",
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    beat_idxs = np.rint(beat_times * FPS).astype(np.int64)
    return beat_idxs, envelope


def extract_beat_features_8d(fpath):
    beat_idxs, onset_strength = extract_beat_indices_and_onset(fpath)
    return beat_features_8d_from_indices(beat_idxs, onset_strength)


def _feature_file_is_valid(path):
    try:
        features = np.load(path)
    except Exception:
        return False
    return features.shape == (TARGET_FRAMES, FEATURE_DIM) and np.isfinite(features).all()


def extract(fpath, skip_completed=True, dest_dir="beat_features_8d_feats"):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")
    if os.path.exists(save_path) and skip_completed and _feature_file_is_valid(save_path):
        return None
    features = extract_beat_features_8d(fpath)
    return features.astype(np.float32), save_path


def extract_folder(src, dest, skip_completed=True, shard_index=0, num_shards=1):
    fpaths = sorted(Path(src).glob("*.wav"))
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    fpaths = fpaths[shard_index::num_shards]
    for fpath in tqdm(fpaths, desc="Beat features 8D", unit="clip"):
        result = extract(fpath, skip_completed=skip_completed, dest_dir=dest)
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
    )
