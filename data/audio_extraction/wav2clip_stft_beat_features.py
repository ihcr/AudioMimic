import os
import sys
import inspect
from functools import partial
from pathlib import Path

import librosa
import librosa as lr
import numpy as np
import scipy.signal
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from feature_config import (
    GAUSSIAN_BEAT_DIM,
    STFT_DIM,
    WAV2CLIP_DIM,
    WAV2CLIP_STFT_BEAT_DIM,
)

FPS = 30
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
WAV2CLIP_SR = 16000
WAV2CLIP_FRAME_LENGTH = WAV2CLIP_SR
WAV2CLIP_HOP_LENGTH = max(1, round(WAV2CLIP_SR / FPS))
STFT_N_FFT = 384
TARGET_SECONDS = 5
TARGET_FRAMES = TARGET_SECONDS * FPS

if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann


def patch_librosa_frame_for_wav2clip():
    frame = librosa.util.frame
    params = inspect.signature(frame).parameters
    frame_length = params.get("frame_length")
    hop_length = params.get("hop_length")
    keyword_only = (
        frame_length is not None
        and hop_length is not None
        and frame_length.kind is inspect.Parameter.KEYWORD_ONLY
        and hop_length.kind is inspect.Parameter.KEYWORD_ONLY
    )
    if not keyword_only or getattr(frame, "_wav2clip_compat", False):
        return

    def frame_compat(x, frame_length, hop_length, *args, **kwargs):
        return frame(
            x,
            *args,
            frame_length=frame_length,
            hop_length=hop_length,
            **kwargs,
        )

    frame_compat._wav2clip_compat = True
    librosa.util.frame = frame_compat


def _get_tempo(audio_name):
    audio_name = audio_name.split("_")[4]
    assert len(audio_name) == 4
    if audio_name[0:3] in [
        "mBR",
        "mPO",
        "mLO",
        "mMH",
        "mLH",
        "mWA",
        "mKR",
        "mJS",
        "mJB",
    ]:
        return int(audio_name[3]) * 10 + 80
    if audio_name[0:3] == "mHO":
        return int(audio_name[3]) * 5 + 110
    raise ValueError(audio_name)


def _crop_or_pad(features, num_frames=TARGET_FRAMES):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] >= num_frames:
        return features[:num_frames]
    pad_shape = (num_frames - features.shape[0],) + features.shape[1:]
    padding = np.zeros(pad_shape, dtype=np.float32)
    return np.concatenate([features, padding], axis=0)


def _resample_time_axis(features, num_frames=TARGET_FRAMES):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == num_frames:
        return features
    if features.shape[0] == 0:
        return np.zeros((num_frames, features.shape[1]), dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, features.shape[0])
    new_x = np.linspace(0.0, 1.0, num_frames)
    channels = [
        np.interp(new_x, old_x, features[:, channel])
        for channel in range(features.shape[1])
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def _coerce_wav2clip_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 3 and embeddings.shape[0] == 1:
        embeddings = embeddings[0]
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    if embeddings.ndim == 2 and embeddings.shape[0] == WAV2CLIP_DIM:
        embeddings = embeddings.T
    return embeddings


def gaussian_beat_from_indices(beat_idxs, num_frames=TARGET_FRAMES, alpha=0.25):
    beat_idxs = np.asarray(sorted(set(int(idx) for idx in beat_idxs if idx >= 0)), dtype=np.int64)
    if len(beat_idxs) == 0:
        return np.zeros((num_frames, GAUSSIAN_BEAT_DIM), dtype=np.float32)

    frames = np.arange(num_frames, dtype=np.float32)
    values = np.zeros((num_frames,), dtype=np.float32)
    augmented = np.concatenate([[0], beat_idxs, [num_frames - 1]])
    for beat_idx in beat_idxs:
        right = augmented[augmented > beat_idx]
        left = augmented[augmented < beat_idx]
        prev_idx = left[-1] if len(left) else 0
        next_idx = right[0] if len(right) else num_frames - 1
        interval = max(next_idx - prev_idx, 1)
        sigma = max(alpha * interval, 1.0)
        beat_values = np.exp(-0.5 * ((frames - beat_idx) / sigma) ** 2)
        values = np.maximum(values, beat_values.astype(np.float32))
    return values[:, None].astype(np.float32)


def extract_stft(fpath):
    data, _ = librosa.load(fpath, sr=SR)
    stft = librosa.stft(
        data,
        n_fft=STFT_N_FFT,
        hop_length=HOP_LENGTH,
        center=True,
    )
    features = np.log1p(np.abs(stft)).T.astype(np.float32)
    features = _crop_or_pad(features)
    assert features.shape == (TARGET_FRAMES, STFT_DIM), features.shape
    return features


def extract_gaussian_beat(fpath):
    audio_name = Path(fpath).stem
    data, _ = librosa.load(fpath, sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)
    try:
        start_bpm = _get_tempo(audio_name)
    except Exception:
        start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]
    _, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    return gaussian_beat_from_indices(beat_idxs)


def load_wav2clip_model(device=None):
    patch_librosa_frame_for_wav2clip()
    try:
        import torch
        import wav2clip
    except ImportError as exc:
        raise RuntimeError(
            "wav2clip is required for --feature_type wav2clip_stft_beat. "
            "Install it in the repo environment with `pip install wav2clip`."
        ) from exc
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return wav2clip.get_model(
        device=device,
        frame_length=WAV2CLIP_FRAME_LENGTH,
        hop_length=WAV2CLIP_HOP_LENGTH,
    )


def extract_wav2clip(fpath, wav2clip_model=None):
    try:
        import wav2clip
    except ImportError as exc:
        raise RuntimeError(
            "wav2clip is required for --feature_type wav2clip_stft_beat. "
            "Install it in the repo environment with `pip install wav2clip`."
        ) from exc
    if wav2clip_model is None:
        wav2clip_model = load_wav2clip_model()
    audio, _ = librosa.load(fpath, sr=WAV2CLIP_SR, mono=True)
    embeddings = wav2clip.embed_audio(audio.astype(np.float32), wav2clip_model)
    embeddings = _coerce_wav2clip_embeddings(embeddings)
    embeddings = _resample_time_axis(embeddings)
    assert embeddings.shape == (TARGET_FRAMES, WAV2CLIP_DIM), embeddings.shape
    return embeddings


def _feature_file_is_valid(path):
    try:
        features = np.load(path)
    except Exception:
        return False
    return (
        features.shape == (TARGET_FRAMES, WAV2CLIP_STFT_BEAT_DIM)
        and np.isfinite(features).all()
    )


def extract(fpath, skip_completed=True, dest_dir="wav2clip_stft_beat_feats", wav2clip_model=None):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")
    if os.path.exists(save_path) and skip_completed:
        if _feature_file_is_valid(save_path):
            return None

    wav2clip_features = extract_wav2clip(fpath, wav2clip_model=wav2clip_model)
    stft_features = extract_stft(fpath)
    beat_features = extract_gaussian_beat(fpath)
    features = np.concatenate([wav2clip_features, stft_features, beat_features], axis=-1)
    assert features.shape == (TARGET_FRAMES, WAV2CLIP_STFT_BEAT_DIM), features.shape
    return features.astype(np.float32), save_path


def extract_folder(src, dest, skip_completed=True, shard_index=0, num_shards=1):
    fpaths = sorted(list(Path(src).glob("*")))
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    fpaths = fpaths[shard_index::num_shards]
    wav2clip_model = load_wav2clip_model()
    extract_ = partial(
        extract,
        skip_completed=skip_completed,
        dest_dir=dest,
        wav2clip_model=wav2clip_model,
    )
    for fpath in tqdm(fpaths):
        result = extract_(fpath)
        if result is None:
            continue
        rep, path = result
        tmp_path = f"{path}.tmp.{os.getpid()}.npy"
        np.save(tmp_path, rep)
        os.replace(tmp_path, path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    extract_folder(
        args.src,
        args.dest,
        skip_completed=not args.overwrite,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
