import os
import re
import subprocess
from functools import partial
from pathlib import Path

import jukemirlib
import numpy as np
from tqdm import tqdm

FPS = 30
LAYER = 66
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILENAMES = ("vqvae.pth.tar", "prior_level_2.pth.tar")
MODELS_READY = False
SLICE_PATTERN = re.compile(r"^(?P<song>.+)_slice(?P<idx>\d+)$")


def _shared_repo_root(repo_root):
    repo_root = Path(repo_root).resolve()
    for parent in [repo_root, *repo_root.parents]:
        if parent.name == ".worktrees":
            return parent.parent
    return repo_root


def _default_cache_dir(repo_root=None):
    repo_root = _shared_repo_root(repo_root or REPO_ROOT)
    return repo_root / ".cache" / "jukemirlib"


def _configure_jukemirlib_cache(cache_dir=None):
    cache_dir = Path(
        cache_dir or os.environ.get("JUKEMIRLIB_CACHE_DIR") or _default_cache_dir()
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JUKEMIRLIB_CACHE_DIR"] = str(cache_dir)
    jukemirlib.constants.CACHE_DIR = str(cache_dir)
    return cache_dir


def _cleanup_incomplete_jukemirlib_downloads(cache_dir):
    for tmp_path in Path(cache_dir).glob("*.tmp"):
        tmp_path.unlink(missing_ok=True)


def _download_with_resume(url, dest_path, runner=subprocess.run):
    dest_path = Path(dest_path)
    partial_path = Path(str(dest_path) + ".partial")
    runner(
        [
            "/usr/bin/curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "-C",
            "-",
            "-o",
            str(partial_path),
            url,
        ],
        check=True,
    )
    partial_path.replace(dest_path)


def ensure_jukebox_models(cache_dir=None, remote_prefix=None, runner=subprocess.run):
    global MODELS_READY
    cache_dir = _configure_jukemirlib_cache(cache_dir)
    if MODELS_READY and all((cache_dir / name).exists() for name in MODEL_FILENAMES):
        return cache_dir

    _cleanup_incomplete_jukemirlib_downloads(cache_dir)
    remote_prefix = remote_prefix or jukemirlib.constants.REMOTE_PREFIX
    for model_name in MODEL_FILENAMES:
        dest_path = cache_dir / model_name
        if dest_path.exists():
            continue
        _download_with_resume(remote_prefix + model_name, dest_path, runner=runner)

    MODELS_READY = True
    return cache_dir


def extract(fpath, skip_completed=True, dest_dir="aist_juke_feats"):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    ensure_jukebox_models()
    audio = jukemirlib.load_audio(fpath)
    reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)

    #np.save(save_path, reps[LAYER])
    return reps[LAYER], save_path


def extract_full_track(wav_path):
    ensure_jukebox_models()
    audio = jukemirlib.load_audio(wav_path)
    reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)
    return reps[LAYER]


def _slice_feature_track(features, slice_idx, stride=0.5, length=5.0, fps=FPS):
    start = int(round(slice_idx * stride * fps))
    window = int(round(length * fps))
    chunk = features[start : start + window]
    if chunk.shape[0] == window:
        return chunk
    if chunk.shape[0] == 0:
        feature_dim = features.shape[1] if features.ndim == 2 else 0
        return np.zeros((window, feature_dim), dtype=features.dtype)
    pad = np.repeat(chunk[-1:], window - chunk.shape[0], axis=0)
    return np.concatenate((chunk, pad), axis=0)


def _parse_slice_name(stem):
    match = SLICE_PATTERN.match(stem)
    if match is None:
        return None
    return match.group("song"), int(match.group("idx"))


def _resolve_source_wav(src_dir, song_name):
    src_dir = Path(src_dir)
    candidate = src_dir.parent / "wavs" / f"{song_name}.wav"
    if candidate.exists():
        return candidate
    candidate = src_dir / f"{song_name}.wav"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve original wav for {song_name!r} from {src_dir}")


def _group_slice_requests(fpaths):
    grouped = {}
    for fpath in fpaths:
        parsed = _parse_slice_name(fpath.stem)
        if parsed is None:
            return None
        song_name, slice_idx = parsed
        grouped.setdefault(song_name, []).append((slice_idx, fpath))
    return {song: sorted(items) for song, items in grouped.items()}


def extract_folder(src, dest, stride=0.5, length=5.0):
    fpaths = Path(src).glob("*")
    fpaths = sorted(path for path in fpaths if path.suffix == ".wav")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    extract_ = partial(extract, dest_dir=dest)
    ensure_jukebox_models()
    grouped = _group_slice_requests(fpaths)
    if grouped:
        grouped_items = sorted(grouped.items())
        for song_name, slice_items in tqdm(
            grouped_items,
            total=len(grouped_items),
            desc="Jukebox songs",
            unit="song",
        ):
            missing_items = [
                (slice_idx, fpath)
                for slice_idx, fpath in slice_items
                if not (dest / f"{fpath.stem}.npy").exists()
            ]
            if not missing_items:
                continue
            source_wav = _resolve_source_wav(src, song_name)
            full_features = extract_full_track(source_wav)
            for slice_idx, fpath in missing_items:
                slice_features = _slice_feature_track(
                    full_features,
                    slice_idx=slice_idx,
                    stride=stride,
                    length=length,
                )
                np.save(dest / f"{fpath.stem}.npy", slice_features)
        return

    for fpath in tqdm(
        fpaths,
        total=len(fpaths),
        desc="Jukebox slices",
        unit="clip",
    ):
        rep, path = extract_(fpath)
        if rep is not None:
            np.save(path, rep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0)

    args = parser.parse_args()

    extract_folder(args.src, args.dest, stride=args.stride, length=args.length)
