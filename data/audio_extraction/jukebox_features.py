import os
import subprocess
from pathlib import Path

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm
import jukemirlib.lib as jukemirlib_lib

FPS = 30
LAYER = 66
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILENAMES = ("vqvae.pth.tar", "prior_level_2.pth.tar")
MODELS_READY = False
DEFAULT_BATCH_SIZE = 8


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


def _prefer_runtime_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jukemirlib.constants.DEVICE = device
    return device


def _ensure_condition_cache_matches_batch_size(batch_size):
    for condition_store in _jukemirlib_condition_stores():
        _ensure_condition_store_matches_batch_size(condition_store, batch_size)


def _jukemirlib_condition_stores():
    seen = set()
    for condition_store in (
        getattr(jukemirlib.extract, "__globals__", {}),
        jukemirlib_lib.__dict__,
        jukemirlib.__dict__,
    ):
        store_id = id(condition_store)
        if store_id in seen:
            continue
        seen.add(store_id)
        yield condition_store


def _ensure_condition_store_matches_batch_size(condition_store, batch_size):
    if "x_cond" not in condition_store and "y_cond" not in condition_store:
        return
    x_cond = condition_store.get("x_cond")
    y_cond = condition_store.get("y_cond")
    if x_cond is None and y_cond is None:
        return
    if (
        getattr(x_cond, "shape", (None,))[0] == batch_size
        and getattr(y_cond, "shape", (None,))[0] == batch_size
    ):
        return
    if "x_cond" in condition_store:
        condition_store["x_cond"] = None
    if "y_cond" in condition_store:
        condition_store["y_cond"] = None


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
    _prefer_runtime_device()
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
    _prefer_runtime_device()
    audio = jukemirlib.load_audio(fpath)
    reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)

    #np.save(save_path, reps[LAYER])
    return reps[LAYER], save_path


def _resolve_batch_size(batch_size=None):
    if batch_size is not None:
        return max(int(batch_size), 1)
    env_value = os.environ.get("EDGE_JUKEBOX_BATCH_SIZE", "").strip()
    if env_value:
        return max(int(env_value), 1)
    return DEFAULT_BATCH_SIZE


def _chunked(sequence, batch_size):
    for start in range(0, len(sequence), batch_size):
        yield sequence[start : start + batch_size]


def extract_batch(fpaths, dest_dir="aist_juke_feats", skip_completed=True):
    fpaths = [Path(fpath) for fpath in fpaths]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    pending_paths = []
    outputs = []
    for fpath in fpaths:
        save_path = dest_dir / f"{fpath.stem}.npy"
        if skip_completed and save_path.exists():
            continue
        pending_paths.append(fpath)
        outputs.append((None, save_path))

    if not pending_paths:
        return []

    ensure_jukebox_models()
    _prefer_runtime_device()
    audios = [jukemirlib.load_audio(str(fpath)) for fpath in pending_paths]

    if len(audios) == 1:
        _ensure_condition_cache_matches_batch_size(2)
        reps = jukemirlib.extract(
            audio=[audios[0], audios[0]],
            layers=[LAYER],
            downsample_target_rate=FPS,
        )[LAYER]
        return [(reps[0], outputs[0][1])]

    _ensure_condition_cache_matches_batch_size(len(audios))
    reps = jukemirlib.extract(
        audio=audios,
        layers=[LAYER],
        downsample_target_rate=FPS,
    )[LAYER]
    return list(zip(reps, [save_path for _, save_path in outputs]))


def extract_folder(src, dest, batch_size=None, skip_completed=True):
    fpaths = Path(src).glob("*")
    fpaths = sorted(path for path in fpaths if path.suffix == ".wav")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not fpaths:
        return

    batch_size = _resolve_batch_size(batch_size)
    ensure_jukebox_models()
    print(f"Using Jukebox batch_size={batch_size} on device={jukemirlib.constants.DEVICE}")
    with tqdm(total=len(fpaths), desc="Jukebox slices", unit="clip") as progress:
        for batch in _chunked(fpaths, batch_size):
            for rep, path in extract_batch(batch, dest_dir=dest, skip_completed=skip_completed):
                np.save(path, rep)
            progress.update(len(batch))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")
    args = parser.parse_args()

    extract_folder(args.src, args.dest)
