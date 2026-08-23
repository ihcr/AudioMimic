"""Parallel, resumable extraction of FineDance-G1 beat metadata."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.audio_extraction.beat_features import (
    extract_audio_beats_librosa,
    load_audio_beat_frames_spectral,
    _beat_mask,
    nearest_beat_distance,
    local_beat_spacing,
    extract_motion_beats_from_motion_pkl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--g1-fk-model-path", type=Path, required=True)
    parser.add_argument("--root-quat-order", choices=("xyzw", "wxyz"), default="xyzw")
    return parser.parse_args()


def extract_one(item: tuple[str, str, str, str, int, int, str]) -> str:
    clip_name, motion_path, wav_path, output_dir, fps, seq_len, model_path = item
    output_path = Path(output_dir) / f"{clip_name}.npz"
    if output_path.exists():
        return "skipped"
    motion_beats, motion_mask, motion_dist, motion_spacing = extract_motion_beats_from_motion_pkl(
        motion_path,
        fps=fps,
        seq_len=seq_len,
        g1_motion_beat_source="fk",
        g1_fk_model_path=model_path,
        g1_root_quat_order="xyzw",
    )
    audio_beats = load_audio_beat_frames_spectral(wav_path, fps=fps, seq_len=seq_len)
    audio_mask = _beat_mask(audio_beats, seq_len)
    audio_dist = nearest_beat_distance(audio_beats, seq_len)
    audio_spacing = local_beat_spacing(audio_beats, seq_len)
    tmp_path = output_path.with_suffix(".npz.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            motion_beats=np.asarray(motion_beats, dtype=np.int64),
            motion_mask=np.asarray(motion_mask, dtype=np.float32),
            motion_dist=np.asarray(motion_dist, dtype=np.int64),
            motion_spacing=np.asarray(motion_spacing, dtype=np.float32),
            audio_beats=np.asarray(audio_beats, dtype=np.int64),
            audio_mask=np.asarray(audio_mask, dtype=np.float32),
            audio_dist=np.asarray(audio_dist, dtype=np.int64),
            audio_spacing=np.asarray(audio_spacing, dtype=np.float32),
        )
    tmp_path.replace(output_path)
    return "written"


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    motion_map = {path.stem: path for path in sorted(args.motion_dir.glob("*.pkl"))}
    wav_map = {path.stem: path for path in sorted(args.wav_dir.glob("*.wav"))}
    if set(motion_map) != set(wav_map):
        raise ValueError("motion and WAV basenames do not match")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = [
        (
            clip_name,
            str(motion_map[clip_name]),
            str(wav_map[clip_name]),
            str(args.output_dir),
            args.fps,
            args.seq_len,
            str(args.g1_fk_model_path),
        )
        for clip_name in sorted(motion_map)
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        counts = {"written": 0, "skipped": 0}
        for result in pool.imap_unordered(extract_one, items, chunksize=4):
            counts[result] += 1
            total = counts["written"] + counts["skipped"]
            if total % 1000 == 0:
                print(f"beat features {total}/{len(items)} written={counts['written']}", flush=True)
    print(counts)


if __name__ == "__main__":
    main()
