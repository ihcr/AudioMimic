#!/usr/bin/env python3
"""Generate beat variants and (optionally) run `test.py` to produce motions for ablation experiments.

Usage examples:

# Dry-run: generate beat jsons only
python scripts/run_beat_ablation.py --wav custom_music/song.wav --out_dir ablation_out

# Execute test.py for each variant (may be slow)
python scripts/run_beat_ablation.py --wav custom_music/song.wav --out_dir ablation_out --execute \
  --test_cmd "python test.py --music_dir custom_music/ --checkpoint runs/train/<run>/weights/train-2000.pt --feature_type baseline --use_beats --beat_rep distance --beat_source user --no_render --save_motions --motion_save_dir renders/ablation/{variant}"

"""
import argparse
import json
import os
import random
from pathlib import Path
from statistics import median

from data.audio_extraction.beat_features import load_audio_beat_frames

FPS = 30


def write_beat_json(frames, out_path, fps=FPS):
    payload = {"fps": fps, "beat_frames": [int(x) for x in frames.tolist()]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def shift_beats(frames, shift, seq_len=None):
    import numpy as np
    frames = np.asarray(frames, dtype=int)
    if seq_len is not None:
        frames = frames[frames < seq_len]
    frames = frames + int(shift)
    frames = frames[frames >= 0]
    return sorted(set(frames.tolist()))


def jitter_beats(frames, jitter_range=5):
    import numpy as np
    frames = np.asarray(frames, dtype=int)
    out = []
    for f in frames:
        out_f = int(f + random.randint(-jitter_range, jitter_range))
        if out_f >= 0:
            out.append(out_f)
    return sorted(set(out))


def constant_beats(frames, total_frames, fps=FPS):
    # build evenly spaced beats using median spacing
    import numpy as np
    if len(frames) < 2:
        # fallback to a simple periodic 1s spacing
        spacing = fps
    else:
        diffs = sorted([b2 - b1 for b1, b2 in zip(frames[:-1], frames[1:]) if b2>b1])
        spacing = int(median(diffs)) if diffs else fps
    beats = list(range(0, total_frames, spacing))
    return beats


def build_variants(wav_path, out_dir, seq_len=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_audio_beat_frames(wav_path, fps=FPS)
    total_frames = seq_len if seq_len is not None else max(frames.max() + 1 if len(frames) else 0, 150)

    variants = {}
    variants["real"] = sorted(set(frames.tolist()))
    variants["shift_plus5"] = shift_beats(frames, 5, seq_len=seq_len)
    variants["shift_plus10"] = shift_beats(frames, 10, seq_len=seq_len)
    variants["jitter"] = jitter_beats(frames, jitter_range=5)
    variants["random_sparse"] = sorted(random.sample(list(range(max(1, total_frames))), min( max(1, len(frames)//2), total_frames )))
    variants["constant"] = constant_beats(frames, total_frames)
    variants["none"] = []

    json_paths = {}
    for name, beats in variants.items():
        p = out_dir / f"beats_{name}.json"
        write_beat_json(frames=beats if isinstance(beats, list) else beats, out_path=p, fps=FPS)
        json_paths[name] = str(p)
    return json_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Source wav file")
    parser.add_argument("--out_dir", required=True, help="Where to write beat jsons and optionally outputs")
    parser.add_argument("--execute", action="store_true", help="If set, actually invoke the provided test command for each variant")
    parser.add_argument(
        "--test_cmd",
        type=str,
        default=None,
        help=(
            "Command template to run generation. Use {variant} and {beat_file} in the template. Example:\n"
            "python test.py --music_dir custom_music/ --checkpoint runs/train/<run>/weights/train-2000.pt --feature_type baseline --use_beats --beat_rep distance --beat_source user --beat_file {beat_file} --no_render --save_motions --motion_save_dir renders/ablation/{variant}"
        ),
    )
    parser.add_argument("--seq_len", type=int, default=150, help="(Optional) sequence length in frames to constrain beats")
    args = parser.parse_args()

    json_paths = build_variants(args.wav, args.out_dir, seq_len=args.seq_len)
    print("Written beat variants:")
    for k, v in json_paths.items():
        print(f" - {k}: {v}")

    if args.execute:
        if not args.test_cmd:
            raise SystemExit("--execute requires --test_cmd to be provided")
        import subprocess
        for variant, beat_file in json_paths.items():
            cmd = args.test_cmd.format(variant=variant, beat_file=beat_file)
            print(f"Running variant {variant}")
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
