import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FPS = 30
WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.5
HORIZON_FRAMES = int(WINDOW_SECONDS * FPS)
STRIDE_FRAMES = int(STRIDE_SECONDS * FPS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--feature_type", choices=("baseline", "jukebox"), default="jukebox")
    parser.add_argument("--data_path", default="data", type=str)
    parser.add_argument("--render_dir", default="eval/full_song_renders", type=str)
    parser.add_argument("--motion_save_dir", default="eval/full_song_motions", type=str)
    parser.add_argument("--slice_work_dir", default="", type=str)
    parser.add_argument("--metrics_path", default="eval/full_song_metrics.json", type=str)
    parser.add_argument("--edge_table_path", default="eval/full_song_edge_table.json", type=str)
    parser.add_argument("--beatit_table_path", default="eval/full_song_beatit_table.json", type=str)
    parser.add_argument("--paper_report_path", default="eval/full_song_report.md", type=str)
    parser.add_argument("--pfc_audit_path", default="eval/full_song_pfc_audit.json", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--beat_source", choices=("audio", "user"), default="audio")
    parser.add_argument("--beat_file", default="", type=str)
    parser.add_argument("--max_songs", default=0, type=int)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_full_song_wavs(data_path):
    wav_dir = Path(data_path) / "test" / "wavs"
    songs = sorted(wav_dir.glob("*.wav"))
    if not songs:
        raise FileNotFoundError(f"No full test-set wav files found in {wav_dir}")
    return songs


def slice_sort_key(path):
    stem = Path(path).stem
    if "_slice" not in stem:
        return stem, -1
    prefix, idx = stem.rsplit("_slice", 1)
    return prefix, int(idx)


def expected_long_frame_count(num_slices, horizon=HORIZON_FRAMES, stride_frames=STRIDE_FRAMES):
    if num_slices <= 0:
        return 0
    return horizon + stride_frames * (num_slices - 1)


def long_window_count(duration_seconds, window_seconds=WINDOW_SECONDS, stride_seconds=STRIDE_SECONDS):
    if duration_seconds <= 0:
        return 0
    if duration_seconds <= window_seconds:
        return 1
    return int(np.ceil((duration_seconds - window_seconds) / stride_seconds)) + 1


def slice_audio_for_long_generation(wav_path, out_dir):
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(wav_path, always_2d=False)
    if len(audio) == 0:
        raise RuntimeError(f"Empty audio file: {wav_path}")

    window = int(round(WINDOW_SECONDS * sr))
    stride = int(round(STRIDE_SECONDS * sr))
    num_windows = long_window_count(len(audio) / float(sr))
    output_paths = []
    for idx in range(num_windows):
        start = idx * stride
        audio_slice = audio[start : start + window]
        if len(audio_slice) < window:
            pad_shape = (window - len(audio_slice),) + audio_slice.shape[1:]
            audio_slice = np.concatenate(
                [audio_slice, np.zeros(pad_shape, dtype=audio_slice.dtype)],
                axis=0,
            )
        out_path = out_dir / f"{Path(wav_path).stem}_slice{idx}.wav"
        sf.write(out_path, audio_slice, sr)
        output_paths.append(out_path)
    return output_paths


def feature_extractor(feature_type):
    if feature_type == "jukebox":
        from data.audio_extraction.jukebox_features import extract as jukebox_extract

        return jukebox_extract
    from data.audio_extraction.baseline_features import extract as baseline_extract

    return baseline_extract


def prepare_song_features(wav_path, slice_dir, feature_type):
    import torch

    slice_audio_for_long_generation(wav_path, slice_dir)
    wav_slices = sorted(slice_dir.glob("*.wav"), key=slice_sort_key)
    if not wav_slices:
        raise RuntimeError(f"No 5-second windows were created for {wav_path}")

    extract = feature_extractor(feature_type)
    features = []
    for wav_slice in tqdm(
        wav_slices,
        desc=f"Features {wav_path.stem}",
        unit="slice",
        leave=False,
    ):
        feature_path = wav_slice.with_suffix(".npy")
        if feature_path.is_file():
            reps = np.load(feature_path)
        else:
            reps, _ = extract(str(wav_slice))
            np.save(feature_path, reps)
        features.append(reps)
    return wav_slices, torch.from_numpy(np.asarray(features)).float()


def build_condition(args, wav_path, wav_slices, music_cond):
    if not args.use_beats:
        return music_cond
    from test import build_beat_condition_slices

    beat_cond = build_beat_condition_slices(
        beat_source=args.beat_source,
        beat_rep=args.beat_rep,
        wav_path=str(wav_path),
        beat_file=args.beat_file,
        total_slices=len(wav_slices),
        start_idx=0,
        num_slices=len(wav_slices),
        fps=FPS,
        horizon=HORIZON_FRAMES,
        stride_frames=STRIDE_FRAMES,
    )
    return {"music": music_cond, "beat": beat_cond}


def render_full_song_motion(model, cond, wav_slices, wav_path, args):
    from model.diffusion import move_cond_to_device

    render_count = len(wav_slices)
    shape = (render_count, model.horizon, model.repr_dim)
    cond = move_cond_to_device(cond, model.accelerator.device)
    model.diffusion.render_sample(
        shape,
        cond,
        model.normalizer,
        "fullsong",
        args.render_dir,
        fk_out=args.motion_save_dir,
        name=[str(path) for path in wav_slices],
        sound=True,
        mode="long",
        render=args.render,
        metadata_audio_path=str(wav_path),
        metadata_stride_frames=STRIDE_FRAMES,
    )


def run_full_song_evaluation(args):
    from EDGE import EDGE
    from eval.run_benchmark_eval import run_benchmark_evaluation

    set_seed(args.seed)
    motion_dir = clear_dir(args.motion_save_dir)
    Path(args.render_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    slice_root = clear_dir(
        args.slice_work_dir
        or str(Path(args.motion_save_dir).parent / "full_song_slices")
    )

    songs = discover_full_song_wavs(args.data_path)
    if args.max_songs:
        songs = songs[: args.max_songs]

    model = EDGE(
        args.feature_type,
        args.checkpoint,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        lambda_beat=0.0,
    )
    model.eval()

    generated_frames = {}
    for wav_path in tqdm(songs, desc="Full-song eval", unit="song"):
        song_slice_dir = clear_dir(slice_root / wav_path.stem)
        wav_slices, music_cond = prepare_song_features(
            wav_path=wav_path,
            slice_dir=song_slice_dir,
            feature_type=args.feature_type,
        )
        cond = build_condition(args, wav_path, wav_slices, music_cond)
        render_full_song_motion(model, cond, wav_slices, wav_path, args)
        generated_frames[wav_path.stem] = expected_long_frame_count(len(wav_slices))

    return run_benchmark_evaluation(
        motion_path=str(motion_dir),
        metrics_path=args.metrics_path,
        edge_table_path=args.edge_table_path,
        beatit_table_path=args.beatit_table_path,
        paper_report_path=args.paper_report_path,
        pfc_audit_path=args.pfc_audit_path,
        method_name=Path(args.checkpoint).stem,
        feature_type=args.feature_type,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        reference_motion_path=str(Path(args.data_path) / "test" / "motions"),
        seed=args.seed,
        checkpoint=args.checkpoint,
        motion_source="generated",
        extra_metadata={
            "eval_mode": "full_song",
            "num_full_songs": len(songs),
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "generated_frames": generated_frames,
        },
    )


if __name__ == "__main__":
    run_full_song_evaluation(parse_args())
