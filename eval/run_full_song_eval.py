import argparse
import random
import re
import shutil
import sys
import wave
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
    parser.add_argument("--motion_format", choices=("smpl", "g1"), default="smpl")
    parser.add_argument("--data_path", default="data", type=str)
    parser.add_argument("--full_wav_dir", default="", type=str)
    parser.add_argument("--full_music_dir", default="", type=str)
    parser.add_argument("--render_dir", default="eval/full_song_renders", type=str)
    parser.add_argument("--motion_save_dir", default="eval/full_song_motions", type=str)
    parser.add_argument("--slice_work_dir", default="", type=str)
    parser.add_argument("--use_precomputed_test_slices", action="store_true")
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
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument(
        "--g1_root_quat_order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
    )
    parser.add_argument(
        "--g1_render_backend",
        choices=("mujoco", "stick"),
        default="mujoco",
    )
    parser.add_argument("--g1_render_width", default=960, type=int)
    parser.add_argument("--g1_render_height", default=720, type=int)
    parser.add_argument("--g1_mujoco_gl", default="egl", type=str)
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


def discover_full_song_wavs(data_path, full_wav_dir=""):
    wav_dir = Path(full_wav_dir) if full_wav_dir else Path(data_path) / "test" / "wavs"
    songs = sorted(wav_dir.glob("*.wav"))
    if not songs:
        raise FileNotFoundError(f"No full test-set wav files found in {wav_dir}")
    return songs


def music_id_from_motion_stem(stem):
    match = re.search(r"_m([A-Z0-9]+)_", stem)
    if not match:
        raise ValueError(f"Cannot infer AIST music id from motion name: {stem}")
    return f"m{match.group(1)}"


def find_music_file(music_dir, music_id):
    music_dir = Path(music_dir)
    for suffix in (".wav", ".mp3", ".flac", ".ogg"):
        candidate = music_dir / f"{music_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No full music file found for {music_id} in {music_dir}")


def discover_full_song_inputs(data_path, full_wav_dir="", full_music_dir=""):
    if not full_music_dir:
        return [(path.stem, path) for path in discover_full_song_wavs(data_path, full_wav_dir)]

    motion_dir = Path(data_path) / "test" / "motions"
    motions = sorted(motion_dir.glob("*.pkl"))
    if not motions:
        raise FileNotFoundError(f"No test motion files found in {motion_dir}")
    return [
        (motion_path.stem, find_music_file(full_music_dir, music_id_from_motion_stem(motion_path.stem)))
        for motion_path in motions
    ]


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


def wav_duration_seconds(wav_path):
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def audio_duration_seconds(audio_path):
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() == ".wav":
        return wav_duration_seconds(audio_path)
    import soundfile as sf

    info = sf.info(str(audio_path))
    return info.frames / float(info.samplerate)


def song_frame_count(audio_path, fps=FPS):
    return int(audio_duration_seconds(audio_path) * fps)


def slice_audio_for_long_generation(wav_path, out_dir, output_stem=None):
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(wav_path, always_2d=False)
    if len(audio) == 0:
        raise RuntimeError(f"Empty audio file: {wav_path}")

    window = int(round(WINDOW_SECONDS * sr))
    stride = int(round(STRIDE_SECONDS * sr))
    num_windows = long_window_count(len(audio) / float(sr))
    prefix = output_stem or Path(wav_path).stem
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
        out_path = out_dir / f"{prefix}_slice{idx}.wav"
        sf.write(out_path, audio_slice, sr)
        output_paths.append(out_path)
    return output_paths


def feature_extractor(feature_type):
    if feature_type == "jukebox":
        from data.audio_extraction.jukebox_features import extract as jukebox_extract

        return jukebox_extract
    from data.audio_extraction.baseline_features import extract as baseline_extract

    return baseline_extract


def _feature_dir_name(feature_type):
    return "jukebox_feats" if feature_type == "jukebox" else "baseline_feats"


def prepare_song_features(
    wav_path,
    slice_dir,
    feature_type,
    precomputed_data_path=None,
    output_stem=None,
):
    import torch

    output_stem = output_stem or Path(wav_path).stem
    if precomputed_data_path is not None:
        data_root = Path(precomputed_data_path)
        wav_slices = sorted(
            (data_root / "test" / "wavs_sliced").glob(f"{output_stem}_slice*.wav"),
            key=slice_sort_key,
        )
        if not wav_slices:
            raise RuntimeError(f"No precomputed test slices found for {wav_path}")
        expected_slices = long_window_count(audio_duration_seconds(wav_path))
        if len(wav_slices) < expected_slices:
            raise RuntimeError(
                f"Only found {len(wav_slices)} precomputed slices for {wav_path}; "
                f"expected {expected_slices} from the full-song audio duration"
            )
        wav_slices = wav_slices[:expected_slices]
        feature_dir = data_root / "test" / _feature_dir_name(feature_type)
        features = []
        for wav_slice in wav_slices:
            feature_path = feature_dir / f"{wav_slice.stem}.npy"
            if not feature_path.is_file():
                raise FileNotFoundError(f"Missing precomputed feature file: {feature_path}")
            features.append(np.load(feature_path))
        return wav_slices, torch.from_numpy(np.asarray(features)).float()

    slice_audio_for_long_generation(wav_path, slice_dir, output_stem=output_stem)
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
        metadata_total_frames=song_frame_count(wav_path),
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        g1_render_backend=args.g1_render_backend,
        g1_render_width=args.g1_render_width,
        g1_render_height=args.g1_render_height,
        g1_mujoco_gl=args.g1_mujoco_gl,
    )


def run_full_song_evaluation(args):
    from EDGE import EDGE
    from eval.run_benchmark_eval import run_benchmark_evaluation

    set_seed(args.seed)
    motion_dir = clear_dir(args.motion_save_dir)
    clear_dir(args.render_dir)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    slice_root = clear_dir(
        args.slice_work_dir
        or str(Path(args.motion_save_dir).parent / "full_song_slices")
    )

    songs = discover_full_song_inputs(
        args.data_path,
        full_wav_dir=args.full_wav_dir,
        full_music_dir=args.full_music_dir,
    )
    if args.max_songs:
        songs = songs[: args.max_songs]
    if args.full_music_dir and args.use_precomputed_test_slices:
        raise ValueError(
            "--full_music_dir uses raw full-length music; do not combine it with "
            "--use_precomputed_test_slices from the short choreography wav tree."
        )

    model = EDGE(
        args.feature_type,
        args.checkpoint,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        lambda_beat=0.0,
        motion_format=args.motion_format,
    )
    model.eval()

    generated_frames = {}
    for output_stem, wav_path in tqdm(songs, desc="Full-song eval", unit="song"):
        song_slice_dir = clear_dir(slice_root / output_stem)
        wav_slices, music_cond = prepare_song_features(
            wav_path=wav_path,
            slice_dir=song_slice_dir,
            feature_type=args.feature_type,
            precomputed_data_path=args.data_path if args.use_precomputed_test_slices else None,
            output_stem=output_stem,
        )
        cond = build_condition(args, wav_path, wav_slices, music_cond)
        render_full_song_motion(model, cond, wav_slices, wav_path, args)
        generated_frames[output_stem] = song_frame_count(wav_path)

    if args.motion_format == "g1":
        from eval.g1_metrics import run_g1_motion_evaluation

        metrics = run_g1_motion_evaluation(
            motion_path=motion_dir,
            reference_motion_path=Path(args.data_path) / "test" / "motions",
            metrics_path=args.metrics_path,
            g1_table_path=args.edge_table_path,
            motion_audit_path=args.pfc_audit_path,
            paper_report_path=args.paper_report_path,
            render_dir=args.render_dir,
            diagnostic_count=0,
            checkpoint=args.checkpoint,
            feature_type=args.feature_type,
            use_beats=args.use_beats,
            beat_rep=args.beat_rep,
            seed=args.seed,
            enable_fk_metrics=False,
            fk_model_path=args.g1_fk_model_path,
            root_quat_order=args.g1_root_quat_order,
        )
        metrics.update(
            {
                "eval_mode": "full_song",
                "num_full_songs": len(songs),
                "window_seconds": WINDOW_SECONDS,
                "stride_seconds": STRIDE_SECONDS,
                "generated_frames": generated_frames,
            }
        )
        from eval.benchmark_report import write_json

        write_json(args.metrics_path, metrics)
        return metrics

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
