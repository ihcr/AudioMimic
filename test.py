import glob
import json
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import librosa
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.audio_extraction.beat_features import nearest_beat_distance
from data.slice import slice_audio

EDGE = None
baseline_extract = None
juke_extract = None

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)
FPS = 30
SLICE_LENGTH_FRAMES = 150
SLICE_STRIDE_FRAMES = 75


def _load_edge():
    global EDGE
    if EDGE is None:
        from EDGE import EDGE as edge_cls

        EDGE = edge_cls
    return EDGE


def _load_baseline_extract():
    global baseline_extract
    if baseline_extract is None:
        from data.audio_extraction.baseline_features import extract as extract

        baseline_extract = extract
    return baseline_extract


def _load_jukebox_extract():
    global juke_extract
    if juke_extract is None:
        from data.audio_extraction.jukebox_features import extract as extract

        juke_extract = extract
    return juke_extract


def load_user_beat_frames(beat_file, target_fps=FPS):
    with open(beat_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    has_times = "beat_times_sec" in payload
    has_frames = "beat_frames" in payload
    if has_times == has_frames:
        raise ValueError("Beat JSON must contain exactly one of beat_times_sec or beat_frames")

    if has_times:
        frames = np.rint(np.asarray(payload["beat_times_sec"], dtype=np.float32) * target_fps)
    else:
        source_fps = int(payload.get("fps", target_fps))
        frames = np.asarray(payload["beat_frames"], dtype=np.float32)
        if source_fps != target_fps:
            frames = frames * (target_fps / source_fps)
    return np.unique(np.clip(np.rint(frames).astype(np.int64), 0, None))


def load_audio_beat_frames(wav_path, fps=FPS):
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    _, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    frames = np.rint(np.asarray(beat_times) * fps).astype(np.int64)
    return np.unique(np.clip(frames, 0, None))


def build_full_song_beat_track(beat_frames, total_frames, beat_rep):
    beat_frames = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
    if total_frames <= 0:
        if beat_rep == "pulse":
            return torch.zeros((0, 1), dtype=torch.float32)
        return torch.zeros(0, dtype=torch.int64)

    beat_frames = np.clip(beat_frames, 0, total_frames - 1)
    beat_frames = np.unique(beat_frames)
    if beat_rep == "distance":
        return torch.from_numpy(nearest_beat_distance(beat_frames, total_frames))
    if beat_rep == "pulse":
        mask = np.zeros(total_frames, dtype=np.float32)
        mask[beat_frames] = 1.0
        return torch.from_numpy(mask[:, None])
    raise ValueError(f"Unsupported beat representation: {beat_rep}")


def slice_beat_track(full_track, start_frame, horizon):
    chunk = full_track[start_frame : start_frame + horizon]
    if chunk.shape[0] >= horizon:
        return chunk

    pad_amount = horizon - chunk.shape[0]
    if full_track.ndim == 1:
        pad_value = int(chunk[-1].item()) if chunk.numel() > 0 else horizon
        return torch.cat(
            (chunk, torch.full((pad_amount,), pad_value, dtype=full_track.dtype)),
            dim=0,
        )

    return torch.cat(
        (chunk, torch.zeros((pad_amount, full_track.shape[1]), dtype=full_track.dtype)),
        dim=0,
    )


def build_beat_condition_slices(
    beat_source,
    beat_rep,
    wav_path,
    beat_file,
    total_slices,
    start_idx,
    num_slices,
    fps=FPS,
    horizon=SLICE_LENGTH_FRAMES,
    stride_frames=SLICE_STRIDE_FRAMES,
):
    if beat_source == "audio":
        beat_frames = load_audio_beat_frames(wav_path, fps=fps)
    elif beat_source == "user":
        if not beat_file:
            raise ValueError("--beat_file is required when --beat_source user")
        beat_frames = load_user_beat_frames(beat_file, target_fps=fps)
    else:
        raise ValueError("Beat source must be 'audio' or 'user' when beats are enabled")

    total_frames = max((total_slices - 1) * stride_frames + horizon, horizon)
    full_track = build_full_song_beat_track(beat_frames, total_frames, beat_rep)
    slices = [
        slice_beat_track(full_track, idx * stride_frames, horizon)
        for idx in range(start_idx, start_idx + num_slices)
    ]
    return torch.stack(slices, dim=0)


def resolve_cached_source_wav(cache_dir, music_dir):
    songname = Path(cache_dir).name
    wav_path = Path(music_dir) / f"{songname}.wav"
    if not wav_path.is_file():
        raise FileNotFoundError(
            f"Could not find original wav for cached feature directory {cache_dir!r} at {wav_path}"
        )
    return str(wav_path)


def set_inference_seed(seed):
    if seed is None or seed < 0:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def choose_slice_start(total_slices, sample_size, rng):
    max_start = max(total_slices - sample_size, 0)
    return rng.randint(0, max_start)


def test(opt):
    feature_func = (
        _load_jukebox_extract()
        if opt.feature_type == "jukebox"
        else _load_baseline_extract()
    )
    rng = set_inference_seed(getattr(opt, "seed", -1)) or random
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = choose_slice_start(len(file_list), sample_size, rng)
            slice_wavs = file_list[rand_idx : rand_idx + sample_size]
            slice_features = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in slice_features]
            music_cond = torch.from_numpy(np.array(cond_list)).float()
            if opt.use_beats:
                song_wav = resolve_cached_source_wav(dir.rstrip("/"), opt.music_dir)
                beat_cond = build_beat_condition_slices(
                    beat_source=opt.beat_source,
                    beat_rep=opt.beat_rep,
                    wav_path=song_wav,
                    beat_file=opt.beat_file,
                    total_slices=len(file_list),
                    start_idx=rand_idx,
                    num_slices=sample_size,
                )
                all_cond.append({"music": music_cond, "beat": beat_cond})
            else:
                all_cond.append(music_cond)
            all_filenames.append(slice_wavs)
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = choose_slice_start(len(file_list), sample_size, rng)
            cond_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(
                tqdm(
                    file_list,
                    total=len(file_list),
                    desc=f"Feature slices {os.path.basename(wav_file)}",
                    unit="slice",
                )
            ):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            music_cond = torch.from_numpy(np.array(cond_list)).float()
            if opt.use_beats:
                beat_cond = build_beat_condition_slices(
                    beat_source=opt.beat_source,
                    beat_rep=opt.beat_rep,
                    wav_path=wav_file,
                    beat_file=opt.beat_file,
                    total_slices=len(file_list),
                    start_idx=rand_idx,
                    num_slices=sample_size,
                )
                all_cond.append({"music": music_cond, "beat": beat_cond})
            else:
                all_cond.append(music_cond)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    if opt.use_beats and opt.beat_source == "none":
        raise ValueError("--beat_source must be set to audio or user when --use_beats is enabled")

    model = _load_edge()(
        opt.feature_type,
        opt.checkpoint,
        use_beats=opt.use_beats,
        beat_rep=opt.beat_rep,
        lambda_beat=0.0,
        motion_format=opt.motion_format,
    )
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple,
            "test",
            opt.render_dir,
            render_count=-1,
            fk_out=fk_out,
            render=not opt.no_render,
            g1_fk_model_path=opt.g1_fk_model_path,
            g1_root_quat_order=opt.g1_root_quat_order,
            g1_render_backend=opt.g1_render_backend,
            g1_render_width=opt.g1_render_width,
            g1_render_height=opt.g1_render_height,
            g1_mujoco_gl=opt.g1_mujoco_gl,
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
