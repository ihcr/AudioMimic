import glob
import os
import pickle
from pathlib import Path

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def _valid_motion_slice(path, expected_frames):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return False
    if set(payload) != {"pos", "q"}:
        return False
    pos = np.asarray(payload["pos"])
    q = np.asarray(payload["q"])
    return pos.shape == (expected_frames, 3) and q.shape == (expected_frames, 72)


def _valid_audio_slice(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return sf.info(path).frames > 0
    except Exception:
        return False


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        out_path = Path(out_dir) / f"{file_name}_slice{idx}.wav"
        if _valid_audio_slice(out_path):
            start_idx += stride_step
            idx += 1
            continue
        tmp_path = out_path.with_name(out_path.name + ".tmp.wav")
        sf.write(tmp_path, audio_slice, sr)
        tmp_path.replace(out_path)
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice}
        out_path = Path(out_dir) / f"{file_name}_slice{slice_count}.pkl"
        if _valid_motion_slice(out_path, expected_frames=window):
            start_idx += stride_step
            slice_count += 1
            continue
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump(out, handle, pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(out_path)
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    split_name = os.path.basename(motion_dir)
    for wav, motion in tqdm(
        zip(wavs, motions),
        total=len(wavs),
        desc=f"Slicing {split_name}",
        unit="clip",
    ):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs, total=len(wavs), desc="Slicing audio", unit="file"):
        audio_slices = slice_audio(wav, stride, length, wav_out)
