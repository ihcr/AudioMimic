import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.audio_extraction.beat_features import load_audio_beat_frames as _load_audio_beat_frames
from eval.eval_pfc import load_full_pose

FPS = 30
DEFAULT_BAS_SIGMA_SQUARED = 9.0
DEFAULT_BAP_TOLERANCE = 2


def mean_joint_speed_curve(full_pose):
    full_pose = np.asarray(full_pose, dtype=np.float32)
    if full_pose.ndim != 3 or full_pose.shape[-1] != 3:
        raise ValueError("full_pose must have shape [T, J, 3]")
    if full_pose.shape[0] <= 1:
        return np.zeros(0, dtype=np.float32)
    return np.mean(
        np.sqrt(np.sum((full_pose[1:] - full_pose[:-1]) ** 2, axis=2)),
        axis=1,
    )


def detect_motion_beat_frames_from_pose(full_pose):
    velocity = mean_joint_speed_curve(full_pose)
    if velocity.size == 0:
        return np.zeros(0, dtype=np.int64)
    smoothed = gaussian_filter1d(velocity, sigma=5)
    beats = argrelextrema(smoothed, np.less)[0]
    return np.asarray(beats, dtype=np.int64)


def load_audio_beat_frames_from_features(audio_path, seq_len=None):
    audio_path = Path(audio_path)
    beat_feat_path = Path(str(audio_path).replace("wavs_sliced", "beat_feats")).with_suffix(".npz")
    if beat_feat_path.is_file():
        payload = np.load(beat_feat_path)
        if "audio_beats" in payload:
            beats = np.asarray(payload["audio_beats"], dtype=np.int64).reshape(-1)
            if seq_len is not None:
                beats = beats[beats < seq_len]
            return beats
        if "audio_mask" in payload:
            mask = np.asarray(payload["audio_mask"], dtype=np.float32).reshape(-1)
            if seq_len is not None:
                mask = mask[:seq_len]
            return np.flatnonzero(mask > 0.5).astype(np.int64)

    baseline_feat_path = Path(str(audio_path).replace("wavs_sliced", "baseline_feats")).with_suffix(".npy")
    if baseline_feat_path.is_file():
        feature = np.load(baseline_feat_path)
        channel = feature[:, 53]
        if seq_len is not None:
            channel = channel[:seq_len]
        return np.flatnonzero(channel.astype(bool)).astype(np.int64)

    return None


def load_audio_beat_frames(wav_path, fps=FPS, seq_len=None):
    feature_beats = load_audio_beat_frames_from_features(wav_path, seq_len=seq_len)
    if feature_beats is not None:
        return feature_beats
    return _load_audio_beat_frames(wav_path, fps=fps, seq_len=seq_len)


def compute_bas_score(music_beats, motion_beats, sigma_squared=DEFAULT_BAS_SIGMA_SQUARED):
    music_beats = np.asarray(music_beats, dtype=np.int64).reshape(-1)
    motion_beats = np.asarray(motion_beats, dtype=np.int64).reshape(-1)
    if music_beats.size == 0 or motion_beats.size == 0:
        return 0.0

    total = 0.0
    for beat in music_beats:
        total += np.exp(-np.min((motion_beats - beat) ** 2) / (2.0 * sigma_squared))
    return float(total / len(music_beats))


def greedy_match_count(generated_beats, designated_beats, tolerance=DEFAULT_BAP_TOLERANCE):
    generated_beats = np.sort(np.asarray(generated_beats, dtype=np.int64).reshape(-1))
    designated_beats = np.sort(np.asarray(designated_beats, dtype=np.int64).reshape(-1))
    generated_idx = 0
    designated_idx = 0
    matched = 0

    while generated_idx < len(generated_beats) and designated_idx < len(designated_beats):
        delta = generated_beats[generated_idx] - designated_beats[designated_idx]
        if abs(delta) <= tolerance:
            matched += 1
            generated_idx += 1
            designated_idx += 1
        elif generated_beats[generated_idx] < designated_beats[designated_idx] - tolerance:
            generated_idx += 1
        else:
            designated_idx += 1
    return matched


def evaluate_motion_file(path, bas_sigma_squared=DEFAULT_BAS_SIGMA_SQUARED, bap_tolerance=DEFAULT_BAP_TOLERANCE):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    audio_path = payload.get("audio_path")
    if not audio_path:
        return None

    full_pose = load_full_pose(payload)
    motion_beats = detect_motion_beat_frames_from_pose(full_pose)
    music_beats = load_audio_beat_frames(audio_path, fps=FPS, seq_len=full_pose.shape[0])

    result = {
        "path": path,
        "BAS": compute_bas_score(
            music_beats=music_beats,
            motion_beats=motion_beats,
            sigma_squared=bas_sigma_squared,
        ),
        "num_generated_beats": int(len(motion_beats)),
        "num_audio_beats": int(len(music_beats)),
    }

    designated_beats = payload.get("designated_beat_frames")
    if designated_beats is not None:
        designated_beats = np.asarray(designated_beats, dtype=np.int64).reshape(-1)
        matched = greedy_match_count(
            motion_beats, designated_beats, tolerance=bap_tolerance
        )
        result.update(
            {
                "matched_designated_beats": int(matched),
                "num_designated_beats": int(len(designated_beats)),
            }
        )
    return result


def evaluate_motion_dir(motion_path, bas_sigma_squared=DEFAULT_BAS_SIGMA_SQUARED, bap_tolerance=DEFAULT_BAP_TOLERANCE):
    bas_scores = []
    scored_files = 0
    skipped_files = 0
    generated_beats = 0
    designated_beats = 0
    matched_designated = 0
    eligible_bap_files = 0

    for motion_file in tqdm(
        sorted(glob.glob(str(Path(motion_path) / "*.pkl"))),
        desc="BAS/BAP",
        unit="file",
    ):
        result = evaluate_motion_file(
            motion_file, bas_sigma_squared=bas_sigma_squared, bap_tolerance=bap_tolerance
        )
        if result is None:
            skipped_files += 1
            continue

        bas_scores.append(result["BAS"])
        scored_files += 1

        if "num_designated_beats" in result:
            eligible_bap_files += 1
            generated_beats += result["num_generated_beats"]
            designated_beats += result["num_designated_beats"]
            matched_designated += result["matched_designated_beats"]

    aggregate = {
        "BAS": float(np.mean(bas_scores)) if bas_scores else float("nan"),
        "BAP": (
            matched_designated / max(generated_beats, 1) if eligible_bap_files else float("nan")
        ),
        "BAP_precision": (
            matched_designated / max(generated_beats, 1) if eligible_bap_files else float("nan")
        ),
        "BAP_recall": (
            matched_designated / max(designated_beats, 1) if eligible_bap_files else float("nan")
        ),
        "num_scored_files": scored_files,
        "num_skipped_files": skipped_files,
        "num_generated_beats": generated_beats,
        "num_designated_beats": designated_beats,
    }
    return aggregate


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", type=str, default="eval/motions")
    parser.add_argument("--bas_sigma_squared", type=float, default=DEFAULT_BAS_SIGMA_SQUARED)
    parser.add_argument("--bap_tolerance", type=int, default=DEFAULT_BAP_TOLERANCE)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_eval_opt()
    result = evaluate_motion_dir(
        opt.motion_path,
        bas_sigma_squared=opt.bas_sigma_squared,
        bap_tolerance=opt.bap_tolerance,
    )
    print(f"BAS: {result['BAS']}")
    print(f"BAP_precision: {result['BAP_precision']}")
    print(f"BAP_recall: {result['BAP_recall']}")
    print(f"num_scored_files: {result['num_scored_files']}")
    print(f"num_skipped_files: {result['num_skipped_files']}")
