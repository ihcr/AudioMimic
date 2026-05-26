import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.audio_extraction.beat_features import (detect_motion_beat_frames_from_pose as
                                                 _detect_motion_beat_frames_from_pose)
from data.audio_extraction.beat_features import load_audio_beat_frames as _load_audio_beat_frames
from data.audio_extraction.beat_features import mean_joint_speed_curve

FPS = 30


def detect_motion_beat_frames_from_pose(full_pose, min_gap=5):
    return _detect_motion_beat_frames_from_pose(full_pose, min_gap=min_gap)


def load_audio_beat_frames(wav_path, fps=FPS, seq_len=None):
    return _load_audio_beat_frames(wav_path, fps=fps, seq_len=seq_len)


def plot_velocity_vs_beats(motion_file, output_path=None):
    motion_file = Path(motion_file)
    with open(motion_file, "rb") as handle:
        payload = pickle.load(handle)

    full_pose = np.asarray(payload["full_pose"], dtype=np.float32)
    velocity = mean_joint_speed_curve(full_pose)
    motion_beats = detect_motion_beat_frames_from_pose(full_pose)
    designated_beats = np.asarray(
        payload.get("designated_beat_frames", np.zeros(0, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)

    audio_beats = np.zeros(0, dtype=np.int64)
    if payload.get("audio_path"):
        audio_beats = load_audio_beat_frames(
            payload["audio_path"], fps=FPS, seq_len=full_pose.shape[0]
        )

    output_path = Path(output_path) if output_path else motion_file.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    frames = np.arange(len(velocity))
    ax.plot(frames, velocity, color="black", linewidth=1.5, label="mean joint speed")

    def draw_markers(indices, color, label, linestyle="-"):
        if len(indices) == 0:
            return
        for frame_idx in indices:
            ax.axvline(frame_idx, color=color, linestyle=linestyle, alpha=0.55)
        ax.plot([], [], color=color, linestyle=linestyle, label=label)

    draw_markers(audio_beats, "tab:orange", "audio beats")
    draw_markers(motion_beats, "tab:blue", "motion beats")
    draw_markers(designated_beats, "tab:green", "designated beats", linestyle="--")

    ax.set_title(motion_file.stem)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean joint speed")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def parse_plot_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True, type=str)
    parser.add_argument("--output_path", default="", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_plot_opt()
    result = plot_velocity_vs_beats(
        opt.motion_file,
        output_path=opt.output_path or None,
    )
    print(result)
