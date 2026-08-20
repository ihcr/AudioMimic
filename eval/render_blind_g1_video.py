"""Render a label-free G1 reference or SONIC execution with a fixed camera."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_motion_music_execution import load_execution_pair
from eval.g1_kinematics import build_g1_qpos, load_g1_mujoco_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion_pkl", type=Path)
    source.add_argument("--tracking_run", type=Path)
    parser.add_argument(
        "--tracking_representation",
        choices=("target", "execution"),
        default="execution",
    )
    parser.add_argument("--output_mp4", type=Path, required=True)
    parser.add_argument("--audio_wav", type=Path)
    parser.add_argument("--audio_start_seconds", type=float, default=0.0)
    parser.add_argument("--start_seconds", type=float, default=4.0)
    parser.add_argument("--duration_seconds", type=float, default=16.0)
    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--tracking_analysis_fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--distance", type=float, default=3.0)
    parser.add_argument(
        "--model_xml",
        type=Path,
        default=REPO_ROOT / "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    return parser.parse_args()


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0.0:
            values[index] *= -1.0
    return values


def _xyzw(values: np.ndarray, order: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if order == "xyzw":
        return values
    if order == "wxyz":
        return values[:, [1, 2, 3, 0]]
    raise ValueError(f"unsupported quaternion order: {order}")


def load_source(args: argparse.Namespace) -> tuple[dict, str]:
    if args.motion_pkl is not None:
        path = args.motion_pkl.expanduser().resolve()
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        motion = {
            "fps": float(payload.get("fps", 30.0)),
            "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
            "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
            "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
        }
        quat_order = "xyzw"
    else:
        target, execution, _ = load_execution_pair(
            args.tracking_run.expanduser().resolve(), float(args.tracking_analysis_fps)
        )
        motion = target if args.tracking_representation == "target" else execution
        quat_order = "wxyz"
    frames = len(motion["dof_pos"])
    expected = {
        "root_pos": (frames, 3),
        "root_rot": (frames, 4),
        "dof_pos": (frames, 29),
    }
    for name, shape in expected.items():
        if np.asarray(motion[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {np.asarray(motion[name]).shape}")
        if not np.isfinite(motion[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    return motion, quat_order


def resample_window(
    motion: dict,
    *,
    quat_order: str,
    start_seconds: float,
    duration_seconds: float,
    output_fps: float,
) -> dict:
    if start_seconds < 0 or duration_seconds <= 0 or output_fps <= 0:
        raise ValueError("start must be non-negative; duration and output_fps must be positive")
    source_fps = float(motion["fps"])
    source_frames = len(motion["dof_pos"])
    source_t = np.arange(source_frames, dtype=np.float64) / source_fps
    output_frames = int(round(duration_seconds * output_fps))
    target_t = start_seconds + np.arange(output_frames, dtype=np.float64) / output_fps
    if not len(target_t) or target_t[-1] > source_t[-1] + 1e-8:
        raise ValueError(
            f"requested window [{start_seconds:.3f}, {start_seconds + duration_seconds:.3f}) "
            f"exceeds source duration {source_frames / source_fps:.3f} s"
        )

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(target_t, source_t, values[:, index]) for index in range(values.shape[1])],
            axis=1,
        )

    source_quat = _normalize_quaternions(_xyzw(motion["root_rot"], quat_order))
    rotations = Slerp(source_t, Rotation.from_quat(source_quat))(target_t).as_quat()
    return {
        "fps": float(output_fps),
        "root_pos": interpolate(np.asarray(motion["root_pos"], dtype=np.float64)),
        "root_rot": rotations,
        "dof_pos": interpolate(np.asarray(motion["dof_pos"], dtype=np.float64)),
    }


def _render_silent(args: argparse.Namespace, motion: dict, output_path: Path) -> None:
    import mujoco

    model = load_g1_mujoco_model(args.model_xml.expanduser().resolve())
    model.vis.global_.offwidth = max(int(args.width), int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(int(args.height), int(model.vis.global_.offheight))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width))
    qpos = build_g1_qpos(
        motion["root_pos"], motion["root_rot"], motion["dof_pos"], root_quat_order="xyzw"
    )
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        f"{motion['fps']:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = float(args.azimuth)
    camera.elevation = float(args.elevation)
    camera.distance = float(args.distance)
    try:
        for index, frame_qpos in enumerate(qpos, start=1):
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.body("pelvis").xpos
            renderer.update_scene(data, camera=camera)
            assert writer.stdin is not None
            writer.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
            if index % 300 == 0:
                print(f"Rendered {index}/{len(qpos)} frames", flush=True)
    finally:
        renderer.close()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg renderer exited with status {return_code}")


def _mux_audio(args: argparse.Namespace, silent_path: Path, output_path: Path) -> None:
    audio_start = float(args.audio_start_seconds) + float(args.start_seconds)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent_path),
        "-ss",
        f"{audio_start:.6f}",
        "-t",
        f"{float(args.duration_seconds):.6f}",
        "-i",
        str(args.audio_wav.expanduser().resolve()),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def render(args: argparse.Namespace) -> None:
    motion, quat_order = load_source(args)
    window = resample_window(
        motion,
        quat_order=quat_order,
        start_seconds=float(args.start_seconds),
        duration_seconds=float(args.duration_seconds),
        output_fps=float(args.output_fps),
    )
    output_path = args.output_mp4.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(f".{output_path.stem}.silent.mp4")
    try:
        _render_silent(args, window, silent_path)
        if args.audio_wav is None:
            silent_path.replace(output_path)
        else:
            _mux_audio(args, silent_path, output_path)
    finally:
        silent_path.unlink(missing_ok=True)
    print(f"Rendered blinded clip to {output_path}")


if __name__ == "__main__":
    render(parse_args())
