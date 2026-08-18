"""Render an exported G1 reference trajectory with the SONIC front camera.

This is a kinematic MuJoCo render of the generator reference: it does not
start SONIC or simulate tracking dynamics.  The camera parameters match the
offscreen third-person camera used by the SONIC Sim2Sim bridge.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

# Must be chosen before MuJoCo is imported by the kinematics helpers.
os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.g1_kinematics import build_g1_qpos, load_g1_mujoco_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion_pkl")
    source.add_argument(
        "--motion_npy_yaw_delta",
        help="raw [T,34] g1_yaw_delta array emitted by the online runtime",
    )
    parser.add_argument("--output_mp4", required=True)
    parser.add_argument(
        "--model_xml",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--fps", default=30.0, type=float)
    parser.add_argument("--azimuth", default=180.0, type=float)
    parser.add_argument("--elevation", default=-12.0, type=float)
    parser.add_argument("--distance", default=3.0, type=float)
    parser.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="0 renders all frames; otherwise render this prefix for diagnostics",
    )
    parser.add_argument(
        "--start_frame",
        default=0,
        type=int,
        help="source frame at which to start rendering",
    )
    parser.add_argument(
        "--frame_stride",
        default=1,
        type=int,
        help="render every Nth source frame; output fps is adjusted to preserve duration",
    )
    return parser.parse_args()


def _load_motion(path: Path) -> dict[str, np.ndarray | float]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = ("root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing motion fields {missing}")
    motion = {
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float32),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float32),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float32),
        "fps": float(payload.get("fps", 30.0)),
    }
    frames = motion["root_pos"].shape[0]
    if (
        motion["root_pos"].shape != (frames, 3)
        or motion["root_rot"].shape != (frames, 4)
        or motion["dof_pos"].shape != (frames, 29)
    ):
        raise ValueError("motion must contain root_pos[T,3], root_rot[T,4], dof_pos[T,29]")
    if not all(np.isfinite(value).all() for key, value in motion.items() if key != "fps"):
        raise ValueError("motion contains non-finite values")
    return motion


def _load_yaw_delta_motion(path: Path, fps: float) -> dict[str, np.ndarray | float]:
    raw = np.asarray(np.load(path), dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 34 or not np.isfinite(raw).all():
        raise ValueError(f"{path}: online motion must be finite [T,34], got {raw.shape}")
    frames = raw.shape[0]
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    current_xy = np.zeros(2, dtype=np.float32)
    current_yaw = 0.0
    for frame in range(frames):
        if frame > 0:
            cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)
            local_x, local_y = raw[frame, :2]
            current_xy += (
                cos_yaw * local_x - sin_yaw * local_y,
                sin_yaw * local_x + cos_yaw * local_y,
            )
            current_yaw += float(np.arctan2(raw[frame, 3], raw[frame, 4]))
        root_pos[frame] = (current_xy[0], current_xy[1], raw[frame, 2])
        root_rot[frame] = (0.0, 0.0, np.sin(current_yaw / 2.0), np.cos(current_yaw / 2.0))
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": raw[:, 5:],
        "fps": float(fps),
    }


def render(args: argparse.Namespace) -> None:
    import mujoco

    output_path = Path(args.output_mp4).expanduser().resolve()
    if args.motion_pkl:
        motion = _load_motion(Path(args.motion_pkl).expanduser().resolve())
    else:
        motion = _load_yaw_delta_motion(
            Path(args.motion_npy_yaw_delta).expanduser().resolve(),
            float(args.fps),
        )
    source_fps = float(motion["fps"])
    if not np.isclose(source_fps, float(args.fps), rtol=0.0, atol=1e-4):
        raise ValueError(f"motion fps {source_fps:g} must match render fps {args.fps:g}")

    model = load_g1_mujoco_model(Path(args.model_xml))
    model.vis.global_.offwidth = max(int(args.width), int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(int(args.height), int(model.vis.global_.offheight))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width))
    qpos = build_g1_qpos(
        motion["root_pos"], motion["root_rot"], motion["dof_pos"], root_quat_order="xyzw"
    )
    if args.start_frame < 0 or args.max_frames < 0 or args.frame_stride <= 0:
        renderer.close()
        raise ValueError("start_frame/max_frames must be non-negative and frame_stride positive")
    start_frame = int(args.start_frame)
    if start_frame >= len(qpos):
        renderer.close()
        raise ValueError(f"start_frame {start_frame} exceeds {len(qpos)} source frames")
    stop_frame = len(qpos) if args.max_frames == 0 else start_frame + int(args.max_frames)
    qpos = qpos[start_frame:stop_frame:int(args.frame_stride)]
    render_fps = source_fps / int(args.frame_stride)
    if qpos.shape[1] != model.nq:
        renderer.close()
        raise ValueError(f"motion has {qpos.shape[1]} qpos values but model expects {model.nq}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-video_size", f"{args.width}x{args.height}",
        "-framerate", f"{render_fps:g}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = float(args.azimuth)
    camera.elevation = float(args.elevation)
    camera.distance = float(args.distance)
    try:
        for frame_index, frame_qpos in enumerate(qpos, start=1):
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.body("pelvis").xpos
            renderer.update_scene(data, camera=camera)
            assert writer.stdin is not None
            writer.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
            if frame_index % 300 == 0:
                print(f"Rendered {frame_index}/{len(qpos)} frames", flush=True)
    finally:
        renderer.close()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"Rendered {len(qpos)} reference frames to {output_path}")


if __name__ == "__main__":
    render(parse_args())
