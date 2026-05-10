import pickle
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import numpy as np
import soundfile as sf

from eval.g1_kinematics import (
    build_g1_qpos,
    forward_g1_kinematics,
    load_g1_mujoco_model,
)


DEFAULT_G1_FK_MODEL_PATH = "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"
DEFAULT_ROOT_QUAT_ORDER = "xyzw"
DEFAULT_RENDER_BACKEND = "mujoco"
DEFAULT_RENDER_WIDTH = 960
DEFAULT_RENDER_HEIGHT = 720
DEFAULT_MUJOCO_GL = "egl"


def _as_float_array(value, name, ndim=None):
    if value is None:
        raise ValueError(f"G1 motion payload is missing {name}")
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} expected {ndim} dimensions, got {array.ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def normalize_g1_motion_payload(motion):
    if isinstance(motion, (str, Path)):
        with open(motion, "rb") as handle:
            motion = pickle.load(handle)
    if not isinstance(motion, dict):
        raise ValueError("G1 motion payload must be a dict or pickle path")

    root_pos = _as_float_array(motion.get("root_pos"), "root_pos", ndim=2)
    root_rot = _as_float_array(motion.get("root_rot"), "root_rot", ndim=2)
    dof_pos = _as_float_array(motion.get("dof_pos"), "dof_pos", ndim=2)
    if root_pos.shape[-1] != 3:
        raise ValueError(f"root_pos expected 3 channels, got {root_pos.shape[-1]}")
    if root_rot.shape[-1] != 4:
        raise ValueError(f"root_rot expected 4 channels, got {root_rot.shape[-1]}")
    if dof_pos.shape[-1] != 29:
        raise ValueError(f"dof_pos expected 29 channels, got {dof_pos.shape[-1]}")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError("G1 root_pos, root_rot, and dof_pos must have matching frames")

    return {
        "motion_format": "g1",
        "fps": float(motion.get("fps", 30.0)),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def compute_axes_limits(poses, min_span=1.6, padding=0.15):
    poses = _as_float_array(poses, "poses", ndim=3)
    if poses.shape[-1] != 3:
        raise ValueError(f"poses expected 3 channels, got {poses.shape[-1]}")

    flat = poses.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    span = max(span + 2.0 * float(padding), float(min_span))
    half = span / 2.0
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
        (float(center[2] - half), float(center[2] + half)),
    )


def select_g1_render_bodies(fk_result, drop_body_names=("world",)):
    bodies = _as_float_array(fk_result.get("bodies"), "bodies", ndim=3)
    body_names = list(fk_result.get("body_names") or [])
    body_parent_ids = np.asarray(fk_result.get("body_parent_ids"), dtype=np.int64)
    if len(body_names) != bodies.shape[1]:
        raise ValueError("G1 FK body_names must match bodies shape")
    if body_parent_ids.shape != (len(body_names),):
        raise ValueError("G1 FK body_parent_ids must match body_names")

    dropped = set(drop_body_names)
    keep_old_ids = [
        body_id
        for body_id, body_name in enumerate(body_names)
        if body_name not in dropped
    ]
    if not keep_old_ids:
        raise ValueError("G1 FK result has no renderable bodies")
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_old_ids)}

    parents = []
    for old_id in keep_old_ids:
        old_parent = int(body_parent_ids[old_id])
        parents.append(old_to_new.get(old_parent, -1))

    poses = bodies[:, keep_old_ids, :]
    names = [body_names[old_id] for old_id in keep_old_ids]
    return poses, np.asarray(parents, dtype=np.int64), names


def _set_line_data_3d(line, points):
    line.set_data(points[:, :2].T)
    line.set_3d_properties(points[:, 2])


def _plot_frame(frame_idx, poses, parents, lines, points):
    pose = poses[frame_idx]
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            continue
        segment = np.stack((pose[joint_idx], pose[parent_idx]), axis=0)
        _set_line_data_3d(lines[joint_idx], segment)
    points._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
    return [*lines, points]


def _render_gif(poses, parents, output_path, fps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    axes_limits = compute_axes_limits(poses)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    try:
        xlim, ylim, zlim = axes_limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-70)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        floor_x, floor_y = np.meshgrid(
            np.linspace(*xlim, 2),
            np.linspace(*ylim, 2),
        )
        floor_z = np.zeros_like(floor_x)
        ax.plot_surface(
            floor_x,
            floor_y,
            floor_z,
            color="lightgray",
            alpha=0.25,
            zorder=-10,
        )

        lines = [
            ax.plot([], [], [], color="black", linewidth=1.6, zorder=10)[0]
            for _ in parents
        ]
        points = ax.scatter([], [], [], color="tab:red", s=10, zorder=11)
        frame_rate = max(int(round(fps)), 1)
        anim = animation.FuncAnimation(
            fig,
            _plot_frame,
            frames=poses.shape[0],
            fargs=(poses, parents, lines, points),
            interval=1000 // frame_rate,
            blit=False,
        )
        anim.save(str(output_path), writer=animation.PillowWriter(fps=frame_rate))
    finally:
        plt.close(fig)


def _audio_path(name):
    if not name:
        raise ValueError("G1 rendering with sound requires an audio filename")
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".wav")
    if not path.is_file():
        raise FileNotFoundError(f"G1 render audio file not found: {path}")
    return path


def _stitched_audio_path(names, temp_dir):
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("Long G1 rendering with sound requires a non-empty audio filename list")

    audio_paths = [_audio_path(name) for name in names]
    audio, sample_rate = lr.load(str(audio_paths[0]), sr=None)
    first_length = len(audio)
    half = first_length // 2
    total_audio = np.zeros(first_length + half * (len(audio_paths) - 1), dtype=audio.dtype)
    total_audio[:first_length] = audio
    cursor = first_length
    for path in audio_paths[1:]:
        next_audio, next_sample_rate = lr.load(str(path), sr=None)
        if next_sample_rate != sample_rate:
            raise ValueError("Cannot stitch G1 render audio slices with different sample rates")
        total_audio[cursor : cursor + half] = next_audio[half : half + half]
        cursor += half

    output_path = Path(temp_dir) / "g1_stitched_audio.wav"
    sf.write(str(output_path), total_audio, sample_rate)
    return output_path


def _output_stem(epoch, num, name=None, stitch=False):
    if stitch:
        first_name = name[0] if isinstance(name, (list, tuple)) and name else ""
        stem = Path(first_name).stem if first_name else "sample"
        parts = stem.split("_")
        if len(parts) > 1:
            stem = "_".join(parts[:-1]) or stem
    else:
        stem = Path(name).stem if name else f"sample{num}"
    return f"{epoch}_{num}_{stem}_g1"


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
    except ImportError:
        return "ffmpeg"
    return imageio_ffmpeg.get_ffmpeg_exe()


def _mux_gif_with_audio(gif_path, audio_path, output_path):
    command = [
        _ffmpeg_exe(),
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(gif_path),
        "-i",
        str(audio_path),
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-q:a",
        "4",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _mux_video_with_audio(video_path, audio_path, output_path):
    command = [
        _ffmpeg_exe(),
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-q:a",
        "4",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _render_audio_path(name, stitch, temp_dir):
    return _stitched_audio_path(name, temp_dir) if stitch else _audio_path(name)


def _smooth_root_xy(root_pos, window=15):
    root_pos = _as_float_array(root_pos, "root_pos", ndim=2)
    if root_pos.shape[0] <= 1:
        return root_pos[:, :2].copy()
    window = min(int(window), root_pos.shape[0])
    if window <= 1:
        return root_pos[:, :2].copy()
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(root_pos[:, :2], ((pad, pad), (0, 0)), mode="edge")
    return np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(2)
        ],
        axis=-1,
    )


def _set_mujoco_gl_env(mujoco_gl):
    if not mujoco_gl:
        return
    if "mujoco" in sys.modules:
        return
    os.environ.setdefault("MUJOCO_GL", str(mujoco_gl))


def _mujoco_camera(mujoco):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 115
    camera.elevation = -20
    camera.lookat[:] = np.asarray([0.0, 0.0, 0.85], dtype=np.float64)
    return camera


def _write_video_frames(video_path, frames, fps, width, height):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ImportError(
            "Native G1 MuJoCo rendering requires imageio-ffmpeg. "
            "Install it in the EDGE environment before rendering."
        ) from exc

    frame_rate = max(int(round(float(fps))), 1)
    writer = imageio_ffmpeg.write_frames(
        str(video_path),
        size=(int(width), int(height)),
        fps=frame_rate,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _render_mujoco_video(
    motion,
    output_path,
    fps,
    model_path,
    root_quat_order,
    width,
    height,
    mujoco_gl,
):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"G1 MuJoCo model not found: {model_path}")
    _set_mujoco_gl_env(mujoco_gl)
    try:
        import mujoco

        model = load_g1_mujoco_model(model_path)
        model.vis.global_.offwidth = max(int(width), int(model.vis.global_.offwidth))
        model.vis.global_.offheight = max(int(height), int(model.vis.global_.offheight))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    except Exception as exc:
        raise RuntimeError(
            "Native G1 MuJoCo rendering failed to initialize. "
            f"Set a valid MUJOCO_GL backend, currently {os.environ.get('MUJOCO_GL')!r}, "
            "and ensure the G1 MJCF and OpenGL dependencies are available."
        ) from exc

    qpos = build_g1_qpos(
        motion["root_pos"],
        motion["root_rot"],
        motion["dof_pos"],
        root_quat_order=root_quat_order,
    )
    if qpos.shape[-1] != model.nq:
        renderer.close()
        raise ValueError(f"G1 qpos expected {model.nq} channels, got {qpos.shape[-1]}")

    lookat_xy = _smooth_root_xy(motion["root_pos"])
    camera = _mujoco_camera(mujoco)

    def frame_iterator():
        try:
            for frame_idx, frame_qpos in enumerate(qpos):
                data.qpos[:] = frame_qpos
                mujoco.mj_forward(model, data)
                camera.lookat[:] = np.asarray(
                    [lookat_xy[frame_idx, 0], lookat_xy[frame_idx, 1], 0.85],
                    dtype=np.float64,
                )
                renderer.update_scene(data, camera=camera)
                yield renderer.render()
        finally:
            renderer.close()

    frames = frame_iterator()
    try:
        _write_video_frames(output_path, frames, fps, width, height)
    finally:
        frames.close()


def render_g1_motion(
    motion,
    out,
    epoch=0,
    num=0,
    name=None,
    sound=True,
    stitch=False,
    model_path=DEFAULT_G1_FK_MODEL_PATH,
    root_quat_order=DEFAULT_ROOT_QUAT_ORDER,
    render_backend=DEFAULT_RENDER_BACKEND,
    width=DEFAULT_RENDER_WIDTH,
    height=DEFAULT_RENDER_HEIGHT,
    mujoco_gl=DEFAULT_MUJOCO_GL,
    output_name=None,
):
    motion = normalize_g1_motion_payload(motion)

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_name = name if output_name is None else output_name
    stem_uses_stitch_name = stitch or isinstance(stem_name, (list, tuple))
    output_stem = _output_stem(
        epoch,
        num,
        name=stem_name,
        stitch=stem_uses_stitch_name,
    )
    fps = motion["fps"]

    if render_backend not in ("mujoco", "stick"):
        raise ValueError(f"Unsupported G1 render backend: {render_backend}")

    if render_backend == "stick":
        fk_result = forward_g1_kinematics(
            motion,
            model_path=model_path,
            root_quat_order=root_quat_order,
        )
        poses, parents, _ = select_g1_render_bodies(fk_result)

        if not sound:
            gif_path = output_dir / f"{output_stem}.gif"
            _render_gif(poses, parents, gif_path, fps=fps)
            return str(gif_path)

        output_path = output_dir / f"{output_stem}.mp4"
        with TemporaryDirectory() as temp_dir:
            audio_path = _render_audio_path(name, stitch, temp_dir)
            gif_path = Path(temp_dir) / f"{output_stem}.gif"
            _render_gif(poses, parents, gif_path, fps=fps)
            _mux_gif_with_audio(gif_path, audio_path, output_path)
        return str(output_path)

    output_path = output_dir / f"{output_stem}.mp4"
    if not sound:
        _render_mujoco_video(
            motion,
            output_path,
            fps=fps,
            model_path=model_path,
            root_quat_order=root_quat_order,
            width=width,
            height=height,
            mujoco_gl=mujoco_gl,
        )
        return str(output_path)

    with TemporaryDirectory() as temp_dir:
        audio_path = _render_audio_path(name, stitch, temp_dir)
        video_path = Path(temp_dir) / f"{output_stem}.mp4"
        _render_mujoco_video(
            motion,
            video_path,
            fps=fps,
            model_path=model_path,
            root_quat_order=root_quat_order,
            width=width,
            height=height,
            mujoco_gl=mujoco_gl,
        )
        _mux_video_with_audio(video_path, audio_path, output_path)
    return str(output_path)
