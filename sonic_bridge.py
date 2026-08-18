"""Transport helpers between exported G1 motion and the SONIC ZMQ deploy API.

The motion generator stays independent of SONIC.  This module owns only the
wire-format conversion and the execution-state boundary used by a later
plan-commit-replan runtime.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HEADER_SIZE = 1280
SONIC_INPUT_PROTOCOL_VERSION = 1
G1_DOF_DIM = 29
G1_FPS = 30.0
G1_COMMIT_FRAMES = 8
SONIC_REFERENCE_FPS = 50.0

# The V6f-X export uses MuJoCo/URDF order.  SONIC's streamed-motion policy
# consumes its own IsaacLab order.  This is the exact gather order used by
# LocalMotionKPlanner::ResampleGeneratedSequence50Hz in SONIC deploy.
SONIC_REFERENCE_FROM_MUJOCO = np.asarray(
    (
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ),
    dtype=np.int64,
)

# Unitree G1 29-DoF limits in the same MuJoCo order as the V6f-X export.
G1_MUJOCO_JOINT_LOWER = np.asarray(
    (
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.618, -0.52, -0.52,
        -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
        -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
    ),
    dtype=np.float32,
)
G1_MUJOCO_JOINT_UPPER = np.asarray(
    (
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
        2.618, 0.52, 0.52,
        2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
        2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
    ),
    dtype=np.float32,
)

class SonicProtocolError(ValueError):
    """Raised when a payload cannot satisfy the SONIC reference contract."""


class SonicExecutionStateError(ValueError):
    """Raised when SONIC feedback cannot form a physical S66 boundary state."""


def _as_float_array(value: Any, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < len(shape_tail) or tuple(array.shape[-len(shape_tail) :]) != shape_tail:
        raise SonicProtocolError(f"{name} must end in {shape_tail}, got {array.shape}")
    if not np.isfinite(array).all():
        raise SonicProtocolError(f"{name} contains non-finite values")
    return array


def _wire_dtype(array: np.ndarray) -> str:
    if array.dtype == np.float32:
        return "f32"
    if array.dtype == np.float64:
        return "f64"
    if array.dtype == np.int32:
        return "i32"
    if array.dtype == np.int64:
        return "i64"
    if array.dtype == np.uint8:
        return "u8"
    if array.dtype == np.bool_:
        return "bool"
    raise SonicProtocolError(f"unsupported SONIC wire dtype: {array.dtype}")


def pack_zmq_message(
    fields: Mapping[str, np.ndarray],
    *,
    topic: str = "pose",
    version: int = SONIC_INPUT_PROTOCOL_VERSION,
) -> bytes:
    """Pack a single SONIC ZMQ message with its fixed-size JSON header."""
    field_metadata = []
    binary_parts = []
    count = None
    for name, value in fields.items():
        array = np.ascontiguousarray(value)
        if array.ndim == 0:
            raise SonicProtocolError(f"{name} must have at least one dimension")
        if count is None or (count == 1 and int(array.shape[0]) > 1):
            count = int(array.shape[0])
        elif int(array.shape[0]) not in (1, count):
            raise SonicProtocolError(
                "SONIC fields must have either the packet frame count or one value"
            )
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        field_metadata.append(
            {"name": str(name), "dtype": _wire_dtype(array), "shape": list(array.shape)}
        )
        binary_parts.append(array.tobytes())
    if count is None or count <= 0:
        raise SonicProtocolError("SONIC message requires at least one frame")
    header = {
        "v": int(version),
        "endian": "le",
        "count": count,
        "fields": field_metadata,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise SonicProtocolError("SONIC JSON header exceeds the fixed 1280-byte limit")
    return topic.encode("utf-8") + header_bytes.ljust(HEADER_SIZE, b"\x00") + b"".join(binary_parts)


def unpack_zmq_message(message: bytes, *, topic: str = "pose") -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Decode a packed message for protocol smoke tests and diagnostics."""
    topic_bytes = topic.encode("utf-8")
    if not message.startswith(topic_bytes):
        raise SonicProtocolError(f"message does not start with topic {topic!r}")
    payload = memoryview(message)[len(topic_bytes) :]
    if len(payload) < HEADER_SIZE:
        raise SonicProtocolError("message is shorter than the SONIC header")
    header = json.loads(bytes(payload[:HEADER_SIZE]).split(b"\x00", 1)[0].decode("utf-8"))
    dtype_map = {
        "f32": np.dtype("<f4"),
        "f64": np.dtype("<f8"),
        "i32": np.dtype("<i4"),
        "i64": np.dtype("<i8"),
        "u8": np.dtype("u1"),
        "bool": np.dtype("?"),
    }
    offset = HEADER_SIZE
    fields: dict[str, np.ndarray] = {}
    for field in header["fields"]:
        dtype = dtype_map.get(field["dtype"])
        if dtype is None:
            raise SonicProtocolError(f"unsupported SONIC wire dtype: {field['dtype']}")
        shape = tuple(int(size) for size in field["shape"])
        byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        fields[field["name"]] = np.frombuffer(payload[offset : offset + byte_count], dtype=dtype).reshape(shape).copy()
        offset += byte_count
    if offset != len(payload):
        raise SonicProtocolError("SONIC payload byte count does not match its header")
    return header, fields


@dataclass(frozen=True)
class G1Motion:
    root_rot_xyzw: np.ndarray
    dof_pos_mujoco: np.ndarray
    fps: float = G1_FPS
    root_pos: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return int(self.dof_pos_mujoco.shape[0])


def load_g1_motion(path: str | Path) -> G1Motion:
    """Load the standard G1 pickle emitted by the V6f-X evaluator."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    try:
        root_rot = _as_float_array(payload["root_rot"], "root_rot", (4,))
        dof_pos = _as_float_array(payload["dof_pos"], "dof_pos", (G1_DOF_DIM,))
    except KeyError as error:
        raise SonicProtocolError(f"{path}: missing required G1 field {error.args[0]!r}") from error
    if root_rot.shape[0] != dof_pos.shape[0]:
        raise SonicProtocolError(f"{path}: root_rot and dof_pos have different frame counts")
    root_pos = None
    if "root_pos" in payload:
        root_pos = _as_float_array(payload["root_pos"], "root_pos", (3,))
        if root_pos.shape[0] != dof_pos.shape[0]:
            raise SonicProtocolError(
                f"{path}: root_pos and dof_pos have different frame counts"
            )
    fps = float(payload.get("fps", G1_FPS) or G1_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise SonicProtocolError(f"{path}: fps must be positive")
    return G1Motion(
        root_rot_xyzw=root_rot,
        dof_pos_mujoco=dof_pos,
        fps=fps,
        root_pos=root_pos,
    )


def _xyzw_to_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion_xyzw = _as_float_array(quaternion_xyzw, "root_rot", (4,))
    norm = np.linalg.norm(quaternion_xyzw, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise SonicProtocolError("root_rot contains a zero-length quaternion")
    normalized = quaternion_xyzw / norm
    return normalized[:, [3, 0, 1, 2]].astype(np.float32, copy=False)


def _yaw_from_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion_xyzw = _as_float_array(quaternion_xyzw, "root_rot", (4,))
    norm = np.linalg.norm(quaternion_xyzw, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise SonicProtocolError("root_rot contains a zero-length quaternion")
    x, y, z, w = (quaternion_xyzw / norm).T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _slerp_xyzw(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + float(fraction) * (right - left)
    else:
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        result = (
            np.sin((1.0 - float(fraction)) * theta) / sin_theta * left
            + np.sin(float(fraction) * theta) / sin_theta * right
        )
    return (result / np.linalg.norm(result)).astype(np.float32, copy=False)


@dataclass
class SonicReferenceAdapter:
    """Convert 30 Hz MuJoCo trajectories into a safe 50 Hz SONIC stream.

    SONIC's own local planner performs this timebase conversion before its
    policy sees a reference.  The adapter additionally makes the first packet
    continuous with the measured robot pose and applies transparent, bounded
    position/yaw-rate constraints for simulation bring-up.
    """

    target_fps: float = SONIC_REFERENCE_FPS
    safety_enabled: bool = True
    max_joint_velocity_rad_s: float = 1.5
    max_yaw_velocity_rad_s: float = 0.5
    ramp_seconds: float = 1.0
    joint_limit_margin_rad: float = 0.05

    def __post_init__(self) -> None:
        if self.target_fps <= 0.0:
            raise SonicProtocolError("target_fps must be positive")
        if self.safety_enabled and self.max_joint_velocity_rad_s <= 0.0:
            raise SonicProtocolError("max_joint_velocity_rad_s must be positive")
        if self.safety_enabled and self.max_yaw_velocity_rad_s <= 0.0:
            raise SonicProtocolError("max_yaw_velocity_rad_s must be positive")
        if self.ramp_seconds < 0.0:
            raise SonicProtocolError("ramp_seconds must be non-negative")
        if not 0.0 <= self.joint_limit_margin_rad < 0.25:
            raise SonicProtocolError("joint_limit_margin_rad must be in [0, 0.25)")
        self._source_frames = 0
        self._target_frames = 0
        self._previous_source_dof: np.ndarray | None = None
        self._previous_source_root_pos: np.ndarray | None = None
        self._previous_source_root_rot: np.ndarray | None = None
        self._previous_reference_dof: np.ndarray | None = None
        self._previous_reference_yaw: float | None = None
        self._ramp_dof_anchor: np.ndarray | None = None
        self._ramp_yaw_anchor: float | None = None
        self.last_diagnostics: dict[str, float | int] = {}

    def adapt(
        self,
        motion: G1Motion,
        *,
        execution_dof_mujoco: np.ndarray | list[float],
        execution_quat_wxyz: np.ndarray | list[float],
    ) -> G1Motion:
        """Resample and bound one contiguous source commit.

        The adapter preserves the exact global 30:50 phase across C4 packets,
        so emitted packet lengths follow 13, 13, 14, ... rather than drifting.
        """
        if not np.isclose(motion.fps, G1_FPS, rtol=0.0, atol=1e-4):
            raise SonicProtocolError(
                f"SONIC adapter expects {G1_FPS:g} Hz source motion, got {motion.fps:g} Hz"
            )
        execution_dof = _as_float_array(
            execution_dof_mujoco, "execution_dof_mujoco", (G1_DOF_DIM,)
        )
        if execution_dof.ndim != 1:
            raise SonicProtocolError("execution_dof_mujoco must have shape [29]")
        execution_quat = _as_float_array(
            execution_quat_wxyz, "execution_quat_wxyz", (4,)
        )
        if execution_quat.ndim != 1 or float(np.linalg.norm(execution_quat)) < 1e-6:
            raise SonicProtocolError("execution_quat_wxyz must have shape [4] and non-zero norm")

        source_start = self._source_frames
        source_stop = source_start + motion.frames
        target_start = self._target_frames
        target_stop = int(np.floor(source_stop * self.target_fps / motion.fps + 1e-8))
        if target_stop <= target_start:
            target_stop = target_start + 1
        target_indices = np.arange(target_start, target_stop, dtype=np.float64)
        source_samples = target_indices * (motion.fps / self.target_fps)
        dof, root_pos, root_rot = self._sample_source_motion(
            motion,
            source_start=source_start,
            source_samples=source_samples,
        )

        if self._previous_reference_dof is None:
            self._previous_reference_dof = execution_dof.copy()
            self._ramp_dof_anchor = execution_dof.copy()
        if self._previous_reference_yaw is None:
            self._previous_reference_yaw = _yaw_from_wxyz(
                execution_quat / np.linalg.norm(execution_quat)
            )
            self._ramp_yaw_anchor = self._previous_reference_yaw

        reference_dof_before_commit = self._previous_reference_dof.copy()
        if not self.safety_enabled:
            raw_joint_velocity = np.diff(
                np.concatenate((reference_dof_before_commit[None], dof), axis=0), axis=0
            ) * self.target_fps
            self.last_diagnostics = {
                "source_frames": int(motion.frames),
                "target_frames": int(dof.shape[0]),
                "source_fps": float(motion.fps),
                "target_fps": float(self.target_fps),
                "raw_max_joint_velocity_rad_s": float(np.max(np.abs(raw_joint_velocity))),
                "safe_max_joint_velocity_rad_s": float(np.max(np.abs(raw_joint_velocity))),
                "position_clipped_values": 0,
                "safety_enabled": 0,
            }
            self._previous_reference_dof = dof[-1].copy()
            self._previous_reference_yaw = float(_yaw_from_xyzw(root_rot[-1:])[0])
            self._advance_source_history(motion, source_stop, target_stop)
            return G1Motion(
                root_rot_xyzw=root_rot,
                dof_pos_mujoco=dof,
                fps=self.target_fps,
                root_pos=root_pos,
            )

        lower = G1_MUJOCO_JOINT_LOWER + self.joint_limit_margin_rad
        upper = G1_MUJOCO_JOINT_UPPER - self.joint_limit_margin_rad
        raw_dof = dof.copy()
        dof = np.clip(dof, lower, upper)
        raw_yaw = _yaw_from_xyzw(root_rot)
        safe_yaw = np.empty_like(raw_yaw, dtype=np.float32)
        safe_dof = np.empty_like(dof, dtype=np.float32)
        max_joint_step = self.max_joint_velocity_rad_s / self.target_fps
        max_yaw_step = self.max_yaw_velocity_rad_s / self.target_fps
        for frame, target_index in enumerate(target_indices.astype(np.int64)):
            if self.ramp_seconds == 0.0:
                ramp = 1.0
            else:
                ramp = min(1.0, (int(target_index) + 1) / self.target_fps / self.ramp_seconds)
            candidate_dof = self._ramp_dof_anchor + ramp * (dof[frame] - self._ramp_dof_anchor)
            step = np.clip(
                candidate_dof - self._previous_reference_dof,
                -max_joint_step,
                max_joint_step,
            )
            self._previous_reference_dof = np.clip(
                self._previous_reference_dof + step,
                lower,
                upper,
            ).astype(np.float32, copy=False)
            safe_dof[frame] = self._previous_reference_dof

            yaw_delta = float(_wrap_angle(raw_yaw[frame] - self._ramp_yaw_anchor))
            candidate_yaw = self._ramp_yaw_anchor + ramp * yaw_delta
            yaw_step = float(_wrap_angle(candidate_yaw - self._previous_reference_yaw))
            yaw_step = float(np.clip(yaw_step, -max_yaw_step, max_yaw_step))
            self._previous_reference_yaw += yaw_step
            safe_yaw[frame] = self._previous_reference_yaw

        safe_root_rot = np.zeros((safe_yaw.shape[0], 4), dtype=np.float32)
        safe_root_rot[:, 2] = np.sin(safe_yaw * 0.5)
        safe_root_rot[:, 3] = np.cos(safe_yaw * 0.5)
        safe_joint_velocity = np.diff(
            np.concatenate((reference_dof_before_commit[None], safe_dof), axis=0), axis=0
        ) * self.target_fps
        self.last_diagnostics = {
            "source_frames": int(motion.frames),
            "target_frames": int(safe_dof.shape[0]),
            "source_fps": float(motion.fps),
            "target_fps": float(self.target_fps),
            "raw_max_joint_velocity_rad_s": float(
                np.max(np.abs(np.diff(raw_dof, axis=0) * self.target_fps))
                if raw_dof.shape[0] > 1
                else 0.0
            ),
            "safe_max_joint_velocity_rad_s": float(np.max(np.abs(safe_joint_velocity))),
            "position_clipped_values": int(np.count_nonzero(raw_dof != dof)),
            "safety_enabled": 1,
        }
        self._advance_source_history(motion, source_stop, target_stop)
        return G1Motion(
            root_rot_xyzw=safe_root_rot,
            dof_pos_mujoco=safe_dof,
            fps=self.target_fps,
            root_pos=root_pos,
        )

    def _advance_source_history(
        self,
        motion: G1Motion,
        source_stop: int,
        target_stop: int,
    ) -> None:
        self._source_frames = source_stop
        self._target_frames = target_stop
        self._previous_source_dof = motion.dof_pos_mujoco[-1].copy()
        self._previous_source_root_rot = motion.root_rot_xyzw[-1].copy()
        self._previous_source_root_pos = (
            motion.root_pos[-1].copy() if motion.root_pos is not None else None
        )
    def _sample_source_motion(
        self,
        motion: G1Motion,
        *,
        source_start: int,
        source_samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        available_indices = np.arange(source_start, source_start + motion.frames, dtype=np.float64)
        dof_values = motion.dof_pos_mujoco
        rot_values = motion.root_rot_xyzw
        root_values = motion.root_pos
        if self._previous_source_dof is not None:
            available_indices = np.concatenate((np.asarray([source_start - 1.0]), available_indices))
            dof_values = np.concatenate((self._previous_source_dof[None], dof_values), axis=0)
            rot_values = np.concatenate((self._previous_source_root_rot[None], rot_values), axis=0)
            if root_values is not None and self._previous_source_root_pos is not None:
                root_values = np.concatenate((self._previous_source_root_pos[None], root_values), axis=0)
        dof = np.stack(
            [np.interp(source_samples, available_indices, dof_values[:, index]) for index in range(G1_DOF_DIM)],
            axis=1,
        ).astype(np.float32)
        root_pos = None
        if root_values is not None:
            root_pos = np.stack(
                [np.interp(source_samples, available_indices, root_values[:, index]) for index in range(3)],
                axis=1,
            ).astype(np.float32)
        root_rot = np.empty((source_samples.shape[0], 4), dtype=np.float32)
        for index, source_sample in enumerate(source_samples):
            right_index = int(np.searchsorted(available_indices, source_sample, side="right"))
            right_index = min(max(right_index, 1), len(available_indices) - 1)
            left_index = right_index - 1
            left_time = available_indices[left_index]
            right_time = available_indices[right_index]
            if source_sample <= available_indices[0]:
                root_rot[index] = rot_values[0]
            elif source_sample >= available_indices[-1]:
                root_rot[index] = rot_values[-1]
            else:
                root_rot[index] = _slerp_xyzw(
                    rot_values[left_index],
                    rot_values[right_index],
                    float((source_sample - left_time) / (right_time - left_time)),
                )
        return dof, root_pos, root_rot


def reference_fields_from_commit(
    motion: G1Motion,
    *,
    start_frame: int,
    stop_frame: int,
    previous_dof_pos: np.ndarray | None = None,
    frame_index_start: int | None = None,
) -> dict[str, np.ndarray]:
    """Create one atomic SONIC v1 reference packet from a generated commit.

    The export remains in MuJoCo order, while SONIC's policy expects its
    IsaacLab order.  Convert only at the wire boundary.  `catch_up=false`
    preserves real-time playback instead of interpreting a newly arrived C4
    as stale buffered data.
    """
    start_frame = int(start_frame)
    stop_frame = int(stop_frame)
    if not 0 <= start_frame < stop_frame <= motion.frames:
        raise SonicProtocolError(f"invalid commit frame range [{start_frame}, {stop_frame})")
    dof_pos_mujoco = motion.dof_pos_mujoco[start_frame:stop_frame]
    if previous_dof_pos is None:
        previous_dof_pos = dof_pos_mujoco[0]
    previous_dof_pos = _as_float_array(previous_dof_pos, "previous_dof_pos", (G1_DOF_DIM,))
    if previous_dof_pos.ndim != 1:
        raise SonicProtocolError("previous_dof_pos must have shape [29]")
    previous_and_current = np.concatenate((previous_dof_pos[None], dof_pos_mujoco), axis=0)
    joint_vel_mujoco = np.diff(previous_and_current, axis=0) * float(motion.fps)
    if frame_index_start is None:
        frame_index_start = start_frame
    return {
        "joint_pos": dof_pos_mujoco[:, SONIC_REFERENCE_FROM_MUJOCO].astype(np.float32, copy=False),
        "joint_vel": joint_vel_mujoco[:, SONIC_REFERENCE_FROM_MUJOCO].astype(np.float32, copy=False),
        "body_quat_w": _xyzw_to_wxyz(motion.root_rot_xyzw[start_frame:stop_frame]),
        "frame_index": np.arange(
            int(frame_index_start),
            int(frame_index_start) + (stop_frame - start_frame),
            dtype=np.int64,
        ),
        "catch_up": np.asarray([0], dtype=np.uint8),
    }


def iter_reference_commits(motion: G1Motion, commit_frames: int = G1_COMMIT_FRAMES):
    """Yield contiguous reference commits with finite-difference continuity."""
    commit_frames = int(commit_frames)
    if commit_frames <= 0:
        raise SonicProtocolError("commit_frames must be positive")
    previous_dof_pos = None
    for start_frame in range(0, motion.frames, commit_frames):
        stop_frame = min(start_frame + commit_frames, motion.frames)
        fields = reference_fields_from_commit(
            motion,
            start_frame=start_frame,
            stop_frame=stop_frame,
            previous_dof_pos=previous_dof_pos,
        )
        previous_dof_pos = motion.dof_pos_mujoco[stop_frame - 1]
        yield start_frame, stop_frame, fields


def g1_motion_from_yaw_delta_commit(
    raw_commit: np.ndarray,
    *,
    base_position: np.ndarray | list[float],
    base_quat_wxyz: np.ndarray | list[float],
    fps: float = G1_FPS,
) -> G1Motion:
    """Anchor a canonical g1_yaw_delta C4 commit at the measured base pose.

    The V6f-X codec emits local root XY increments, yaw increments, height and
    29 MuJoCo-order joints.  Its S66 condition intentionally has no global XY
    or absolute yaw, so those are taken from the synchronized execution state
    only when constructing SONIC's absolute reference payload.
    """
    raw_commit = np.asarray(raw_commit, dtype=np.float32)
    if raw_commit.ndim != 2 or raw_commit.shape[1] != 34:
        raise SonicProtocolError(
            f"g1_yaw_delta commit must have shape [T, 34], got {raw_commit.shape}"
        )
    if raw_commit.shape[0] <= 0 or not np.isfinite(raw_commit).all():
        raise SonicProtocolError("g1_yaw_delta commit must be non-empty and finite")
    base_position = _as_feedback_vector(
        {"base_position": base_position}, "base_position", 3
    )
    base_quat = _as_feedback_vector(
        {"base_quat": base_quat_wxyz}, "base_quat", 4
    )
    quaternion_norm = float(np.linalg.norm(base_quat))
    if quaternion_norm < 1e-6:
        raise SonicProtocolError("base_quat must not have zero length")
    current_xy = base_position[:2].astype(np.float32, copy=True)
    current_yaw = _yaw_from_wxyz(base_quat / quaternion_norm)
    root_rot_xyzw = np.empty((raw_commit.shape[0], 4), dtype=np.float32)
    root_pos = np.empty((raw_commit.shape[0], 3), dtype=np.float32)
    dof_pos = raw_commit[:, 5:].astype(np.float32, copy=True)
    for frame in range(raw_commit.shape[0]):
        if frame > 0:
            local_delta = raw_commit[frame, :2]
            cos_yaw = np.cos(current_yaw)
            sin_yaw = np.sin(current_yaw)
            current_xy += np.asarray(
                (
                    cos_yaw * local_delta[0] - sin_yaw * local_delta[1],
                    sin_yaw * local_delta[0] + cos_yaw * local_delta[1],
                ),
                dtype=np.float32,
            )
            delta_yaw_sincos = raw_commit[frame, 3:5]
            if float(np.linalg.norm(delta_yaw_sincos)) < 1e-6:
                raise SonicProtocolError(
                    "g1_yaw_delta commit contains a zero-length yaw sin/cos pair"
                )
            current_yaw += float(
                np.arctan2(delta_yaw_sincos[0], delta_yaw_sincos[1])
            )
        root_rot_xyzw[frame] = (
            0.0,
            0.0,
            np.sin(current_yaw * 0.5),
            np.cos(current_yaw * 0.5),
        )
        root_pos[frame] = (current_xy[0], current_xy[1], raw_commit[frame, 2])
    return G1Motion(
        root_rot_xyzw=root_rot_xyzw,
        dof_pos_mujoco=dof_pos,
        fps=float(fps),
        # SONIC v1 does not transmit root_pos.  Retain this integrated path for
        # diagnostics and for a future protocol that supports root tracking.
        root_pos=root_pos,
    )


def _as_feedback_vector(payload: Mapping[str, Any], name: str, size: int) -> np.ndarray:
    if name not in payload:
        raise SonicExecutionStateError(f"SONIC feedback is missing {name!r}")
    value = np.asarray(payload[name], dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise SonicExecutionStateError(f"SONIC feedback {name!r} must have finite shape [{size}]")
    return value


def _yaw_from_wxyz(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = quaternion_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class SonicS66Builder:
    """Build the generator's physical S66 condition from SONIC execution feedback.

    The current deploy feedback does not publish a real base position or base
    linear velocity.  The builder deliberately requires both fields so a caller
    cannot accidentally close the loop with the deploy's fixed visualisation
    translation.
    """

    def __init__(self, fps: float = G1_FPS):
        self.fps = float(fps)
        if self.fps <= 0:
            raise SonicExecutionStateError("fps must be positive")
        self._previous_planar_velocity: np.ndarray | None = None
        self._previous_yaw_velocity: float | None = None

    def build(self, feedback: Mapping[str, Any]) -> np.ndarray:
        base_position = _as_feedback_vector(feedback, "base_position", 3)
        base_linear_velocity = _as_feedback_vector(feedback, "base_linear_velocity", 3)
        base_quat = _as_feedback_vector(feedback, "base_quat", 4)
        base_ang_vel = _as_feedback_vector(feedback, "base_ang_vel", 3)
        joint_pos = _as_feedback_vector(feedback, "body_q", G1_DOF_DIM)
        joint_vel = _as_feedback_vector(feedback, "body_dq", G1_DOF_DIM)
        yaw = _yaw_from_wxyz(base_quat / max(float(np.linalg.norm(base_quat)), 1e-6))
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        planar_velocity = np.asarray(
            [
                cos_yaw * base_linear_velocity[0] + sin_yaw * base_linear_velocity[1],
                -sin_yaw * base_linear_velocity[0] + cos_yaw * base_linear_velocity[1],
            ],
            dtype=np.float32,
        )
        yaw_velocity = float(base_ang_vel[2])
        if self._previous_planar_velocity is None:
            planar_acceleration = np.zeros(2, dtype=np.float32)
            yaw_acceleration = 0.0
        else:
            planar_acceleration = (planar_velocity - self._previous_planar_velocity) * self.fps
            yaw_acceleration = (yaw_velocity - float(self._previous_yaw_velocity)) * self.fps
        self._previous_planar_velocity = planar_velocity.copy()
        self._previous_yaw_velocity = yaw_velocity
        return np.concatenate(
            (
                base_position[2:3],
                planar_velocity,
                base_linear_velocity[2:3],
                np.asarray([yaw_velocity], dtype=np.float32),
                joint_pos,
                joint_vel,
                planar_acceleration,
                np.asarray([yaw_acceleration], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)


def merge_sonic_execution_feedback(
    sonic_feedback: Mapping[str, Any], sim_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge SONIC joint feedback with true base state from the MuJoCo side channel."""
    required_sim_keys = (
        "base_position",
        "base_linear_velocity",
        "base_quat",
        "base_ang_vel",
    )
    missing = [key for key in required_sim_keys if key not in sim_state]
    if missing:
        raise SonicExecutionStateError(
            f"SONIC simulation state is missing required fields: {missing}"
        )
    merged = dict(sonic_feedback)
    for key in required_sim_keys:
        merged[key] = sim_state[key]
    merged["_sonic_feedback_index"] = sonic_feedback.get("index")
    merged["_sim_time"] = sim_state.get("sim_time")
    return merged


class SonicS66Synchronizer:
    """Pair the latest true simulation base state with SONIC joint feedback."""

    def __init__(self, fps: float = G1_FPS):
        self.builder = SonicS66Builder(fps=fps)
        self._latest_sim_state: Mapping[str, Any] | None = None

    def update_sim_state(self, sim_state: Mapping[str, Any]) -> None:
        self._latest_sim_state = dict(sim_state)

    @property
    def latest_sim_state(self) -> dict[str, Any] | None:
        """Latest true base pose used to synchronize an S66 sample."""
        if self._latest_sim_state is None:
            return None
        return dict(self._latest_sim_state)

    def update_sonic_feedback(self, sonic_feedback: Mapping[str, Any]) -> dict[str, Any] | None:
        if self._latest_sim_state is None:
            return None
        merged = merge_sonic_execution_feedback(sonic_feedback, self._latest_sim_state)
        return {
            "sonic_feedback_index": merged["_sonic_feedback_index"],
            "sim_time": merged["_sim_time"],
            "s66": self.builder.build(merged).tolist(),
        }


class SonicFeedbackSubscriber:
    """Non-blocking subscriber for SONIC's msgpack execution-state stream."""

    def __init__(self, host: str = "localhost", port: int = 5557, topic: str = "g1_debug"):
        try:
            import msgpack
            import zmq
        except ImportError as error:
            raise RuntimeError("SONIC feedback requires the msgpack and pyzmq Python packages") from error
        self._msgpack = msgpack
        self._topic = str(topic)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic.encode("utf-8"))
        self._socket.connect(f"tcp://{host}:{int(port)}")

    def poll(self, timeout_ms: int = 0) -> dict[str, Any] | None:
        if not self._socket.poll(timeout=max(int(timeout_ms), 0)):
            return None
        payload = unpack_sonic_feedback_message(
            self._socket.recv(), topic=self._topic, msgpack_module=self._msgpack
        )
        if payload is None:
            return None
        payload["_received_monotonic_seconds"] = time.monotonic()
        return payload

    def close(self) -> None:
        self._socket.close()
        self._context.term()


def unpack_sonic_feedback_message(
    message: bytes,
    *,
    topic: str = "g1_debug",
    msgpack_module=None,
) -> dict[str, Any] | None:
    """Decode one topic-prefixed SONIC feedback packet without opening a socket."""
    prefix = str(topic).encode("utf-8")
    if not message.startswith(prefix):
        return None
    if msgpack_module is None:
        try:
            import msgpack as msgpack_module
        except ImportError as error:
            raise RuntimeError("SONIC feedback requires the msgpack Python package") from error
    payload = msgpack_module.unpackb(message[len(prefix) :], raw=False)
    if not isinstance(payload, dict):
        raise SonicProtocolError("SONIC feedback payload must be a msgpack map")
    return payload


def jsonable_feedback(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a feedback packet to plain JSON-compatible values for logging."""
    converted: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            converted[key] = value.item()
        else:
            converted[key] = value
    return converted
