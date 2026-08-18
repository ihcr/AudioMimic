from dataclasses import dataclass

import numpy as np
import torch


G1_STREAMING_STATE_DIM = 66
G1_STREAMING_VELOCITY_DIM = 33
G1_STREAMING_ACCELERATION_DIM = 33
G1_STREAMING_JERK_DIM = 3
G1_STREAMING_FPS = 30.0
NORMALIZER_EPS = 1e-6


@dataclass(frozen=True)
class G1BoundaryStateSpec:
    """Serialized boundary-state layout used by codecs and generators."""

    name: str
    dim: int
    include_root_yaw_acceleration: bool

    def asdict(self):
        return {
            "name": self.name,
            "dim": int(self.dim),
            "include_root_yaw_acceleration": bool(
                self.include_root_yaw_acceleration
            ),
        }


LEGACY_S66 = G1BoundaryStateSpec(
    name="S66_physical",
    dim=66,
    include_root_yaw_acceleration=True,
)
REDUCED_B63 = G1BoundaryStateSpec(
    name="B63_reduced_qv_v1",
    dim=63,
    include_root_yaw_acceleration=False,
)
G1_BOUNDARY_STATE_SPECS = {
    LEGACY_S66.name: LEGACY_S66,
    REDUCED_B63.name: REDUCED_B63,
}


def resolve_boundary_state_spec(state_spec=None, state_dim=None):
    if state_spec is None:
        if state_dim is None or int(state_dim) == LEGACY_S66.dim:
            return LEGACY_S66
        if int(state_dim) == REDUCED_B63.dim:
            return REDUCED_B63
        raise ValueError(f"unsupported G1 boundary state dimension {state_dim}")
    if isinstance(state_spec, G1BoundaryStateSpec):
        resolved = state_spec
    elif isinstance(state_spec, str):
        try:
            resolved = G1_BOUNDARY_STATE_SPECS[state_spec]
        except KeyError as error:
            raise ValueError(f"unsupported G1 boundary state layout {state_spec!r}") from error
    elif isinstance(state_spec, dict):
        name = state_spec.get("name")
        resolved = resolve_boundary_state_spec(name)
        if state_spec != resolved.asdict():
            raise ValueError(f"boundary state spec payload does not match {name!r}")
    else:
        raise TypeError("state_spec must be a G1BoundaryStateSpec, name, dict, or None")
    if state_dim is not None and int(state_dim) != resolved.dim:
        raise ValueError(
            f"boundary state layout {resolved.name} requires {resolved.dim} dimensions, "
            f"got {state_dim}"
        )
    return resolved


@dataclass(frozen=True)
class G1StreamingStateStatistics:
    state_mean: np.ndarray
    state_std: np.ndarray
    velocity_std: np.ndarray
    acceleration_std: np.ndarray
    jerk_std: np.ndarray
    fps: float = G1_STREAMING_FPS
    state_layout: str = LEGACY_S66.name

    def __post_init__(self):
        spec = resolve_boundary_state_spec(self.state_layout)
        if np.asarray(self.state_mean).shape != (spec.dim,):
            raise ValueError(
                f"{spec.name} statistics require state_mean shape ({spec.dim},)"
            )
        if np.asarray(self.state_std).shape != (spec.dim,):
            raise ValueError(
                f"{spec.name} statistics require state_std shape ({spec.dim},)"
            )

    @property
    def state_spec(self):
        return resolve_boundary_state_spec(self.state_layout)

    def normalize_state(self, state):
        mean, std = self.state_tensors(state.device, state.dtype)
        return (state - mean) / std

    def denormalize_state(self, state):
        mean, std = self.state_tensors(state.device, state.dtype)
        return state * std + mean

    def state_tensors(self, device, dtype):
        mean = torch.as_tensor(self.state_mean, device=device, dtype=dtype)
        std = torch.as_tensor(self.state_std, device=device, dtype=dtype)
        return mean, std

    def transition_scale_tensors(self, device, dtype):
        return (
            torch.as_tensor(self.velocity_std, device=device, dtype=dtype),
            torch.as_tensor(self.acceleration_std, device=device, dtype=dtype),
            torch.as_tensor(self.jerk_std, device=device, dtype=dtype),
        )

    def state_dict(self):
        return {
            "state_mean": np.asarray(self.state_mean, dtype=np.float32),
            "state_std": np.asarray(self.state_std, dtype=np.float32),
            "velocity_std": np.asarray(self.velocity_std, dtype=np.float32),
            "acceleration_std": np.asarray(self.acceleration_std, dtype=np.float32),
            "jerk_std": np.asarray(self.jerk_std, dtype=np.float32),
            "fps": float(self.fps),
            "state_layout": self.state_layout,
            "state_spec": self.state_spec.asdict(),
        }

    @classmethod
    def from_state_dict(cls, state):
        state_layout = state.get("state_layout", LEGACY_S66.name)
        if "state_spec" in state:
            state_spec = resolve_boundary_state_spec(state["state_spec"])
            if state_spec.name != state_layout:
                raise ValueError("state_layout and state_spec disagree")
        return cls(
            state_mean=np.asarray(state["state_mean"], dtype=np.float32),
            state_std=np.asarray(state["state_std"], dtype=np.float32),
            velocity_std=np.asarray(state["velocity_std"], dtype=np.float32),
            acceleration_std=np.asarray(state["acceleration_std"], dtype=np.float32),
            jerk_std=np.asarray(state["jerk_std"], dtype=np.float32),
            fps=float(state.get("fps", G1_STREAMING_FPS)),
            state_layout=state_layout,
        )


class _RunningMoments:
    def __init__(self, dim):
        self.dim = int(dim)
        self.count = 0
        self.total = np.zeros(self.dim, dtype=np.float64)
        self.total_square = np.zeros(self.dim, dtype=np.float64)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.dim)
        if values.size == 0:
            return
        self.count += int(values.shape[0])
        self.total += values.sum(axis=0)
        self.total_square += np.square(values).sum(axis=0)

    def mean_std(self):
        if self.count <= 0:
            raise ValueError("cannot finalize empty running moments")
        mean = self.total / float(self.count)
        variance = self.total_square / float(self.count) - np.square(mean)
        std = np.sqrt(np.maximum(variance, NORMALIZER_EPS**2))
        return mean.astype(np.float32), std.astype(np.float32)


def _require_yaw_delta_motion(motion):
    if not torch.is_tensor(motion):
        motion = torch.as_tensor(motion, dtype=torch.float32)
    if motion.ndim < 2 or motion.shape[-1] != 34:
        raise ValueError(f"g1_yaw_delta motion must end in [T, 34], got {tuple(motion.shape)}")
    return motion


def _rotate_into_next_frame(vector, delta_yaw):
    cos_yaw = torch.cos(delta_yaw)
    sin_yaw = torch.sin(delta_yaw)
    x = cos_yaw * vector[..., 0] + sin_yaw * vector[..., 1]
    y = -sin_yaw * vector[..., 0] + cos_yaw * vector[..., 1]
    return torch.stack((x, y), dim=-1)


def _prepend_zero_difference(values):
    zero = torch.zeros_like(values[..., :1, :])
    if values.shape[-2] <= 1:
        return zero
    return torch.cat((zero, values[..., 1:, :] - values[..., :-1, :]), dim=-2)


def motion_to_streaming_state(motion, fps=G1_STREAMING_FPS, state_spec=LEGACY_S66):
    """Convert raw 34D g1_yaw_delta motion to a declared boundary-state sequence."""
    motion = _require_yaw_delta_motion(motion)
    state_spec = resolve_boundary_state_spec(state_spec)
    fps = float(fps)
    delta_xy = motion[..., :2]
    height = motion[..., 2:3]
    delta_yaw = torch.atan2(motion[..., 3], motion[..., 4])
    joints = motion[..., 5:]

    planar_velocity_previous_frame = delta_xy * fps
    planar_velocity = _rotate_into_next_frame(planar_velocity_previous_frame, delta_yaw)
    vertical_velocity = _prepend_zero_difference(height) * fps
    yaw_velocity = delta_yaw.unsqueeze(-1) * fps
    joint_velocity = _prepend_zero_difference(joints) * fps

    previous_planar_velocity = torch.zeros_like(planar_velocity)
    if motion.shape[-2] > 1:
        previous_planar_velocity[..., 1:, :] = _rotate_into_next_frame(
            planar_velocity[..., :-1, :],
            delta_yaw[..., 1:],
        )
    planar_acceleration = (planar_velocity - previous_planar_velocity) * fps
    planar_acceleration[..., 0, :] = 0.0
    yaw_acceleration = _prepend_zero_difference(yaw_velocity) * fps

    state = torch.cat(
        (
            height,
            planar_velocity,
            vertical_velocity,
            yaw_velocity,
            joints,
            joint_velocity,
            planar_acceleration,
            yaw_acceleration,
        ),
        dim=-1,
    )
    if state.shape[-1] != G1_STREAMING_STATE_DIM:
        raise RuntimeError(f"S66 construction produced {state.shape[-1]} dimensions")
    return state[..., : state_spec.dim]


def streaming_state_velocity(state):
    resolve_boundary_state_spec(state_dim=state.shape[-1])
    return torch.cat((state[..., 1:4], state[..., 4:5], state[..., 34:63]), dim=-1)


def boundary_state_from_motion(
    motion,
    frame_start,
    fps=G1_STREAMING_FPS,
    state_spec=LEGACY_S66,
):
    motion = _require_yaw_delta_motion(motion)
    state_spec = resolve_boundary_state_spec(state_spec)
    frame_start = torch.as_tensor(frame_start, device=motion.device, dtype=torch.long)
    if frame_start.ndim == 0:
        frame_start = frame_start.expand(motion.shape[0])
    if motion.ndim != 3 or frame_start.shape != (motion.shape[0],):
        raise ValueError("batched boundary extraction expects motion [B,T,34] and frame_start [B]")
    if torch.any(frame_start < 0) or torch.any(frame_start > motion.shape[1]):
        raise ValueError("frame_start is outside the motion sequence")
    state_sequence = motion_to_streaming_state(motion, fps=fps, state_spec=state_spec)
    boundary_index = torch.clamp(frame_start - 1, min=0)
    batch_index = torch.arange(motion.shape[0], device=motion.device)
    state = state_sequence[batch_index, boundary_index]
    cold = frame_start.eq(0)
    if cold.any():
        state = state.clone()
        state[cold, 1:5] = 0.0
        state[cold, 34:] = 0.0
    return state


def canonical_streaming_state(
    batch_size,
    device=None,
    dtype=torch.float32,
    root_height=0.78,
    joint_angles=None,
    state_spec=LEGACY_S66,
):
    state_spec = resolve_boundary_state_spec(state_spec)
    state = torch.zeros(int(batch_size), state_spec.dim, device=device, dtype=dtype)
    state[:, 0] = float(root_height)
    if joint_angles is not None:
        joint_angles = torch.as_tensor(joint_angles, device=device, dtype=dtype)
        if joint_angles.shape != (29,):
            raise ValueError("joint_angles must have shape [29]")
        state[:, 5:34] = joint_angles
    return state


def transition_features_from_plan(boundary_state, plan_motion, fps=G1_STREAMING_FPS):
    """Return physical velocity, acceleration, and root/yaw jerk for a plan."""
    plan_motion = _require_yaw_delta_motion(plan_motion)
    if plan_motion.ndim != 3 or boundary_state.ndim != 2:
        raise ValueError("expected boundary_state [B,D] and plan_motion [B,T,34]")
    state_spec = resolve_boundary_state_spec(state_dim=boundary_state.shape[-1])
    if boundary_state.shape[0] != plan_motion.shape[0]:
        raise ValueError("boundary_state and plan_motion batch sizes differ")
    fps = float(fps)
    delta_yaw = torch.atan2(plan_motion[..., 3], plan_motion[..., 4])
    planar_velocity = _rotate_into_next_frame(plan_motion[..., :2] * fps, delta_yaw)

    previous_height = torch.cat((boundary_state[:, None, 0:1], plan_motion[:, :-1, 2:3]), dim=1)
    vertical_velocity = (plan_motion[..., 2:3] - previous_height) * fps
    yaw_velocity = delta_yaw.unsqueeze(-1) * fps
    previous_joints = torch.cat((boundary_state[:, None, 5:34], plan_motion[:, :-1, 5:]), dim=1)
    joint_velocity = (plan_motion[..., 5:] - previous_joints) * fps
    velocity = torch.cat((planar_velocity, vertical_velocity, yaw_velocity, joint_velocity), dim=-1)

    previous_velocity = torch.cat((streaming_state_velocity(boundary_state)[:, None], velocity[:, :-1]), dim=1)
    previous_planar = previous_velocity[..., :2]
    previous_planar = _rotate_into_next_frame(previous_planar, delta_yaw)
    previous_velocity_aligned = previous_velocity.clone()
    previous_velocity_aligned[..., :2] = previous_planar
    acceleration = (velocity - previous_velocity_aligned) * fps

    if state_spec.include_root_yaw_acceleration:
        previous_root_yaw_acceleration = torch.cat(
            (boundary_state[:, 63:65], boundary_state[:, 65:66]),
            dim=-1,
        )
    else:
        previous_root_yaw_acceleration = torch.zeros_like(velocity[:, 0, :3])
    root_yaw_acceleration = torch.cat((acceleration[..., :2], acceleration[..., 3:4]), dim=-1)
    previous_acceleration = torch.cat(
        (previous_root_yaw_acceleration[:, None], root_yaw_acceleration[:, :-1]),
        dim=1,
    )
    previous_acceleration_planar = _rotate_into_next_frame(
        previous_acceleration[..., :2],
        delta_yaw,
    )
    previous_acceleration_aligned = previous_acceleration.clone()
    previous_acceleration_aligned[..., :2] = previous_acceleration_planar
    jerk = (root_yaw_acceleration - previous_acceleration_aligned) * fps
    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "jerk": jerk,
    }


def fit_streaming_state_statistics(
    normalized_motion,
    motion_mean,
    motion_std,
    fps=G1_STREAMING_FPS,
    batch_size=256,
    minimum_frame=3,
    state_spec=LEGACY_S66,
):
    normalized_motion = np.asarray(normalized_motion, dtype=np.float32)
    motion_mean = np.asarray(motion_mean, dtype=np.float32).reshape(1, 1, 34)
    motion_std = np.asarray(motion_std, dtype=np.float32).reshape(1, 1, 34)
    if normalized_motion.ndim != 3 or normalized_motion.shape[-1] != 34:
        raise ValueError("normalized_motion must have shape [N,T,34]")
    state_spec = resolve_boundary_state_spec(state_spec)
    state_moments = _RunningMoments(state_spec.dim)
    velocity_moments = _RunningMoments(G1_STREAMING_VELOCITY_DIM)
    acceleration_moments = _RunningMoments(G1_STREAMING_ACCELERATION_DIM)
    jerk_moments = _RunningMoments(G1_STREAMING_JERK_DIM)
    minimum_frame = max(int(minimum_frame), 1)

    for start in range(0, normalized_motion.shape[0], int(batch_size)):
        raw = normalized_motion[start : start + int(batch_size)] * motion_std + motion_mean
        raw_tensor = torch.from_numpy(raw)
        state = motion_to_streaming_state(raw_tensor, fps=fps, state_spec=state_spec)
        boundary = state[:, minimum_frame - 1]
        plan = raw_tensor[:, minimum_frame:]
        features = transition_features_from_plan(boundary, plan, fps=fps)
        state_moments.update(state[:, minimum_frame:].numpy())
        velocity_moments.update(features["velocity"].numpy())
        acceleration_moments.update(features["acceleration"].numpy())
        jerk_moments.update(features["jerk"].numpy())

    state_mean, state_std = state_moments.mean_std()
    _, velocity_std = velocity_moments.mean_std()
    _, acceleration_std = acceleration_moments.mean_std()
    _, jerk_std = jerk_moments.mean_std()
    return G1StreamingStateStatistics(
        state_mean=state_mean,
        state_std=state_std,
        velocity_std=velocity_std,
        acceleration_std=acceleration_std,
        jerk_std=jerk_std,
        fps=float(fps),
        state_layout=state_spec.name,
    )
