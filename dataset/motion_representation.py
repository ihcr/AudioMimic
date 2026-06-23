import torch
import torch.nn.functional as F

from rotation_transforms import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    standardize_quaternion,
)


SMPL_MOTION_FORMAT = "smpl"
G1_MOTION_FORMAT = "g1"
G1_ROOT_DELTA_MOTION_FORMAT = "g1_root_delta"
G1_YAW_DELTA_MOTION_FORMAT = "g1_yaw_delta"
SMPL_REPR_DIM = 3 + 24 * 6 + 4
G1_DOF_DIM = 29
G1_REPR_DIM = 3 + 6 + G1_DOF_DIM
G1_YAW_DELTA_REPR_DIM = 2 + 1 + 2 + G1_DOF_DIM
G1_MOTION_FORMATS = (
    G1_MOTION_FORMAT,
    G1_ROOT_DELTA_MOTION_FORMAT,
    G1_YAW_DELTA_MOTION_FORMAT,
)
VALID_MOTION_FORMATS = (SMPL_MOTION_FORMAT, *G1_MOTION_FORMATS)


def validate_motion_format(motion_format):
    if motion_format not in VALID_MOTION_FORMATS:
        raise ValueError(
            f"Unsupported motion_format {motion_format!r}; expected one of {VALID_MOTION_FORMATS}"
        )
    return motion_format


def is_g1_motion_format(motion_format):
    return validate_motion_format(motion_format) in G1_MOTION_FORMATS


def motion_repr_dim(motion_format):
    validate_motion_format(motion_format)
    if motion_format in (G1_MOTION_FORMAT, G1_ROOT_DELTA_MOTION_FORMAT):
        return G1_REPR_DIM
    if motion_format == G1_YAW_DELTA_MOTION_FORMAT:
        return G1_YAW_DELTA_REPR_DIM
    return SMPL_REPR_DIM


def _float_tensor(value):
    if torch.is_tensor(value):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


def _require_shape(tensor, suffix, name):
    if tuple(tensor.shape[-len(suffix) :]) != tuple(suffix):
        raise ValueError(f"{name} expected shape ending in {suffix}, got {tuple(tensor.shape)}")


def _xyzw_to_wxyz(quaternions):
    return quaternions[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(quaternions):
    return quaternions[..., [1, 2, 3, 0]]


def _normalize_xyzw_quaternion(quaternions):
    quaternions = F.normalize(quaternions, dim=-1)
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def _yaw_from_xyzw_quaternion(quaternions):
    quaternions = _normalize_xyzw_quaternion(quaternions)
    x, y, z, w = torch.unbind(quaternions, dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw_xyzw_quaternion(yaw):
    half_yaw = 0.5 * yaw
    zeros = torch.zeros_like(half_yaw)
    return torch.stack(
        (zeros, zeros, torch.sin(half_yaw), torch.cos(half_yaw)),
        dim=-1,
    )


def _identity_matrices_like(root_pos, frame_count=1):
    leading_shape = root_pos.shape[:-2]
    eye = torch.eye(3, device=root_pos.device, dtype=root_pos.dtype)
    view_shape = (1,) * len(leading_shape) + (1, 3, 3)
    return eye.reshape(view_shape).expand(leading_shape + (frame_count, 3, 3))


def encode_g1_motion(root_pos, root_rot, dof_pos):
    root_pos = _float_tensor(root_pos)
    root_rot = _float_tensor(root_rot)
    dof_pos = _float_tensor(dof_pos)
    _require_shape(root_pos, (3,), "root_pos")
    _require_shape(root_rot, (4,), "root_rot")
    _require_shape(dof_pos, (G1_DOF_DIM,), "dof_pos")
    if root_pos.shape[:-1] != root_rot.shape[:-1] or root_pos.shape[:-1] != dof_pos.shape[:-1]:
        raise ValueError("G1 root_pos, root_rot, and dof_pos must share leading dimensions")

    root_rot = standardize_quaternion(F.normalize(root_rot, dim=-1))
    root_rot_6d = matrix_to_rotation_6d(quaternion_to_matrix(root_rot))
    return torch.cat((root_pos, root_rot_6d, dof_pos), dim=-1)


def encode_g1_root_delta_motion(root_pos, root_rot, dof_pos):
    root_pos = _float_tensor(root_pos)
    root_rot = _float_tensor(root_rot)
    dof_pos = _float_tensor(dof_pos)
    _require_shape(root_pos, (3,), "root_pos")
    _require_shape(root_rot, (4,), "root_rot")
    _require_shape(dof_pos, (G1_DOF_DIM,), "dof_pos")
    if root_pos.shape[:-1] != root_rot.shape[:-1] or root_pos.shape[:-1] != dof_pos.shape[:-1]:
        raise ValueError("G1 root_pos, root_rot, and dof_pos must share leading dimensions")
    if root_pos.ndim < 2:
        raise ValueError("G1 root-delta encoding requires an explicit frame dimension")

    root_rot_xyzw = _normalize_xyzw_quaternion(root_rot)
    root_mats = quaternion_to_matrix(_xyzw_to_wxyz(root_rot_xyzw))

    frames = root_pos.shape[-2]
    zero_xy = root_pos.new_zeros(root_pos.shape[:-2] + (1, 2))
    if frames > 1:
        displacement = root_pos[..., 1:, :] - root_pos[..., :-1, :]
        local_displacement = torch.matmul(
            root_mats[..., :-1, :, :].transpose(-1, -2),
            displacement.unsqueeze(-1),
        ).squeeze(-1)
        delta_xy = torch.cat((zero_xy, local_displacement[..., :2]), dim=-2)
        relative_mats = torch.matmul(
            root_mats[..., :-1, :, :].transpose(-1, -2),
            root_mats[..., 1:, :, :],
        )
        delta_mats = torch.cat(
            (_identity_matrices_like(root_pos, frame_count=1), relative_mats),
            dim=-3,
        )
    else:
        delta_xy = zero_xy
        delta_mats = _identity_matrices_like(root_pos, frame_count=1)

    root_height = root_pos[..., 2:3]
    delta_rot_6d = matrix_to_rotation_6d(delta_mats)
    return torch.cat((delta_xy, root_height, delta_rot_6d, dof_pos), dim=-1)


def encode_g1_yaw_delta_motion(root_pos, root_rot, dof_pos):
    root_pos = _float_tensor(root_pos)
    root_rot = _float_tensor(root_rot)
    dof_pos = _float_tensor(dof_pos)
    _require_shape(root_pos, (3,), "root_pos")
    _require_shape(root_rot, (4,), "root_rot")
    _require_shape(dof_pos, (G1_DOF_DIM,), "dof_pos")
    if root_pos.shape[:-1] != root_rot.shape[:-1] or root_pos.shape[:-1] != dof_pos.shape[:-1]:
        raise ValueError("G1 root_pos, root_rot, and dof_pos must share leading dimensions")
    if root_pos.ndim < 2:
        raise ValueError("G1 yaw-delta encoding requires an explicit frame dimension")

    yaw = _yaw_from_xyzw_quaternion(root_rot)
    frames = root_pos.shape[-2]
    zero_xy = root_pos.new_zeros(root_pos.shape[:-2] + (1, 2))
    zero_yaw = yaw.new_zeros(yaw.shape[:-1] + (1,))
    if frames > 1:
        displacement = root_pos[..., 1:, :2] - root_pos[..., :-1, :2]
        prev_yaw = yaw[..., :-1]
        cos_yaw = torch.cos(prev_yaw)
        sin_yaw = torch.sin(prev_yaw)
        local_x = cos_yaw * displacement[..., 0] + sin_yaw * displacement[..., 1]
        local_y = -sin_yaw * displacement[..., 0] + cos_yaw * displacement[..., 1]
        delta_xy = torch.cat((zero_xy, torch.stack((local_x, local_y), dim=-1)), dim=-2)
        delta_yaw = torch.cat((zero_yaw, _wrap_angle(yaw[..., 1:] - yaw[..., :-1])), dim=-1)
    else:
        delta_xy = zero_xy
        delta_yaw = zero_yaw

    root_height = root_pos[..., 2:3]
    delta_yaw_sincos = torch.stack((torch.sin(delta_yaw), torch.cos(delta_yaw)), dim=-1)
    return torch.cat((delta_xy, root_height, delta_yaw_sincos, dof_pos), dim=-1)


def encode_g1_motion_for_format(root_pos, root_rot, dof_pos, motion_format=G1_MOTION_FORMAT):
    motion_format = validate_motion_format(motion_format)
    if motion_format == G1_ROOT_DELTA_MOTION_FORMAT:
        return encode_g1_root_delta_motion(root_pos, root_rot, dof_pos)
    if motion_format == G1_YAW_DELTA_MOTION_FORMAT:
        return encode_g1_yaw_delta_motion(root_pos, root_rot, dof_pos)
    if motion_format == G1_MOTION_FORMAT:
        return encode_g1_motion(root_pos, root_rot, dof_pos)
    raise ValueError(f"{motion_format!r} is not a G1 motion format")


def decode_g1_absolute_motion(samples):
    samples = _float_tensor(samples)
    _require_shape(samples, (G1_REPR_DIM,), "samples")
    root_pos = samples[..., :3]
    root_rot_6d = samples[..., 3:9]
    dof_pos = samples[..., 9:]
    root_rot = matrix_to_quaternion(rotation_6d_to_matrix(root_rot_6d))
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def decode_g1_root_delta_motion(samples):
    samples = _float_tensor(samples)
    _require_shape(samples, (G1_REPR_DIM,), "samples")
    if samples.ndim < 2:
        raise ValueError("G1 root-delta decoding requires an explicit frame dimension")

    delta_xy = samples[..., :2]
    root_height = samples[..., 2:3]
    delta_rot_6d = samples[..., 3:9]
    dof_pos = samples[..., 9:]
    delta_mats = rotation_6d_to_matrix(delta_rot_6d)

    leading_shape = samples.shape[:-2]
    frames = samples.shape[-2]
    current_mat = _identity_matrices_like(samples, frame_count=1)[..., 0, :, :]
    current_xy = samples.new_zeros(leading_shape + (2,))
    root_positions = []
    root_mats = []
    for frame in range(frames):
        if frame > 0:
            local_delta = torch.cat(
                (
                    delta_xy[..., frame, :],
                    samples.new_zeros(leading_shape + (1,)),
                ),
                dim=-1,
            )
            world_delta = torch.matmul(current_mat, local_delta.unsqueeze(-1)).squeeze(-1)
            current_xy = current_xy + world_delta[..., :2]
            current_mat = torch.matmul(current_mat, delta_mats[..., frame, :, :])
        root_positions.append(torch.cat((current_xy, root_height[..., frame, :]), dim=-1))
        root_mats.append(current_mat)

    root_pos = torch.stack(root_positions, dim=-2)
    root_mat = torch.stack(root_mats, dim=-3)
    root_rot = _wxyz_to_xyzw(matrix_to_quaternion(root_mat))
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def decode_g1_yaw_delta_motion(samples):
    samples = _float_tensor(samples)
    _require_shape(samples, (G1_YAW_DELTA_REPR_DIM,), "samples")
    if samples.ndim < 2:
        raise ValueError("G1 yaw-delta decoding requires an explicit frame dimension")

    delta_xy = samples[..., :2]
    root_height = samples[..., 2:3]
    delta_yaw_sincos = samples[..., 3:5]
    dof_pos = samples[..., 5:]
    delta_yaw = torch.atan2(delta_yaw_sincos[..., 0], delta_yaw_sincos[..., 1])

    leading_shape = samples.shape[:-2]
    frames = samples.shape[-2]
    current_xy = samples.new_zeros(leading_shape + (2,))
    current_yaw = samples.new_zeros(leading_shape)
    root_positions = []
    root_rots = []
    for frame in range(frames):
        if frame > 0:
            cos_yaw = torch.cos(current_yaw)
            sin_yaw = torch.sin(current_yaw)
            local_delta = delta_xy[..., frame, :]
            world_delta = torch.stack(
                (
                    cos_yaw * local_delta[..., 0] - sin_yaw * local_delta[..., 1],
                    sin_yaw * local_delta[..., 0] + cos_yaw * local_delta[..., 1],
                ),
                dim=-1,
            )
            current_xy = current_xy + world_delta
            current_yaw = current_yaw + delta_yaw[..., frame]
        root_positions.append(torch.cat((current_xy, root_height[..., frame, :]), dim=-1))
        root_rots.append(_yaw_xyzw_quaternion(current_yaw))

    root_pos = torch.stack(root_positions, dim=-2)
    root_rot = torch.stack(root_rots, dim=-2)
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def decode_g1_motion(samples, motion_format=G1_MOTION_FORMAT):
    motion_format = validate_motion_format(motion_format)
    if motion_format == G1_ROOT_DELTA_MOTION_FORMAT:
        return decode_g1_root_delta_motion(samples)
    if motion_format == G1_YAW_DELTA_MOTION_FORMAT:
        return decode_g1_yaw_delta_motion(samples)
    if motion_format == G1_MOTION_FORMAT:
        return decode_g1_absolute_motion(samples)
    raise ValueError(f"{motion_format!r} is not a G1 motion format")
