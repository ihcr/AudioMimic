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
SMPL_REPR_DIM = 3 + 24 * 6 + 4
G1_DOF_DIM = 29
G1_REPR_DIM = 3 + 6 + G1_DOF_DIM
VALID_MOTION_FORMATS = (SMPL_MOTION_FORMAT, G1_MOTION_FORMAT)


def validate_motion_format(motion_format):
    if motion_format not in VALID_MOTION_FORMATS:
        raise ValueError(
            f"Unsupported motion_format {motion_format!r}; expected one of {VALID_MOTION_FORMATS}"
        )
    return motion_format


def motion_repr_dim(motion_format):
    validate_motion_format(motion_format)
    if motion_format == G1_MOTION_FORMAT:
        return G1_REPR_DIM
    return SMPL_REPR_DIM


def _float_tensor(value):
    if torch.is_tensor(value):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


def _require_shape(tensor, suffix, name):
    if tuple(tensor.shape[-len(suffix) :]) != tuple(suffix):
        raise ValueError(f"{name} expected shape ending in {suffix}, got {tuple(tensor.shape)}")


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


def decode_g1_motion(samples):
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
