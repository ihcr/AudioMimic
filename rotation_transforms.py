import math

import torch
import torch.nn.functional as F


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., :1] < 0, -quaternions, quaternions)


def _normalize_quaternion(quaternions):
    return F.normalize(quaternions, dim=-1)


def _quaternion_conjugate(quaternions):
    result = quaternions.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def _copysign(a, b):
    return torch.where(b < 0, -a, a)


def axis_angle_to_quaternion(axis_angle):
    angles = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    small = angles < 1e-8
    scale = torch.empty_like(angles)
    scale[~small] = torch.sin(half_angles[~small]) / angles[~small]
    scale[small] = 0.5 - (angles[small] * angles[small]) / 48.0
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * scale], dim=-1)
    return _normalize_quaternion(quaternions)


def quaternion_to_axis_angle(quaternions):
    quaternions = _normalize_quaternion(quaternions)
    xyz = quaternions[..., 1:]
    norms = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2.0 * half_angles
    small = norms < 1e-8
    scale = torch.empty_like(norms)
    scale[~small] = angles[~small] / norms[~small]
    scale[small] = 2.0
    return xyz * scale


def quaternion_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, dim=-1)
    bw, bx, by, bz = torch.unbind(b, dim=-1)
    product = torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )
    return standardize_quaternion(product)


def quaternion_apply(quaternions, points):
    quaternions = _normalize_quaternion(quaternions)
    point_quaternions = torch.cat([torch.zeros_like(points[..., :1]), points], dim=-1)
    rotated = quaternion_multiply(
        quaternion_multiply(quaternions, point_quaternions),
        _quaternion_conjugate(quaternions),
    )
    return rotated[..., 1:]


def quaternion_to_matrix(quaternions):
    quaternions = _normalize_quaternion(quaternions)
    w, x, y, z = torch.unbind(quaternions, dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected rotation matrices with shape (..., 3, 3)")

    batch_dim = matrix.shape[:-2]
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    denom = torch.clamp(2.0 * q_abs[..., None], min=0.1)
    quat_candidates = quat_by_rijk / denom
    best = F.one_hot(q_abs.argmax(dim=-1), num_classes=4).bool()
    quaternions = quat_candidates[best].reshape(batch_dim + (4,))
    return standardize_quaternion(_normalize_quaternion(quaternions))


def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class RotateAxisAngle:
    def __init__(self, angle, axis="X", degrees=True):
        if degrees:
            angle = math.radians(angle)
        axis_map = {
            "X": torch.tensor([1.0, 0.0, 0.0]),
            "Y": torch.tensor([0.0, 1.0, 0.0]),
            "Z": torch.tensor([0.0, 0.0, 1.0]),
        }
        if axis not in axis_map:
            raise ValueError(f"Unsupported axis {axis!r}")
        self.axis_angle = axis_map[axis] * angle

    def transform_points(self, points):
        axis_angle = self.axis_angle.to(device=points.device, dtype=points.dtype)
        quat = axis_angle_to_quaternion(axis_angle.expand(points.shape[:-1] + (3,)))
        return quaternion_apply(quat, points)
