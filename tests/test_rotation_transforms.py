import math
import unittest

import torch

from rotation_transforms import (
    RotateAxisAngle,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


class RotationTransformTests(unittest.TestCase):
    def test_axis_angle_matrix_round_trip(self):
        axis_angle = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32)
        matrix = axis_angle_to_matrix(axis_angle)
        recovered = matrix_to_axis_angle(matrix)
        self.assertTrue(torch.allclose(recovered, axis_angle, atol=1e-5, rtol=1e-5))

    def test_quaternion_matrix_round_trip(self):
        quat = torch.tensor([[0.9238795, 0.0, 0.3826834, 0.0]], dtype=torch.float32)
        matrix = quaternion_to_matrix(quat)
        recovered = matrix_to_quaternion(matrix)
        self.assertTrue(torch.allclose(recovered, quat, atol=1e-5, rtol=1e-5))

    def test_rotation_6d_round_trip(self):
        axis_angle = torch.tensor([[0.0, math.pi / 4, 0.0]], dtype=torch.float32)
        matrix = axis_angle_to_matrix(axis_angle)
        rot6d = matrix_to_rotation_6d(matrix)
        recovered = rotation_6d_to_matrix(rot6d)
        self.assertTrue(torch.allclose(recovered, matrix, atol=1e-5, rtol=1e-5))

    def test_quaternion_apply_and_multiply(self):
        half_turn_z = torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=torch.float32)
        quat = axis_angle_to_quaternion(half_turn_z)
        point = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        rotated = quaternion_apply(quat, point)
        expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(rotated, expected, atol=1e-5, rtol=1e-5))

        doubled = quaternion_multiply(quat, quat)
        recovered = quaternion_to_axis_angle(doubled)
        expected_axis_angle = torch.tensor([[0.0, 0.0, math.pi]], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(recovered.abs(), expected_axis_angle.abs(), atol=1e-5, rtol=1e-5)
        )

    def test_rotate_axis_angle_points(self):
        rot = RotateAxisAngle(90, axis="X", degrees=True)
        points = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        rotated = rot.transform_points(points)
        expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(rotated, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
