import unittest

import torch

from rotation_transforms import (RotateAxisAngle, matrix_to_axis_angle,
                                 rotation_6d_to_matrix)


class RotationTransformStabilityTests(unittest.TestCase):
    def test_zero_rotation_6d_maps_to_zero_axis_angle_with_finite_gradients(self):
        d6 = torch.zeros(2, 6, requires_grad=True)

        axis_angle = matrix_to_axis_angle(rotation_6d_to_matrix(d6))
        loss = axis_angle.square().sum()
        loss.backward()

        self.assertTrue(torch.isfinite(axis_angle).all())
        self.assertTrue(torch.allclose(axis_angle, torch.zeros_like(axis_angle), atol=1e-6))
        self.assertIsNotNone(d6.grad)
        self.assertTrue(torch.isfinite(d6.grad).all())

    def test_rotate_axis_angle_x_matches_expected_point_rotation(self):
        rot = RotateAxisAngle(90, axis="X", degrees=True)
        points = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

        rotated = rot.transform_points(points)

        expected = torch.tensor([[1.0, -3.0, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(rotated, expected, atol=1e-5))

    def test_collinear_rotation_6d_keeps_gradients_finite(self):
        d6 = torch.tensor(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            requires_grad=True,
        )

        axis_angle = matrix_to_axis_angle(rotation_6d_to_matrix(d6))
        loss = axis_angle.square().sum()
        loss.backward()

        self.assertTrue(torch.isfinite(axis_angle).all())
        self.assertIsNotNone(d6.grad)
        self.assertTrue(torch.isfinite(d6.grad).all())


if __name__ == "__main__":
    unittest.main()
