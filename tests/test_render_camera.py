import unittest

import numpy as np

import render_camera


class RenderCameraTests(unittest.TestCase):
    def test_compute_axes_limits_covers_full_pose_sequence(self):
        poses = np.zeros((2, 24, 3), dtype=np.float32)
        poses[:, :, 2] = 2.5
        poses[1, :, 0] = 4.0
        poses[1, :, 1] = -2.0

        xlim, ylim, zlim = render_camera.compute_axes_limits(poses)

        self.assertLessEqual(xlim[0], float(poses[:, :, 0].min()))
        self.assertGreaterEqual(xlim[1], float(poses[:, :, 0].max()))
        self.assertLessEqual(ylim[0], float(poses[:, :, 1].min()))
        self.assertGreaterEqual(ylim[1], float(poses[:, :, 1].max()))
        self.assertLessEqual(zlim[0], float(poses[:, :, 2].min()))
        self.assertGreaterEqual(zlim[1], float(poses[:, :, 2].max()))

    def test_compute_axes_limits_keeps_minimum_view_span(self):
        poses = np.zeros((1, 24, 3), dtype=np.float32)
        poses[:, :, 2] = 2.5

        xlim, ylim, zlim = render_camera.compute_axes_limits(poses, min_span=3.0)

        self.assertGreaterEqual(xlim[1] - xlim[0], 3.0)
        self.assertGreaterEqual(ylim[1] - ylim[0], 3.0)
        self.assertGreaterEqual(zlim[1] - zlim[0], 3.0)


if __name__ == "__main__":
    unittest.main()
