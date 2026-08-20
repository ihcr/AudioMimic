import unittest

import numpy as np

from eval.render_blind_g1_video import resample_window


class BlindG1VideoRenderTests(unittest.TestCase):
    def test_resample_window_converts_wxyz_and_preserves_requested_clock(self):
        frames = 6
        motion = {
            "fps": 2.0,
            "root_pos": np.column_stack(
                (np.arange(frames, dtype=float), np.zeros(frames), np.ones(frames))
            ),
            "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
            "dof_pos": np.tile(np.arange(frames, dtype=float)[:, None], (1, 29)),
        }

        output = resample_window(
            motion,
            quat_order="wxyz",
            start_seconds=0.5,
            duration_seconds=1.0,
            output_fps=2.0,
        )

        np.testing.assert_allclose(output["root_pos"][:, 0], [1.0, 2.0])
        np.testing.assert_allclose(output["root_rot"], [[0.0, 0.0, 0.0, 1.0]] * 2)
        self.assertEqual(output["dof_pos"].shape, (2, 29))

    def test_resample_window_rejects_window_past_source(self):
        motion = {
            "fps": 2.0,
            "root_pos": np.zeros((4, 3)),
            "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
            "dof_pos": np.zeros((4, 29)),
        }

        with self.assertRaisesRegex(ValueError, "exceeds source duration"):
            resample_window(
                motion,
                quat_order="xyzw",
                start_seconds=1.0,
                duration_seconds=2.0,
                output_fps=2.0,
            )


if __name__ == "__main__":
    unittest.main()
