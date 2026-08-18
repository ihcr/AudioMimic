import unittest

import numpy as np

from sonic_bridge import G1Motion, SONIC_REFERENCE_FROM_MUJOCO
from stream_to_sonic import (
    build_offline_reference_packets,
    prepend_feedback_alignment,
    retime_g1_motion,
)


class OfflineSonicPlaybackTests(unittest.TestCase):
    def test_retime_motion_changes_duration_and_interpolates_at_30hz(self):
        motion = G1Motion(
            root_rot_xyzw=np.tile(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (12, 1)
            ),
            dof_pos_mujoco=np.arange(12, dtype=np.float32)[:, None]
            * np.ones((1, 29), dtype=np.float32),
            root_pos=np.arange(12, dtype=np.float32)[:, None]
            * np.ones((1, 3), dtype=np.float32),
            fps=30.0,
        )
        faster = retime_g1_motion(motion, 1.5)
        slower = retime_g1_motion(motion, 0.75)
        self.assertEqual(faster.frames, 8)
        self.assertEqual(slower.frames, 16)
        self.assertEqual(faster.fps, 30.0)
        self.assertAlmostEqual(float(faster.dof_pos_mujoco[1, 0]), 1.5)
        self.assertAlmostEqual(float(slower.dof_pos_mujoco[1, 0]), 0.75)
        self.assertTrue(np.allclose(np.linalg.norm(faster.root_rot_xyzw, axis=1), 1.0))

    def test_fixed_motion_is_resampled_and_packetized_without_feedback(self):
        frames = 24
        motion = G1Motion(
            root_rot_xyzw=np.tile(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)
            ),
            dof_pos_mujoco=np.arange(frames * 29, dtype=np.float32).reshape(frames, 29),
            root_pos=np.zeros((frames, 3), dtype=np.float32),
            fps=30.0,
        )
        packets = list(
            build_offline_reference_packets(
                motion,
                target_fps=50.0,
                safety_enabled=False,
                max_joint_velocity_rad_s=0.0,
                max_yaw_velocity_rad_s=0.0,
                ramp_seconds=0.0,
                joint_limit_margin_rad=0.0,
            )
        )
        self.assertEqual(
            [(start, stop) for start, stop, *_ in packets],
            [(0, 8), (8, 16), (16, 24)],
        )
        self.assertEqual([reference.frames for _, _, reference, _, _ in packets], [13, 13, 14])
        fields = [packet[3] for packet in packets]
        self.assertEqual(fields[0]["frame_index"].tolist(), list(range(13)))
        self.assertEqual(fields[1]["frame_index"].tolist(), list(range(13, 26)))
        self.assertEqual(fields[2]["frame_index"].tolist(), list(range(26, 40)))
        self.assertTrue(
            np.array_equal(
                fields[0]["joint_pos"][0],
                motion.dof_pos_mujoco[0, SONIC_REFERENCE_FROM_MUJOCO],
            )
        )
        self.assertEqual(packets[0][4]["safety_enabled"], 0)

    def test_oracle_preview_slides_while_frame_index_advances_by_c4(self):
        frames = 40
        motion = G1Motion(
            root_rot_xyzw=np.tile(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)
            ),
            dof_pos_mujoco=np.arange(frames * 29, dtype=np.float32).reshape(frames, 29),
            root_pos=np.zeros((frames, 3), dtype=np.float32),
            fps=30.0,
        )
        packets = list(
            build_offline_reference_packets(
                motion,
                target_fps=50.0,
                safety_enabled=False,
                max_joint_velocity_rad_s=0.0,
                max_yaw_velocity_rad_s=0.0,
                ramp_seconds=0.0,
                joint_limit_margin_rad=0.0,
                preview_seconds=1.0,
            )
        )
        self.assertEqual([(p[0], p[1]) for p in packets[:2]], [(0, 30), (8, 38)])
        self.assertEqual([p[2].frames for p in packets[:2]], [50, 50])
        self.assertEqual(int(packets[0][3]["frame_index"][0]), 0)
        self.assertEqual(int(packets[1][3]["frame_index"][0]), 13)

    def test_full_mode_sends_one_continuous_reference(self):
        frames = 24
        motion = G1Motion(
            root_rot_xyzw=np.tile(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)
            ),
            dof_pos_mujoco=np.zeros((frames, 29), dtype=np.float32),
            root_pos=np.zeros((frames, 3), dtype=np.float32),
            fps=30.0,
        )
        packets = list(
            build_offline_reference_packets(
                motion,
                target_fps=50.0,
                safety_enabled=False,
                max_joint_velocity_rad_s=0.0,
                max_yaw_velocity_rad_s=0.0,
                ramp_seconds=0.0,
                joint_limit_margin_rad=0.0,
                packet_mode="full",
            )
        )
        self.assertEqual(len(packets), 1)
        self.assertEqual((packets[0][0], packets[0][1]), (0, frames))
        self.assertEqual(packets[0][2].frames, 40)
        self.assertEqual(packets[0][3]["frame_index"].tolist(), list(range(40)))

    def test_feedback_alignment_starts_measured_and_ends_at_motion(self):
        motion = G1Motion(
            root_rot_xyzw=np.tile(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (8, 1)
            ),
            dof_pos_mujoco=np.ones((8, 29), dtype=np.float32),
            root_pos=np.zeros((8, 3), dtype=np.float32),
            fps=30.0,
        )
        measured_sonic = np.arange(29, dtype=np.float32) / 100.0
        aligned, prefix_frames = prepend_feedback_alignment(
            motion,
            {
                "body_q_measured": measured_sonic,
                "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
            },
            align_seconds=1.0,
            hold_seconds=0.5,
        )
        expected_mujoco = np.empty(29, dtype=np.float32)
        expected_mujoco[SONIC_REFERENCE_FROM_MUJOCO] = measured_sonic
        self.assertEqual(prefix_frames, 45)
        self.assertTrue(np.allclose(aligned.dof_pos_mujoco[0], expected_mujoco))
        self.assertTrue(np.allclose(aligned.dof_pos_mujoco[29], motion.dof_pos_mujoco[0]))
        self.assertTrue(np.allclose(aligned.dof_pos_mujoco[44], motion.dof_pos_mujoco[0]))
        self.assertTrue(np.allclose(aligned.dof_pos_mujoco[45], motion.dof_pos_mujoco[0]))


if __name__ == "__main__":
    unittest.main()
