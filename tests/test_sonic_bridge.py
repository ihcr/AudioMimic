import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sonic_bridge import (
    G1Motion,
    SONIC_REFERENCE_FROM_MUJOCO,
    SonicReferenceAdapter,
    SonicExecutionStateError,
    SonicS66Builder,
    SonicS66Synchronizer,
    g1_motion_from_yaw_delta_commit,
    iter_reference_commits,
    load_g1_motion,
    pack_zmq_message,
    reference_fields_from_commit,
    unpack_sonic_feedback_message,
    unpack_zmq_message,
)


class SonicBridgeTests(unittest.TestCase):
    def _motion(self, frames=10):
        root_rot = np.zeros((frames, 4), dtype=np.float32)
        root_rot[:, 3] = 1.0
        dof_pos = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 100.0
        return G1Motion(root_rot_xyzw=root_rot, dof_pos_mujoco=dof_pos, fps=30.0)

    def test_v1_packet_round_trip_preserves_commit_fields(self):
        motion = self._motion(frames=9)
        _, _, fields = next(iter_reference_commits(motion, commit_frames=8))
        header, unpacked = unpack_zmq_message(pack_zmq_message(fields, topic="pose", version=1))
        self.assertEqual(header["v"], 1)
        self.assertEqual(header["count"], 8)
        self.assertEqual(tuple(unpacked["joint_pos"].shape), (8, 29))
        self.assertEqual(tuple(unpacked["joint_vel"].shape), (8, 29))
        self.assertEqual(tuple(unpacked["body_quat_w"].shape), (8, 4))
        self.assertEqual(unpacked["body_quat_w"][0].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(unpacked["catch_up"].tolist(), [0])
        self.assertEqual(unpacked["frame_index"].tolist(), list(range(8)))

    def test_commit_velocity_is_continuous_across_packet_boundary(self):
        motion = self._motion(frames=10)
        joint_slopes = np.arange(1, 30, dtype=np.float32)
        motion = G1Motion(
            root_rot_xyzw=motion.root_rot_xyzw,
            dof_pos_mujoco=np.arange(motion.frames, dtype=np.float32)[:, None] * joint_slopes[None],
            fps=motion.fps,
        )
        commits = list(iter_reference_commits(motion, commit_frames=8))
        first = commits[0][2]
        second = commits[1][2]
        expected = (
            (motion.dof_pos_mujoco[8] - motion.dof_pos_mujoco[7]) * motion.fps
        )[SONIC_REFERENCE_FROM_MUJOCO]
        self.assertTrue(np.allclose(second["joint_vel"][0], expected))
        self.assertEqual(first["frame_index"].tolist(), list(range(8)))
        self.assertEqual(second["frame_index"].tolist(), [8, 9])

    def test_yaw_delta_commit_is_anchored_at_execution_pose(self):
        raw = np.zeros((3, 34), dtype=np.float32)
        raw[:, 2] = [0.79, 0.80, 0.81]
        raw[:, 4] = 1.0
        raw[1, :2] = [1.0, 0.0]
        raw[1, 3:5] = [1.0, 0.0]
        raw[2, :2] = [1.0, 0.0]
        raw[:, 5:] = np.arange(29, dtype=np.float32)
        motion = g1_motion_from_yaw_delta_commit(
            raw,
            base_position=[10.0, 20.0, 0.78],
            base_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        self.assertTrue(
            np.allclose(
                motion.root_pos,
                [[10.0, 20.0, 0.79], [11.0, 20.0, 0.80], [11.0, 21.0, 0.81]],
            )
        )
        self.assertTrue(
            np.allclose(
                motion.root_rot_xyzw[1],
                [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            )
        )
        fields = reference_fields_from_commit(
            motion,
            start_frame=0,
            stop_frame=3,
            previous_dof_pos=np.full(29, -1.0, dtype=np.float32),
            frame_index_start=80,
        )
        self.assertEqual(fields["frame_index"].tolist(), [80, 81, 82])
        self.assertTrue(
            np.allclose(
                fields["joint_pos"][0], raw[0, 5:][SONIC_REFERENCE_FROM_MUJOCO]
            )
        )

    def test_reference_adapter_resamples_30hz_commits_without_drift(self):
        source = self._motion(frames=8)
        source = G1Motion(
            root_rot_xyzw=source.root_rot_xyzw,
            dof_pos_mujoco=(np.arange(8, dtype=np.float32) / 10.0)[:, None]
            * np.ones((1, 29), dtype=np.float32),
            fps=30.0,
        )
        adapter = SonicReferenceAdapter(
            target_fps=50.0,
            max_joint_velocity_rad_s=1000.0,
            max_yaw_velocity_rad_s=1000.0,
            ramp_seconds=0.0,
            joint_limit_margin_rad=0.0,
        )
        first = adapter.adapt(
            source,
            execution_dof_mujoco=source.dof_pos_mujoco[0],
            execution_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        second_source = G1Motion(
            root_rot_xyzw=source.root_rot_xyzw,
            dof_pos_mujoco=(np.arange(8, 16, dtype=np.float32) / 10.0)[:, None]
            * np.ones((1, 29), dtype=np.float32),
            fps=30.0,
        )
        second = adapter.adapt(
            second_source,
            execution_dof_mujoco=source.dof_pos_mujoco[-1],
            execution_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(first.fps, 50.0)
        self.assertEqual(first.frames, 13)
        self.assertEqual(second.frames, 13)
        self.assertAlmostEqual(float(second.dof_pos_mujoco[0, 0]), 0.78, places=5)

    def test_reference_packet_uses_sonic_policy_order(self):
        motion = self._motion(frames=1)
        fields = reference_fields_from_commit(motion, start_frame=0, stop_frame=1)
        self.assertTrue(
            np.array_equal(
                fields["joint_pos"][0],
                motion.dof_pos_mujoco[0, SONIC_REFERENCE_FROM_MUJOCO],
            )
        )

    def test_reference_adapter_can_preserve_unlimited_diffusion_values(self):
        source = self._motion(frames=8)
        source = G1Motion(
            root_rot_xyzw=source.root_rot_xyzw,
            dof_pos_mujoco=np.full((8, 29), 3.5, dtype=np.float32),
            fps=30.0,
        )
        adapter = SonicReferenceAdapter(
            target_fps=50.0,
            safety_enabled=False,
            max_joint_velocity_rad_s=0.0,
            max_yaw_velocity_rad_s=0.0,
            ramp_seconds=0.0,
            joint_limit_margin_rad=0.0,
        )
        reference = adapter.adapt(
            source,
            execution_dof_mujoco=np.zeros(29, dtype=np.float32),
            execution_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        self.assertTrue(np.array_equal(reference.dof_pos_mujoco, np.full((13, 29), 3.5)))
        self.assertEqual(adapter.last_diagnostics["position_clipped_values"], 0)
        self.assertEqual(adapter.last_diagnostics["safety_enabled"], 0)

    def test_load_g1_motion_accepts_v6fx_export_shape(self):
        motion = self._motion(frames=3)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.pkl"
            with path.open("wb") as handle:
                pickle.dump(
                    {
                        "fps": 30.0,
                        "root_pos": np.zeros((3, 3), dtype=np.float32),
                        "root_rot": motion.root_rot_xyzw,
                        "dof_pos": motion.dof_pos_mujoco,
                    },
                    handle,
                )
            loaded = load_g1_motion(path)
        self.assertEqual(loaded.frames, 3)
        self.assertIsNotNone(loaded.root_pos)
        self.assertTrue(np.array_equal(loaded.root_pos, np.zeros((3, 3), dtype=np.float32)))
        self.assertTrue(np.array_equal(loaded.dof_pos_mujoco, motion.dof_pos_mujoco))

    def test_s66_builder_requires_true_base_translation_and_velocity(self):
        builder = SonicS66Builder()
        feedback = {
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "base_ang_vel": [0.0, 0.0, 0.2],
            "body_q": [0.0] * 29,
            "body_dq": [0.0] * 29,
        }
        with self.assertRaisesRegex(SonicExecutionStateError, "base_position"):
            builder.build(feedback)

    def test_s66_builder_matches_declared_layout(self):
        builder = SonicS66Builder(fps=30.0)
        feedback = {
            "base_position": [2.0, 3.0, 0.78],
            "base_linear_velocity": [1.0, 0.0, 0.1],
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "base_ang_vel": [0.0, 0.0, 0.2],
            "body_q": list(np.linspace(-0.2, 0.2, 29)),
            "body_dq": list(np.linspace(-0.3, 0.3, 29)),
        }
        state = builder.build(feedback)
        self.assertEqual(state.shape, (66,))
        self.assertAlmostEqual(float(state[0]), 0.78, places=6)
        self.assertTrue(np.allclose(state[1:3], [1.0, 0.0]))
        self.assertAlmostEqual(float(state[3]), 0.1, places=6)
        self.assertAlmostEqual(float(state[4]), 0.2, places=6)
        self.assertTrue(np.allclose(state[5:34], feedback["body_q"]))
        self.assertTrue(np.allclose(state[34:63], feedback["body_dq"]))
        self.assertTrue(np.allclose(state[63:], 0.0))

    def test_s66_synchronizer_requires_both_execution_streams(self):
        synchronizer = SonicS66Synchronizer(fps=30.0)
        sonic_feedback = {
            "index": 42,
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "base_ang_vel": [0.0, 0.0, 0.2],
            "body_q": [0.0] * 29,
            "body_dq": [0.0] * 29,
        }
        self.assertIsNone(synchronizer.update_sonic_feedback(sonic_feedback))
        synchronizer.update_sim_state(
            {
                "sim_time": 3.5,
                "base_position": [0.0, 0.0, 0.78],
                "base_linear_velocity": [0.2, 0.0, 0.0],
                "base_quat": [1.0, 0.0, 0.0, 0.0],
                "base_ang_vel": [0.0, 0.0, 0.2],
            }
        )
        result = synchronizer.update_sonic_feedback(sonic_feedback)
        self.assertEqual(result["sonic_feedback_index"], 42)
        self.assertEqual(result["sim_time"], 3.5)
        self.assertEqual(len(result["s66"]), 66)

    def test_msgpack_feedback_uses_topic_prefix(self):
        import msgpack

        message = b"g1_debug" + msgpack.packb({"body_q": [0.1, 0.2]}, use_bin_type=True)
        self.assertEqual(
            unpack_sonic_feedback_message(message), {"body_q": [0.1, 0.2]}
        )
        self.assertIsNone(unpack_sonic_feedback_message(message, topic="other"))


if __name__ == "__main__":
    unittest.main()
