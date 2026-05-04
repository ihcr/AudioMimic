import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class G1ModelFetchTests(unittest.TestCase):
    def test_extracts_mesh_urls_from_mjcf(self):
        fetcher = reload_module("scripts.fetch_unitree_g1_description")

        xml = '<mujoco><asset><mesh file="pelvis.STL"/><mesh file="leg.STL"/></asset></mujoco>'

        self.assertEqual(fetcher.referenced_mesh_files(xml), ["leg.STL", "pelvis.STL"])


class G1KinematicsTests(unittest.TestCase):
    def test_build_g1_qpos_uses_mujoco_free_joint_wxyz_order(self):
        kinematics = reload_module("eval.g1_kinematics")
        root_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        root_rot = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        dof_pos = np.arange(29, dtype=np.float32)[None]

        qpos = kinematics.build_g1_qpos(
            root_pos,
            root_rot,
            dof_pos,
            root_quat_order="wxyz",
        )

        self.assertEqual(qpos.shape, (1, 36))
        np.testing.assert_allclose(qpos[0, :7], [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(qpos[0, 7:], np.arange(29, dtype=np.float32))

    def test_build_g1_qpos_converts_xyzw_to_mujoco_wxyz(self):
        kinematics = reload_module("eval.g1_kinematics")
        root_pos = np.zeros((1, 3), dtype=np.float32)
        root_rot = np.array([[0.0, 0.0, 0.0, 2.0]], dtype=np.float32)
        dof_pos = np.zeros((1, 29), dtype=np.float32)

        qpos = kinematics.build_g1_qpos(
            root_pos,
            root_rot,
            dof_pos,
            root_quat_order="xyzw",
        )

        np.testing.assert_allclose(qpos[0, 3:7], [1.0, 0.0, 0.0, 0.0])

    def test_validate_g1_joint_order_rejects_mismatched_order(self):
        kinematics = reload_module("eval.g1_kinematics")
        wrong = list(kinematics.EXPECTED_G1_29DOF_JOINTS)
        wrong[0], wrong[1] = wrong[1], wrong[0]

        with self.assertRaisesRegex(ValueError, "G1 joint order mismatch"):
            kinematics.validate_g1_joint_order(wrong)

    def test_forward_kinematics_reports_missing_model_path(self):
        kinematics = reload_module("eval.g1_kinematics")
        motion = {
            "root_pos": np.zeros((2, 3), dtype=np.float32),
            "root_rot": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
            "dof_pos": np.zeros((2, 29), dtype=np.float32),
            "fps": 30.0,
        }

        with TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.xml"
            with self.assertRaisesRegex(FileNotFoundError, "G1 MuJoCo model"):
                kinematics.forward_g1_kinematics(motion, missing)


if __name__ == "__main__":
    unittest.main()
