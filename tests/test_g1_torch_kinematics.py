import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class G1TorchKinematicsTests(unittest.TestCase):
    def test_matches_mujoco_keypoints_with_xyzw_root_quaternion(self):
        torch_kinematics = reload_module("model.g1_torch_kinematics")
        mujoco_kinematics = reload_module("eval.g1_kinematics")
        model_path = Path("third_party/unitree_g1_description/g1_29dof_rev_1_0.xml")
        if not model_path.is_file():
            self.skipTest(f"missing G1 MuJoCo model: {model_path}")

        root_pos = np.array(
            [[0.0, 0.0, 0.84], [0.03, -0.02, 0.85]],
            dtype=np.float32,
        )
        root_rot = np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.24740396, 0.9689124]],
            dtype=np.float32,
        )
        dof_pos = np.zeros((2, 29), dtype=np.float32)
        dof_pos[:, 0] = [0.1, -0.2]
        dof_pos[:, 3] = [0.2, 0.35]
        dof_pos[:, 15] = [-0.15, 0.1]

        expected = mujoco_kinematics.forward_g1_kinematics(
            {"root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos},
            model_path,
            root_quat_order="xyzw",
        )
        model = torch_kinematics.G1TorchKinematics(model_path, root_quat_order="xyzw")
        actual = model(
            torch.from_numpy(root_pos),
            torch.from_numpy(root_rot),
            torch.from_numpy(dof_pos),
        )

        np.testing.assert_allclose(
            actual["keypoints"].detach().numpy(),
            expected["keypoints"],
            atol=1e-5,
        )

    def test_outputs_are_differentiable(self):
        torch_kinematics = reload_module("model.g1_torch_kinematics")
        model_path = Path("third_party/unitree_g1_description/g1_29dof_rev_1_0.xml")
        if not model_path.is_file():
            self.skipTest(f"missing G1 MuJoCo model: {model_path}")

        model = torch_kinematics.G1TorchKinematics(model_path, root_quat_order="xyzw")
        root_pos = torch.tensor([[[0.0, 0.0, 0.84]]], requires_grad=True)
        root_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], requires_grad=True)
        dof_pos = torch.zeros(1, 1, 29, requires_grad=True)
        output = model(root_pos, root_rot, dof_pos)
        loss = output["keypoints"].pow(2).sum()

        loss.backward()

        self.assertIsNotNone(root_pos.grad)
        self.assertIsNotNone(root_rot.grad)
        self.assertIsNotNone(dof_pos.grad)
        self.assertTrue(torch.isfinite(root_pos.grad).all())
        self.assertTrue(torch.isfinite(root_rot.grad).all())
        self.assertTrue(torch.isfinite(dof_pos.grad).all())


if __name__ == "__main__":
    unittest.main()
