import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def write_motion(path, dof_offset=0.0, root_offset=0.0):
    frames = 4
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 0] = root_offset
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    dof_pos = np.zeros((frames, 29), dtype=np.float32)
    dof_pos[:, 0] = np.linspace(0.0, 0.3, frames, dtype=np.float32) + dof_offset
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "q": np.concatenate([root_rot, dof_pos], axis=-1),
            },
            handle,
        )


class G1ConditionSensitivityTests(unittest.TestCase):
    def test_infer_motion_key_handles_labels_with_underscores(self):
        from eval.g1_condition_sensitivity import infer_motion_key

        self.assertEqual(
            infer_motion_key("no_beat_uncond_12_036_slice34_g1.pkl"),
            "036_slice34",
        )
        self.assertEqual(
            infer_motion_key("g1_eval_0_137_slice62_g1.pkl"),
            "137_slice62",
        )

    def test_summarize_variant_compares_paired_dof_outputs(self):
        from eval.g1_condition_sensitivity import summarize_variant

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "no_beat_uncond"
            variant = root / "random"
            baseline.mkdir()
            variant.mkdir()
            write_motion(baseline / "no_beat_uncond_0_001_slice0_g1.pkl", dof_offset=0.0)
            write_motion(variant / "random_0_001_slice0_g1.pkl", dof_offset=0.2)

            summary = summarize_variant(
                "random",
                variant,
                baseline,
                active_dof_threshold=0.05,
            )

        self.assertEqual(summary["num_pairs"], 1)
        self.assertGreater(summary["DofRMSEMean"], 0.0)
        self.assertEqual(summary["ActiveDofDeltaCountMean"], 1.0)
        self.assertEqual(
            summary["top_changed_dofs"][0]["joint_name"],
            "left_hip_pitch_joint",
        )


if __name__ == "__main__":
    unittest.main()
