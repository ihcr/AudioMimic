import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.prepare_sonic_capability_references import convert_reference
from eval.analyze_gt_sonic_capability import _planned_runs
from eval.select_gt_capability_references import _select_candidates
from sonic_bridge import SONIC_REFERENCE_FROM_MUJOCO
from scripts.run_sonic_capability_suite import _motion_spec


class GTCapabilitySelectionTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path, values):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"value_{index}" for index in range(values.shape[1])])
            writer.writerows(values)

    def test_sonic_reference_conversion_inverts_wire_joint_order(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_dir = Path(directory) / "reference"
            reference_dir.mkdir()
            frames = 10
            sonic_joints = np.tile(np.arange(29, dtype=np.float32), (frames, 1))
            body_pos = np.zeros((frames, 14 * 3), dtype=np.float32)
            body_pos[:, 2] = 0.8
            body_quat = np.zeros((frames, 14 * 4), dtype=np.float32)
            body_quat[:, 0::4] = 1.0
            self._write_csv(reference_dir / "joint_pos.csv", sonic_joints)
            self._write_csv(reference_dir / "body_pos.csv", body_pos)
            self._write_csv(reference_dir / "body_quat.csv", body_quat)

            payload, metrics = convert_reference(reference_dir)

            np.testing.assert_allclose(
                payload["dof_pos"][:, SONIC_REFERENCE_FROM_MUJOCO],
                np.tile(np.arange(29, dtype=np.float32), (payload["dof_pos"].shape[0], 1)),
            )
            np.testing.assert_allclose(payload["root_rot"][0], [0.0, 0.0, 0.0, 1.0])
            self.assertEqual(payload["fps"], 30.0)
            self.assertAlmostEqual(metrics["root_height_min_m"], 0.8)

    def test_native_manifest_expands_to_three_repeats_per_tier(self):
        manifest = {
            "selected": {
                level: {"name": f"motion_{level}"}
                for level in ("low", "medium", "high")
            }
        }

        planned = _planned_runs(manifest)

        self.assertEqual(len(planned), 9)
        self.assertIn("sonic_native_high_motion_high_r03", planned)
        self.assertEqual(planned["sonic_native_low_motion_low_r02"]["repeat"], 2)

    def test_suite_routes_retargeted_gt_to_separate_v2_runs(self):
        args = argparse.Namespace(source="retargeted_gt")

        name, path, prefix, runs_root, logs_root = _motion_spec(
            args, Path("/tmp/repo"), "high"
        )

        self.assertEqual(name, "gMH_sBM_cAll_d22_mMH0_ch01")
        self.assertEqual(path.suffix, ".pkl")
        self.assertEqual(prefix, "gt_cap_high_gMH_sBM_cAll_d22_mMH0_ch01")
        self.assertEqual(runs_root.name, "retargeted_gt_runs_v2")
        self.assertEqual(logs_root.name, "retargeted_gt_v2")

    def test_selects_near_target_percentiles_with_distinct_genres(self):
        records = []
        for index in range(21):
            value = float(index + 1)
            records.append(
                {
                    "motion_id": f"motion_{index}",
                    "genre": f"g{index:02d}",
                    "motion_energy_rad2_s2": value,
                    "velocity_p95_rad_s": value,
                    "acceleration_p95_rad_s2": value,
                    "jerk_p95_rad_s3": value,
                }
            )

        selected = _select_candidates(records, count=2)

        self.assertAlmostEqual(selected["low"][0]["dynamic_percentile"], 0.20)
        self.assertAlmostEqual(selected["medium"][0]["dynamic_percentile"], 0.50)
        self.assertAlmostEqual(selected["high"][0]["dynamic_percentile"], 0.80)
        self.assertEqual(
            len({selected[level][0]["genre"] for level in ("low", "medium", "high")}),
            3,
        )


if __name__ == "__main__":
    unittest.main()
