import importlib
import json
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def write_g1_motion(path, root_x=0.0, dof_offset=0.0, audio_path="clip.wav"):
    frames = 6
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(root_x, root_x + 0.5, frames, dtype=np.float32)
    root_pos[:, 2] = 0.78 + np.linspace(0.0, 0.05, frames, dtype=np.float32)
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 0] = 1.0
    dof_pos = np.zeros((frames, 29), dtype=np.float32)
    dof_pos[:, 0] = np.linspace(dof_offset, dof_offset + 0.3, frames, dtype=np.float32)
    dof_pos[:, 1] = np.linspace(-0.2, 0.2, frames, dtype=np.float32)
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "motion_format": "g1",
                "fps": 30.0,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "pos": root_pos,
                "q": np.concatenate([root_rot, dof_pos], axis=-1),
                "audio_path": audio_path,
                "designated_beat_frames": np.array([1, 3], dtype=np.int64),
                "beat_rep": "distance",
            },
            handle,
            pickle.HIGHEST_PROTOCOL,
        )


class G1MetricTests(unittest.TestCase):
    def test_beat_timing_reports_precision_recall_f1_and_offsets(self):
        g1_module = reload_module("eval.g1_metrics")

        report = g1_module.compute_beat_timing_report(
            generated_beats=np.array([10, 31, 80], dtype=np.int64),
            target_beats=np.array([12, 30, 60], dtype=np.int64),
            tolerance=2,
        )

        self.assertAlmostEqual(report["precision"], 2 / 3)
        self.assertAlmostEqual(report["recall"], 2 / 3)
        self.assertAlmostEqual(report["f1"], 2 / 3)
        self.assertAlmostEqual(report["timing_mean_frames"], -0.5)
        self.assertAlmostEqual(report["timing_std_frames"], 1.5)

    def test_fk_metrics_are_optional_and_reported_when_enabled(self):
        g1_module = reload_module("eval.g1_metrics")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            motion_dir = root / "motions"
            reference_dir = root / "reference"
            render_dir = root / "renders"
            motion_dir.mkdir()
            reference_dir.mkdir()
            write_g1_motion(motion_dir / "gen.pkl", audio_path="gen.wav")
            write_g1_motion(reference_dir / "ref.pkl", audio_path="ref.wav")

            fk_keypoints = np.array(
                [
                    [[0.0, 0.0, 0.2], [0.0, 0.0, 0.1]],
                    [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]],
                    [[0.0, 0.0, 0.2], [0.2, 0.0, 0.1]],
                    [[0.0, 0.0, 0.1], [0.3, 0.0, 0.1]],
                    [[0.0, 0.0, 0.2], [0.4, 0.0, 0.1]],
                    [[0.0, 0.0, 0.1], [0.5, 0.0, 0.1]],
                ],
                dtype=np.float32,
            )
            fk_result = {
                "keypoints": fk_keypoints,
                "keypoint_names": ["pelvis", "left_lowest_foot_geom"],
                "left_foot_points": fk_keypoints[:, 1],
                "right_foot_points": fk_keypoints[:, 1],
                "metadata": {
                    "model_path": "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
                    "root_quat_order": "wxyz",
                    "joint_names": ["joint"],
                },
            }

            with patch.object(
                g1_module,
                "load_audio_beat_frames",
                return_value=np.array([1, 3], dtype=np.int64),
            ), patch.object(
                g1_module,
                "detect_fk_motion_beat_frames",
                return_value=np.array([1, 3], dtype=np.int64),
            ), patch.object(
                g1_module,
                "forward_g1_kinematics",
                return_value=fk_result,
            ):
                without_fk = g1_module.run_g1_motion_evaluation(
                    motion_path=motion_dir,
                    reference_motion_path=reference_dir,
                    metrics_path=root / "metrics_no_fk.json",
                    g1_table_path=root / "table_no_fk.json",
                    motion_audit_path=root / "audit_no_fk.json",
                    paper_report_path=root / "report_no_fk.md",
                    render_dir=render_dir,
                    diagnostic_count=0,
                    use_beats=True,
                )
                with_fk = g1_module.run_g1_motion_evaluation(
                    motion_path=motion_dir,
                    reference_motion_path=reference_dir,
                    metrics_path=root / "metrics_fk.json",
                    g1_table_path=root / "table_fk.json",
                    motion_audit_path=root / "audit_fk.json",
                    paper_report_path=root / "report_fk.md",
                    render_dir=render_dir,
                    diagnostic_count=0,
                    use_beats=True,
                    enable_fk_metrics=True,
                    fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
                    root_quat_order="wxyz",
                )

            audit = json.loads((root / "audit_fk.json").read_text(encoding="utf-8"))

        self.assertNotIn("G1FKBAS", without_fk)
        self.assertIn("G1FKBAS", with_fk)
        self.assertIn("G1FKRoboPerformBAS", with_fk)
        self.assertIn("G1BeatF1", with_fk)
        self.assertIn("G1FootSliding", with_fk)
        self.assertTrue(np.isfinite(with_fk["G1FKBAS"]))
        self.assertAlmostEqual(with_fk["G1FKRoboPerformBAS"], 1.0)
        self.assertEqual(audit["fk_model_path"], "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml")

    def test_g1_metrics_are_finite_and_do_not_report_smpl_only_names(self):
        g1_module = reload_module("eval.g1_metrics")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            motion_dir = root / "motions"
            reference_dir = root / "reference"
            render_dir = root / "renders"
            motion_dir.mkdir()
            reference_dir.mkdir()
            for idx in range(2):
                write_g1_motion(
                    motion_dir / f"gen{idx}.pkl",
                    root_x=float(idx),
                    dof_offset=float(idx) * 0.2,
                    audio_path=f"gen{idx}.wav",
                )
            for idx in range(3):
                write_g1_motion(
                    reference_dir / f"ref{idx}.pkl",
                    root_x=float(idx) * 0.5,
                    dof_offset=float(idx) * 0.1,
                    audio_path=f"ref{idx}.wav",
                )

            with patch.object(
                g1_module,
                "load_audio_beat_frames",
                return_value=np.array([1, 3], dtype=np.int64),
            ), patch.object(
                g1_module,
                "detect_g1_motion_beat_frames",
                return_value=np.array([1, 3], dtype=np.int64),
            ), patch.object(
                g1_module,
                "detect_g1_roboperform_motion_beat_frames",
                return_value=np.array([1, 3], dtype=np.int64),
            ), patch.object(
                reload_module("eval.eval_bas_bap"),
                "load_full_pose",
                side_effect=AssertionError("SMPL loader must not be used for G1"),
            ):
                result = g1_module.run_g1_motion_evaluation(
                    motion_path=motion_dir,
                    reference_motion_path=reference_dir,
                    metrics_path=root / "metrics.json",
                    g1_table_path=root / "g1_table.json",
                    motion_audit_path=root / "motion_audit.json",
                    paper_report_path=root / "paper_report.md",
                    render_dir=render_dir,
                    diagnostic_count=1,
                    checkpoint="train-2000.pt",
                    feature_type="jukebox",
                    use_beats=True,
                    beat_rep="distance",
                    seed=1234,
                )

            saved_metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            saved_table = json.loads((root / "g1_table.json").read_text(encoding="utf-8"))
            audit = json.loads((root / "motion_audit.json").read_text(encoding="utf-8"))
            wrote_render = any(render_dir.glob("*.png"))

        self.assertEqual(result["num_motion_files"], 2)
        self.assertEqual(saved_metrics["num_motion_files"], 2)
        self.assertEqual(saved_table["Method"], "G1 train-2000")
        self.assertEqual(audit["num_files"], 2)
        for forbidden in ("PFC", "Distg", "Distk", "Divk", "Divm"):
            self.assertNotIn(forbidden, saved_metrics)
            self.assertNotIn(forbidden, saved_table)
        for value in saved_metrics.values():
            if isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value), value)
        self.assertGreaterEqual(saved_metrics["G1BAS"], 0.0)
        self.assertEqual(saved_metrics["G1RoboPerformBAS"], 1.0)
        self.assertEqual(saved_table["G1 RoboPerform BAS"], 1.0)
        self.assertEqual(saved_metrics["G1BAP_precision"], 1.0)
        self.assertTrue(wrote_render)

    def test_reference_range_violation_counts_joint_values_outside_test_range(self):
        g1_module = reload_module("eval.g1_metrics")

        reference = [
            {
                "dof_pos": np.zeros((4, 2), dtype=np.float32),
                "root_pos": np.zeros((4, 3), dtype=np.float32),
                "root_rot": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
                "fps": 30.0,
            }
        ]
        generated = {
            "dof_pos": np.array(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, -2.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "root_pos": np.zeros((4, 3), dtype=np.float32),
            "root_rot": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
            "fps": 30.0,
        }

        bounds = g1_module.compute_reference_bounds(reference, margin_fraction=0.0)
        violation_rate = g1_module.compute_joint_range_violation_rate(
            generated, bounds
        )

        self.assertEqual(violation_rate, 0.25)


if __name__ == "__main__":
    unittest.main()
