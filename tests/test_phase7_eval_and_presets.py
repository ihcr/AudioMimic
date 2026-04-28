import importlib
import os
import pickle
import random
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class SavedMotionMetadataTests(unittest.TestCase):
    def test_build_saved_motion_metadata_includes_audio_and_distance_beats(self):
        diffusion_module = reload_module("model.diffusion")
        cond = {
            "beat": np.array(
                [
                    [3, 2, 1, 0, 1, 0],
                    [1, 0, 1, 2, 3, 4],
                ],
                dtype=np.int64,
            )
        }

        metadata = diffusion_module.build_saved_motion_metadata(
            cond=cond,
            sample_idx=0,
            audio_path="data/test/wavs_sliced/song_slice0.wav",
            beat_rep="distance",
        )

        self.assertEqual(
            metadata["audio_path"], "data/test/wavs_sliced/song_slice0.wav"
        )
        np.testing.assert_array_equal(
            metadata["designated_beat_frames"],
            np.array([3, 5], dtype=np.int64),
        )
        self.assertEqual(metadata["beat_rep"], "distance")

    def test_build_saved_motion_metadata_includes_pulse_beats(self):
        diffusion_module = reload_module("model.diffusion")
        cond = {
            "beat": np.array(
                [[[0.0], [1.0], [0.4], [0.7]]],
                dtype=np.float32,
            )
        }

        metadata = diffusion_module.build_saved_motion_metadata(
            cond=cond,
            sample_idx=0,
            audio_path="clip.wav",
            beat_rep="pulse",
        )

        np.testing.assert_array_equal(
            metadata["designated_beat_frames"],
            np.array([1, 3], dtype=np.int64),
        )
        self.assertEqual(metadata["beat_rep"], "pulse")


class EvalBasBapTests(unittest.TestCase):
    def test_evaluate_motion_dir_reports_bas_and_bap(self):
        eval_module = reload_module("eval.eval_bas_bap")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_path = tmp_path / "motions"
            motion_path.mkdir()

            with open(motion_path / "sample.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "full_pose": np.zeros((150, 24, 3), dtype=np.float32),
                        "audio_path": "sample.wav",
                        "designated_beat_frames": np.array([10, 50], dtype=np.int64),
                        "beat_rep": "distance",
                    },
                    handle,
                )

            with patch.object(
                eval_module,
                "detect_motion_beat_frames_from_pose",
                return_value=np.array([10, 50], dtype=np.int64),
            ), patch.object(
                eval_module,
                "load_audio_beat_frames",
                return_value=np.array([10, 50], dtype=np.int64),
            ):
                result = eval_module.evaluate_motion_dir(str(motion_path))

        self.assertAlmostEqual(result["BAS"], 1.0, places=6)
        self.assertAlmostEqual(result["BAP"], 1.0, places=6)
        self.assertAlmostEqual(result["BAP_precision"], 1.0, places=6)
        self.assertAlmostEqual(result["BAP_recall"], 1.0, places=6)

    def test_evaluate_motion_dir_skips_files_without_audio_path(self):
        eval_module = reload_module("eval.eval_bas_bap")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_path = tmp_path / "motions"
            motion_path.mkdir()

            with open(motion_path / "sample.pkl", "wb") as handle:
                pickle.dump({"full_pose": np.zeros((150, 24, 3), dtype=np.float32)}, handle)

            result = eval_module.evaluate_motion_dir(str(motion_path))

        self.assertEqual(result["num_scored_files"], 0)
        self.assertTrue(np.isnan(result["BAP"]))
        self.assertTrue(np.isnan(result["BAP_precision"]))
        self.assertTrue(np.isnan(result["BAP_recall"]))

    def test_compute_bas_score_matches_bailando_music_to_motion_direction(self):
        eval_module = reload_module("eval.eval_bas_bap")

        score = eval_module.compute_bas_score(
            music_beats=np.array([0, 5], dtype=np.int64),
            motion_beats=np.array([0], dtype=np.int64),
        )

        expected = (1.0 + np.exp(-(25.0) / 18.0)) / 2.0
        self.assertAlmostEqual(score, expected, places=6)


class EvalPfcAuditTests(unittest.TestCase):
    def test_compute_physical_score_can_return_component_details(self):
        pfc_module = reload_module("eval.eval_pfc")

        with TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir)
            full_pose = np.zeros((8, 24, 3), dtype=np.float32)
            full_pose[:, 7, 0] = np.linspace(0.0, 0.3, 8, dtype=np.float32)
            full_pose[:, 8, 0] = np.linspace(0.0, 0.2, 8, dtype=np.float32)
            full_pose[:, 10, 0] = np.linspace(0.0, 0.1, 8, dtype=np.float32)
            full_pose[:, 11, 0] = np.linspace(0.0, 0.4, 8, dtype=np.float32)
            with open(motion_path / "sample.pkl", "wb") as handle:
                pickle.dump({"full_pose": full_pose}, handle)

            details = pfc_module.compute_physical_score(
                str(motion_path), return_details=True
            )

        self.assertIn("PFC", details)
        self.assertIn("num_files", details)
        self.assertIn("mean_root_acceleration", details)
        self.assertIn("mean_left_foot_min", details)
        self.assertIn("mean_right_foot_min", details)
        self.assertEqual(details["num_files"], 1)
        self.assertGreaterEqual(details["PFC"], 0.0)

    def test_compute_physical_score_accepts_ground_truth_motion_pickles(self):
        pfc_module = reload_module("eval.eval_pfc")

        with TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir)
            with open(motion_path / "sample.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "pos": np.zeros((10, 3), dtype=np.float32),
                        "q": np.zeros((10, 72), dtype=np.float32),
                    },
                    handle,
                )

            details = pfc_module.compute_physical_score(
                str(motion_path), return_details=True
            )

        self.assertEqual(details["num_files"], 1)
        self.assertGreaterEqual(details["PFC"], 0.0)


class EvalDiversityTests(unittest.TestCase):
    def test_compute_diversity_metrics_reports_paper_scaled_distances(self):
        div_module = reload_module("eval.eval_diversity")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reference_path = tmp_path / "reference"
            motion_path = tmp_path / "motions"
            reference_path.mkdir()
            motion_path.mkdir()
            for idx in range(3):
                with open(reference_path / f"ref{idx}.pkl", "wb") as handle:
                    pickle.dump({"full_pose": np.zeros((10, 24, 3), dtype=np.float32)}, handle)
            for idx in range(2):
                with open(motion_path / f"clip{idx}.pkl", "wb") as handle:
                    pickle.dump({"full_pose": np.zeros((10, 24, 3), dtype=np.float32)}, handle)

            feature_map = {
                "ref0.pkl": (
                    np.array([0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0], dtype=np.float32),
                ),
                "ref1.pkl": (
                    np.array([2.0, 0.0], dtype=np.float32),
                    np.array([0.0, 2.0], dtype=np.float32),
                ),
                "ref2.pkl": (
                    np.array([4.0, 0.0], dtype=np.float32),
                    np.array([0.0, 4.0], dtype=np.float32),
                ),
                "clip0.pkl": (
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                ),
                "clip1.pkl": (
                    np.array([3.0, 0.0], dtype=np.float32),
                    np.array([0.0, 3.0], dtype=np.float32),
                ),
            }

            def fake_extract(path):
                return feature_map[Path(path).name]

            with patch.object(div_module, "extract_diversity_features", side_effect=fake_extract):
                result = div_module.compute_diversity_metrics(
                    str(motion_path),
                    reference_motion_path=str(reference_path),
                    seed=1234,
                )

        expected = 2.0 / np.std(np.array([0.0, 2.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(result["Distk"], expected, places=6)
        self.assertAlmostEqual(result["Distg"], expected, places=6)
        self.assertAlmostEqual(result["Divk"], expected, places=6)
        self.assertAlmostEqual(result["Divm"], expected, places=6)
        self.assertEqual(result["num_motion_files"], 2)
        self.assertEqual(result["num_reference_files"], 3)

    def test_compute_diversity_metrics_ignores_zero_variance_reference_dimensions(self):
        div_module = reload_module("eval.eval_diversity")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_path = tmp_path / "motions"
            reference_path = tmp_path / "reference"
            motion_path.mkdir()
            reference_path.mkdir()

            for idx in range(2):
                with open(motion_path / f"clip{idx}.pkl", "wb") as handle:
                    pickle.dump({"full_pose": np.zeros((10, 24, 3), dtype=np.float32)}, handle)
            for idx in range(3):
                with open(reference_path / f"ref{idx}.pkl", "wb") as handle:
                    pickle.dump({"full_pose": np.zeros((10, 24, 3), dtype=np.float32)}, handle)

            feature_map = {
                "ref0.pkl": (
                    np.array([0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 5.0], dtype=np.float32),
                ),
                "ref1.pkl": (
                    np.array([2.0, 0.0], dtype=np.float32),
                    np.array([2.0, 5.0], dtype=np.float32),
                ),
                "ref2.pkl": (
                    np.array([4.0, 0.0], dtype=np.float32),
                    np.array([4.0, 5.0], dtype=np.float32),
                ),
                "clip0.pkl": (
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([1.0, 0.0], dtype=np.float32),
                ),
                "clip1.pkl": (
                    np.array([3.0, 0.0], dtype=np.float32),
                    np.array([3.0, 10.0], dtype=np.float32),
                ),
            }

            def fake_extract(path):
                return feature_map[Path(path).name]

            with patch.object(div_module, "extract_diversity_features", side_effect=fake_extract):
                result = div_module.compute_diversity_metrics(
                    str(motion_path),
                    reference_motion_path=str(reference_path),
                    seed=1234,
                )

        expected = 2.0 / np.std(np.array([0.0, 2.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(result["Distg"], expected, places=6)
        self.assertEqual(result["zero_variance_dims_manual"], 1)
        self.assertTrue(np.isfinite(result["Distg"]))

    def test_extract_diversity_features_accepts_ground_truth_motion_pickles(self):
        div_module = reload_module("eval.eval_diversity")

        with TemporaryDirectory() as tmpdir:
            motion_file = Path(tmpdir) / "sample.pkl"
            with open(motion_file, "wb") as handle:
                pickle.dump(
                    {
                        "pos": np.zeros((10, 3), dtype=np.float32),
                        "q": np.zeros((10, 72), dtype=np.float32),
                    },
                    handle,
                )

            kinetic_feature, geometric_feature = div_module.extract_diversity_features(
                str(motion_file)
            )

        self.assertEqual(kinetic_feature.ndim, 1)
        self.assertEqual(geometric_feature.ndim, 1)

    def test_load_full_pose_returns_saved_full_pose_when_it_is_the_only_source(self):
        pfc_module = reload_module("eval.eval_pfc")

        payload = {
            "full_pose": np.full((2, 24, 3), 7.0, dtype=np.float32),
        }

        full_pose = pfc_module.load_full_pose(payload)

        np.testing.assert_allclose(full_pose, payload["full_pose"])

    def test_load_full_pose_rejects_inconsistent_generated_payload_when_validating(self):
        pfc_module = reload_module("eval.eval_pfc")

        payload = {
            "full_pose": np.full((2, 24, 3), 7.0, dtype=np.float32),
            "smpl_poses": np.zeros((2, 72), dtype=np.float32),
            "smpl_trans": np.zeros((2, 3), dtype=np.float32),
        }

        with self.assertRaises(ValueError):
            pfc_module.load_full_pose(payload, validate_consistency=True)


class BenchmarkPreparationTests(unittest.TestCase):
    def test_prepare_generated_motion_dir_uses_existing_files_directly(self):
        benchmark_module = reload_module("eval.run_benchmark_eval")

        with TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motions"
            motion_path.mkdir()
            with open(motion_path / "sample.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "full_pose": np.zeros((10, 24, 3), dtype=np.float32),
                        "audio_path": "sample.wav",
                    },
                    handle,
                )

            prepared_path, temp_dir = benchmark_module.prepare_motion_dir_for_benchmark(
                str(motion_path),
                motion_source="generated",
            )

        self.assertEqual(prepared_path, str(motion_path))
        self.assertIsNone(temp_dir)

    def test_prepare_generated_motion_dir_rejects_inconsistent_joint_and_smpl_payloads(self):
        benchmark_module = reload_module("eval.run_benchmark_eval")

        with TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motions"
            motion_path.mkdir()
            with open(motion_path / "sample.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "full_pose": np.full((10, 24, 3), 5.0, dtype=np.float32),
                        "smpl_poses": np.zeros((10, 72), dtype=np.float32),
                        "smpl_trans": np.zeros((10, 3), dtype=np.float32),
                        "audio_path": "sample.wav",
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                benchmark_module.prepare_motion_dir_for_benchmark(
                    str(motion_path),
                    motion_source="generated",
                )


class BenchmarkReportTests(unittest.TestCase):
    def test_build_benchmark_report_produces_edge_and_beatit_views(self):
        benchmark_module = reload_module("eval.benchmark_report")
        metrics = {
            "PFC": 12.5,
            "PFC_internal_only": True,
            "BAS": 0.25,
            "BAP": 0.4,
            "BAP_precision": 0.4,
            "BAP_recall": 0.7,
            "Distk": 9.6,
            "Distg": 7.1,
            "Divk": 9.6,
            "Divm": 7.1,
        }

        report = benchmark_module.build_benchmark_report(
            metrics=metrics,
            method_name="EDGE+BeatDistance",
            use_beats=True,
            beat_rep="distance",
        )

        self.assertIn("edge_table", report)
        self.assertIn("beatit_table", report)
        self.assertEqual(report["edge_table"]["Method"], "EDGE+BeatDistance")
        self.assertEqual(report["beatit_table"]["Methods"], "EDGE+BeatDistance")
        self.assertEqual(report["edge_table"]["Distk"], 9.6)
        self.assertEqual(report["edge_table"]["Distg"], 7.1)
        self.assertEqual(report["beatit_table"]["Divk"], 9.6)
        self.assertEqual(report["beatit_table"]["Divm"], 7.1)
        self.assertEqual(report["beatit_table"]["BAP"], 0.4)
        self.assertIsNone(report["beatit_table"]["KPD"])
        self.assertTrue(report["notes"]["pfc_internal_only"])


class PfcAuditRunnerTests(unittest.TestCase):
    def test_audit_pfc_sources_collects_named_runs(self):
        audit_module = reload_module("eval.audit_pfc")

        with patch.object(
            audit_module,
            "compute_physical_score",
            side_effect=[
                {"PFC": 76.0, "num_files": 20},
                {"PFC": 114.0, "num_files": 186},
            ],
        ):
            report = audit_module.audit_pfc_sources(
                {
                    "ground_truth": "data/test/motions",
                    "official_checkpoint": "slurm/evals/official/motions",
                }
            )

        self.assertEqual(report["ground_truth"]["PFC"], 76.0)
        self.assertEqual(report["official_checkpoint"]["PFC"], 114.0)
        self.assertEqual(report["ground_truth"]["num_files"], 20)


class PlotVelocityTests(unittest.TestCase):
    def test_plot_velocity_vs_beats_writes_png(self):
        plot_module = reload_module("eval.plot_velocity_vs_beats")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_file = tmp_path / "sample.pkl"
            output_path = tmp_path / "plot.png"
            with open(motion_file, "wb") as handle:
                pickle.dump(
                    {
                        "full_pose": np.zeros((150, 24, 3), dtype=np.float32),
                        "audio_path": "sample.wav",
                        "designated_beat_frames": np.array([10, 50], dtype=np.int64),
                    },
                    handle,
                )

            with patch.object(
                plot_module,
                "detect_motion_beat_frames_from_pose",
                return_value=np.array([10, 50], dtype=np.int64),
            ), patch.object(
                plot_module,
                "load_audio_beat_frames",
                return_value=np.array([12, 52], dtype=np.int64),
            ):
                plot_module.plot_velocity_vs_beats(
                    str(motion_file), output_path=str(output_path)
                )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


class InferenceSeedingTests(unittest.TestCase):
    def test_choose_slice_start_is_seedable(self):
        test_module = reload_module("test")
        rng_a = random.Random(1234)
        rng_b = random.Random(1234)

        start_a = test_module.choose_slice_start(20, 11, rng_a)
        start_b = test_module.choose_slice_start(20, 11, rng_b)

        self.assertEqual(start_a, start_b)
        self.assertEqual(start_a, 7)


class DatasetEvalRunnerTests(unittest.TestCase):
    def test_batch_to_render_tuple_keeps_cond_and_wavname(self):
        eval_module = reload_module("eval.run_dataset_eval")
        cond = {
            "music": torch.zeros((1, 150, 4), dtype=torch.float32),
            "beat": torch.zeros((1, 150), dtype=torch.int64),
        }
        batch = (
            torch.zeros((1, 150, 151), dtype=torch.float32),
            cond,
            ["feature.npy"],
            ["song.wav"],
        )

        render_tuple = eval_module.batch_to_render_tuple(batch)

        self.assertEqual(len(render_tuple), 3)
        self.assertIs(render_tuple[1], cond)
        self.assertEqual(render_tuple[2], ["song.wav"])

    def test_render_dataset_batch_uses_normal_mode(self):
        eval_module = reload_module("eval.run_dataset_eval")

        class DummyDiffusion:
            def __init__(self):
                self.calls = []

            def render_sample(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        class DummyModel:
            def __init__(self):
                self.diffusion = DummyDiffusion()
                self.normalizer = object()
                self.horizon = 150
                self.repr_dim = 151
                self.accelerator = type("Accel", (), {"device": torch.device("cpu")})()

        model = DummyModel()
        cond = {
            "music": torch.zeros((1, 150, 4), dtype=torch.float32),
            "beat": torch.zeros((1, 150), dtype=torch.int64),
        }
        batch = (
            torch.zeros((1, 150, 151), dtype=torch.float32),
            cond,
            ["feature.npy"],
            ["song_slice0.wav"],
        )

        eval_module.render_dataset_batch(
            model,
            batch,
            render_dir="renders/eval",
            motion_dir="eval/motions",
        )

        self.assertEqual(len(model.diffusion.calls), 1)
        _, kwargs = model.diffusion.calls[0]
        self.assertEqual(kwargs["mode"], "normal")
        self.assertEqual(kwargs["name"], ["song_slice0.wav"])


if __name__ == "__main__":
    unittest.main()
