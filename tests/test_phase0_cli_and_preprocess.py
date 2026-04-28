import importlib
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import args

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ENV_PYTHON = REPO_ROOT.parents[1] / ".venv311" / "bin" / "python"


@contextmanager
def argv_context(*argv):
    with patch.object(sys, "argv", list(argv)):
        yield


def reload_data_module(module_name):
    sys.modules.pop(module_name, None)
    sys.modules.pop("filter_split_data", None)
    sys.modules.pop("slice", None)

    sys.path.insert(0, str(DATA_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)


def load_train_module():
    sys.modules.pop("train", None)
    return importlib.import_module("train")


class TrainArgParserTests(unittest.TestCase):
    def test_train_parser_uses_official_style_training_defaults(self):
        with argv_context("train.py", "--use_beats"):
            beat_opt = args.parse_train_opt()

        with argv_context("train.py"):
            baseline_opt = args.parse_train_opt()

        self.assertEqual(beat_opt.learning_rate, 2e-4)
        self.assertEqual(baseline_opt.learning_rate, 2e-4)
        self.assertEqual(beat_opt.batch_size, 128)
        self.assertEqual(baseline_opt.batch_size, 128)
        self.assertEqual(beat_opt.gradient_accumulation_steps, 1)
        self.assertEqual(baseline_opt.gradient_accumulation_steps, 1)
        self.assertEqual(beat_opt.lambda_acc, 0.0)
        self.assertEqual(baseline_opt.lambda_acc, 0.0)
        self.assertFalse(beat_opt.learning_rate_was_explicit)
        self.assertFalse(baseline_opt.learning_rate_was_explicit)

    def test_train_parser_tracks_explicit_learning_rate(self):
        with argv_context("train.py", "--learning_rate", "0.001"):
            opt = args.parse_train_opt()

        self.assertEqual(opt.learning_rate, 0.001)
        self.assertTrue(opt.learning_rate_was_explicit)

    def test_train_parser_uses_dynamic_worker_defaults_when_unset(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}, clear=False):
            with argv_context("train.py"):
                opt = args.parse_train_opt()

        self.assertEqual(opt.train_num_workers, 2)
        self.assertEqual(opt.test_num_workers, 2)

    def test_train_parser_accepts_mixed_precision_argument(self):
        with argv_context("train.py", "--mixed_precision", "no"):
            opt = args.parse_train_opt()

        self.assertEqual(opt.mixed_precision, "no")

    def test_train_parser_accepts_phase0_arguments(self):
        with argv_context(
            "train.py",
            "--learning_rate",
            "0.001",
            "--weight_decay",
            "0.03",
            "--use_beats",
            "--beat_rep",
            "pulse",
            "--lambda_acc",
            "0.2",
            "--lambda_beat",
            "0.7",
            "--beat_a",
            "12.0",
            "--beat_c",
            "0.2",
            "--beat_estimator_ckpt",
            "weights/beat_estimator.pt",
            "--train_num_workers",
            "3",
            "--test_num_workers",
            "1",
            "--gradient_accumulation_steps",
            "3",
            "--mixed_precision",
            "bf16",
        ):
            opt = args.parse_train_opt()

        self.assertEqual(opt.learning_rate, 0.001)
        self.assertEqual(opt.weight_decay, 0.03)
        self.assertTrue(opt.use_beats)
        self.assertEqual(opt.beat_rep, "pulse")
        self.assertEqual(opt.lambda_acc, 0.2)
        self.assertEqual(opt.lambda_beat, 0.7)
        self.assertEqual(opt.beat_a, 12.0)
        self.assertEqual(opt.beat_c, 0.2)
        self.assertEqual(opt.beat_estimator_ckpt, "weights/beat_estimator.pt")
        self.assertEqual(opt.train_num_workers, 3)
        self.assertEqual(opt.test_num_workers, 1)
        self.assertEqual(opt.gradient_accumulation_steps, 3)
        self.assertEqual(opt.mixed_precision, "bf16")

    def test_test_parser_accepts_phase0_beat_arguments(self):
        with argv_context(
            "test.py",
            "--use_beats",
            "--beat_rep",
            "distance",
            "--beat_source",
            "user",
            "--beat_file",
            "beats/example.json",
        ):
            opt = args.parse_test_opt()

        self.assertTrue(opt.use_beats)
        self.assertEqual(opt.beat_rep, "distance")
        self.assertEqual(opt.beat_source, "user")
        self.assertEqual(opt.beat_file, "beats/example.json")


class TrainWiringTests(unittest.TestCase):
    def test_train_passes_phase0_options_into_edge(self):
        opt = SimpleNamespace(
            feature_type="jukebox",
            checkpoint="weights/example.pt",
            learning_rate=1e-3,
            learning_rate_was_explicit=True,
            weight_decay=0.05,
            use_beats=True,
            beat_rep="distance",
            lambda_acc=0.1,
            lambda_beat=0.5,
            beat_a=10.0,
            beat_c=0.1,
            beat_estimator_ckpt="weights/beat_estimator.pt",
            gradient_accumulation_steps=4,
            mixed_precision="bf16",
            train_num_workers=0,
            test_num_workers=0,
        )
        fake_model = MagicMock()
        fake_edge_ctor = MagicMock(return_value=fake_model)
        train_module = load_train_module()

        with patch.object(train_module, "_load_edge", return_value=fake_edge_ctor):
            train_module.train(opt)

        fake_edge_ctor.assert_called_once_with(
            opt.feature_type,
            checkpoint_path=opt.checkpoint,
            learning_rate=opt.learning_rate,
            learning_rate_was_explicit=opt.learning_rate_was_explicit,
            weight_decay=opt.weight_decay,
            use_beats=opt.use_beats,
            beat_rep=opt.beat_rep,
            lambda_acc=opt.lambda_acc,
            lambda_beat=opt.lambda_beat,
            beat_a=opt.beat_a,
            beat_c=opt.beat_c,
            beat_estimator_ckpt=opt.beat_estimator_ckpt,
            gradient_accumulation_steps=opt.gradient_accumulation_steps,
            mixed_precision=opt.mixed_precision,
            resume_training_state=True,
        )
        fake_model.train_loop.assert_called_once_with(opt)


class CreateDatasetEntrypointTests(unittest.TestCase):
    def test_dataset_folder_falls_back_to_shared_repo_data_in_worktree(self):
        create_dataset_module = reload_data_module("create_dataset")
        shared_root = REPO_ROOT.parents[1] / "data" / "edge_aistpp"

        with patch.object(create_dataset_module, "DATA_DIR", DATA_DIR):
            resolved = create_dataset_module._resolve_dataset_folder("edge_aistpp")

        self.assertEqual(resolved, shared_root)

    def test_train_help_runs_from_repo_root(self):
        completed = subprocess.run(
            [str(ENV_PYTHON), "train.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--learning_rate", completed.stdout)

    def test_test_help_runs_from_repo_root(self):
        completed = subprocess.run(
            [str(ENV_PYTHON), "test.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--beat_source", completed.stdout)

    def test_create_dataset_help_runs_from_repo_root(self):
        completed = subprocess.run(
            [str(ENV_PYTHON), "data/create_dataset.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--extract-jukebox", completed.stdout)

    def test_create_dataset_uses_data_relative_paths(self):
        create_dataset_module = reload_data_module("create_dataset")
        opt = SimpleNamespace(
            dataset_folder="edge_aistpp",
            extract_baseline=True,
            extract_jukebox=True,
            extract_beats=False,
            stride=0.5,
            length=5.0,
        )
        expected_root = DATA_DIR
        expected_dataset_root = REPO_ROOT.parents[1] / "data" / "edge_aistpp"

        with patch.object(create_dataset_module, "split_data") as split_data, patch.object(
            create_dataset_module, "slice_aistpp"
        ) as slice_aistpp, patch.object(
            create_dataset_module, "baseline_extract"
        ) as baseline_extract, patch.object(
            create_dataset_module, "_load_jukebox_setup", return_value=lambda: None
        ), patch.object(
            create_dataset_module, "jukebox_extract"
        ) as jukebox_extract:
            create_dataset_module.create_dataset(opt)

        split_data.assert_called_once_with(
            str(expected_dataset_root), output_root=expected_root
        )
        slice_aistpp.assert_any_call(
            str(expected_root / "train" / "motions"),
            str(expected_root / "train" / "wavs"),
            opt.stride,
            opt.length,
        )
        slice_aistpp.assert_any_call(
            str(expected_root / "test" / "motions"),
            str(expected_root / "test" / "wavs"),
            opt.stride,
            opt.length,
        )
        baseline_extract.assert_any_call(
            str(expected_root / "train" / "wavs_sliced"),
            str(expected_root / "train" / "baseline_feats"),
        )
        baseline_extract.assert_any_call(
            str(expected_root / "test" / "wavs_sliced"),
            str(expected_root / "test" / "baseline_feats"),
        )
        jukebox_extract.assert_any_call(
            str(expected_root / "train" / "wavs_sliced"),
            str(expected_root / "train" / "jukebox_feats"),
        )
        jukebox_extract.assert_any_call(
            str(expected_root / "test" / "wavs_sliced"),
            str(expected_root / "test" / "jukebox_feats"),
        )

    def test_create_dataset_invokes_beat_extraction_when_requested(self):
        create_dataset_module = reload_data_module("create_dataset")
        opt = SimpleNamespace(
            dataset_folder="edge_aistpp",
            extract_baseline=False,
            extract_jukebox=False,
            extract_beats=True,
            stride=0.5,
            length=5.0,
        )
        expected_root = DATA_DIR

        with patch.object(create_dataset_module, "split_data"), patch.object(
            create_dataset_module, "slice_aistpp"
        ), patch.object(create_dataset_module, "beat_extract") as beat_extract:
            create_dataset_module.create_dataset(opt)

        beat_extract.assert_any_call(
            str(expected_root / "train" / "motions_sliced"),
            str(expected_root / "train" / "wavs_sliced"),
            str(expected_root / "train" / "beat_feats"),
        )
        beat_extract.assert_any_call(
            str(expected_root / "test" / "motions_sliced"),
            str(expected_root / "test" / "wavs_sliced"),
            str(expected_root / "test" / "beat_feats"),
        )

    def test_create_dataset_preflights_jukebox_models_before_processing(self):
        create_dataset_module = reload_data_module("create_dataset")
        opt = SimpleNamespace(
            dataset_folder="edge_aistpp",
            extract_baseline=False,
            extract_jukebox=True,
            extract_beats=False,
            stride=0.5,
            length=5.0,
        )
        call_order = []

        def record(name):
            def inner(*args, **kwargs):
                call_order.append(name)
            return inner

        with patch.object(
            create_dataset_module, "_load_jukebox_setup", return_value=record("setup")
        ), patch.object(
            create_dataset_module, "split_data", side_effect=record("split")
        ), patch.object(
            create_dataset_module, "slice_aistpp", side_effect=record("slice")
        ), patch.object(
            create_dataset_module, "_load_jukebox_extract", return_value=record("extract")
        ):
            create_dataset_module.create_dataset(opt)

        self.assertIn("setup", call_order)
        self.assertIn("split", call_order)
        self.assertLess(call_order.index("setup"), call_order.index("split"))


if __name__ == "__main__":
    unittest.main()
