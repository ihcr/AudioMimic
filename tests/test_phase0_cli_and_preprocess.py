import importlib
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
            weight_decay=0.05,
            use_beats=True,
            beat_rep="distance",
            lambda_acc=0.1,
            lambda_beat=0.5,
            beat_a=10.0,
            beat_c=0.1,
            beat_estimator_ckpt="weights/beat_estimator.pt",
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
            weight_decay=opt.weight_decay,
            use_beats=opt.use_beats,
            beat_rep=opt.beat_rep,
            lambda_acc=opt.lambda_acc,
            lambda_beat=opt.lambda_beat,
            beat_a=opt.beat_a,
            beat_c=opt.beat_c,
            beat_estimator_ckpt=opt.beat_estimator_ckpt,
        )
        fake_model.train_loop.assert_called_once_with(opt)


class CreateDatasetEntrypointTests(unittest.TestCase):
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

        with patch.object(create_dataset_module, "split_data") as split_data, patch.object(
            create_dataset_module, "slice_aistpp"
        ) as slice_aistpp, patch.object(
            create_dataset_module, "baseline_extract"
        ) as baseline_extract, patch.object(
            create_dataset_module, "jukebox_extract"
        ) as jukebox_extract:
            create_dataset_module.create_dataset(opt)

        split_data.assert_called_once_with(
            str(expected_root / "edge_aistpp"), output_root=expected_root
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


if __name__ == "__main__":
    unittest.main()
