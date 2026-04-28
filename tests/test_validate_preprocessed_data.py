import importlib.util
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "data" / "validate_preprocessed_data.py"


def load_module():
    module_name = "validate_preprocessed_data"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_motion(path, min_height=1.2, max_height=1.8):
    pos = np.zeros((300, 3), dtype=np.float32)
    pos[:, 1] = np.linspace(min_height, max_height, 300, dtype=np.float32)
    q = np.zeros((300, 72), dtype=np.float32)
    with open(path, "wb") as handle:
        pickle.dump({"pos": pos, "q": q}, handle, pickle.HIGHEST_PROTOCOL)


def write_wav_placeholder(path):
    path.write_bytes(b"RIFF")


def write_feature(path, feature_type):
    dim = 35 if feature_type == "baseline" else 4800
    np.save(path, np.zeros((150, dim), dtype=np.float32))


def write_beat(path):
    np.savez(
        path,
        motion_beats=np.array([10, 30, 60], dtype=np.int64),
        motion_mask=np.zeros((150,), dtype=np.float32),
        motion_dist=np.zeros((150,), dtype=np.int64),
        motion_spacing=np.ones((150,), dtype=np.float32) * 10,
        audio_beats=np.array([9, 29, 61], dtype=np.int64),
        audio_mask=np.zeros((150,), dtype=np.float32),
        audio_dist=np.zeros((150,), dtype=np.int64),
        audio_spacing=np.ones((150,), dtype=np.float32) * 10,
    )


def build_minimal_dataset(root, feature_type="jukebox", use_beats=False):
    names = ["clip_a_slice0", "clip_b_slice0"]
    for split in ("train", "test"):
        split_dir = root / split
        (split_dir / "motions_sliced").mkdir(parents=True)
        (split_dir / "wavs_sliced").mkdir(parents=True)
        (split_dir / f"{feature_type}_feats").mkdir(parents=True)
        if use_beats:
            (split_dir / "beat_feats").mkdir(parents=True)

        for name in names:
            write_motion(split_dir / "motions_sliced" / f"{name}.pkl")
            write_wav_placeholder(split_dir / "wavs_sliced" / f"{name}.wav")
            write_feature(split_dir / f"{feature_type}_feats" / f"{name}.npy", feature_type)
            if use_beats:
                write_beat(split_dir / "beat_feats" / f"{name}.npz")


class ValidatePreprocessedDataTests(unittest.TestCase):
    def test_validate_preprocessed_dataset_accepts_matching_feature_and_motion_files(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            processed_root = Path(tmpdir) / "dataset_backups"
            build_minimal_dataset(data_root, feature_type="jukebox", use_beats=True)
            processed_root.mkdir()

            summary = module.validate_preprocessed_dataset(
                data_path=data_root,
                processed_data_dir=processed_root,
                feature_type="jukebox",
                use_beats=True,
                beat_rep="distance",
                sample_count=2,
            )

        self.assertEqual(summary["train"]["count"], 2)
        self.assertEqual(summary["test"]["count"], 2)
        self.assertEqual(summary["train"]["feature_dir"], "jukebox_feats")
        self.assertEqual(summary["train"]["beat_count"], 2)

    def test_validate_preprocessed_dataset_rejects_negative_root_heights(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            processed_root = Path(tmpdir) / "dataset_backups"
            build_minimal_dataset(data_root, feature_type="jukebox", use_beats=False)
            processed_root.mkdir()
            bad_motion = data_root / "train" / "motions_sliced" / "clip_a_slice0.pkl"
            write_motion(bad_motion, min_height=-0.5, max_height=0.1)

            with self.assertRaisesRegex(ValueError, "root height"):
                module.validate_preprocessed_dataset(
                    data_path=data_root,
                    processed_data_dir=processed_root,
                    feature_type="jukebox",
                    use_beats=False,
                    beat_rep="distance",
                    sample_count=2,
                )

    def test_validate_preprocessed_dataset_rejects_legacy_matching_cache_files(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            processed_root = Path(tmpdir) / "dataset_backups"
            build_minimal_dataset(data_root, feature_type="jukebox", use_beats=True)
            processed_root.mkdir()
            (processed_root / "train_tensor_dataset_jukebox_beat_distance_v4.pkl").write_bytes(b"old")

            with self.assertRaisesRegex(ValueError, "legacy cache"):
                module.validate_preprocessed_dataset(
                    data_path=data_root,
                    processed_data_dir=processed_root,
                    feature_type="jukebox",
                    use_beats=True,
                    beat_rep="distance",
                    sample_count=2,
                )


if __name__ == "__main__":
    unittest.main()
