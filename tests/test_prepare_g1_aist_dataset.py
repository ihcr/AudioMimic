import importlib.util
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "data" / "prepare_g1_aist_dataset.py"


def load_module():
    module_name = "prepare_g1_aist_dataset"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_raw_g1_motion(path, frames=180):
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0.0, 0.2, frames, dtype=np.float32)
    root_pos[:, 2] = 0.8
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 0] = 1.0
    dof_pos = np.zeros((frames, 29), dtype=np.float32)
    dof_pos[:, 0] = np.linspace(-0.1, 0.1, frames, dtype=np.float32)
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "fps": 30.0,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
            },
            handle,
            pickle.HIGHEST_PROTOCOL,
        )


def write_aist_slice_sources(aist_root, split_name, stem, slices):
    split_root = aist_root / split_name
    (split_root / "wavs_sliced").mkdir(parents=True, exist_ok=True)
    (split_root / "jukebox_feats").mkdir(parents=True, exist_ok=True)
    for index in range(slices):
        slice_stem = f"{stem}_slice{index}"
        (split_root / "wavs_sliced" / f"{slice_stem}.wav").write_bytes(b"RIFF")
        np.save(
            split_root / "jukebox_feats" / f"{slice_stem}.npy",
            np.zeros((150, 4800), dtype=np.float32),
        )


class PrepareG1AISTDatasetTests(unittest.TestCase):
    def test_module_adds_repo_root_for_file_path_execution(self):
        original_path = list(sys.path)
        try:
            sys.path = [entry for entry in sys.path if entry != str(REPO_ROOT)]
            load_module()
            self.assertIn(str(REPO_ROOT), sys.path)
        finally:
            sys.path = original_path

    def test_prepare_uses_official_split_and_skips_ignored_train_sequences(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_g1 = root / "aist-g1-retargeted"
            aist_root = root / "aist"
            split_dir = root / "splits"
            output_root = root / "g1_aistpp_full"
            raw_g1.mkdir()
            split_dir.mkdir()

            train_keep = "gBR_sBM_cAll_d04_mBR1_ch01"
            train_ignore = "gBR_sBM_cAll_d04_mBR1_ch02"
            test_keep = "gLH_sBM_cAll_d17_mLH4_ch02"
            outside = "gWA_sBM_cAll_d25_mWA0_ch01"
            for stem in (train_keep, train_ignore, test_keep, outside):
                write_raw_g1_motion(raw_g1 / f"{stem}.pkl", frames=180)

            (split_dir / "crossmodal_train.txt").write_text(
                f"{train_keep}\n{train_ignore}\n", encoding="utf-8"
            )
            (split_dir / "crossmodal_test.txt").write_text(
                f"{test_keep}\n", encoding="utf-8"
            )
            (split_dir / "ignore_list.txt").write_text(
                f"{train_ignore}\n", encoding="utf-8"
            )

            write_aist_slice_sources(aist_root, "train", train_keep, slices=3)
            write_aist_slice_sources(aist_root, "test", test_keep, slices=3)

            summary = module.prepare_g1_aist_dataset(
                g1_motion_dir=raw_g1,
                aist_data_root=aist_root,
                output_root=output_root,
                split_dir=split_dir,
                feature_type="jukebox",
                clean=True,
            )

            self.assertEqual(summary["raw_g1_sequences"], 4)
            self.assertEqual(summary["train"]["sequences"], 1)
            self.assertEqual(summary["test"]["sequences"], 1)
            self.assertEqual(summary["ignored_train_sequences"], 1)
            self.assertEqual(summary["unused_sequences_outside_official_split"], 1)
            self.assertEqual(summary["train"]["slices"], 3)
            self.assertEqual(summary["test"]["slices"], 3)

            train_slices = sorted((output_root / "train" / "motions_sliced").glob("*.pkl"))
            self.assertEqual([path.stem for path in train_slices], [
                f"{train_keep}_slice0",
                f"{train_keep}_slice1",
                f"{train_keep}_slice2",
            ])
            self.assertFalse((output_root / "train" / "motions" / f"{train_ignore}.pkl").exists())

            with open(train_slices[0], "rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["motion_format"], "g1")
            self.assertEqual(payload["root_pos"].shape, (150, 3))
            self.assertEqual(payload["root_rot"].shape, (150, 4))
            self.assertEqual(payload["dof_pos"].shape, (150, 29))
            self.assertEqual(payload["q"].shape, (150, 33))
            self.assertTrue((output_root / "train" / "wavs_sliced" / f"{train_keep}_slice0.wav").is_symlink())
            self.assertTrue((output_root / "train" / "jukebox_feats" / f"{train_keep}_slice0.npy").is_symlink())
            self.assertTrue((output_root / "metadata.json").is_file())

    def test_prepare_pads_one_frame_retarget_rounding_gap(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_g1 = root / "aist-g1-retargeted"
            aist_root = root / "aist"
            split_dir = root / "splits"
            output_root = root / "g1_aistpp_full"
            raw_g1.mkdir()
            split_dir.mkdir()

            train_keep = "gBR_sBM_cAll_d04_mBR1_ch01"
            write_raw_g1_motion(raw_g1 / f"{train_keep}.pkl", frames=179)
            (split_dir / "crossmodal_train.txt").write_text(f"{train_keep}\n", encoding="utf-8")
            (split_dir / "crossmodal_test.txt").write_text("", encoding="utf-8")
            (split_dir / "ignore_list.txt").write_text("", encoding="utf-8")
            write_aist_slice_sources(aist_root, "train", train_keep, slices=3)
            (aist_root / "test" / "wavs_sliced").mkdir(parents=True)
            (aist_root / "test" / "jukebox_feats").mkdir(parents=True)

            summary = module.prepare_g1_aist_dataset(
                g1_motion_dir=raw_g1,
                aist_data_root=aist_root,
                output_root=output_root,
                split_dir=split_dir,
                feature_type="jukebox",
                clean=True,
            )

            self.assertEqual(summary["train"]["padded_sequences"], 1)
            self.assertEqual(summary["train"]["padded_frames"], 1)
            self.assertEqual(summary["train"]["slices"], 3)
            with open(
                output_root / "train" / "motions_sliced" / f"{train_keep}_slice2.pkl",
                "rb",
            ) as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["root_pos"].shape, (150, 3))

    def test_prepare_can_write_fk_beat_metadata(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_g1 = root / "aist-g1-retargeted"
            aist_root = root / "aist"
            split_dir = root / "splits"
            output_root = root / "g1_aistpp_full_fkbeats"
            raw_g1.mkdir()
            split_dir.mkdir()

            train_keep = "gBR_sBM_cAll_d04_mBR1_ch01"
            write_raw_g1_motion(raw_g1 / f"{train_keep}.pkl", frames=180)
            (split_dir / "crossmodal_train.txt").write_text(f"{train_keep}\n", encoding="utf-8")
            (split_dir / "crossmodal_test.txt").write_text("", encoding="utf-8")
            (split_dir / "ignore_list.txt").write_text("", encoding="utf-8")
            write_aist_slice_sources(aist_root, "train", train_keep, slices=3)
            (aist_root / "test" / "wavs_sliced").mkdir(parents=True)
            (aist_root / "test" / "jukebox_feats").mkdir(parents=True)

            beat_calls = []

            def fake_beat_extract(motion_dir, wav_dir, out_dir, **kwargs):
                beat_calls.append((Path(motion_dir), Path(wav_dir), Path(out_dir), kwargs))
                Path(out_dir).mkdir(parents=True, exist_ok=True)

            with patch.object(module, "extract_beat_metadata", side_effect=fake_beat_extract):
                summary = module.prepare_g1_aist_dataset(
                    g1_motion_dir=raw_g1,
                    aist_data_root=aist_root,
                    output_root=output_root,
                    split_dir=split_dir,
                    feature_type="jukebox",
                    clean=True,
                    extract_beats=True,
                    g1_motion_beat_source="fk",
                    g1_fk_model_path="third_party/unitree_g1_description/g1.xml",
                    g1_root_quat_order="xyzw",
                )

            self.assertEqual(len(beat_calls), 2)
            self.assertEqual(beat_calls[0][0], output_root / "train" / "motions_sliced")
            self.assertEqual(beat_calls[0][2], output_root / "train" / "beat_feats")
            self.assertEqual(beat_calls[0][3]["g1_motion_beat_source"], "fk")
            self.assertEqual(beat_calls[0][3]["g1_root_quat_order"], "xyzw")
            self.assertEqual(summary["beat_metadata"]["motion_beat_source"], "fk")


if __name__ == "__main__":
    unittest.main()
