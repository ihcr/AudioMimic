import importlib
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


def reload_module():
    sys.modules.pop("data.prepare_finedance_dataset", None)
    return importlib.import_module("data.prepare_finedance_dataset")


def write_wav(path, seconds=2.0, sample_rate=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    sf.write(path, samples, sample_rate)


def write_finedance_motion(path, frames=60, moving=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    motion = np.zeros((frames, 315), dtype=np.float32)
    motion[:, 1] = 1.0
    axis_angle = np.zeros((frames, 52, 3), dtype=np.float32)
    if moving:
        ramp = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        motion[:, 0] = ramp
        axis_angle[:, :22, 0] = (2.0 * ramp)[:, None]
        axis_angle[:, 22, 1] = 2.5 * ramp
        axis_angle[:, 37, 2] = 3.0 * ramp
    motion[:, 3:] = axis_angle_x_y_z_to_rot6d(axis_angle).reshape(frames, -1)
    np.save(path, motion)
    return motion


def axis_angle_x_y_z_to_rot6d(axis_angle):
    matrix = np.zeros(axis_angle.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1, 1] = 1.0
    matrix[..., 2, 2] = 1.0

    x_mask = axis_angle[..., 0] != 0
    angle = axis_angle[..., 0]
    matrix[x_mask, 1, 1] = np.cos(angle[x_mask])
    matrix[x_mask, 1, 2] = -np.sin(angle[x_mask])
    matrix[x_mask, 2, 1] = np.sin(angle[x_mask])
    matrix[x_mask, 2, 2] = np.cos(angle[x_mask])

    y_mask = axis_angle[..., 1] != 0
    angle = axis_angle[..., 1]
    matrix[y_mask, 0, 0] = np.cos(angle[y_mask])
    matrix[y_mask, 0, 2] = np.sin(angle[y_mask])
    matrix[y_mask, 2, 0] = -np.sin(angle[y_mask])
    matrix[y_mask, 2, 2] = np.cos(angle[y_mask])

    z_mask = axis_angle[..., 2] != 0
    angle = axis_angle[..., 2]
    matrix[z_mask, 0, 0] = np.cos(angle[z_mask])
    matrix[z_mask, 0, 1] = -np.sin(angle[z_mask])
    matrix[z_mask, 1, 0] = np.sin(angle[z_mask])
    matrix[z_mask, 1, 1] = np.cos(angle[z_mask])
    return matrix[..., :2, :].reshape(axis_angle.shape[:-1] + (6,))


def write_prepared_clip(root, split, stem, feature_type="jukebox", use_beats=True):
    split_dir = root / split
    for name in ("motions_sliced", "wavs_sliced", f"{feature_type}_feats"):
        (split_dir / name).mkdir(parents=True, exist_ok=True)
    with open(split_dir / "motions_sliced" / f"{stem}.pkl", "wb") as handle:
        pickle.dump(
            {
                "pos": np.zeros((150, 3), dtype=np.float32),
                "q": np.zeros((150, 72), dtype=np.float32),
            },
            handle,
        )
    write_wav(split_dir / "wavs_sliced" / f"{stem}.wav", seconds=5.0)
    np.save(
        split_dir / f"{feature_type}_feats" / f"{stem}.npy",
        np.zeros((150, 4800), dtype=np.float32),
    )
    if use_beats:
        (split_dir / "beat_feats").mkdir(parents=True, exist_ok=True)
        np.savez(
            split_dir / "beat_feats" / f"{stem}.npz",
            motion_beats=np.array([0], dtype=np.int64),
            motion_mask=np.zeros(150, dtype=np.float32),
            motion_dist=np.zeros(150, dtype=np.int64),
            motion_spacing=np.ones(150, dtype=np.float32),
            audio_beats=np.array([0], dtype=np.int64),
            audio_mask=np.zeros(150, dtype=np.float32),
            audio_dist=np.zeros(150, dtype=np.int64),
            audio_spacing=np.ones(150, dtype=np.float32),
        )


class FineDanceConversionTests(unittest.TestCase):
    def test_axis_angle_conversion_uses_the_same_body_mapping_as_finedance_smpl_rendering(self):
        module = reload_module()
        raw = np.tile(np.arange(159, dtype=np.float32), (2, 1))

        pos, q = module.convert_finedance_body_motion_to_edge(raw)

        expected_q = np.concatenate(
            (raw[:, 3:69], raw[:, 69:72], raw[:, 114:117]),
            axis=1,
        )
        self.assertEqual(pos.shape, (2, 3))
        self.assertEqual(q.shape, (2, 72))
        np.testing.assert_array_equal(pos, raw[:, :3])
        np.testing.assert_array_equal(q, expected_q)

    def test_6d_conversion_uses_the_downloaded_finedance_body_mapping(self):
        module = reload_module()
        with TemporaryDirectory() as tmpdir:
            raw = write_finedance_motion(Path(tmpdir) / "motion.npy", frames=2)

            pos, q = module.convert_finedance_body_motion_to_edge(raw)

        axis_angle = np.zeros((2, 52, 3), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, 2, dtype=np.float32)
        axis_angle[:, :22, 0] = (2.0 * ramp)[:, None]
        axis_angle[:, 22, 1] = 2.5 * ramp
        axis_angle[:, 37, 2] = 3.0 * ramp
        expected_q = axis_angle[:, module.EDGE_BODY_JOINT_INDICES, :].reshape(2, -1)
        self.assertEqual(pos.shape, (2, 3))
        self.assertEqual(q.shape, (2, 72))
        np.testing.assert_array_equal(pos, raw[:, :3])
        np.testing.assert_allclose(q, expected_q, atol=1e-5)

    def test_conversion_rejects_unexpected_motion_width(self):
        module = reload_module()

        with self.assertRaisesRegex(ValueError, "expected 159 axis-angle columns or 315"):
            module.convert_finedance_body_motion_to_edge(np.zeros((10, 158), dtype=np.float32))

    def test_cross_genre_split_applies_official_ignore_list(self):
        module = reload_module()

        split = module.finedance_split_ids("cross_genre")

        self.assertIn("063", split.test_ids)
        self.assertNotIn("120", split.test_ids)
        self.assertNotIn("130", split.test_ids)
        self.assertNotIn("116", split.train_ids)
        self.assertNotIn("063", split.train_ids)


class FineDancePreparationTests(unittest.TestCase):
    def test_prepare_finedance_dataset_drops_static_train_clips_and_keeps_body_only_payloads(self):
        module = reload_module()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "finedance"
            output = tmp_path / "prepared"
            write_finedance_motion(source / "motion" / "001.npy", moving=False)
            write_finedance_motion(source / "motion" / "002.npy", moving=True)
            write_finedance_motion(source / "motion" / "063.npy", moving=False)
            write_wav(source / "music_wav" / "001.wav")
            write_wav(source / "music_wav" / "002.wav")
            write_wav(source / "music_wav" / "063.wav")
            stale = output / "train" / "motions_sliced" / "stale.pkl"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"stale")

            summary = module.prepare_finedance_dataset(
                finedance_root=source,
                output_root=output,
                split_name="cross_genre",
                length_seconds=1.0,
                stride_seconds=1.0,
                min_motion_std=0.07,
                root_height_offset=0.0,
            )

            train_motions = sorted((output / "train" / "motions_sliced").glob("*.pkl"))
            test_motions = sorted((output / "test" / "motions_sliced").glob("*.pkl"))
            self.assertFalse(stale.exists())
            self.assertTrue(train_motions)
            self.assertFalse(any(path.name.startswith("001_") for path in train_motions))
            self.assertTrue(any(path.name.startswith("002_") for path in train_motions))
            self.assertTrue(any(path.name.startswith("063_") for path in test_motions))
            self.assertGreater(summary["train"]["dropped_static_clips"], 0)

            with open(train_motions[0], "rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(set(payload), {"pos", "q"})
            self.assertEqual(payload["pos"].shape, (60, 3))
            self.assertEqual(payload["q"].shape, (60, 72))
            np.testing.assert_array_equal(payload["pos"][::2], payload["pos"][1::2])
            np.testing.assert_array_equal(payload["q"][::2], payload["q"][1::2])

    def test_build_mixed_dataset_uses_aist_test_as_main_test_split(self):
        module = reload_module()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            aist = tmp_path / "aist"
            finedance = tmp_path / "finedance_prepared"
            mixed = tmp_path / "mixed"
            write_prepared_clip(aist, "train", "aist_train")
            write_prepared_clip(aist, "test", "aist_test")
            write_prepared_clip(finedance, "train", "fd_train")
            write_prepared_clip(finedance, "test", "fd_test")
            stale = mixed / "train" / "motions_sliced" / "stale.pkl"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"stale")

            summary = module.build_mixed_aist_finedance_dataset(
                aist_root=aist,
                finedance_root=finedance,
                output_root=mixed,
                feature_type="jukebox",
                use_beats=True,
            )

            train_stems = sorted(path.stem for path in (mixed / "train" / "motions_sliced").glob("*.pkl"))
            test_stems = sorted(path.stem for path in (mixed / "test" / "motions_sliced").glob("*.pkl"))
            finedance_test_stems = sorted(
                path.stem for path in (mixed / "finedance_test" / "motions_sliced").glob("*.pkl")
            )
            self.assertEqual(train_stems, ["aist__aist_train", "finedance__fd_train"])
            self.assertFalse(stale.exists())
            self.assertEqual(test_stems, ["aist__aist_test"])
            self.assertEqual(finedance_test_stems, ["finedance__fd_test"])
            self.assertEqual(summary["train"]["clips"], 2)
            self.assertEqual(summary["test"]["source"], "aist")


if __name__ == "__main__":
    unittest.main()
