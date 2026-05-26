import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.audio_extraction.motion_energy_features import (
    TARGET_FRAMES,
    extract_motion_energy_features,
    parse_args,
)


def _write_motion(path, speed_scale):
    frames = np.arange(TARGET_FRAMES, dtype=np.float32)
    root_pos = np.zeros((TARGET_FRAMES, 3), dtype=np.float32)
    root_pos[:, 0] = frames * float(speed_scale)
    root_rot = np.zeros((TARGET_FRAMES, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    dof_pos = np.zeros((TARGET_FRAMES, 29), dtype=np.float32)
    q = np.concatenate((root_rot, dof_pos), axis=-1)
    payload = {
        "motion_format": "g1",
        "fps": 30,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "pos": root_pos,
        "q": q,
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _write_gaussian(path, beat_frames):
    values = np.zeros((TARGET_FRAMES, 1), dtype=np.float32)
    for frame in beat_frames:
        values[int(frame), 0] = 1.0
    np.save(path, values)


class MotionEnergyFeatureTests(unittest.TestCase):
    def test_extract_motion_energy_cache_writes_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "test"):
                for folder in ("motions_sliced", "gaussian_beat_feats"):
                    (root / split / folder).mkdir(parents=True)

            for idx, speed in enumerate((0.001, 0.002, 0.003)):
                stem = f"{idx:03d}_slice000"
                _write_motion(root / "train" / "motions_sliced" / f"{stem}.pkl", speed)
                _write_gaussian(root / "train" / "gaussian_beat_feats" / f"{stem}.npy", [20, 70, 120])
            _write_motion(root / "test" / "motions_sliced" / "100_slice000.pkl", 0.002)
            _write_gaussian(root / "test" / "gaussian_beat_feats" / "100_slice000.npy", [30, 90])

            args = parse_args(
                [
                    "--data_path",
                    str(root),
                    "--device",
                    "cpu",
                    "--batch_size",
                    "2",
                    "--g1_fk_model_path",
                    "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
                ]
            )
            metadata = extract_motion_energy_features(args)

            self.assertEqual(metadata["cache_version"], "motion_energy_v1")
            metadata_path = root / "motion_energy_metadata.json"
            self.assertTrue(metadata_path.is_file())
            with open(metadata_path, "r", encoding="utf-8") as handle:
                saved_metadata = json.load(handle)
            self.assertEqual(saved_metadata["normalization"]["train_peak_count"], 9)

            feature_path = root / "test" / "motion_energy_feats" / "100_slice000.npz"
            self.assertTrue(feature_path.is_file())
            with np.load(feature_path) as data:
                self.assertEqual(data["beat_energy_envelope"].shape, (TARGET_FRAMES, 1))
                self.assertEqual(data["weighted_fk_speed"].shape, (TARGET_FRAMES,))
                self.assertEqual(data["audio_beat_frames"].tolist(), [30, 90])
                self.assertEqual(data["beat_energy_peaks"].shape, (2,))
                self.assertTrue(np.isfinite(data["beat_energy_envelope"]).all())


if __name__ == "__main__":
    unittest.main()
