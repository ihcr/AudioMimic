import importlib
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def load_beat_features():
    return importlib.import_module("data.audio_extraction.beat_features")


class BeatDistanceUtilityTests(unittest.TestCase):
    def test_nearest_beat_distance_uses_clamped_unique_beats(self):
        beat_features = load_beat_features()

        result = beat_features.nearest_beat_distance(
            np.array([-2, 2, 2, 10]), seq_len=6
        )

        np.testing.assert_array_equal(result, np.array([0, 1, 0, 1, 1, 0]))

    def test_local_beat_spacing_uses_adjacent_intervals(self):
        beat_features = load_beat_features()

        result = beat_features.local_beat_spacing(np.array([1, 4, 6]), seq_len=8)

        np.testing.assert_array_equal(
            result, np.array([3, 3, 3, 3, 2, 2, 2, 2], dtype=np.float32)
        )


class BeatFeatureFolderTests(unittest.TestCase):
    def test_extract_folder_writes_expected_npz_payload(self):
        beat_features = load_beat_features()

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_dir = tmp_path / "motions_sliced"
            wav_dir = tmp_path / "wavs_sliced"
            out_dir = tmp_path / "beat_feats"
            motion_dir.mkdir()
            wav_dir.mkdir()

            with open(motion_dir / "clipA.pkl", "wb") as handle:
                pickle.dump({"pos": np.zeros((300, 3)), "q": np.zeros((300, 72))}, handle)
            (wav_dir / "clipA.wav").write_bytes(b"placeholder")

            motion_payload = (
                np.array([10, 40], dtype=np.int64),
                np.zeros(150, dtype=np.float32),
                np.arange(150, dtype=np.int64),
                np.full(150, 30.0, dtype=np.float32),
            )
            audio_payload = (
                np.array([12, 42], dtype=np.int64),
                np.ones(150, dtype=np.float32),
                np.arange(150, dtype=np.int64),
                np.full(150, 20.0, dtype=np.float32),
            )

            with patch.object(
                beat_features,
                "extract_motion_beats_from_motion_pkl",
                return_value=motion_payload,
            ), patch.object(
                beat_features,
                "extract_audio_beats_librosa",
                return_value=audio_payload,
            ):
                beat_features.extract_folder(
                    str(motion_dir), str(wav_dir), str(out_dir), fps=30, seq_len=150
                )

            payload = np.load(out_dir / "clipA.npz")

            self.assertEqual(
                set(payload.files),
                {
                    "motion_beats",
                    "motion_mask",
                    "motion_dist",
                    "motion_spacing",
                    "audio_beats",
                    "audio_mask",
                    "audio_dist",
                    "audio_spacing",
                },
            )
            self.assertEqual(payload["motion_dist"].shape, (150,))
            self.assertEqual(payload["audio_mask"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
