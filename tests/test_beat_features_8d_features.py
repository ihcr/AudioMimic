import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data.audio_extraction.beat_features_8d_features import (
    FEATURE_NAMES,
    beat_features_8d_from_indices,
    extract_folder,
)


class BeatFeatures8DTests(unittest.TestCase):
    def test_channel_order_and_shape_are_stable(self):
        self.assertEqual(
            FEATURE_NAMES,
            (
                "beat_pulse",
                "gaussian_beat",
                "dist_to_prev_beat_norm",
                "dist_to_next_beat_norm",
                "beat_phase_sin",
                "beat_phase_cos",
                "beat_interval_norm",
                "onset_strength_norm",
            ),
        )
        onset = np.linspace(0.0, 1.0, 150, dtype=np.float32)
        features = beat_features_8d_from_indices([30, 60], onset)

        self.assertEqual(features.shape, (150, 8))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(float(features[30, 0]), 1.0)
        self.assertEqual(float(features[60, 0]), 1.0)
        self.assertEqual(float(features[30, 1]), 1.0)
        self.assertAlmostEqual(float(features[:, 7].min()), 0.0)
        self.assertAlmostEqual(float(features[:, 7].max()), 1.0)

    def test_no_beat_clip_stays_finite(self):
        features = beat_features_8d_from_indices([], np.zeros(150, dtype=np.float32))

        self.assertEqual(features.shape, (150, 8))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, 0], 0.0)
        np.testing.assert_allclose(features[:, 1], 0.0)
        np.testing.assert_allclose(features[:, 2], 1.0)
        np.testing.assert_allclose(features[:, 3], 1.0)
        np.testing.assert_allclose(features[:, 4], 0.0)
        np.testing.assert_allclose(features[:, 5], 1.0)
        np.testing.assert_allclose(features[:, 6], 1.0)
        np.testing.assert_allclose(features[:, 7], 0.0)

    def test_extract_folder_writes_expected_cache_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_dir = root / "wavs"
            out_dir = root / "beat_features_8d_feats"
            wav_dir.mkdir()
            (wav_dir / "clip.wav").write_bytes(b"placeholder")
            fake_features = np.ones((150, 8), dtype=np.float32)

            with patch(
                "data.audio_extraction.beat_features_8d_features.extract_beat_features_8d",
                return_value=fake_features,
            ):
                extract_folder(wav_dir, out_dir)

            written = np.load(out_dir / "clip.npy")
            self.assertEqual(written.shape, (150, 8))
            np.testing.assert_allclose(written, fake_features)


if __name__ == "__main__":
    unittest.main()
