import importlib
import sys
import unittest

import numpy as np


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class BeatFeatures8DTests(unittest.TestCase):
    def setUp(self):
        self.module = reload_module("data.audio_extraction.beat_features_8d_features")

    def test_channel_order_and_shape_are_stable(self):
        self.assertEqual(
            self.module.FEATURE_NAMES,
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
        features = self.module.beat_features_8d_from_indices(
            np.array([0, 75, 149]),
            np.linspace(0.0, 1.0, 150, dtype=np.float32),
        )
        self.assertEqual(features.shape, (150, 8))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0, 0], 1.0)
        self.assertEqual(features[75, 0], 1.0)

    def test_no_beat_clip_is_finite(self):
        features = self.module.beat_features_8d_from_indices(
            np.array([], dtype=np.int64),
            np.zeros(150, dtype=np.float32),
        )
        self.assertEqual(features.shape, (150, 8))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.allclose(features[:, 0], 0.0))
        self.assertTrue(np.allclose(features[:, 2], 1.0))
        self.assertTrue(np.allclose(features[:, 3], 1.0))


if __name__ == "__main__":
    unittest.main()
