import unittest

import numpy as np

from eval.analyze_music_pairing_sensitivity import (
    _circular_audio_shift,
    _metric_contrast,
    _summarize,
)


class MusicPairingSensitivityTests(unittest.TestCase):
    def test_circular_shift_preserves_onset_values_and_wraps_beats(self):
        onset = np.arange(8, dtype=np.float64)
        beats = np.asarray([0.5, 3.5], dtype=np.float64)

        shifted_onset, shifted_beats = _circular_audio_shift(
            onset, beats, shift_seconds=1.0, fps=2.0
        )

        np.testing.assert_array_equal(shifted_onset, np.roll(onset, 2))
        np.testing.assert_allclose(shifted_beats, [0.5, 1.5])

    def test_metric_contrast_reports_rank_without_calling_it_a_p_value(self):
        contrast = _metric_contrast(0.5, [0.1, 0.4, 0.6, 0.2], 0.3)

        self.assertAlmostEqual(contrast["paired_rank_percentile"], 75.0)
        self.assertAlmostEqual(contrast["paired_minus_wrong_song"], 0.2)
        self.assertNotIn("p_value", contrast)

    def test_summary_keeps_reference_and_execution_separate(self):
        metric_payload = {
            metric: {
                "paired": 0.5,
                "shift_null_mean": 0.2,
                "paired_minus_null_mean": 0.3,
                "paired_rank_percentile": 80.0,
                "wrong_song": 0.1,
                "paired_minus_wrong_song": 0.4,
            }
            for metric in (
                "impact_zero_lag_correlation",
                "impact_best_correlation",
                "speed_zero_lag_correlation",
                "bas_music_to_motion",
                "bas_motion_to_music",
            )
        }
        records = [
            {"source": "reference", "route": "M2", "pairing": {"metrics": metric_payload}},
            {"source": "execution", "route": "M2", "pairing": {"metrics": metric_payload}},
        ]

        summary = _summarize(records)

        self.assertEqual(summary["reference:M2"]["samples"], 1)
        self.assertEqual(summary["execution:M2"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
