import unittest

import numpy as np

from eval.analyze_motion_music_execution import (
    _binary_metrics,
    _boundary_indices,
    _repeated_pose_ratio,
    _transition_timing_metrics,
    best_correlation,
)


class MotionMusicExecutionTests(unittest.TestCase):
    def test_positive_lag_means_motion_follows_audio(self):
        fps = 50.0
        time = np.arange(500) / fps
        audio = np.sin(2.0 * np.pi * 0.73 * time) + 0.2 * np.sin(2.0 * np.pi * 1.31 * time)
        motion = np.concatenate((np.repeat(audio[:1], 5), audio[:-5]))
        result = best_correlation(audio, motion, fps, max_lag_seconds=0.3)
        self.assertAlmostEqual(result["best_lag_seconds"], 5 / fps)
        self.assertGreater(result["best_correlation"], 0.99)

    def test_c4_boundaries_preserve_30_to_50_phase(self):
        indices = _boundary_indices(100, 50.0)
        self.assertEqual(indices[:6].tolist(), [13, 27, 40, 53, 67, 80])

    def test_repetition_ignores_local_neighbors(self):
        fps = 10.0
        pattern = np.linspace(0.0, 1.0, 30)[:, None]
        repeated = np.concatenate((pattern, pattern), axis=0)
        ratio, median = _repeated_pose_ratio(repeated, fps, exclusion_seconds=2.0)
        self.assertGreater(ratio, 0.95)
        self.assertLess(median, 1e-8)

    def test_contact_metrics_separate_detection_and_timing(self):
        target = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool)
        measured = np.array([0, 0, 0, 1, 1, 0, 0, 1], dtype=bool)
        classification = _binary_metrics(target, measured)
        timing = _transition_timing_metrics(target, measured, fps=10.0)
        self.assertAlmostEqual(classification["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(classification["recall"], 0.5)
        self.assertEqual(timing["target_transitions"], 3)
        self.assertEqual(timing["measured_transitions"], 3)
        self.assertEqual(timing["matched_ratio"], 1.0)
        self.assertEqual(timing["median_absolute_error_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
