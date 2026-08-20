import unittest

import numpy as np

from eval.analyze_generation_execution_gap import estimate_lag_seconds


class GenerationExecutionGapTests(unittest.TestCase):
    def test_positive_lag_means_execution_follows_reference(self):
        fps = 50.0
        t = np.arange(500) / fps
        target = np.stack((np.sin(2 * np.pi * 0.7 * t), np.cos(2 * np.pi * 1.3 * t)), axis=1)
        measured = np.concatenate((np.repeat(target[:1], 4, axis=0), target[:-4]), axis=0)
        lag, _ = estimate_lag_seconds(target, measured, fps, 0.25)
        self.assertAlmostEqual(lag, 4 / fps)


if __name__ == "__main__":
    unittest.main()
