import unittest

from eval.build_module_benchmark import _score, _tempo


class ModuleBenchmarkTest(unittest.TestCase):
    def test_tempo_bins_match_frozen_protocol(self):
        self.assertEqual(_tempo(89.9), "slow_<90")
        self.assertEqual(_tempo(90.0), "medium_90-130")
        self.assertEqual(_tempo(130.0), "medium_90-130")
        self.assertEqual(_tempo(130.1), "fast_>130")

    def test_gt_match_score_is_high_at_reference_center(self):
        center = _score(5.0, [4.0, 5.0, 6.0])
        away = _score(10.0, [4.0, 5.0, 6.0])
        self.assertIsNotNone(center)
        self.assertIsNotNone(away)
        self.assertGreater(center, away)
        self.assertAlmostEqual(center, 1.0)


if __name__ == "__main__":
    unittest.main()
