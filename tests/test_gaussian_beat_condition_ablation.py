import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch


class GaussianBeatConditionAblationTests(unittest.TestCase):
    def test_shift_condition_delays_with_edge_padding(self):
        from eval.run_gaussian_beat_condition_ablation import shift_condition

        cond = torch.arange(5, dtype=torch.float32).view(5, 1)
        shifted = shift_condition(cond, 2)

        self.assertTrue(
            torch.equal(
                shifted.squeeze(-1),
                torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]),
            )
        )

    def test_constant_condition_like_uses_existing_shape_and_dtype(self):
        from eval.run_gaussian_beat_condition_ablation import constant_condition_like

        cond = torch.zeros((150, 1), dtype=torch.float32)
        constant = constant_condition_like(cond, 0.75)

        self.assertEqual(tuple(constant.shape), (150, 1))
        self.assertEqual(constant.dtype, torch.float32)
        self.assertAlmostEqual(float(constant.mean().item()), 0.75)

    def test_random_feature_paths_avoids_fixed_points_when_possible(self):
        from eval.run_gaussian_beat_condition_ablation import build_random_feature_paths

        paths = [Path(f"clip_{idx}.npy") for idx in range(16)]
        shuffled = build_random_feature_paths(paths, seed=1234)

        self.assertCountEqual(shuffled, paths)
        self.assertTrue(all(left != right for left, right in zip(paths, shuffled)))

    def test_compute_feature_mean(self):
        from eval.run_gaussian_beat_condition_ablation import compute_feature_mean

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "a.npy", np.full((2, 1), 0.25, dtype=np.float32))
            np.save(root / "b.npy", np.full((2, 1), 0.75, dtype=np.float32))

            self.assertAlmostEqual(compute_feature_mean(root), 0.5)


if __name__ == "__main__":
    unittest.main()
