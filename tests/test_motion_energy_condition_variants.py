import unittest

import torch

from eval.run_g1_dataset_eval import apply_motion_energy_condition_variant


class _Predictor:
    def predict_energy(self, cond):
        return torch.full_like(cond["control"]["beat_energy_envelope"], 0.75)


class _Diffusion:
    model = _Predictor()


class _Model:
    diffusion = _Diffusion()


class _ControlPredictor:
    def predict_controls(self, cond):
        return {
            "motion_intensity": torch.full_like(cond["control"]["motion_intensity"], 0.25),
            "motion_beatness": torch.full_like(cond["control"]["motion_beatness"], 0.75),
        }


class _ControlDiffusion:
    model = _ControlPredictor()


class _ControlModel:
    diffusion = _ControlDiffusion()


class MotionEnergyConditionVariantTest(unittest.TestCase):
    def make_condition(self):
        return {
            "semantic": {
                "wav2clip": torch.ones(2, 4, 3),
            },
            "control": {
                "gaussian_beat": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
                "beat_energy_envelope": torch.tensor(
                    [
                        [[0.0], [0.2], [0.4], [0.6]],
                        [[0.1], [0.3], [0.5], [0.7]],
                    ],
                    dtype=torch.float32,
                ),
            },
        }

    def test_pred_energy_uses_model_predictor(self):
        cond = self.make_condition()
        updated = apply_motion_energy_condition_variant(_Model(), cond, "pred_energy")
        self.assertTrue(
            torch.allclose(
                updated["control"]["beat_energy_envelope"],
                torch.full((2, 4, 1), 0.75),
            )
        )
        self.assertTrue(
            torch.equal(
                updated["control"]["gaussian_beat"],
                cond["control"]["gaussian_beat"],
            )
        )

    def test_flat_energy_keeps_per_sample_mean(self):
        cond = self.make_condition()
        updated = apply_motion_energy_condition_variant(_Model(), cond, "flat_energy")
        expected = torch.tensor(
            [
                [[0.3], [0.3], [0.3], [0.3]],
                [[0.4], [0.4], [0.4], [0.4]],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(updated["control"]["beat_energy_envelope"], expected))

    def test_zero_control_zeros_gaussian_and_energy(self):
        cond = self.make_condition()
        updated = apply_motion_energy_condition_variant(_Model(), cond, "zero_control")
        self.assertTrue(torch.count_nonzero(updated["control"]["gaussian_beat"]).item() == 0)
        self.assertTrue(torch.count_nonzero(updated["control"]["beat_energy_envelope"]).item() == 0)

    def make_v3_condition(self):
        return {
            "semantic": {
                "wav2clip": torch.ones(2, 4, 3),
            },
            "control": {
                "gaussian_beat": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
                "motion_intensity": torch.tensor(
                    [
                        [[0.0], [0.2], [0.4], [0.6]],
                        [[0.1], [0.3], [0.5], [0.7]],
                    ],
                    dtype=torch.float32,
                ),
                "motion_beatness": torch.tensor(
                    [
                        [[0.9], [0.7], [0.5], [0.3]],
                        [[0.8], [0.6], [0.4], [0.2]],
                    ],
                    dtype=torch.float32,
                ),
            },
        }

    def test_pred_controls_uses_two_head_predictor(self):
        cond = self.make_v3_condition()
        updated = apply_motion_energy_condition_variant(_ControlModel(), cond, "pred_controls")
        self.assertTrue(
            torch.allclose(
                updated["control"]["motion_intensity"],
                torch.full((2, 4, 1), 0.25),
            )
        )
        self.assertTrue(
            torch.allclose(
                updated["control"]["motion_beatness"],
                torch.full((2, 4, 1), 0.75),
            )
        )

    def test_flat_intensity_and_zero_beatness_keep_other_controls(self):
        cond = self.make_v3_condition()
        flat = apply_motion_energy_condition_variant(_ControlModel(), cond, "flat_intensity")
        expected = torch.tensor(
            [
                [[0.3], [0.3], [0.3], [0.3]],
                [[0.4], [0.4], [0.4], [0.4]],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(flat["control"]["motion_intensity"], expected))
        self.assertTrue(torch.equal(flat["control"]["motion_beatness"], cond["control"]["motion_beatness"]))

        zero = apply_motion_energy_condition_variant(_ControlModel(), cond, "zero_beatness")
        self.assertTrue(torch.count_nonzero(zero["control"]["motion_beatness"]).item() == 0)
        self.assertTrue(torch.equal(zero["control"]["motion_intensity"], cond["control"]["motion_intensity"]))

    def test_zero_all_controls_zeros_v3_control_streams(self):
        cond = self.make_v3_condition()
        updated = apply_motion_energy_condition_variant(_ControlModel(), cond, "zero_all_controls")
        self.assertTrue(torch.count_nonzero(updated["control"]["gaussian_beat"]).item() == 0)
        self.assertTrue(torch.count_nonzero(updated["control"]["motion_intensity"]).item() == 0)
        self.assertTrue(torch.count_nonzero(updated["control"]["motion_beatness"]).item() == 0)


if __name__ == "__main__":
    unittest.main()
