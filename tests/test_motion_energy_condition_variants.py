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


class _BeatnessOnlyPredictor:
    def predict_controls(self, cond):
        return {
            "motion_beatness": torch.full_like(cond["control"]["motion_beatness"], 0.65),
        }


class _BeatnessOnlyDiffusion:
    model = _BeatnessOnlyPredictor()


class _BeatnessOnlyModel:
    diffusion = _BeatnessOnlyDiffusion()


class _BodySupportPredictor:
    def predict_controls(self, cond):
        control = cond["control"]
        return {
            "body_intensity": torch.full_like(control["body_intensity"], 0.15),
            "support_beatness": torch.full_like(control["support_beatness"], 0.35),
            "upper_beatness": torch.full_like(control["upper_beatness"], 0.55),
            "support_contact": torch.full_like(control["support_contact"], 0.75),
        }


class _BodySupportDiffusion:
    model = _BodySupportPredictor()


class _BodySupportModel:
    diffusion = _BodySupportDiffusion()


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

    def make_beatness_only_condition(self):
        return {
            "semantic": {
                "beat_features_8d": torch.ones(2, 4, 8),
            },
            "control": {
                "motion_beatness": torch.tensor(
                    [
                        [[0.9], [0.7], [0.5], [0.3]],
                        [[0.8], [0.6], [0.4], [0.2]],
                    ],
                    dtype=torch.float32,
                ),
            },
        }

    def test_auto_pred_controls_handles_beatness_only_predictor(self):
        cond = self.make_beatness_only_condition()
        updated = apply_motion_energy_condition_variant(_BeatnessOnlyModel(), cond, "auto")

        self.assertTrue(
            torch.allclose(
                updated["control"]["motion_beatness"],
                torch.full((2, 4, 1), 0.65),
            )
        )
        self.assertTrue(torch.equal(updated["semantic"]["beat_features_8d"], cond["semantic"]["beat_features_8d"]))
        self.assertNotIn("motion_intensity", updated["control"])

    def test_zero_all_controls_zeros_beatness_only_semantic_and_control(self):
        cond = self.make_beatness_only_condition()
        updated = apply_motion_energy_condition_variant(
            _BeatnessOnlyModel(),
            cond,
            "zero_all_controls",
        )

        self.assertTrue(torch.count_nonzero(updated["semantic"]["beat_features_8d"]).item() == 0)
        self.assertTrue(torch.count_nonzero(updated["control"]["motion_beatness"]).item() == 0)

    def make_body_support_condition(self):
        return {
            "semantic": {
                "wav2clip": torch.ones(2, 4, 3),
            },
            "control": {
                "gaussian_beat": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
                "body_intensity": torch.full((2, 4, 1), 0.2),
                "support_beatness": torch.full((2, 4, 1), 0.4),
                "upper_beatness": torch.full((2, 4, 1), 0.6),
                "support_contact": torch.ones(2, 4, 2),
            },
        }

    def test_body_support_pred_controls_and_diagnostic_variants(self):
        cond = self.make_body_support_condition()
        pred = apply_motion_energy_condition_variant(_BodySupportModel(), cond, "auto")
        self.assertTrue(torch.allclose(pred["control"]["body_intensity"], torch.full((2, 4, 1), 0.15)))
        self.assertTrue(torch.allclose(pred["control"]["support_beatness"], torch.full((2, 4, 1), 0.35)))
        self.assertTrue(torch.allclose(pred["control"]["upper_beatness"], torch.full((2, 4, 1), 0.55)))
        self.assertTrue(torch.allclose(pred["control"]["support_contact"], torch.full((2, 4, 2), 0.75)))

        flat = apply_motion_energy_condition_variant(_BodySupportModel(), cond, "flat_body_intensity")
        self.assertTrue(torch.allclose(flat["control"]["body_intensity"], torch.full((2, 4, 1), 0.2)))
        self.assertTrue(torch.equal(flat["control"]["support_beatness"], cond["control"]["support_beatness"]))

        zero_support = apply_motion_energy_condition_variant(
            _BodySupportModel(),
            cond,
            "zero_support_beatness",
        )
        self.assertEqual(torch.count_nonzero(zero_support["control"]["support_beatness"]).item(), 0)
        self.assertTrue(torch.equal(zero_support["control"]["upper_beatness"], cond["control"]["upper_beatness"]))

        zero_upper = apply_motion_energy_condition_variant(
            _BodySupportModel(),
            cond,
            "zero_upper_beatness",
        )
        self.assertEqual(torch.count_nonzero(zero_upper["control"]["upper_beatness"]).item(), 0)
        self.assertTrue(torch.equal(zero_upper["control"]["support_beatness"], cond["control"]["support_beatness"]))

        zero_contact = apply_motion_energy_condition_variant(
            _BodySupportModel(),
            cond,
            "zero_support_contact",
        )
        self.assertEqual(torch.count_nonzero(zero_contact["control"]["support_contact"]).item(), 0)

        zero_all = apply_motion_energy_condition_variant(_BodySupportModel(), cond, "zero_all_controls")
        for key in (
            "gaussian_beat",
            "body_intensity",
            "support_beatness",
            "upper_beatness",
            "support_contact",
        ):
            self.assertEqual(torch.count_nonzero(zero_all["control"][key]).item(), 0)


if __name__ == "__main__":
    unittest.main()
