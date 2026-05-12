import importlib
import sys
import unittest


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class G1CheckpointScreeningTests(unittest.TestCase):
    def test_quality_gate_accepts_beat_gain_with_sane_motion(self):
        screen = reload_module("eval.screen_g1_checkpoints")
        reference = {
            "G1BAS": 0.54,
            "G1FKBAS": 0.55,
            "G1BeatF1": 0.33,
            "G1FootSliding": 0.60,
            "RootVelocityMean": 0.35,
            "RootSmoothnessJerkMean": 720.0,
            "G1Dist": 6.8,
            "RootHeightViolationRate": 0.05,
            "G1GroundPenetration": 0.06,
        }
        candidate = {
            **reference,
            "G1BAS": 0.60,
            "G1FKBAS": 0.64,
            "G1BeatF1": 0.44,
            "G1FootSliding": 0.55,
            "RootSmoothnessJerkMean": 760.0,
        }

        acceptance = screen.evaluate_g1_acceptance(candidate, reference)

        self.assertTrue(acceptance["accepted"])
        self.assertGreater(
            screen.g1_checkpoint_score(candidate, reference),
            screen.g1_checkpoint_score(reference, reference),
        )

    def test_quality_gate_rejects_motion_collapse_even_with_high_beat(self):
        screen = reload_module("eval.screen_g1_checkpoints")
        reference = {
            "G1BAS": 0.54,
            "G1FKBAS": 0.55,
            "G1BeatF1": 0.33,
            "G1FootSliding": 0.60,
            "RootVelocityMean": 0.35,
            "RootSmoothnessJerkMean": 720.0,
            "G1Dist": 6.8,
            "RootHeightViolationRate": 0.05,
            "G1GroundPenetration": 0.06,
        }
        collapsed = {
            **reference,
            "G1BAS": 0.73,
            "G1FKBAS": 0.88,
            "G1BeatF1": 0.87,
            "G1FootSliding": 3.0,
            "RootVelocityMean": 3.0,
            "RootSmoothnessJerkMean": 3500.0,
            "G1Dist": 56.0,
            "RootHeightViolationRate": 0.33,
        }

        acceptance = screen.evaluate_g1_acceptance(collapsed, reference)

        self.assertFalse(acceptance["accepted"])
        self.assertIn("G1FootSliding", acceptance["failed_rules"])
        self.assertIn("G1Dist", acceptance["failed_rules"])
        self.assertLess(
            screen.g1_checkpoint_score(collapsed, reference),
            screen.g1_checkpoint_score(reference, reference),
        )

    def test_parse_epoch_list_rejects_empty_input(self):
        screen = reload_module("eval.screen_g1_checkpoints")

        with self.assertRaises(ValueError):
            screen.parse_epoch_list(" , ")


if __name__ == "__main__":
    unittest.main()
