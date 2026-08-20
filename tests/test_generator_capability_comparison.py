import unittest

from eval.compare_generator_to_sonic_capability import (
    _execution_summary,
    _gate_diagnostics,
    _nearest_gt_tier,
)


class GeneratorCapabilityComparisonTests(unittest.TestCase):
    def test_nearest_gt_tier_uses_all_dynamic_statistics(self):
        dynamic = {
            "energy": 2.0,
            "velocity_p95": 3.0,
            "acceleration_p95": 30.0,
            "jerk_p95": 300.0,
        }
        primary = {
            "low": {
                "motion_energy_rad2_s2": 1.0,
                "velocity_p95_rad_s": 2.0,
                "acceleration_p95_rad_s2": 20.0,
                "jerk_p95_rad_s3": 200.0,
            },
            "medium": {
                "motion_energy_rad2_s2": 2.0,
                "velocity_p95_rad_s": 3.0,
                "acceleration_p95_rad_s2": 30.0,
                "jerk_p95_rad_s3": 300.0,
            },
            "high": {
                "motion_energy_rad2_s2": 4.0,
                "velocity_p95_rad_s": 6.0,
                "acceleration_p95_rad_s2": 60.0,
                "jerk_p95_rad_s3": 600.0,
            },
        }

        nearest, distances = _nearest_gt_tier(dynamic, primary)

        self.assertEqual(nearest, "medium")
        self.assertAlmostEqual(distances["medium"]["log_euclidean_distance"], 0.0)
        self.assertEqual(
            distances["medium"]["ratios"],
            {
                "energy": 1.0,
                "velocity_p95": 1.0,
                "acceleration_p95": 1.0,
                "jerk_p95": 1.0,
            },
        )

    def test_execution_summary_uses_per_joint_medians_and_converts_lag(self):
        def record(value, fell=False):
            return {
                "tracking": {
                    "fell": fell,
                    "lag_compensated_position_rmse_rad": value,
                    "global_lag_seconds": value / 10.0,
                    "minimum_base_height_m": 0.7,
                    "fk": {"root_relative_empkpe_m": value / 2.0},
                    "body_groups": {
                        "full_body": {
                            "amplitude_retention": {"median": 0.9},
                            "energy_retention": {"median": value},
                            "band_power_retention": {
                                "low_0_1": {"median": 0.8},
                                "mid_1_3": {"median": 0.6},
                                "high_3_8": {"median": 0.4},
                            },
                        },
                        "arms": {
                            "band_power_retention": {
                                "high_3_8": {"median": 0.3},
                            }
                        },
                    },
                }
            }

        summary = _execution_summary([record(0.2), record(0.4, fell=True)])

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["successes"], 1)
        self.assertAlmostEqual(summary["energy_retention"]["mean"], 0.3)
        self.assertAlmostEqual(summary["lag_ms"]["mean"], 30.0)

    def test_gate_diagnostics_respect_metric_directions(self):
        execution = {
            "aligned_rmse_rad": {"mean": 0.1},
            "empkpe_m": {"mean": 0.05},
            "amplitude_retention": {"mean": 0.9},
            "energy_retention": {"mean": 0.8},
            "low_band_retention": {"mean": 0.7},
            "mid_band_retention": {"mean": 0.6},
            "high_band_retention": {"mean": 0.5},
        }
        capability = {
            "empirical_gates": {
                "aligned_joint_rmse_p95_rad": 0.2,
                "root_relative_empkpe_p95_m": 0.1,
                "amplitude_retention_p05": 0.8,
                "energy_retention_p05": 0.7,
                "low_band_retention_p05": 0.6,
                "mid_band_retention_p05": 0.5,
                "high_band_retention_p05": 0.4,
            }
        }

        diagnostics = _gate_diagnostics(execution, capability)

        self.assertEqual(diagnostics["passed"], 7)
        self.assertTrue(all(diagnostics["checks"].values()))


if __name__ == "__main__":
    unittest.main()
