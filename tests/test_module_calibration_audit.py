import unittest

from eval.build_module_calibration_audit import _annotate, _metric_inventory, _status


class ModuleCalibrationAuditTest(unittest.TestCase):
    def test_known_checks_map_to_frozen_modules(self):
        row = _annotate(
            {
                "dataset": "aistpp",
                "check": "jitter_jerk_p95_rad_s3",
                "expected": "increase",
                "delta_high_minus_clean": 1.0,
                "result": "PASS",
            }
        )
        self.assertEqual(row["module"], "D")
        self.assertEqual(row["submodule"], "smoothness")
        self.assertEqual(row["metric"], "jerk_p95")

    def test_warn_keeps_module_as_pass_with_caveat(self):
        status = _status(
            [
                {"module": "D", "result": "PASS"},
                {"module": "M", "result": "PASS"},
                {"module": "M", "result": "WARN"},
            ]
        )
        self.assertEqual(status["D"]["status"], "PASS")
        self.assertEqual(status["M"]["status"], "PASS_WITH_CAVEAT")

    def test_inventory_covers_all_taxonomy_metrics(self):
        taxonomy = {
            "modules": [
                {"id": "D", "name_zh": "动作", "submodules": [{"id": "D.x", "name_zh": "子项", "metric_ids": ["G-X"]}]},
                {"id": "X", "name_zh": "执行", "submodules": [{"id": "X.x", "name_zh": "子项", "metric_ids": ["T-X"]}]},
            ]
        }
        evaluation = {"metrics": [{"id": "G-X", "name": "gx", "current_status": "implemented"}, {"id": "T-X", "name": "tx", "current_status": "missing"}]}
        inventory = _metric_inventory(taxonomy, evaluation, {})
        self.assertEqual({row["metric_id"] for row in inventory}, {"G-X", "T-X"})
        self.assertEqual(len(inventory), 2)


if __name__ == "__main__":
    unittest.main()
