import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MetricTaxonomyTest(unittest.TestCase):
    def test_every_registered_metric_has_exactly_one_taxonomy_location(self):
        evaluation_map = json.loads((ROOT / "eval/evaluation_map_v1.json").read_text(encoding="utf-8"))
        taxonomy = json.loads((ROOT / "eval/metric_taxonomy_v1.json").read_text(encoding="utf-8"))
        registered = {row["id"] for row in evaluation_map["metrics"]}
        assigned = [
            metric_id
            for module in taxonomy["modules"]
            for submodule in module["submodules"]
            for metric_id in submodule["metric_ids"]
        ]
        self.assertEqual(len(assigned), len(set(assigned)), "a metric is assigned more than once")
        self.assertEqual(registered, set(assigned))

    def test_calibration_methods_reference_registered_metrics(self):
        evaluation_map = json.loads((ROOT / "eval/evaluation_map_v1.json").read_text(encoding="utf-8"))
        taxonomy = json.loads((ROOT / "eval/metric_taxonomy_v1.json").read_text(encoding="utf-8"))
        registered = {row["id"] for row in evaluation_map["metrics"]}
        for method in taxonomy["calibration_methods"]:
            self.assertTrue(method["validates"])
            self.assertLessEqual(set(method["validates"]), registered)


if __name__ == "__main__":
    unittest.main()
