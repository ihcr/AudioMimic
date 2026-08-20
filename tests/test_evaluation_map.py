import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "eval" / "evaluation_map_v1.json"


class EvaluationMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(MAP_PATH.read_text(encoding="utf-8"))

    def test_metric_registry_is_well_formed(self):
        required_fields = {
            "id",
            "stage",
            "name",
            "definition",
            "direction",
            "unit",
            "level",
            "source",
            "required_inputs",
            "aggregation",
            "applicability",
            "current_status",
            "acceptance_rule",
        }
        metrics = self.payload["metrics"]
        ids = [metric["id"] for metric in metrics]

        self.assertEqual(len(ids), len(set(ids)), "metric IDs must be unique")
        self.assertGreaterEqual(len(metrics), 40)
        for metric in metrics:
            self.assertFalse(required_fields - set(metric), metric["id"])
            self.assertIn(metric["direction"], self.payload["directions"])
            self.assertIn(metric["level"], self.payload["evidence_levels"])
            self.assertTrue(metric["source"], metric["id"])
            self.assertTrue(metric["required_inputs"], metric["id"])

    def test_all_pipeline_stages_are_represented(self):
        expected = {
            "protocol",
            "generator_quality",
            "music_alignment",
            "executability",
            "tracker",
            "executed_quality",
            "realtime_system",
            "human_evaluation",
        }
        actual = {metric["stage"] for metric in self.payload["metrics"]}
        self.assertEqual(expected, actual)

    def test_gate_metric_references_exist(self):
        ids = {metric["id"] for metric in self.payload["metrics"]}
        direct_metric_gates = self.payload["gates"]["data_and_realtime"]
        self.assertLessEqual(set(direct_metric_gates), ids)

        tracker_aliases = {
            "T-SUCC": "T-SUCC",
            "T-LAG_P95_MS": "T-LAG",
            "T-AMP_MEDIAN": "T-AMP",
            "T-ENERGY_MEDIAN": "T-ENERGY",
            "T-BAND_HIGH_MEDIAN": "T-BAND",
        }
        self.assertLessEqual(set(tracker_aliases.values()), ids)

    def test_protocol_separates_generation_and_tracking_repeats(self):
        protocol = self.payload["protocol"]
        self.assertGreaterEqual(protocol["minimum_generation_seeds_per_song"], 3)
        self.assertGreaterEqual(protocol["minimum_tracker_repeats_per_reference"], 3)
        self.assertIn("hold M_ref fixed", protocol["comparison_policy"])


if __name__ == "__main__":
    unittest.main()
