import unittest

from eval.build_grouped_gt_benchmark import _styles, _tempo
from eval.audit_finedance_root_height import classify_root_height


class GroupedGtBenchmarkTest(unittest.TestCase):
    def test_aist_genre_is_derived_from_sequence_id(self):
        row = {"dataset": "aistpp", "sequence_id": "gBR_sBM_cAll_d04_mBR0_ch01"}
        self.assertEqual(_styles(row), ["Breaking"])

    def test_finedance_multilabel_styles_are_preserved_and_canonicalized(self):
        row = {"dataset": "finedance", "sequence_id": "001", "style": ["Street", "jazz"]}
        self.assertEqual(_styles(row), ["Jazz", "Street"])

    def test_tempo_bins_are_not_style_labels(self):
        def row(bpm):
            return {"music": {"audio_bpm": bpm}}

        self.assertEqual(_tempo(row(89.9)), "slow_<90")
        self.assertEqual(_tempo(row(90.0)), "medium_90-130")
        self.assertEqual(_tempo(row(130.0)), "medium_90-130")
        self.assertEqual(_tempo(row(130.1)), "fast_>130")

    def test_root_height_audit_separates_transient_drop(self):
        common = dict(
            root_min=-0.30,
            root_median=0.80,
            negative_fraction=0.01,
            penetration_ratio=0.01,
            foot_min=-0.05,
            median_height_warning_m=0.60,
            max_transient_negative_fraction=0.05,
            max_penetration_ratio=0.02,
            min_foot_height_warning_m=-0.12,
        )
        self.assertEqual(
            classify_root_height(**common), "transient_root_drop_feet_near_ground"
        )
        common["penetration_ratio"] = 0.10
        self.assertEqual(
            classify_root_height(**common), "root_drop_with_physical_warning"
        )


if __name__ == "__main__":
    unittest.main()
