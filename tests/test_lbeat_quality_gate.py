import unittest


class LBeatEvalExitCodeTests(unittest.TestCase):
    def test_rejected_report_exits_success_by_default(self):
        from eval.lbeat_quality_gate import exit_code_for_lbeat_report

        self.assertEqual(
            exit_code_for_lbeat_report({"accepted": False}),
            0,
        )

    def test_rejected_report_can_fail_with_explicit_strict_mode(self):
        from eval.lbeat_quality_gate import exit_code_for_lbeat_report

        self.assertEqual(
            exit_code_for_lbeat_report({"accepted": False}, fail_on_rejection=True),
            1,
        )


if __name__ == "__main__":
    unittest.main()
