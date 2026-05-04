import importlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class FullSongEvalHelperTests(unittest.TestCase):
    def test_discovers_full_test_wavs_not_sliced_wavs(self):
        eval_module = reload_module("eval.run_full_song_eval")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            full_dir = root / "test" / "wavs"
            sliced_dir = root / "test" / "wavs_sliced"
            full_dir.mkdir(parents=True)
            sliced_dir.mkdir(parents=True)
            (full_dir / "song_b.wav").touch()
            (full_dir / "song_a.wav").touch()
            (sliced_dir / "song_a_slice0.wav").touch()
            (sliced_dir / "song_a_slice1.wav").touch()

            songs = eval_module.discover_full_song_wavs(root)

        self.assertEqual(
            [path.name for path in songs],
            ["song_a.wav", "song_b.wav"],
        )

    def test_expected_long_frame_count_matches_overlap_stitching(self):
        eval_module = reload_module("eval.run_full_song_eval")

        self.assertEqual(eval_module.expected_long_frame_count(1), 150)
        self.assertEqual(eval_module.expected_long_frame_count(2), 225)
        self.assertEqual(eval_module.expected_long_frame_count(4), 375)

    def test_long_window_count_covers_song_tail(self):
        eval_module = reload_module("eval.run_full_song_eval")

        self.assertEqual(eval_module.long_window_count(4.0), 1)
        self.assertEqual(eval_module.long_window_count(5.0), 1)
        self.assertEqual(eval_module.long_window_count(7.1), 2)
        self.assertEqual(eval_module.long_window_count(11.99), 4)


class MetricComparisonTests(unittest.TestCase):
    def test_metric_comparison_writes_all_requested_rows(self):
        compare_module = reload_module("eval.write_metric_comparison")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            beatd = root / "beatd.json"
            official = root / "official.json"
            beatd.write_text(
                json.dumps(
                    {
                        "num_motion_files": 20,
                        "num_scored_files": 20,
                        "num_full_songs": 20,
                        "PFC": 1.2,
                        "BAS": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            official.write_text(
                json.dumps(
                    {
                        "num_motion_files": 20,
                        "num_scored_files": 20,
                        "num_full_songs": 20,
                        "PFC": 1.5,
                        "BAS": 0.4,
                    }
                ),
                encoding="utf-8",
            )

            rows = compare_module.write_comparison(
                entries=[("Official EDGE", official), ("BeatDistance", beatd)],
                json_path=root / "comparison.json",
                markdown_path=root / "comparison.md",
            )

            saved = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
            report = (root / "comparison.md").read_text(encoding="utf-8")

        self.assertEqual([row["label"] for row in rows], ["Official EDGE", "BeatDistance"])
        self.assertEqual(
            [row["label"] for row in saved["rows"]],
            ["Official EDGE", "BeatDistance"],
        )
        self.assertIn("Official EDGE", report)
        self.assertIn("BeatDistance", report)


if __name__ == "__main__":
    unittest.main()
