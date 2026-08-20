import tempfile
import unittest
from pathlib import Path

from eval.build_human_pairwise_study import (
    _asset_issues,
    _readiness,
    build_assignments,
    build_candidates,
)


def _asset(route, representation="reference", repeat=None, seed=1234):
    payload = {
        "asset_id": f"{route}_{representation}_{repeat}",
        "path": "clip.mp4",
        "route": route,
        "representation": representation,
        "song_id": "098",
        "generation_seed": seed,
        "window_id": "w1",
        "duration_seconds": 16.0,
        "camera_profile": "front_follow",
        "render_profile": "paper_v1",
        "label_free": True,
        "audio_embedded": True,
        "eligible": True,
    }
    if repeat is not None:
        payload["tracker_repeat"] = repeat
    return payload


class HumanPairwiseStudyTests(unittest.TestCase):
    def test_debug_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.touch()
            asset = _asset("M2")
            asset["path"] = str(path)
            asset["label_free"] = False

            issues = _asset_issues(asset, Path(directory))

            self.assertIn("video contains identifying labels or debug composition", issues)

    def test_matched_three_route_design_builds_all_route_pairs_and_tracker_trials(self):
        references = [_asset(route) for route in ("M0", "M2", "M4")]
        executions = [
            _asset(route, "execution", repeat)
            for route in ("M0", "M2", "M4")
            for repeat in (1, 2, 3)
        ]

        candidates = build_candidates(references + executions)

        by_task = {
            task: [trial for trial in candidates if trial["task"] == task]
            for task in ("generator_quality", "music_match", "execution_retention")
        }
        self.assertEqual(len(by_task["generator_quality"]), 3)
        self.assertEqual(len(by_task["music_match"]), 3)
        self.assertEqual(len(by_task["execution_retention"]), 9)

    def test_different_generation_seeds_do_not_form_route_pair(self):
        candidates = build_candidates([_asset("M0", seed=1234), _asset("M2", seed=2345)])

        self.assertEqual(candidates, [])

    def test_coherence_question_is_reserved_for_long_clips(self):
        short = build_candidates([_asset("M0"), _asset("M2")])
        long_assets = [_asset("M0"), _asset("M2")]
        for asset in long_assets:
            asset["duration_seconds"] = 30.0
        long = build_candidates(long_assets)

        short_quality = next(trial for trial in short if trial["task"] == "generator_quality")
        long_quality = next(trial for trial in long if trial["task"] == "generator_quality")
        self.assertNotIn("H-COHERENCE", short_quality["questions"])
        self.assertIn("H-COHERENCE", long_quality["questions"])

    def test_two_participants_receive_opposite_sides_for_single_trial(self):
        candidates = build_candidates([_asset("M0"), _asset("M2")])
        quality = [trial for trial in candidates if trial["task"] == "generator_quality"]

        assignments = build_assignments(
            quality, participants=2, trials_per_task=1, seed=7
        )

        first = assignments[0]["trials"][0]
        second = assignments[1]["trials"][0]
        self.assertEqual(first["left_asset"], second["right_asset"])
        self.assertEqual(first["right_asset"], second["left_asset"])

    def test_pilot_and_paper_readiness_are_separate(self):
        assets = [_asset(route) for route in ("M0", "M2", "M4")]
        assets.extend(
            _asset(route, "execution", 1) for route in ("M0", "M2", "M4")
        )
        candidates = build_candidates(assets)

        pilot = _readiness(
            assets,
            candidates,
            {"songs": 3, "generation_seeds_per_song": 3, "tracker_repeats_per_reference": 3},
        )
        minimal = _readiness(
            assets,
            candidates,
            {"songs": 1, "generation_seeds_per_song": 1, "tracker_repeats_per_reference": 1},
        )

        self.assertTrue(pilot["pilot_ready"])
        self.assertFalse(pilot["paper_ready"])
        self.assertTrue(minimal["paper_ready"])


if __name__ == "__main__":
    unittest.main()
