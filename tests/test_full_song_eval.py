import importlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np


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

    def test_discovers_full_test_wavs_from_explicit_full_wav_dir(self):
        eval_module = reload_module("eval.run_full_song_eval")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "prepared"
            full_wav_dir = root / "full_wavs"
            (data_root / "test" / "wavs_sliced").mkdir(parents=True)
            full_wav_dir.mkdir()
            (full_wav_dir / "song_b.wav").touch()
            (full_wav_dir / "song_a.wav").touch()

            songs = eval_module.discover_full_song_wavs(data_root, full_wav_dir=full_wav_dir)

        self.assertEqual(
            [path.name for path in songs],
            ["song_a.wav", "song_b.wav"],
        )

    def test_discovers_full_music_inputs_from_test_motion_names(self):
        eval_module = reload_module("eval.run_full_song_eval")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "prepared"
            motion_dir = data_root / "test" / "motions"
            music_dir = root / "music"
            motion_dir.mkdir(parents=True)
            music_dir.mkdir()
            (motion_dir / "gHO_sBM_cAll_d21_mHO5_ch02.pkl").touch()
            (motion_dir / "gBR_sBM_cAll_d04_mBR0_ch02.pkl").touch()
            (music_dir / "mHO5.mp3").touch()
            (music_dir / "mBR0.wav").touch()

            songs = eval_module.discover_full_song_inputs(
                data_root,
                full_music_dir=music_dir,
            )

        self.assertEqual(
            [(stem, path.name) for stem, path in songs],
            [
                ("gBR_sBM_cAll_d04_mBR0_ch02", "mBR0.wav"),
                ("gHO_sBM_cAll_d21_mHO5_ch02", "mHO5.mp3"),
            ],
        )

    def test_prepare_song_features_can_reuse_precomputed_test_slices(self):
        eval_module = reload_module("eval.run_full_song_eval")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "prepared"
            wav_dir = data_root / "test" / "wavs_sliced"
            feat_dir = data_root / "test" / "jukebox_feats"
            wav_dir.mkdir(parents=True)
            feat_dir.mkdir(parents=True)
            for idx in range(6):
                (wav_dir / f"song_slice{idx}.wav").touch()
                (feat_dir / f"song_slice{idx}.npy").touch()
            with patch.object(eval_module, "audio_duration_seconds", return_value=5.1), patch.object(
                eval_module.np,
                "load",
                side_effect=[
                    [[float(0)] * 4800] * 150,
                    [[float(1)] * 4800] * 150,
                ],
            ) as load_mock:
                wav_slices, music_cond = eval_module.prepare_song_features(
                    wav_path=root / "song.wav",
                    slice_dir=root / "unused",
                    feature_type="jukebox",
                    precomputed_data_path=data_root,
                )

        self.assertEqual([path.name for path in wav_slices], ["song_slice0.wav", "song_slice1.wav"])
        self.assertEqual(tuple(music_cond.shape), (2, 150, 4800))
        self.assertEqual(load_mock.call_count, 2)

    def test_slice_audio_uses_requested_output_stem_for_full_music(self):
        eval_module = reload_module("eval.run_full_song_eval")

        with TemporaryDirectory() as tmpdir, patch("soundfile.read") as read_mock, patch(
            "soundfile.write"
        ) as write_mock:
            read_mock.return_value = (np.zeros((4, 1), dtype=np.float32), 2)
            paths = eval_module.slice_audio_for_long_generation(
                Path(tmpdir) / "mHO5.mp3",
                Path(tmpdir) / "slices",
                output_stem="gHO_sBM_cAll_d21_mHO5_ch02",
            )

        self.assertEqual(paths[0].name, "gHO_sBM_cAll_d21_mHO5_ch02_slice0.wav")
        self.assertTrue(write_mock.called)

    def test_render_full_song_motion_threads_g1_render_arguments_and_original_audio(self):
        eval_module = reload_module("eval.run_full_song_eval")
        model = MagicMock()
        model.horizon = 150
        model.repr_dim = 38
        model.accelerator.device = "cpu"
        model.normalizer = object()
        args = MagicMock()
        args.render_dir = "renders"
        args.motion_save_dir = "motions"
        args.render = True
        args.motion_format = "g1"
        args.g1_fk_model_path = "robot.xml"
        args.g1_root_quat_order = "xyzw"
        args.g1_render_backend = "mujoco"
        args.g1_render_width = 640
        args.g1_render_height = 480
        args.g1_mujoco_gl = "egl"

        with patch.object(eval_module, "song_frame_count", return_value=360):
            eval_module.render_full_song_motion(
                model=model,
                cond=MagicMock(),
                wav_slices=[Path("song_slice0.wav"), Path("song_slice1.wav")],
                wav_path=Path("song.wav"),
                args=args,
            )

        _, kwargs = model.diffusion.render_sample.call_args
        self.assertEqual(kwargs["mode"], "long")
        self.assertTrue(kwargs["render"])
        self.assertEqual(kwargs["metadata_audio_path"], "song.wav")
        self.assertEqual(kwargs["g1_fk_model_path"], "robot.xml")
        self.assertEqual(kwargs["g1_root_quat_order"], "xyzw")
        self.assertEqual(kwargs["g1_render_backend"], "mujoco")
        self.assertEqual(kwargs["g1_render_width"], 640)
        self.assertEqual(kwargs["g1_render_height"], 480)
        self.assertEqual(kwargs["g1_mujoco_gl"], "egl")
        self.assertEqual(kwargs["metadata_total_frames"], 360)


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
