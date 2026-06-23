import argparse
import unittest

import numpy as np


class RenderG1CheckpointComparisonTests(unittest.TestCase):
    def test_parse_model_spec_accepts_checkpoint_path(self):
        from eval.render_g1_checkpoint_comparison import parse_model_spec

        spec = parse_model_spec(
            "wav2clip_r02:wav2clip_stft_beat:stream_adapter:runs/train/foo.pt"
        )

        self.assertEqual(spec.label, "wav2clip_r02")
        self.assertEqual(spec.feature_type, "wav2clip_stft_beat")
        self.assertEqual(spec.feature_fusion, "stream_adapter")
        self.assertEqual(spec.checkpoint, "runs/train/foo.pt")
        self.assertEqual(spec.condition_variant, "auto")

    def test_parse_model_spec_accepts_condition_variant(self):
        from eval.render_g1_checkpoint_comparison import parse_model_spec

        spec = parse_model_spec(
            "v3:wav2clip_motion_intensity_beatness:linear:runs/train/foo.pt:pred_controls"
        )

        self.assertEqual(spec.label, "v3")
        self.assertEqual(spec.condition_variant, "pred_controls")

    def test_parse_model_spec_rejects_malformed_value(self):
        from eval.render_g1_checkpoint_comparison import parse_model_spec

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_model_spec("missing-fields")

    def test_checkpoint_motion_format_uses_checkpoint_config(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        import torch

        from eval.render_g1_checkpoint_comparison import _checkpoint_motion_format

        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.pt"
            torch.save({"config": {"motion_format": "g1_yaw_delta"}}, checkpoint)

            self.assertEqual(_checkpoint_motion_format(checkpoint, "g1"), "g1_yaw_delta")

    def test_write_label_banner_creates_png(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from eval.render_g1_checkpoint_comparison import write_label_banner

        with TemporaryDirectory() as tmpdir:
            banner = Path(tmpdir) / "labels.png"
            write_label_banner(["a", "b", "c"], banner, tile_width=100)

            self.assertTrue(banner.is_file())

    def test_write_label_overlay_creates_png(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from eval.render_g1_checkpoint_comparison import write_label_overlay

        with TemporaryDirectory() as tmpdir:
            overlay = Path(tmpdir) / "labels.png"
            write_label_overlay(["gt", "a", "b", "c"], overlay, 100, 80, columns=2)

            self.assertTrue(overlay.is_file())

    def test_resolve_comparison_layout_auto_uses_grid_for_four_tiles(self):
        from eval.render_g1_checkpoint_comparison import _resolve_comparison_layout

        self.assertEqual(_resolve_comparison_layout("auto", 4), "grid2x2")
        self.assertEqual(_resolve_comparison_layout("auto", 3), "horizontal")

    def test_stitch_g1_motion_payloads_uses_stride_overlap(self):
        from eval.render_g1_checkpoint_comparison import _stitch_g1_motion_payloads

        payloads = []
        for index in range(3):
            base = np.arange(150, dtype=np.float32) + index * 1000
            payloads.append(
                {
                    "motion_format": "g1",
                    "motion_rep": "g1",
                    "fps": 30.0,
                    "root_pos": np.repeat(base[:, None], 3, axis=1),
                    "root_rot": np.repeat(base[:, None], 4, axis=1),
                    "dof_pos": np.repeat(base[:, None], 29, axis=1),
                }
            )

        stitched = _stitch_g1_motion_payloads(payloads, stride_frames=75)

        self.assertEqual(stitched["root_pos"].shape, (300, 3))
        self.assertEqual(float(stitched["root_pos"][0, 0]), 0.0)
        self.assertEqual(float(stitched["root_pos"][149, 0]), 149.0)
        self.assertEqual(float(stitched["root_pos"][150, 0]), 1075.0)
        self.assertEqual(float(stitched["root_pos"][-1, 0]), 2149.0)

    def test_prepare_audio_slices_can_use_dataset_cache(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from eval.render_g1_checkpoint_comparison import _prepare_audio_slices

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_dir = root / "data" / "test" / "wavs_sliced"
            wav_dir.mkdir(parents=True)
            for idx in range(20):
                (wav_dir / f"012_slice{idx}.wav").touch()

            plan = _prepare_audio_slices(
                root / "music" / "012.wav",
                root / "renders",
                out_length=10.0,
                seed=1234,
                slice_start=4,
                data_path=root / "data",
                split="test",
                feature_source="cache",
            )

            self.assertEqual(plan["source"], "cache")
            self.assertEqual(plan["start_idx"], 4)
            self.assertEqual([path.name for path in plan["selected"]], [
                "012_slice4.wav",
                "012_slice9.wav",
                "012_slice14.wav",
            ])
            self.assertEqual(plan["slice_step"], 5)
            self.assertEqual(plan["effective_stride_seconds"], 2.5)

    def test_gt_motion_path_keeps_cached_slice_index_for_cache_source(self):
        from pathlib import Path

        from eval.render_g1_checkpoint_comparison import _gt_motion_path_for_audio_slice

        audio_plan = {
            "source": "cache",
            "music": Path("music/012.wav"),
            "effective_stride_seconds": 2.5,
        }

        path = _gt_motion_path_for_audio_slice(
            Path("data/test/wavs_sliced/012_slice15.wav"),
            Path("data"),
            "test",
            audio_plan,
        )

        self.assertEqual(path, Path("data/test/motions_sliced/012_slice15.pkl"))

    def test_gt_motion_path_maps_extract_slice_index_to_cached_stride(self):
        from pathlib import Path

        from eval.render_g1_checkpoint_comparison import _gt_motion_path_for_audio_slice

        audio_plan = {
            "source": "extract",
            "music": Path("music/012.wav"),
            "effective_stride_seconds": 2.5,
        }

        path = _gt_motion_path_for_audio_slice(
            Path("renders/audio_slices/012/012_slice3.wav"),
            Path("data"),
            "test",
            audio_plan,
        )

        self.assertEqual(path, Path("data/test/motions_sliced/012_slice15.pkl"))

    def test_load_cached_features_and_beats(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from eval.render_g1_checkpoint_comparison import (
            _load_cached_beat_condition,
            _load_cached_features,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_dir = root / "data" / "test" / "gaussian_beat_feats"
            beat_dir = root / "data" / "test" / "beat_feats"
            feature_dir.mkdir(parents=True)
            beat_dir.mkdir(parents=True)
            wavs = [root / "012_slice0.wav", root / "012_slice1.wav"]
            for idx, wav in enumerate(wavs):
                np.save(feature_dir / f"{wav.stem}.npy", np.full((150, 1), idx))
                np.savez(
                    beat_dir / f"{wav.stem}.npz",
                    audio_dist=np.full((150,), idx + 1),
                    audio_mask=np.full((150,), idx, dtype=np.float32),
                )

            features = _load_cached_features("gaussian_beat", wavs, root / "data", "test")
            beat = _load_cached_beat_condition(wavs, root / "data", "test", "distance")

            self.assertEqual(tuple(features.shape), (2, 150, 1))
            self.assertEqual(tuple(beat.shape), (2, 150))
            self.assertEqual(int(beat[1, 0].item()), 2)

    def test_build_structured_motion_condition_uses_wav2clip_and_zero_placeholders(self):
        import torch

        from eval.render_g1_checkpoint_comparison import _build_structured_motion_condition

        combined = torch.ones((2, 150, 706))
        gaussian = torch.full((2, 150, 1), 0.5)

        cond = _build_structured_motion_condition(
            "wav2clip_local_motion_intensity_beatness",
            {
                "wav2clip_stft_beat": combined,
                "gaussian_beat": gaussian,
            },
        )

        self.assertEqual(tuple(cond["semantic"]["wav2clip"].shape), (2, 150, 512))
        self.assertTrue(torch.equal(cond["control"]["gaussian_beat"], gaussian))
        self.assertEqual(tuple(cond["control"]["motion_intensity"].shape), (2, 150, 1))
        self.assertEqual(tuple(cond["control"]["motion_beatness"].shape), (2, 150, 1))
        self.assertEqual(float(cond["control"]["motion_intensity"].sum().item()), 0.0)

    def test_build_structured_motion_condition_uses_beat8d_and_zero_beatness(self):
        import torch

        from eval.render_g1_checkpoint_comparison import _build_structured_motion_condition

        beat_features = torch.ones((2, 150, 8))

        cond = _build_structured_motion_condition(
            "beat_features_8d_motion_beatness",
            {
                "beat_features_8d": beat_features,
            },
        )

        self.assertTrue(torch.equal(cond["semantic"]["beat_features_8d"], beat_features))
        self.assertEqual(tuple(cond["control"]["motion_beatness"].shape), (2, 150, 1))
        self.assertEqual(float(cond["control"]["motion_beatness"].sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
