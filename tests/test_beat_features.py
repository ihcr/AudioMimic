import importlib
import os
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf


def load_beat_features():
    return importlib.import_module("data.audio_extraction.beat_features")


class BeatDistanceUtilityTests(unittest.TestCase):
    def test_nearest_beat_distance_uses_clamped_unique_beats(self):
        beat_features = load_beat_features()

        result = beat_features.nearest_beat_distance(
            np.array([-2, 2, 2, 10]), seq_len=6
        )

        np.testing.assert_array_equal(result, np.array([0, 1, 0, 1, 1, 0]))

    def test_local_beat_spacing_uses_adjacent_intervals(self):
        beat_features = load_beat_features()

        result = beat_features.local_beat_spacing(np.array([1, 4, 6]), seq_len=8)

        np.testing.assert_array_equal(
            result, np.array([3, 3, 3, 3, 2, 2, 2, 2], dtype=np.float32)
        )


class BeatFeatureFolderTests(unittest.TestCase):
    def test_extract_folder_writes_expected_npz_payload(self):
        beat_features = load_beat_features()

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_dir = tmp_path / "motions_sliced"
            wav_dir = tmp_path / "wavs_sliced"
            out_dir = tmp_path / "beat_feats"
            motion_dir.mkdir()
            wav_dir.mkdir()

            with open(motion_dir / "clipA.pkl", "wb") as handle:
                pickle.dump({"pos": np.zeros((300, 3)), "q": np.zeros((300, 72))}, handle)
            (wav_dir / "clipA.wav").write_bytes(b"placeholder")

            motion_payload = (
                np.array([10, 40], dtype=np.int64),
                np.zeros(150, dtype=np.float32),
                np.arange(150, dtype=np.int64),
                np.full(150, 30.0, dtype=np.float32),
            )
            audio_payload = (
                np.array([12, 42], dtype=np.int64),
                np.ones(150, dtype=np.float32),
                np.arange(150, dtype=np.int64),
                np.full(150, 20.0, dtype=np.float32),
            )

            with patch.object(
                beat_features,
                "extract_motion_beats_from_motion_pkl",
                return_value=motion_payload,
            ), patch.object(
                beat_features,
                "extract_audio_beats_librosa",
                return_value=audio_payload,
            ):
                beat_features.extract_folder(
                    str(motion_dir), str(wav_dir), str(out_dir), fps=30, seq_len=150
                )

            payload = np.load(out_dir / "clipA.npz")

            self.assertEqual(
                set(payload.files),
                {
                    "motion_beats",
                    "motion_mask",
                    "motion_dist",
                    "motion_spacing",
                    "audio_beats",
                    "audio_mask",
                    "audio_dist",
                    "audio_spacing",
                },
            )
            self.assertEqual(payload["motion_dist"].shape, (150,))
            self.assertEqual(payload["audio_mask"].dtype, np.float32)

    def test_extract_folder_reports_progress_and_skips_completed_outputs(self):
        beat_features = load_beat_features()

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_dir = tmp_path / "motions_sliced"
            wav_dir = tmp_path / "wavs_sliced"
            out_dir = tmp_path / "beat_feats"
            motion_dir.mkdir()
            wav_dir.mkdir()
            out_dir.mkdir()

            for clip_name in ("clipA", "clipB"):
                with open(motion_dir / f"{clip_name}.pkl", "wb") as handle:
                    pickle.dump({"pos": np.zeros((300, 3)), "q": np.zeros((300, 72))}, handle)
                (wav_dir / f"{clip_name}.wav").write_bytes(b"placeholder")

            np.savez(
                out_dir / "clipA.npz",
                motion_beats=np.array([1], dtype=np.int64),
                motion_mask=np.zeros(150, dtype=np.float32),
                motion_dist=np.zeros(150, dtype=np.int64),
                motion_spacing=np.ones(150, dtype=np.float32),
                audio_beats=np.array([1], dtype=np.int64),
                audio_mask=np.zeros(150, dtype=np.float32),
                audio_dist=np.zeros(150, dtype=np.int64),
                audio_spacing=np.ones(150, dtype=np.float32),
            )

            motion_payload = (
                np.array([10, 40], dtype=np.int64),
                np.zeros(150, dtype=np.float32),
                np.arange(150, dtype=np.int64),
                np.full(150, 30.0, dtype=np.float32),
            )
            audio_payload = (
                np.array([12, 42], dtype=np.int64),
                np.ones(150, dtype=np.float32),
                np.arange(150, dtype=np.int64),
                np.full(150, 20.0, dtype=np.float32),
            )

            progress_calls = []

            def fake_tqdm(iterable, **kwargs):
                progress_calls.append(kwargs)
                return iterable

            with patch.object(
                beat_features,
                "extract_motion_beats_from_motion_pkl",
                return_value=motion_payload,
            ) as motion_extract, patch.object(
                beat_features,
                "extract_audio_beats_librosa",
                return_value=audio_payload,
            ) as audio_extract, patch.object(
                beat_features,
                "tqdm",
                side_effect=fake_tqdm,
                create=True,
            ):
                beat_features.extract_folder(
                    str(motion_dir), str(wav_dir), str(out_dir), fps=30, seq_len=150
                )

            self.assertEqual(len(progress_calls), 1)
            self.assertEqual(progress_calls[0]["desc"], "Beat metadata")
            self.assertEqual(progress_calls[0]["total"], 2)
            motion_extract.assert_called_once()
            audio_extract.assert_called_once()
            self.assertTrue((out_dir / "clipB.npz").is_file())


class BaselineFeatureCompatibilityTests(unittest.TestCase):
    def test_baseline_extract_handles_scipy_hann_compatibility(self):
        baseline_features = importlib.import_module("data.audio_extraction.baseline_features")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wav_path = tmp_path / "gBR_sBM_cAll_d04_mBR1_ch03_slice0.wav"
            audio = np.sin(np.linspace(0, 8 * np.pi, 5 * 30 * 512, dtype=np.float32))
            sf.write(wav_path, audio, 30 * 512)

            feature, save_path = baseline_features.extract(
                str(wav_path), skip_completed=False, dest_dir=str(tmp_path / "baseline_feats")
            )

        self.assertEqual(feature.shape, (150, 35))
        self.assertTrue(save_path.endswith(".npy"))

    def test_baseline_extract_can_drop_binary_beat_channel(self):
        baseline_features = importlib.import_module("data.audio_extraction.baseline_features")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wav_path = tmp_path / "gBR_sBM_cAll_d04_mBR1_ch03_slice0.wav"
            audio = np.sin(np.linspace(0, 8 * np.pi, 5 * 30 * 512, dtype=np.float32))
            sf.write(wav_path, audio, 30 * 512)

            feature, _ = baseline_features.extract(
                str(wav_path),
                skip_completed=False,
                dest_dir=str(tmp_path / "baseline34_feats"),
                feature_dim=34,
            )

        self.assertEqual(feature.shape, (150, 34))

    def test_baseline_extract_folder_reports_progress(self):
        baseline_features = importlib.import_module("data.audio_extraction.baseline_features")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_dir = tmp_path / "wavs_sliced"
            dest_dir = tmp_path / "baseline_feats"
            src_dir.mkdir()
            dest_dir.mkdir()

            for idx in range(2):
                (src_dir / f"clip{idx}.wav").write_bytes(b"slice")

            progress_calls = []

            def fake_tqdm(iterable, **kwargs):
                progress_calls.append(kwargs)
                return iterable

            with patch.object(
                baseline_features, "extract", return_value=(np.zeros((150, 35), dtype=np.float32), str(dest_dir / "clip.npy"))
            ), patch.object(
                baseline_features,
                "tqdm",
                side_effect=fake_tqdm,
                create=True,
            ):
                baseline_features.extract_folder(str(src_dir), str(dest_dir))

        self.assertEqual(len(progress_calls), 1)
        self.assertEqual(progress_calls[0]["desc"], "Baseline features")
        self.assertEqual(progress_calls[0]["total"], 2)
        self.assertEqual(progress_calls[0]["unit"], "clip")


class BeatAudioCompatibilityTests(unittest.TestCase):
    def test_extract_audio_beats_handles_scipy_hann_compatibility(self):
        beat_features = load_beat_features()

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wav_path = tmp_path / "clip.wav"
            sr = 30 * 512
            seconds = 5
            t = np.linspace(0, seconds, sr * seconds, endpoint=False, dtype=np.float32)
            audio = 0.1 * np.sin(2 * np.pi * 2.0 * t)
            audio += 0.9 * (np.sin(2 * np.pi * 2.0 * t) > 0.995).astype(np.float32)
            sf.write(wav_path, audio, sr)

            beat_frames, beat_mask, beat_dist, beat_spacing = (
                beat_features.extract_audio_beats_librosa(str(wav_path), fps=30, seq_len=150)
            )

        self.assertEqual(beat_frames.ndim, 1)
        self.assertEqual(beat_mask.shape, (150,))
        self.assertEqual(beat_dist.shape, (150,))
        self.assertEqual(beat_spacing.shape, (150,))


class JukeboxFeatureCacheTests(unittest.TestCase):
    def test_default_cache_dir_prefers_shared_repo_root_from_worktree(self):
        jukebox_features = importlib.import_module("data.audio_extraction.jukebox_features")

        cache_dir = jukebox_features._default_cache_dir(
            Path("/tmp/project/.worktrees/diffusion")
        )

        self.assertEqual(cache_dir, Path("/tmp/project/.cache/jukemirlib"))

    def test_jukebox_extract_uses_writable_cache_dir(self):
        jukebox_features = importlib.import_module("data.audio_extraction.jukebox_features")

        with TemporaryDirectory() as tmpdir:
            cache_dir = jukebox_features._configure_jukemirlib_cache(tmpdir)

        self.assertEqual(Path(cache_dir), Path(tmpdir))
        self.assertEqual(
            Path(jukebox_features.jukemirlib.constants.CACHE_DIR),
            Path(tmpdir),
        )
        self.assertTrue(os.environ["JUKEMIRLIB_CACHE_DIR"].endswith(Path(tmpdir).name))

    def test_ensure_jukebox_models_cleans_tmp_files_and_skips_existing_checkpoints(self):
        jukebox_features = importlib.import_module("data.audio_extraction.jukebox_features")

        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            (cache_dir / "vqvae.pth.tar").write_bytes(b"vqvae")
            (cache_dir / "prior_level_2.pth.tar").write_bytes(b"prior")
            tmp_file = cache_dir / "prior_level_2.pth.tarabcd.tmp"
            tmp_file.write_bytes(b"partial")

            with patch.object(jukebox_features, "_download_with_resume") as download:
                resolved = jukebox_features.ensure_jukebox_models(cache_dir=cache_dir)

        self.assertEqual(resolved, cache_dir)
        self.assertFalse(tmp_file.exists())
        download.assert_not_called()

    def test_extract_folder_skips_existing_outputs_by_default(self):
        jukebox_features = importlib.import_module("data.audio_extraction.jukebox_features")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sliced_dir = tmp_path / "train" / "wavs_sliced"
            dest_dir = tmp_path / "train" / "jukebox_feats"
            sliced_dir.mkdir(parents=True)
            dest_dir.mkdir(parents=True)

            for name in ["songA_slice0.wav", "songA_slice1.wav", "songB_slice0.wav"]:
                (sliced_dir / name).write_bytes(b"slice")

            stale_path = dest_dir / "songA_slice1.npy"
            np.save(stale_path, np.full((150, 2), -1.0, dtype=np.float32))

            batch_calls = []

            def fake_extract_batch(wav_paths, dest_dir=None, skip_completed=False):
                batch_calls.append([Path(path).name for path in wav_paths])
                outputs = []
                for wav_path in wav_paths:
                    stem = Path(wav_path).stem
                    save_path = Path(dest_dir) / f"{stem}.npy"
                    if skip_completed and save_path.exists():
                        continue
                    feature_value = float(len(stem))
                    feature = np.full((150, 2), feature_value, dtype=np.float32)
                    outputs.append((feature, str(save_path)))
                return outputs

            class FakeProgress:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.updated = []

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self, amount):
                    self.updated.append(amount)

            progress_instances = []

            def fake_tqdm(*args, **kwargs):
                progress = FakeProgress(**kwargs)
                progress_instances.append(progress)
                return progress

            with patch.object(
                jukebox_features, "extract_batch", side_effect=fake_extract_batch
            ), patch.object(
                jukebox_features, "ensure_jukebox_models"
            ), patch.object(
                jukebox_features,
                "tqdm",
                side_effect=fake_tqdm,
                create=True,
            ):
                jukebox_features.extract_folder(str(sliced_dir), str(dest_dir), batch_size=2)

            song_a_slice0 = np.load(dest_dir / "songA_slice0.npy")
            song_a_slice1 = np.load(dest_dir / "songA_slice1.npy")
            song_b_slice0 = np.load(dest_dir / "songB_slice0.npy")

            self.assertEqual(song_a_slice0.shape, (150, 2))
            self.assertEqual(song_a_slice1.shape, (150, 2))
            self.assertEqual(song_b_slice0.shape, (150, 2))
            self.assertEqual(batch_calls, [["songA_slice0.wav", "songA_slice1.wav"], ["songB_slice0.wav"]])
            self.assertEqual(progress_instances[0].updated, [2, 1])
            np.testing.assert_array_equal(song_a_slice0, np.full((150, 2), 12.0, dtype=np.float32))
            np.testing.assert_array_equal(song_a_slice1, np.full((150, 2), -1.0, dtype=np.float32))
            np.testing.assert_array_equal(song_b_slice0, np.full((150, 2), 12.0, dtype=np.float32))

    def test_extract_folder_reports_clip_level_progress_for_jukebox_slices(self):
        jukebox_features = importlib.import_module("data.audio_extraction.jukebox_features")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sliced_dir = tmp_path / "train" / "wavs_sliced"
            dest_dir = tmp_path / "train" / "jukebox_feats"
            sliced_dir.mkdir(parents=True)

            for name in ["songA_slice0.wav", "songA_slice1.wav", "songB_slice0.wav"]:
                (sliced_dir / name).write_bytes(b"slice")

            progress_calls = []

            class FakeProgress:
                def __init__(self, **kwargs):
                    progress_calls.append(kwargs)

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self, amount):
                    return None

            def fake_tqdm(*args, **kwargs):
                progress_calls.append(kwargs)
                return FakeProgress(**kwargs)

            with patch.object(
                jukebox_features,
                "extract_batch",
                return_value=[
                    (np.zeros((150, 2), dtype=np.float32), str(dest_dir / "clip0.npy")),
                    (np.zeros((150, 2), dtype=np.float32), str(dest_dir / "clip1.npy")),
                ],
            ), patch.object(
                jukebox_features, "ensure_jukebox_models"
            ), patch.object(
                jukebox_features,
                "tqdm",
                side_effect=fake_tqdm,
                create=True,
            ):
                jukebox_features.extract_folder(str(sliced_dir), str(dest_dir), batch_size=2)

        self.assertEqual(len(progress_calls), 2)
        self.assertEqual(progress_calls[0]["desc"], "Jukebox slices")
        self.assertEqual(progress_calls[0]["total"], 3)
        self.assertEqual(progress_calls[0]["unit"], "clip")

    def test_extract_batch_uses_batched_jukemirlib_extract(self):
        jukebox_features = importlib.import_module("data.audio_extraction.jukebox_features")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dest_dir = tmp_path / "jukebox_feats"
            wav_paths = [tmp_path / "songA.wav", tmp_path / "songB.wav"]
            for wav_path in wav_paths:
                wav_path.write_bytes(b"slice")

            loaded_audio = {
                str(wav_paths[0]): np.array([1.0, 2.0], dtype=np.float32),
                str(wav_paths[1]): np.array([3.0, 4.0], dtype=np.float32),
            }

            with patch.object(
                jukebox_features, "ensure_jukebox_models"
            ), patch.object(
                jukebox_features, "_prefer_runtime_device"
            ) as prefer_device, patch.object(
                jukebox_features.jukemirlib, "load_audio", side_effect=lambda path: loaded_audio[str(path)]
            ) as load_audio, patch.object(
                jukebox_features.jukemirlib,
                "extract",
                return_value={
                    jukebox_features.LAYER: np.stack(
                        [
                            np.full((150, 2), 11.0, dtype=np.float32),
                            np.full((150, 2), 22.0, dtype=np.float32),
                        ],
                        axis=0,
                    )
                },
            ) as extract:
                outputs = jukebox_features.extract_batch(wav_paths, dest_dir=dest_dir)

        prefer_device.assert_called_once()
        self.assertEqual(load_audio.call_count, 2)
        extract.assert_called_once()
        self.assertEqual(len(outputs), 2)
        np.testing.assert_array_equal(outputs[0][0], np.full((150, 2), 11.0, dtype=np.float32))
        np.testing.assert_array_equal(outputs[1][0], np.full((150, 2), 22.0, dtype=np.float32))
        self.assertTrue(str(outputs[0][1]).endswith("songA.npy"))
        self.assertTrue(str(outputs[1][1]).endswith("songB.npy"))

    def test_extract_batch_resets_cached_jukemirlib_conditions_when_batch_size_changes(self):
        jukebox_features = importlib.import_module("data.audio_extraction.jukebox_features")

        class FakeCondition:
            def __init__(self, batch_size):
                self.shape = (batch_size, 1, 8)

        def fake_extract(audio=None, **kwargs):
            if fake_extract.__globals__["x_cond"] is not None:
                raise AssertionError("stale x_cond was not reset before smaller batch")
            if fake_extract.__globals__["y_cond"] is not None:
                raise AssertionError("stale y_cond was not reset before smaller batch")
            fake_extract.__globals__["x_cond"] = FakeCondition(len(audio))
            fake_extract.__globals__["y_cond"] = FakeCondition(len(audio))
            return {
                jukebox_features.LAYER: np.stack(
                    [np.full((150, 2), idx, dtype=np.float32) for idx in range(len(audio))],
                    axis=0,
                )
            }

        fake_extract.__globals__["x_cond"] = FakeCondition(16)
        fake_extract.__globals__["y_cond"] = FakeCondition(16)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wav_paths = [tmp_path / f"song{idx}.wav" for idx in range(5)]
            for wav_path in wav_paths:
                wav_path.write_bytes(b"slice")

            with patch.object(
                jukebox_features, "ensure_jukebox_models"
            ), patch.object(
                jukebox_features, "_prefer_runtime_device"
            ), patch.object(
                jukebox_features.jukemirlib,
                "load_audio",
                side_effect=lambda path: np.array([1.0, 2.0], dtype=np.float32),
            ), patch.object(
                jukebox_features.jukemirlib,
                "extract",
                new=fake_extract,
            ):
                jukebox_features.extract_batch(wav_paths, dest_dir=tmp_path / "features")

        self.assertEqual(fake_extract.__globals__["x_cond"].shape[0], 5)
        self.assertEqual(fake_extract.__globals__["y_cond"].shape[0], 5)


if __name__ == "__main__":
    unittest.main()
