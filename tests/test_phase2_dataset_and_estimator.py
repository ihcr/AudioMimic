import importlib
import os
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class DatasetBeatSchemaTests(unittest.TestCase):
    def setUp(self):
        self.dataset_module = reload_module("dataset.dance_dataset")

    def test_prune_legacy_processed_dataset_caches_removes_older_matching_versions(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            current = tmp_path / "processed_train_jukebox_beat_distance_v4.pkl"
            legacy_a = tmp_path / "processed_train_jukebox_beat_distance.pkl"
            legacy_b = tmp_path / "processed_train_jukebox_beat_distance_v3.pkl"
            keep_other = tmp_path / "processed_train_jukebox_nobeat_distance_v3.pkl"
            for path in (current, legacy_a, legacy_b, keep_other):
                path.write_bytes(b"cache")

            self.dataset_module.prune_legacy_processed_dataset_caches(
                tmp_path, "train", "jukebox", True, "distance"
            )

            self.assertTrue(current.exists())
            self.assertFalse(legacy_a.exists())
            self.assertFalse(legacy_b.exists())
            self.assertTrue(keep_other.exists())

    def test_dataset_imports_without_pytorch3d(self):
        self.assertTrue(hasattr(self.dataset_module, "AISTPPDataset"))

    def test_dataset_uses_beat_aware_cache_name(self):
        dataset_cls = self.dataset_module.AISTPPDataset
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_path = tmp_path / "clip.npy"
            beat_path = tmp_path / "clip.npz"
            wav_path = tmp_path / "clip.wav"
            np.save(feature_path, np.zeros((150, 4800), dtype=np.float32))
            np.savez(
                beat_path,
                motion_beats=np.array([1], dtype=np.int64),
                motion_mask=np.zeros(150, dtype=np.float32),
                motion_dist=np.zeros(150, dtype=np.int64),
                motion_spacing=np.ones(150, dtype=np.float32),
                audio_beats=np.array([1], dtype=np.int64),
                audio_mask=np.zeros(150, dtype=np.float32),
                audio_dist=np.zeros(150, dtype=np.int64),
                audio_spacing=np.ones(150, dtype=np.float32),
            )
            wav_path.write_bytes(b"wav")

            fake_data = {
                "pos": np.zeros((1, 150, 3), dtype=np.float32),
                "q": np.zeros((1, 150, 72), dtype=np.float32),
                "filenames": [str(feature_path)],
                "wavs": [str(wav_path)],
                "beatnames": [str(beat_path)],
            }
            fake_pose = torch.zeros((1, 150, 151), dtype=torch.float32)

            with patch.object(dataset_cls, "load_aistpp", return_value=fake_data), patch.object(
                dataset_cls, "process_dataset", return_value=fake_pose
            ):
                dataset_cls(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=True,
                    feature_type="jukebox",
                    use_beats=True,
                    beat_rep="distance",
                    force_reload=True,
                )

            self.assertTrue((tmp_path / "processed_train_jukebox_beat_distance_v4.pkl").is_file())

    def test_getitem_uses_preloaded_beat_tensors_and_loads_feature_on_demand(self):
        dataset_cls = self.dataset_module.AISTPPDataset
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_path = tmp_path / "clip.npy"
            np.save(feature_path, np.ones((150, 35), dtype=np.float32))
            fake_data = {
                "pos": np.zeros((1, 150, 3), dtype=np.float32),
                "q": np.zeros((1, 150, 72), dtype=np.float32),
                "filenames": [str(feature_path)],
                "wavs": ["clip.wav"],
                "motion_dist": np.arange(150, dtype=np.int64)[None, :],
                "motion_spacing": np.full((1, 150), 30.0, dtype=np.float32),
                "motion_mask": np.zeros((1, 150), dtype=np.float32),
                "audio_dist": np.arange(149, -1, -1, dtype=np.int64)[None, :],
                "audio_mask": np.zeros((1, 150), dtype=np.float32),
            }
            fake_pose = torch.zeros((1, 150, 151), dtype=torch.float32)

            with patch.object(dataset_cls, "load_aistpp", return_value=fake_data), patch.object(
                dataset_cls, "process_dataset", return_value=fake_pose
            ):
                dataset = dataset_cls(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=True,
                    feature_type="baseline",
                    use_beats=True,
                    beat_rep="distance",
                    force_reload=True,
                )

            original_np_load = self.dataset_module.np.load
            load_calls = []

            def tracked_np_load(path, *args, **kwargs):
                load_calls.append((str(path), kwargs.get("mmap_mode")))
                return original_np_load(path, *args, **kwargs)

            with patch.object(self.dataset_module.np, "load", side_effect=tracked_np_load):
                _, cond, _, _ = dataset[0]

            self.assertEqual(cond["music"].shape, (150, 35))
            self.assertTrue(torch.equal(cond["beat"], torch.arange(150, dtype=torch.int64)))
            self.assertEqual(load_calls, [(str(feature_path), "r")])

    def test_getitem_can_load_music_from_memmap_feature_cache(self):
        dataset_cls = self.dataset_module.AISTPPDataset
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_a = np.ones((4, 3), dtype=np.float32)
            feature_b = np.full((4, 3), 2.0, dtype=np.float32)
            feature_path_a = tmp_path / "clip_a.npy"
            feature_path_b = tmp_path / "clip_b.npy"
            np.save(feature_path_a, feature_a)
            np.save(feature_path_b, feature_b)
            fake_data = {
                "pos": np.zeros((2, 150, 3), dtype=np.float32),
                "q": np.zeros((2, 150, 72), dtype=np.float32),
                "filenames": [str(feature_path_a), str(feature_path_b)],
                "wavs": ["clip_a.wav", "clip_b.wav"],
            }
            fake_pose = torch.zeros((2, 150, 151), dtype=torch.float32)

            with patch.object(dataset_cls, "load_aistpp", return_value=fake_data), patch.object(
                dataset_cls, "process_dataset", return_value=fake_pose
            ):
                dataset = dataset_cls(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=True,
                    feature_type="baseline",
                    force_reload=True,
                    feature_cache_mode="memmap",
                )

            self.assertTrue(Path(dataset.feature_store_path).is_file())
            original_np_load = self.dataset_module.np.load
            load_calls = []

            def tracked_np_load(path, *args, **kwargs):
                load_calls.append((str(path), kwargs.get("mmap_mode")))
                return original_np_load(path, *args, **kwargs)

            with patch.object(self.dataset_module.np, "load", side_effect=tracked_np_load):
                _, cond, _, _ = dataset[1]

            self.assertTrue(torch.equal(cond, torch.from_numpy(feature_b)))
            self.assertEqual(load_calls, [(str(dataset.feature_store_path), "r")])

    def test_memmap_feature_cache_reopens_after_dataset_pickle_roundtrip(self):
        dataset_cls = self.dataset_module.AISTPPDataset
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_path = tmp_path / "clip.npy"
            feature = np.full((4, 3), 3.0, dtype=np.float32)
            np.save(feature_path, feature)
            fake_data = {
                "pos": np.zeros((1, 150, 3), dtype=np.float32),
                "q": np.zeros((1, 150, 72), dtype=np.float32),
                "filenames": [str(feature_path)],
                "wavs": ["clip.wav"],
            }
            fake_pose = torch.zeros((1, 150, 151), dtype=torch.float32)

            with patch.object(dataset_cls, "load_aistpp", return_value=fake_data), patch.object(
                dataset_cls, "process_dataset", return_value=fake_pose
            ):
                dataset = dataset_cls(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=True,
                    feature_type="baseline",
                    force_reload=True,
                    feature_cache_mode="memmap",
                )

            _, before, _, _ = dataset[0]
            reloaded = pickle.loads(pickle.dumps(dataset))
            _, after, _, _ = reloaded[0]

            self.assertTrue(torch.equal(before, torch.from_numpy(feature)))
            self.assertTrue(torch.equal(after, torch.from_numpy(feature)))

    def test_getitem_returns_tensor_condition_when_beats_disabled(self):
        dataset_cls = self.dataset_module.AISTPPDataset
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature = np.ones((150, 35), dtype=np.float32)
            feature_path = tmp_path / "clip.npy"
            np.save(feature_path, feature)
            wav_path = tmp_path / "clip.wav"
            wav_path.write_bytes(b"wav")

            fake_data = {
                "pos": np.zeros((1, 150, 3), dtype=np.float32),
                "q": np.zeros((1, 150, 72), dtype=np.float32),
                "filenames": [str(feature_path)],
                "wavs": [str(wav_path)],
            }
            fake_pose = torch.zeros((1, 150, 151), dtype=torch.float32)

            with patch.object(dataset_cls, "load_aistpp", return_value=fake_data), patch.object(
                dataset_cls, "process_dataset", return_value=fake_pose
            ):
                dataset = dataset_cls(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=True,
                    feature_type="baseline",
                    force_reload=True,
                )

            pose, cond, _, _ = dataset[0]
            self.assertTrue(torch.is_tensor(cond))
            self.assertEqual(cond.shape, (150, 35))
            self.assertTrue(torch.equal(pose, fake_pose[0]))

    def test_getitem_returns_train_distance_condition_dict(self):
        dataset_cls = self.dataset_module.AISTPPDataset
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_path = tmp_path / "clip.npy"
            beat_path = tmp_path / "clip.npz"
            wav_path = tmp_path / "clip.wav"

            np.save(feature_path, np.ones((150, 4800), dtype=np.float32))
            np.savez(
                beat_path,
                motion_beats=np.array([10, 40], dtype=np.int64),
                motion_mask=np.eye(1, 150, 10, dtype=np.float32).reshape(150),
                motion_dist=np.arange(150, dtype=np.int64),
                motion_spacing=np.full(150, 30.0, dtype=np.float32),
                audio_beats=np.array([12, 42], dtype=np.int64),
                audio_mask=np.eye(1, 150, 12, dtype=np.float32).reshape(150),
                audio_dist=np.arange(149, -1, -1, dtype=np.int64),
                audio_spacing=np.full(150, 20.0, dtype=np.float32),
            )
            wav_path.write_bytes(b"wav")

            fake_data = {
                "pos": np.zeros((1, 150, 3), dtype=np.float32),
                "q": np.zeros((1, 150, 72), dtype=np.float32),
                "filenames": [str(feature_path)],
                "wavs": [str(wav_path)],
                "beatnames": [str(beat_path)],
            }
            fake_pose = torch.zeros((1, 150, 151), dtype=torch.float32)

            with patch.object(dataset_cls, "load_aistpp", return_value=fake_data), patch.object(
                dataset_cls, "process_dataset", return_value=fake_pose
            ):
                dataset = dataset_cls(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=True,
                    feature_type="jukebox",
                    use_beats=True,
                    beat_rep="distance",
                    force_reload=True,
                )

            _, cond, _, _ = dataset[0]
            self.assertEqual(set(cond.keys()), {"music", "beat", "beat_target", "beat_spacing", "audio_mask"})
            self.assertEqual(cond["music"].shape, (150, 4800))
            self.assertEqual(cond["beat"].dtype, torch.int64)
            self.assertEqual(cond["beat_target"].dtype, torch.float32)
            self.assertTrue(torch.equal(cond["beat"], torch.arange(150, dtype=torch.int64)))

    def test_getitem_returns_test_pulse_condition_dict(self):
        dataset_cls = self.dataset_module.AISTPPDataset
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_path = tmp_path / "clip.npy"
            beat_path = tmp_path / "clip.npz"
            wav_path = tmp_path / "clip.wav"

            np.save(feature_path, np.ones((150, 35), dtype=np.float32))
            audio_mask = np.zeros(150, dtype=np.float32)
            audio_mask[[5, 25]] = 1.0
            np.savez(
                beat_path,
                motion_beats=np.array([10], dtype=np.int64),
                motion_mask=np.zeros(150, dtype=np.float32),
                motion_dist=np.arange(150, dtype=np.int64),
                motion_spacing=np.full(150, 30.0, dtype=np.float32),
                audio_beats=np.array([5, 25], dtype=np.int64),
                audio_mask=audio_mask,
                audio_dist=np.arange(149, -1, -1, dtype=np.int64),
                audio_spacing=np.full(150, 20.0, dtype=np.float32),
            )
            wav_path.write_bytes(b"wav")

            fake_data = {
                "pos": np.zeros((1, 150, 3), dtype=np.float32),
                "q": np.zeros((1, 150, 72), dtype=np.float32),
                "filenames": [str(feature_path)],
                "wavs": [str(wav_path)],
                "beatnames": [str(beat_path)],
            }
            fake_pose = torch.zeros((1, 150, 151), dtype=torch.float32)

            with patch.object(dataset_cls, "load_aistpp", return_value=fake_data), patch.object(
                dataset_cls, "process_dataset", return_value=fake_pose
            ):
                dataset = dataset_cls(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=False,
                    feature_type="baseline",
                    use_beats=True,
                    beat_rep="pulse",
                    normalizer=object(),
                    force_reload=True,
                )

            _, cond, _, _ = dataset[0]
            self.assertEqual(cond["music"].shape, (150, 35))
            self.assertEqual(cond["beat"].shape, (150, 1))
            self.assertEqual(cond["beat"].dtype, torch.float32)
            self.assertTrue(torch.equal(cond["beat"][:, 0], torch.from_numpy(audio_mask)))


class BeatEstimatorTests(unittest.TestCase):
    def test_beat_estimator_forward_shape_and_nonnegative(self):
        beat_estimator = reload_module("model.beat_estimator")
        model = beat_estimator.BeatDistanceEstimator()
        joints = torch.randn(2, 150, 24, 3)

        output = model(joints)

        self.assertEqual(output.shape, (2, 150))
        self.assertTrue(torch.all(output >= 0))

    def test_motion_beat_dataset_pairs_motion_and_beats(self):
        train_module = reload_module("train_beat_estimator")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_dir = tmp_path / "motions_sliced"
            beat_dir = tmp_path / "beat_feats"
            motion_dir.mkdir()
            beat_dir.mkdir()

            with open(motion_dir / "clipA.pkl", "wb") as handle:
                pickle.dump({"pos": np.zeros((300, 3)), "q": np.zeros((300, 72))}, handle)
            np.savez(
                beat_dir / "clipA.npz",
                motion_dist=np.arange(150, dtype=np.int64),
                motion_spacing=np.full(150, 30.0, dtype=np.float32),
            )

            dataset = train_module.MotionBeatDataset(str(motion_dir), str(beat_dir))
            motion_path, beat_path = dataset.samples[0]

            self.assertEqual(len(dataset), 1)
            self.assertTrue(motion_path.endswith("clipA.pkl"))
            self.assertTrue(beat_path.endswith("clipA.npz"))

    def test_motion_beat_dataset_writes_versioned_cache(self):
        train_module = reload_module("train_beat_estimator")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_dir = tmp_path / "train" / "motions_sliced"
            beat_dir = tmp_path / "train" / "beat_feats"
            cache_dir = tmp_path / "dataset_backups"
            motion_dir.mkdir(parents=True)
            beat_dir.mkdir(parents=True)

            (motion_dir / "clipA.pkl").write_bytes(b"motion")
            np.savez(
                beat_dir / "clipA.npz",
                motion_dist=np.arange(150, dtype=np.int64),
                motion_spacing=np.full(150, 30.0, dtype=np.float32),
            )

            with patch.object(
                train_module,
                "load_motion_joints",
                return_value=torch.zeros(150, 24, 3, dtype=torch.float32),
            ):
                dataset = train_module.MotionBeatDataset(
                    str(motion_dir),
                    str(beat_dir),
                    cache_dir=cache_dir,
                )

            self.assertEqual(len(dataset), 1)
            self.assertTrue(
                (cache_dir / "beat_estimator_train_fps30_seq150_v2.pt").is_file()
            )

    def test_motion_beat_dataset_reuses_cache_without_reloading_motion_files(self):
        train_module = reload_module("train_beat_estimator")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            motion_dir = tmp_path / "train" / "motions_sliced"
            beat_dir = tmp_path / "train" / "beat_feats"
            cache_dir = tmp_path / "dataset_backups"
            motion_dir.mkdir(parents=True)
            beat_dir.mkdir(parents=True)

            (motion_dir / "clipA.pkl").write_bytes(b"motion")
            np.savez(
                beat_dir / "clipA.npz",
                motion_dist=np.arange(150, dtype=np.int64),
                motion_spacing=np.full(150, 30.0, dtype=np.float32),
            )

            with patch.object(
                train_module,
                "load_motion_joints",
                return_value=torch.zeros(150, 24, 3, dtype=torch.float32),
            ):
                train_module.MotionBeatDataset(
                    str(motion_dir),
                    str(beat_dir),
                    cache_dir=cache_dir,
                )

            with patch.object(
                train_module,
                "load_motion_joints",
                side_effect=AssertionError("cache should be reused"),
            ):
                dataset = train_module.MotionBeatDataset(
                    str(motion_dir),
                    str(beat_dir),
                    cache_dir=cache_dir,
                )
                joints, target = dataset[0]

            self.assertEqual(joints.shape, (150, 24, 3))
            self.assertEqual(target.shape, (150,))

    def test_save_checkpoint_roundtrip(self):
        beat_estimator = reload_module("model.beat_estimator")
        train_module = reload_module("train_beat_estimator")
        model = beat_estimator.BeatDistanceEstimator()

        with TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "beat_estimator.pt"
            train_module.save_checkpoint(
                model,
                ckpt_path,
                {"hidden_dim": 128, "num_layers": 6},
            )
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.assertIn("model_state_dict", payload)
        self.assertEqual(payload["config"]["hidden_dim"], 128)

    def test_parse_args_defaults_to_50_epochs_with_validation(self):
        train_module = reload_module("train_beat_estimator")

        with patch.object(
            sys,
            "argv",
            [
                "train_beat_estimator.py",
                "--motion_dir",
                "motions",
                "--beat_dir",
                "beats",
            ],
        ):
            args = train_module.parse_args()

        self.assertEqual(args.epochs, 50)
        self.assertAlmostEqual(args.val_split, 0.1)

    def test_parse_args_accepts_loader_and_precision_flags(self):
        train_module = reload_module("train_beat_estimator")

        with patch.object(
            sys,
            "argv",
            [
                "train_beat_estimator.py",
                "--motion_dir",
                "motions",
                "--beat_dir",
                "beats",
                "--num_workers",
                "5",
                "--mixed_precision",
                "no",
                "--force_rebuild_cache",
            ],
        ):
            args = train_module.parse_args()

        self.assertEqual(args.num_workers, 5)
        self.assertEqual(args.mixed_precision, "no")
        self.assertTrue(args.force_rebuild_cache)

    def test_build_dataloader_kwargs_enables_prefetching_for_estimator(self):
        train_module = reload_module("train_beat_estimator")

        kwargs = train_module.build_dataloader_kwargs(num_workers=6, pin_memory=True)

        self.assertEqual(kwargs["num_workers"], 6)
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 1)

    def test_train_reports_batch_progress(self):
        train_module = reload_module("train_beat_estimator")

        class TinyDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                joints = torch.zeros(150, 24, 3, dtype=torch.float32)
                target = torch.zeros(150, dtype=torch.float32)
                return joints, target

        progress_calls = []

        class FakeProgress:
            def __init__(self, iterable, **kwargs):
                self._iterable = iterable
                self.kwargs = kwargs
                progress_calls.append(kwargs)

            def __iter__(self):
                return iter(self._iterable)

            def set_postfix(self, **kwargs):
                self.postfix = kwargs

        def fake_tqdm(iterable, **kwargs):
            return FakeProgress(iterable, **kwargs)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))

            def forward(self, joints):
                batch, seq_len = joints.shape[:2]
                return self.bias.expand(batch, seq_len)

        with patch.object(
            train_module, "MotionBeatDataset", return_value=TinyDataset()
        ), patch.object(
            train_module, "save_checkpoint"
        ), patch.object(
            train_module, "BeatDistanceEstimator", return_value=TinyModel()
        ), patch.object(
            train_module, "tqdm", side_effect=fake_tqdm, create=True
        ):
            train_module.train(
                motion_dir="unused",
                beat_dir="unused",
                output_path="unused.pt",
                epochs=1,
                batch_size=1,
                val_split=0.0,
                device="cpu",
            )

        self.assertEqual(len(progress_calls), 1)
        self.assertEqual(progress_calls[0]["desc"], "Beat estimator train 1/1")
        self.assertEqual(progress_calls[0]["unit"], "batch")

    def test_train_uses_validation_and_saves_best_checkpoint(self):
        train_module = reload_module("train_beat_estimator")

        class TinyDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                joints = torch.zeros(150, 24, 3, dtype=torch.float32)
                target = torch.zeros(150, dtype=torch.float32)
                return joints, target

        progress_calls = []

        class FakeProgress:
            def __init__(self, iterable, **kwargs):
                self._iterable = iterable
                self.kwargs = kwargs
                progress_calls.append(kwargs)

            def __iter__(self):
                return iter(self._iterable)

            def set_postfix(self, **kwargs):
                self.postfix = kwargs

        def fake_tqdm(iterable, **kwargs):
            return FakeProgress(iterable, **kwargs)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.ones(1))

            def forward(self, joints):
                batch, seq_len = joints.shape[:2]
                return self.bias.expand(batch, seq_len)

        with patch.object(
            train_module, "MotionBeatDataset", return_value=TinyDataset()
        ), patch.object(
            train_module, "save_checkpoint"
        ) as save_checkpoint, patch.object(
            train_module, "BeatDistanceEstimator", return_value=TinyModel()
        ), patch.object(
            train_module, "tqdm", side_effect=fake_tqdm, create=True
        ):
            train_module.train(
                motion_dir="unused",
                beat_dir="unused",
                output_path="unused.pt",
                epochs=2,
                batch_size=1,
                learning_rate=0.0,
                val_split=0.5,
                device="cpu",
            )

        self.assertEqual(save_checkpoint.call_count, 1)
        _, _, config = save_checkpoint.call_args.args
        self.assertEqual(config["best_epoch"], 1)
        self.assertAlmostEqual(config["best_val_loss"], 1.0)
        self.assertAlmostEqual(config["val_split"], 0.5)
        self.assertEqual(
            [call["desc"] for call in progress_calls],
            [
                "Beat estimator train 1/2",
                "Beat estimator val 1/2",
                "Beat estimator train 2/2",
                "Beat estimator val 2/2",
            ],
        )


if __name__ == "__main__":
    unittest.main()
