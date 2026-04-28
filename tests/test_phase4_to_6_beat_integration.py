import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class DecoderBeatConditioningTests(unittest.TestCase):
    def test_beat_dance_decoder_distance_forward(self):
        model_module = reload_module("model.model")
        model = model_module.BeatDanceDecoder(
            nfeats=151,
            seq_len=150,
            latent_dim=64,
            ff_size=128,
            num_layers=2,
            num_heads=4,
            cond_feature_dim=35,
            beat_rep="distance",
            use_rotary=False,
        )
        x = torch.randn(2, 150, 151)
        cond = {
            "music": torch.randn(2, 150, 35),
            "beat": torch.randint(0, 151, (2, 150)),
        }
        times = torch.randint(0, 1000, (2,))

        output = model(x, cond, times)

        self.assertEqual(output.shape, (2, 150, 151))

    def test_beat_dance_decoder_pulse_forward(self):
        model_module = reload_module("model.model")
        model = model_module.BeatDanceDecoder(
            nfeats=151,
            seq_len=150,
            latent_dim=64,
            ff_size=128,
            num_layers=2,
            num_heads=4,
            cond_feature_dim=4800,
            beat_rep="pulse",
            use_rotary=False,
        )
        x = torch.randn(1, 150, 151)
        cond = {
            "music": torch.randn(1, 150, 4800),
            "beat": torch.randn(1, 150, 1),
        }
        times = torch.randint(0, 1000, (1,))

        output = model(x, cond, times)

        self.assertEqual(output.shape, (1, 150, 151))

    def test_mixed_condition_dropout_keeps_null_condition_gradients_finite(self):
        model_module = reload_module("model.model")
        model = model_module.BeatDanceDecoder(
            nfeats=151,
            seq_len=32,
            latent_dim=64,
            ff_size=128,
            num_layers=2,
            num_heads=4,
            cond_feature_dim=35,
            beat_rep="distance",
            use_rotary=False,
        )
        x = torch.randn(2, 32, 151)
        cond = {
            "music": torch.randn(2, 32, 35),
            "beat": torch.randint(0, 151, (2, 32)),
        }
        times = torch.randint(0, 1000, (2,))

        with patch.object(
            model_module,
            "prob_mask_like",
            return_value=torch.tensor([True, False]),
        ):
            output = model(x, cond, times, cond_drop_prob=0.25)
            loss = output.square().mean()

        loss.backward()

        grad = model.null_cond_embed.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())


class ConditionHelperTests(unittest.TestCase):
    def test_condition_helpers_support_tensor_and_dict(self):
        diffusion_module = reload_module("model.diffusion")

        tensor_cond = torch.randn(3, 150, 35)
        dict_cond = {
            "music": torch.randn(3, 150, 35),
            "beat": torch.randint(0, 151, (3, 150)),
            "beat_target": torch.randn(3, 150),
        }

        moved_tensor = diffusion_module.move_cond_to_device(tensor_cond, torch.device("cpu"))
        moved_dict = diffusion_module.move_cond_to_device(dict_cond, torch.device("cpu"))
        sliced_tensor = diffusion_module.slice_cond(tensor_cond, slice(1, 3))
        sliced_dict = diffusion_module.slice_cond(dict_cond, slice(1, 3))

        self.assertEqual(diffusion_module.cond_batch_size(moved_tensor), 3)
        self.assertEqual(diffusion_module.cond_batch_size(moved_dict), 3)
        self.assertEqual(diffusion_module.cond_device(moved_tensor), torch.device("cpu"))
        self.assertEqual(diffusion_module.cond_device(moved_dict), torch.device("cpu"))
        self.assertEqual(sliced_tensor.shape[0], 2)
        self.assertEqual(sliced_dict["music"].shape[0], 2)
        self.assertEqual(sliced_dict["beat"].shape[0], 2)


class CheckpointConfigTests(unittest.TestCase):
    def test_checkpoint_config_overrides_default_runtime_architecture(self):
        edge_module = reload_module("EDGE")

        resolved = edge_module.resolve_model_config(
            feature_type="jukebox",
            use_beats=False,
            beat_rep="distance",
            checkpoint_config={
                "feature_type": "baseline",
                "use_beats": True,
                "beat_rep": "pulse",
            },
        )

        self.assertEqual(resolved["feature_type"], "baseline")
        self.assertTrue(resolved["use_beats"])
        self.assertEqual(resolved["beat_rep"], "pulse")

    def test_checkpoint_config_rejects_explicit_runtime_conflicts(self):
        edge_module = reload_module("EDGE")

        with self.assertRaises(ValueError):
            edge_module.resolve_model_config(
                feature_type="baseline",
                use_beats=True,
                beat_rep="distance",
                checkpoint_config={
                    "feature_type": "jukebox",
                    "use_beats": True,
                    "beat_rep": "pulse",
                },
                )


class RuntimeTrainingConfigTests(unittest.TestCase):
    def test_checkpoint_inferred_beat_run_uses_safe_defaults_when_lr_not_explicit(self):
        edge_module = reload_module("EDGE")

        resolved = edge_module.resolve_runtime_training_config(
            feature_type="jukebox",
            use_beats=False,
            beat_rep="distance",
            learning_rate=4e-4,
            learning_rate_was_explicit=False,
            checkpoint_config={
                "feature_type": "jukebox",
                "use_beats": True,
                "beat_rep": "distance",
            },
        )

        self.assertTrue(resolved["use_beats"])
        self.assertEqual(resolved["learning_rate"], 2e-4)
        self.assertEqual(resolved["grad_clip_norm"], 1.0)

    def test_explicit_learning_rate_is_preserved_across_checkpoint_resolution(self):
        edge_module = reload_module("EDGE")

        resolved = edge_module.resolve_runtime_training_config(
            feature_type="jukebox",
            use_beats=False,
            beat_rep="distance",
            learning_rate=2.5e-4,
            learning_rate_was_explicit=True,
            checkpoint_config={
                "feature_type": "jukebox",
                "use_beats": True,
                "beat_rep": "distance",
            },
        )

        self.assertEqual(resolved["learning_rate"], 2.5e-4)
        self.assertEqual(resolved["grad_clip_norm"], 1.0)


class LoaderConfigTests(unittest.TestCase):
    def test_tensor_dataset_cache_name_is_versioned(self):
        edge_module = reload_module("EDGE")

        cache_name = edge_module.tensor_dataset_cache_name(
            "train", "jukebox", True, "distance"
        )

        self.assertEqual(cache_name, "train_tensor_dataset_jukebox_beat_distance_v5.pkl")

    def test_prune_legacy_tensor_dataset_caches_removes_older_matching_versions(self):
        edge_module = reload_module("EDGE")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            current = tmp_path / "train_tensor_dataset_jukebox_beat_distance_v5.pkl"
            legacy_a = tmp_path / "train_tensor_dataset_jukebox_beat_distance.pkl"
            legacy_b = tmp_path / "train_tensor_dataset_jukebox_beat_distance_v3.pkl"
            keep_other = tmp_path / "train_tensor_dataset_jukebox_nobeat_distance_v3.pkl"
            for path in (current, legacy_a, legacy_b, keep_other):
                path.write_bytes(b"cache")

            edge_module.prune_legacy_tensor_dataset_caches(
                tmp_path, "train", "jukebox", True, "distance"
            )

            self.assertTrue(current.exists())
            self.assertFalse(legacy_a.exists())
            self.assertFalse(legacy_b.exists())
            self.assertTrue(keep_other.exists())


class AccumulationTrainingHelperTests(unittest.TestCase):
    def test_accumulation_helper_only_steps_on_sync_and_checkpoint_tracks_recipe(self):
        edge_module = reload_module("EDGE")
        params = [torch.nn.Parameter(torch.tensor(1.0))]
        model = MagicMock()
        model.parameters.return_value = params
        optim = MagicMock()
        diffusion = MagicMock()
        accelerator = MagicMock()
        accelerator.sync_gradients = False

        stepped = edge_module.maybe_apply_optimization_step(
            accelerator=accelerator,
            model=model,
            optim=optim,
            diffusion=diffusion,
            grad_clip_norm=1.0,
            global_step=0,
            ema_interval=1,
        )

        self.assertFalse(stepped)
        accelerator.clip_grad_norm_.assert_not_called()
        optim.step.assert_not_called()
        diffusion.ema.update_model_average.assert_not_called()

        accelerator.sync_gradients = True
        stepped = edge_module.maybe_apply_optimization_step(
            accelerator=accelerator,
            model=model,
            optim=optim,
            diffusion=diffusion,
            grad_clip_norm=1.0,
            global_step=0,
            ema_interval=1,
        )

        self.assertTrue(stepped)
        accelerator.clip_grad_norm_.assert_called_once_with(params, 1.0)
        optim.step.assert_called_once_with()
        diffusion.ema.update_model_average.assert_called_once_with(
            diffusion.master_model,
            diffusion.model,
        )

        config = edge_module.build_checkpoint_config(
            feature_type="jukebox",
            use_beats=True,
            beat_rep="distance",
            batch_size=128,
            gradient_accumulation_steps=4,
            num_processes=1,
            learning_rate=2e-4,
            weight_decay=0.02,
            lambda_acc=0.1,
            lambda_beat=0.5,
        )

        self.assertEqual(config["feature_type"], "jukebox")
        self.assertTrue(config["use_beats"])
        self.assertEqual(config["beat_rep"], "distance")
        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertEqual(config["effective_batch_size"], 512)
        self.assertEqual(config["learning_rate"], 2e-4)
        self.assertEqual(config["weight_decay"], 0.02)
        self.assertEqual(config["lambda_acc"], 0.1)
        self.assertEqual(config["lambda_beat"], 0.5)

    def test_build_sample_dataloader_keeps_partial_test_batch(self):
        edge_module = reload_module("EDGE")

        dataset = torch.utils.data.TensorDataset(torch.arange(186))

        loader = edge_module.build_sample_dataloader(
            dataset,
            batch_size=256,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))

        self.assertEqual(len(loader), 1)
        self.assertEqual(batch[0].shape[0], 186)

    def test_build_dataloader_kwargs_enables_prefetching_when_workers_present(self):
        edge_module = reload_module("EDGE")

        kwargs = edge_module.build_dataloader_kwargs(num_workers=6, pin_memory=True)

        self.assertEqual(kwargs["num_workers"], 6)
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 1)

    def test_build_dataloader_kwargs_skips_prefetching_without_workers(self):
        edge_module = reload_module("EDGE")

        kwargs = edge_module.build_dataloader_kwargs(num_workers=0, pin_memory=True)

        self.assertEqual(kwargs, {"num_workers": 0, "pin_memory": True})

    def test_resolve_runtime_mixed_precision_disables_bf16_without_cuda(self):
        edge_module = reload_module("EDGE")

        resolved = edge_module.resolve_runtime_mixed_precision(
            "bf16", torch.device("cpu")
        )

        self.assertEqual(resolved, "no")


class WandbFallbackTests(unittest.TestCase):
    def test_safe_wandb_init_returns_none_when_wandb_init_fails(self):
        edge_module = reload_module("EDGE")

        with patch.object(edge_module.wandb, "init", side_effect=RuntimeError("boom")):
            run = edge_module.safe_wandb_init("EDGE", "smoke")

        self.assertIsNone(run)

    def test_safe_wandb_init_skips_init_when_disabled_via_env(self):
        edge_module = reload_module("EDGE")

        with patch.dict(os.environ, {"WANDB_DISABLED": "true"}, clear=False), patch.object(
            edge_module.wandb, "init"
        ) as mock_init:
            run = edge_module.safe_wandb_init("EDGE", "smoke")

        self.assertIsNone(run)
        mock_init.assert_not_called()


class LossGuardTests(unittest.TestCase):
    def test_validate_loss_terms_raises_on_non_finite_loss(self):
        edge_module = reload_module("EDGE")

        with self.assertRaises(FloatingPointError) as ctx:
            edge_module.validate_loss_terms(
                torch.tensor(float("nan")),
                (
                    torch.tensor(1.0),
                    torch.tensor(2.0),
                    torch.tensor(3.0),
                    torch.tensor(4.0),
                    torch.tensor(float("nan")),
                    torch.tensor(6.0),
                ),
                filenames=["feature_a.npy"],
                wavnames=["song_a.wav"],
            )

        self.assertIn("Non-finite loss detected", str(ctx.exception))
        self.assertIn("feature_a.npy", str(ctx.exception))
        self.assertIn("song_a.wav", str(ctx.exception))


class TrainingProgressFormattingTests(unittest.TestCase):
    def test_build_train_postfix_for_baseline_run(self):
        edge_module = reload_module("EDGE")

        postfix = edge_module.build_train_postfix(
            loss=torch.tensor(0.125),
            acc_loss=torch.tensor(0.05),
            beat_loss=torch.tensor(0.0),
            use_beats=False,
            beat_rep="distance",
            lambda_beat=0.5,
        )

        self.assertEqual(postfix["beat_mode"], "none")
        self.assertEqual(postfix["beat_loss"], "n/a")
        self.assertEqual(postfix["loss"], "0.1250")
        self.assertEqual(postfix["acc_loss"], "0.0500")

    def test_build_train_postfix_for_condition_only_beat_run(self):
        edge_module = reload_module("EDGE")

        postfix = edge_module.build_train_postfix(
            loss=torch.tensor(0.03125),
            acc_loss=torch.tensor(0.01),
            beat_loss=torch.tensor(0.0),
            use_beats=True,
            beat_rep="distance",
            lambda_beat=0.0,
        )

        self.assertEqual(postfix["beat_mode"], "distance:cond")
        self.assertEqual(postfix["beat_loss"], "off")
        self.assertEqual(postfix["loss"], "0.0312")

    def test_build_train_postfix_for_lbeat_run(self):
        edge_module = reload_module("EDGE")

        postfix = edge_module.build_train_postfix(
            loss=torch.tensor(0.05234),
            acc_loss=torch.tensor(0.0049),
            beat_loss=torch.tensor(2.2187),
            use_beats=True,
            beat_rep="distance",
            lambda_beat=0.5,
        )

        self.assertEqual(postfix["beat_mode"], "distance:cond+lbeat")
        self.assertEqual(postfix["beat_loss"], "2.2187")
        self.assertEqual(postfix["acc_loss"], "0.0049")


class CheckpointLoadCompatibilityTests(unittest.TestCase):
    def test_load_trusted_checkpoint_uses_weights_only_false(self):
        edge_module = reload_module("EDGE")

        with patch.object(edge_module.torch, "load", return_value={"ok": True}) as mock_load:
            payload = edge_module.load_trusted_checkpoint("weights/example.pt", map_location="cpu")

        self.assertEqual(payload, {"ok": True})
        mock_load.assert_called_once_with(
            "weights/example.pt",
            map_location="cpu",
            weights_only=False,
        )


class CheckpointRestoreTests(unittest.TestCase):
    def test_restore_checkpoint_state_loads_model_optimizer_and_ema(self):
        edge_module = reload_module("EDGE")
        model = MagicMock()
        diffusion = MagicMock()
        optim = MagicMock()
        checkpoint = {
            "ema_state_dict": {"ema": 1},
            "model_state_dict": {"model": 2},
            "optimizer_state_dict": {"optim": 3},
        }

        edge_module.restore_checkpoint_state(
            model,
            diffusion,
            optim,
            checkpoint,
            use_ema_weights=True,
            num_processes=1,
            restore_optimizer=True,
        )

        model.load_state_dict.assert_called_once_with({"ema": 1})
        diffusion.master_model.load_state_dict.assert_called_once_with({"ema": 1})
        optim.load_state_dict.assert_called_once_with({"optim": 3})


class DummyModel(nn.Module):
    def forward(self, x, cond, t, cond_drop_prob=0.0):
        return x


class DummySMPL:
    def forward(self, q, x):
        return x.unsqueeze(2).expand(-1, -1, 24, -1)


class DummyBeatEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, joints):
        return F.softplus(joints[..., 0].mean(dim=2) * self.scale)


class IdentityNormalizer:
    def unnormalize(self, tensor):
        return tensor


class DiffusionBeatLossTests(unittest.TestCase):
    def test_gaussian_diffusion_returns_acc_and_beat_losses_for_dict_cond(self):
        diffusion_module = reload_module("model.diffusion")
        estimator = DummyBeatEstimator()
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=151,
            smpl=DummySMPL(),
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
            beat_estimator=estimator,
            lambda_acc=0.1,
            lambda_beat=0.5,
            beat_a=10.0,
            beat_c=0.1,
        )
        x = torch.randn(2, 150, 151, requires_grad=True)
        cond = {
            "music": torch.randn(2, 150, 35),
            "beat": torch.randint(0, 151, (2, 150)),
            "beat_target": torch.randint(0, 10, (2, 150)).float(),
            "beat_spacing": torch.full((2, 150), 20.0),
            "audio_mask": torch.zeros(2, 150),
        }
        t = torch.zeros(2, dtype=torch.long)

        total_loss, losses = diffusion.p_losses(x, cond, t)
        total_loss.backward()

        self.assertEqual(len(losses), 6)
        self.assertTrue(torch.isfinite(losses[-2]))
        self.assertTrue(torch.isfinite(losses[-1]))
        self.assertFalse(estimator.scale.requires_grad)
        self.assertIsNone(estimator.scale.grad)

    def test_gaussian_diffusion_keeps_zero_beat_terms_without_beats(self):
        diffusion_module = reload_module("model.diffusion")
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=151,
            smpl=DummySMPL(),
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
        )
        x = torch.randn(1, 150, 151)
        cond = torch.randn(1, 150, 35)
        t = torch.zeros(1, dtype=torch.long)

        _, losses = diffusion.p_losses(x, cond, t)

        self.assertEqual(len(losses), 6)
        self.assertTrue(torch.isfinite(losses[-2]))
        self.assertGreaterEqual(losses[-2].item(), 0.0)
        self.assertEqual(losses[-1].item(), 0.0)

    def test_render_sample_respects_no_render_mode_for_sound_and_render_flags(self):
        diffusion_module = reload_module("model.diffusion")
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=151,
            smpl=DummySMPL(),
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
        )
        samples = torch.randn(1, 150, 151)
        cond = torch.randn(1, 150, 35)

        with patch.object(diffusion_module, "p_map", side_effect=lambda fn, items: [fn(item) for item in items]), patch.object(
            diffusion_module, "skeleton_render"
        ) as mock_render:
            diffusion.render_sample(
                samples,
                cond,
                IdentityNormalizer(),
                epoch="test",
                render_out="renders",
                name=["song.wav"],
                sound=True,
                render=False,
            )

        _, kwargs = mock_render.call_args
        self.assertFalse(kwargs["render"])
        self.assertFalse(kwargs["sound"])


class InferenceBeatUtilityTests(unittest.TestCase):
    def test_load_user_beat_frames_supports_seconds_and_frames(self):
        test_module = reload_module("test")

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            seconds_path = tmp_path / "beats_seconds.json"
            frames_path = tmp_path / "beats_frames.json"
            seconds_path.write_text(json.dumps({"fps": 30, "beat_times_sec": [0.5, 1.0, 1.5]}))
            frames_path.write_text(json.dumps({"fps": 30, "beat_frames": [15, 30, 45]}))

            seconds_frames = test_module.load_user_beat_frames(str(seconds_path), target_fps=30)
            raw_frames = test_module.load_user_beat_frames(str(frames_path), target_fps=30)

        self.assertTrue(np.array_equal(seconds_frames, np.array([15, 30, 45], dtype=np.int64)))
        self.assertTrue(np.array_equal(raw_frames, np.array([15, 30, 45], dtype=np.int64)))

    def test_audio_beats_are_built_on_the_full_song_before_slicing(self):
        test_module = reload_module("test")

        with patch.object(test_module.librosa, "load", return_value=(np.zeros(10, dtype=np.float32), 10)), patch.object(
            test_module.librosa.beat,
            "beat_track",
            return_value=(120.0, np.array([0.0, 2.5, 5.0])),
        ):
            beat_frames = test_module.load_audio_beat_frames("song.wav", fps=30)

        full_track = test_module.build_full_song_beat_track(
            beat_frames,
            total_frames=300,
            beat_rep="distance",
        )
        sliced = test_module.slice_beat_track(full_track, start_frame=75, horizon=150)

        self.assertEqual(full_track.shape, (300,))
        self.assertEqual(sliced.shape, (150,))
        self.assertEqual(sliced[0].item(), 0)
        self.assertEqual(sliced[75].item(), 0)

    def test_long_ddim_sample_falls_back_to_ddim_for_single_clip(self):
        diffusion_module = reload_module("model.diffusion")
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=151,
            smpl=DummySMPL(),
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
        )
        cond = torch.randn(1, 150, 35)
        expected = torch.randn(1, 150, 151)

        with patch.object(diffusion, "ddim_sample", return_value=expected) as mock_ddim:
            output = diffusion.long_ddim_sample((1, 150, 151), cond)

        mock_ddim.assert_called_once()
        self.assertTrue(torch.equal(output, expected))

    def test_build_full_song_pulse_track_keeps_channel_dimension(self):
        test_module = reload_module("test")

        full_track = test_module.build_full_song_beat_track(
            np.array([5, 25], dtype=np.int64),
            total_frames=60,
            beat_rep="pulse",
        )
        sliced = test_module.slice_beat_track(full_track, start_frame=0, horizon=30)

        self.assertEqual(full_track.shape, (60, 1))
        self.assertEqual(sliced.shape, (30, 1))
        self.assertEqual(float(sliced[5, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
