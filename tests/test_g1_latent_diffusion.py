import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


def write_g1_motion(path, offset=0.0):
    frames = 150
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0.0, 0.2 + offset, frames, dtype=np.float32)
    root_pos[:, 2] = 0.78
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    dof_pos = np.zeros((frames, 29), dtype=np.float32)
    dof_pos[:, 0] = np.linspace(-0.1, 0.1 + offset, frames, dtype=np.float32)
    payload = {
        "motion_rep": "g1",
        "motion_format": "g1",
        "fps": 30.0,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "pos": root_pos,
        "q": np.concatenate([root_rot, dof_pos], axis=-1),
        "audio_path": "",
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)


def write_dataset(root):
    for split, count in (("train", 2), ("test", 1)):
        motion_dir = root / split / "motions_sliced"
        beat_dir = root / split / "beat_features_8d_feats"
        wav2clip_dir = root / split / "wav2clip_stft_beat_feats"
        wav_dir = root / split / "wavs_sliced"
        motion_dir.mkdir(parents=True)
        beat_dir.mkdir(parents=True)
        wav2clip_dir.mkdir(parents=True)
        wav_dir.mkdir(parents=True)
        for index in range(count):
            stem = f"clip{index:03d}"
            write_g1_motion(motion_dir / f"{stem}.pkl", offset=0.01 * index)
            beat = np.zeros((150, 8), dtype=np.float32)
            beat[:, 4] = np.sin(np.linspace(0, np.pi * 2, 150, dtype=np.float32))
            beat[:, 5] = np.cos(np.linspace(0, np.pi * 2, 150, dtype=np.float32))
            np.save(beat_dir / f"{stem}.npy", beat)
            wav2clip = np.zeros((150, 706), dtype=np.float32)
            wav2clip[:, :512] = (index + 1) * 0.01
            wav2clip[:, 512:] = 99.0
            np.save(wav2clip_dir / f"{stem}.npy", wav2clip)
            (wav_dir / f"{stem}.wav").write_bytes(b"")


def g1_model_path():
    path = Path("third_party/unitree_g1_description/g1_29dof_rev_1_0.xml")
    if not path.is_file():
        raise unittest.SkipTest(f"missing G1 model: {path}")
    return path


class G1LatentDiffusionTests(unittest.TestCase):
    def test_latent_beat_cache_and_diffusion_shapes(self):
        from dataset.g1_latent_beat_dataset import G1LatentBeatDataset, G1MusicControlLatentDataset
        from dataset.g1_motion_prior_dataset import G1MotionPriorDataset
        from model.g1_latent_diffusion import (
            G1Beat8DLatentDenoiser,
            G1LatentDiffusion,
            G1MusicControlLatentDenoiser,
        )
        from model.g1_motion_prior import G1MotionAutoencoder

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_root = tmp / "data"
            motion_cache = tmp / "motion_cache"
            latent_cache = tmp / "latent_cache"
            run_root = tmp / "run"
            write_dataset(data_root)
            motion_dataset = G1MotionPriorDataset(
                data_path=data_root,
                backup_path=motion_cache,
                split="train",
                motion_format="g1_yaw_delta",
                g1_fk_model_path=g1_model_path(),
                cache_batch_size=2,
                cache_device="cpu",
            )
            prior = G1MotionAutoencoder(
                motion_format="g1_yaw_delta",
                latent_dim=128,
                hidden_dim=64,
                temporal_downsample=2,
            )
            prior_checkpoint = run_root / "weights" / "train-1.pt"
            prior_checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "epoch": 1,
                    "model": prior.state_dict(),
                    "config": {
                        "latent_dim": 128,
                        "hidden_dim": 64,
                        "temporal_downsample": 2,
                        "prior_type": "ae",
                        "dropout": 0.0,
                        "motion_format": "g1_yaw_delta",
                    },
                    "normalizer": motion_dataset.normalizer.state_dict(),
                    "metadata": motion_dataset.metadata,
                },
                prior_checkpoint,
            )

            dataset = G1LatentBeatDataset(
                data_path=data_root,
                motion_prior_processed_data_dir=motion_cache,
                latent_processed_data_dir=latent_cache,
                prior_checkpoint=prior_checkpoint,
                split="train",
                motion_format="g1_yaw_delta",
                g1_fk_model_path=g1_model_path(),
                cache_batch_size=2,
                cache_device="cpu",
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["latent"].shape), (75, 128))
            self.assertEqual(tuple(sample["beat_features"].shape), (150, 8))

            control_dataset = G1MusicControlLatentDataset(
                data_path=data_root,
                motion_prior_processed_data_dir=motion_cache,
                latent_processed_data_dir=latent_cache / "control",
                prior_checkpoint=prior_checkpoint,
                split="train",
                motion_format="g1_yaw_delta",
                g1_fk_model_path=g1_model_path(),
                cache_batch_size=2,
                cache_device="cpu",
                use_wav2clip_semantic=False,
            )
            control_sample = control_dataset[0]
            self.assertEqual(tuple(control_sample["latent"].shape), (75, 128))
            self.assertEqual(tuple(control_sample["control_features"].shape), (150, 8))
            self.assertNotIn("semantic_features", control_sample)

            semantic_dataset = G1MusicControlLatentDataset(
                data_path=data_root,
                motion_prior_processed_data_dir=motion_cache,
                latent_processed_data_dir=latent_cache / "semantic",
                prior_checkpoint=prior_checkpoint,
                split="train",
                motion_format="g1_yaw_delta",
                g1_fk_model_path=g1_model_path(),
                cache_batch_size=2,
                cache_device="cpu",
                use_wav2clip_semantic=True,
            )
            semantic_sample = semantic_dataset[0]
            self.assertEqual(tuple(semantic_sample["control_features"].shape), (150, 8))
            self.assertEqual(tuple(semantic_sample["semantic_features"].shape), (150, 512))
            self.assertLess(float(semantic_sample["semantic_features"].max()), 1.0)

            denoiser = G1Beat8DLatentDenoiser(
                latent_dim=128,
                hidden_dim=64,
                num_layers=1,
                num_heads=4,
                ff_size=128,
                dropout=0.0,
            )
            diffusion = G1LatentDiffusion(denoiser, timesteps=10)
            latent = torch.stack([dataset[0]["latent"], dataset[1]["latent"]])
            beat = torch.stack([dataset[0]["beat_features"], dataset[1]["beat_features"]])
            loss, stats = diffusion.p_losses(latent, beat)
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("loss/noise_mse", stats)
            sampled = diffusion.ddim_sample(beat, shape=(2, 75, 128), sampling_steps=2)
            self.assertEqual(tuple(sampled.shape), (2, 75, 128))

            v6bc_denoiser = G1MusicControlLatentDenoiser(
                latent_dim=128,
                hidden_dim=64,
                num_layers=1,
                num_heads=4,
                ff_size=128,
                dropout=0.0,
                use_wav2clip_semantic=True,
            )
            v6bc_diffusion = G1LatentDiffusion(v6bc_denoiser, timesteps=10)
            latent = torch.stack([semantic_dataset[0]["latent"], semantic_dataset[1]["latent"]])
            control = torch.stack(
                [semantic_dataset[0]["control_features"], semantic_dataset[1]["control_features"]]
            )
            semantic = torch.stack(
                [semantic_dataset[0]["semantic_features"], semantic_dataset[1]["semantic_features"]]
            )
            loss, stats = v6bc_diffusion.p_losses(
                latent,
                control,
                semantic_features=semantic,
                control_rank_weight=0.10,
                semantic_rank_weight=0.03,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("delta_shift_control", stats)
            self.assertIn("delta_random_control", stats)
            self.assertIn("delta_random_semantic", stats)
            sampled = v6bc_diffusion.ddim_sample(
                control,
                shape=(2, 75, 128),
                semantic_features=semantic,
                sampling_steps=2,
            )
            self.assertEqual(tuple(sampled.shape), (2, 75, 128))

    def test_eval_variant_conditions(self):
        from eval.run_g1_latent_diffusion_eval import variant_conditions

        control = torch.arange(2 * 150 * 8, dtype=torch.float32).reshape(2, 150, 8)
        semantic = torch.arange(2 * 150 * 512, dtype=torch.float32).reshape(2, 150, 512)
        shifted_control, shifted_semantic = variant_conditions(
            control,
            semantic,
            "shifted_control",
            use_wav2clip_semantic=True,
        )
        self.assertTrue(torch.equal(shifted_control, torch.roll(control, shifts=20, dims=1)))
        self.assertTrue(torch.equal(shifted_semantic, semantic))
        zero_control, real_semantic = variant_conditions(
            control,
            semantic,
            "zero_control",
            use_wav2clip_semantic=True,
        )
        self.assertTrue(torch.equal(zero_control, torch.zeros_like(control)))
        self.assertTrue(torch.equal(real_semantic, semantic))
        zero_control, zero_semantic = variant_conditions(
            control,
            semantic,
            "zero_all",
            use_wav2clip_semantic=True,
        )
        self.assertTrue(torch.equal(zero_control, torch.zeros_like(control)))
        self.assertTrue(torch.equal(zero_semantic, torch.zeros_like(semantic)))
        random_control, real_semantic = variant_conditions(
            control,
            semantic,
            "real_semantic_random_control",
            use_wav2clip_semantic=True,
        )
        self.assertEqual(tuple(random_control.shape), tuple(control.shape))
        self.assertTrue(torch.equal(real_semantic, semantic))


if __name__ == "__main__":
    unittest.main()
