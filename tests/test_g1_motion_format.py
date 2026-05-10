import importlib
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def write_g1_motion(path):
    frames = 150
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 2] = 0.78
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 0] = 1.0
    dof_pos = np.zeros((frames, 29), dtype=np.float32)
    dof_pos[:, 0] = np.linspace(-0.2, 0.2, frames, dtype=np.float32)
    payload = {
        "motion_rep": "g1",
        "fps": 30.0,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "pos": root_pos,
        "q": np.concatenate([root_rot, dof_pos], axis=-1),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)


def write_g1_dataset(root):
    for split in ("train", "test"):
        split_dir = root / split
        (split_dir / "motions_sliced").mkdir(parents=True)
        (split_dir / "wavs_sliced").mkdir(parents=True)
        (split_dir / "baseline_feats").mkdir(parents=True)
        write_g1_motion(split_dir / "motions_sliced" / "clip_slice0.pkl")
        (split_dir / "wavs_sliced" / "clip_slice0.wav").write_bytes(b"RIFF")
        np.save(
            split_dir / "baseline_feats" / "clip_slice0.npy",
            np.zeros((150, 35), dtype=np.float32),
        )


class DummyModel(nn.Module):
    def forward(self, x, cond, t, cond_drop_prob=0.0):
        return x

    def guided_forward(self, x, cond, t, weight):
        return x


class IdentityNormalizer:
    def unnormalize(self, tensor):
        return tensor


class G1MotionRepresentationTests(unittest.TestCase):
    def test_g1_encode_decode_uses_robot_native_dimensions(self):
        motion_module = reload_module("dataset.motion_representation")
        root_pos = torch.zeros(2, 150, 3)
        root_rot = torch.zeros(2, 150, 4)
        root_rot[..., 0] = 1.0
        dof_pos = torch.randn(2, 150, 29)

        encoded = motion_module.encode_g1_motion(root_pos, root_rot, dof_pos)
        decoded = motion_module.decode_g1_motion(encoded)

        self.assertEqual(motion_module.motion_repr_dim("g1"), 38)
        self.assertEqual(encoded.shape, (2, 150, 38))
        self.assertEqual(decoded["root_pos"].shape, (2, 150, 3))
        self.assertEqual(decoded["root_rot"].shape, (2, 150, 4))
        self.assertEqual(decoded["dof_pos"].shape, (2, 150, 29))
        self.assertTrue(torch.allclose(decoded["root_pos"], root_pos, atol=1e-5))
        self.assertTrue(torch.allclose(decoded["dof_pos"], dof_pos, atol=1e-5))
        quat_dot = (decoded["root_rot"] * root_rot).sum(dim=-1).abs()
        self.assertTrue(torch.allclose(quat_dot, torch.ones_like(quat_dot), atol=1e-5))


class G1DatasetTests(unittest.TestCase):
    def test_dataset_loads_g1_clips_without_downsampling_again(self):
        dataset_module = reload_module("dataset.dance_dataset")

        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            processed_root = Path(tmpdir) / "processed"
            write_g1_dataset(data_root)

            dataset = dataset_module.AISTPPDataset(
                data_path=str(data_root),
                backup_path=str(processed_root),
                train=True,
                feature_type="baseline",
                motion_format="g1",
                force_reload=True,
            )

            pose, music, _, _ = dataset[0]
            self.assertEqual(pose.shape, (150, 38))
            self.assertEqual(music.shape, (150, 35))
            self.assertEqual(dataset.normalizer.scaler.data_min_.shape[0], 38)


class G1DiffusionTests(unittest.TestCase):
    def test_g1_loss_skips_human_body_fk_and_foot_losses(self):
        diffusion_module = reload_module("model.diffusion")
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=38,
            smpl=None,
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
            motion_format="g1",
        )
        x = torch.randn(2, 150, 38)
        cond = torch.randn(2, 150, 35)
        t = torch.zeros(2, dtype=torch.long)

        total_loss, losses = diffusion.p_losses(x, cond, t)

        self.assertTrue(torch.isfinite(total_loss))
        self.assertEqual(len(losses), 6)
        self.assertEqual(float(losses[2]), 0.0)
        self.assertEqual(float(losses[3]), 0.0)

    def test_g1_render_sample_saves_robot_motion_payload_without_smpl_render(self):
        diffusion_module = reload_module("model.diffusion")
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=38,
            smpl=None,
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
            motion_format="g1",
        )
        samples = torch.zeros(1, 150, 38)
        samples[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        samples[..., 9:] = torch.randn(1, 150, 29)
        cond = torch.randn(1, 150, 35)

        with TemporaryDirectory() as tmpdir, patch.object(
            diffusion_module, "skeleton_render"
        ) as mock_render:
            diffusion.render_sample(
                samples,
                cond,
                IdentityNormalizer(),
                epoch="sample",
                render_out=tmpdir,
                fk_out=tmpdir,
                name=["song_slice0.wav"],
                sound=True,
                render=False,
            )
            saved = sorted(Path(tmpdir).glob("*.pkl"))
            self.assertEqual(len(saved), 1)
            with open(saved[0], "rb") as handle:
                payload = pickle.load(handle)

        mock_render.assert_not_called()
        self.assertEqual(payload["motion_format"], "g1")
        self.assertEqual(payload["root_pos"].shape, (150, 3))
        self.assertEqual(payload["root_rot"].shape, (150, 4))
        self.assertEqual(payload["dof_pos"].shape, (150, 29))
        self.assertEqual(payload["q"].shape, (150, 33))
        self.assertNotIn("full_pose", payload)

    def test_g1_render_sample_calls_g1_visualizer_when_render_enabled(self):
        diffusion_module = reload_module("model.diffusion")
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=38,
            smpl=None,
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
            motion_format="g1",
        )
        samples = torch.zeros(1, 150, 38)
        samples[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        cond = torch.randn(1, 150, 35)

        with TemporaryDirectory() as tmpdir, patch(
            "eval.g1_visualization.render_g1_motion"
        ) as mock_render:
            diffusion.render_sample(
                samples,
                cond,
                IdentityNormalizer(),
                epoch="sample",
                render_out=tmpdir,
                fk_out=tmpdir,
                name=["song_slice0.wav"],
                sound=True,
                render=True,
                g1_fk_model_path="robot.xml",
                g1_root_quat_order="xyzw",
                g1_render_backend="stick",
                g1_render_width=640,
                g1_render_height=480,
                g1_mujoco_gl="glfw",
            )

        mock_render.assert_called_once()
        _, kwargs = mock_render.call_args
        self.assertEqual(kwargs["out"], tmpdir)
        self.assertEqual(kwargs["epoch"], "sample")
        self.assertEqual(kwargs["num"], 0)
        self.assertEqual(kwargs["name"], "song_slice0.wav")
        self.assertTrue(kwargs["sound"])
        self.assertEqual(kwargs["model_path"], "robot.xml")
        self.assertEqual(kwargs["root_quat_order"], "xyzw")
        self.assertEqual(kwargs["render_backend"], "stick")
        self.assertEqual(kwargs["width"], 640)
        self.assertEqual(kwargs["height"], 480)
        self.assertEqual(kwargs["mujoco_gl"], "glfw")
        self.assertEqual(mock_render.call_args.args[0]["motion_format"], "g1")

    def test_g1_long_render_uses_original_full_song_audio_when_available(self):
        diffusion_module = reload_module("model.diffusion")
        diffusion = diffusion_module.GaussianDiffusion(
            DummyModel(),
            horizon=150,
            repr_dim=38,
            smpl=None,
            schedule="cosine",
            n_timestep=10,
            predict_epsilon=False,
            loss_type="l2",
            cond_drop_prob=0.0,
            motion_format="g1",
        )
        samples = torch.zeros(2, 150, 38)
        samples[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        cond = torch.randn(2, 150, 35)

        with TemporaryDirectory() as tmpdir, patch(
            "eval.g1_visualization.render_g1_motion"
        ) as mock_render:
            diffusion.render_sample(
                samples,
                cond,
                IdentityNormalizer(),
                epoch="fullsong",
                render_out=tmpdir,
                fk_out=tmpdir,
                name=["song_slice0.wav", "song_slice1.wav"],
                sound=True,
                mode="long",
                render=True,
                metadata_audio_path="song.wav",
                metadata_total_frames=160,
                g1_render_backend="mujoco",
            )

        _, kwargs = mock_render.call_args
        self.assertEqual(kwargs["name"], "song.wav")
        self.assertFalse(kwargs["stitch"])
        self.assertEqual(kwargs["render_backend"], "mujoco")
        self.assertEqual(kwargs["output_name"], ["song_slice0.wav", "song_slice1.wav"])
        self.assertEqual(mock_render.call_args.args[0]["root_pos"].shape[0], 160)


if __name__ == "__main__":
    unittest.main()
