import argparse
import importlib
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


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
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)


def write_dataset(root):
    for split, count in (("train", 2), ("test", 1)):
        motion_dir = root / split / "motions_sliced"
        motion_dir.mkdir(parents=True)
        for index in range(count):
            write_g1_motion(motion_dir / f"clip{index:03d}.pkl", offset=0.01 * index)


def g1_model_path():
    path = Path("third_party/unitree_g1_description/g1_29dof_rev_1_0.xml")
    if not path.is_file():
        raise unittest.SkipTest(f"missing G1 model: {path}")
    return path


class G1MotionPriorDatasetTests(unittest.TestCase):
    def test_cache_fields_shapes_and_metadata(self):
        dataset_module = reload_module("dataset.g1_motion_prior_dataset")
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            cache_root = Path(tmpdir) / "cache"
            write_dataset(data_root)

            dataset = dataset_module.G1MotionPriorDataset(
                data_path=data_root,
                backup_path=cache_root,
                split="train",
                motion_format="g1_yaw_delta",
                g1_fk_model_path=g1_model_path(),
                cache_batch_size=2,
                cache_device="cpu",
            )

            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(tuple(sample["motion"].shape), (150, 34))
            self.assertEqual(tuple(sample["contact"].shape), (150, 2))
            self.assertEqual(tuple(sample["near_support"].shape), (150, 2))
            self.assertEqual(tuple(sample["lowest_foot_heights"].shape), (150, 2))
            self.assertEqual(dataset.metadata["cache_version"], dataset_module.MOTION_PRIOR_CACHE_VERSION)
            self.assertEqual(dataset.metadata["motion_format"], "g1_yaw_delta")
            self.assertEqual(dataset.metadata["repr_dim"], 34)
            self.assertTrue(np.isfinite(dataset.normalizer.mean).all())
            self.assertTrue(np.isfinite(dataset.normalizer.std).all())


class G1MotionPriorModelTests(unittest.TestCase):
    def test_ae_and_vae_forward_shapes(self):
        model_module = reload_module("model.g1_motion_prior")
        x = torch.randn(2, 150, 34)
        ae = model_module.G1MotionAutoencoder(
            motion_format="g1_yaw_delta",
            latent_dim=64,
            hidden_dim=64,
            temporal_downsample=2,
            prior_type="ae",
        )
        ae_out = ae(x)
        self.assertEqual(tuple(ae_out["latent"].shape), (2, 75, 64))
        self.assertEqual(tuple(ae_out["recon"].shape), (2, 150, 34))
        self.assertEqual(tuple(ae_out["contact_logits"].shape), (2, 150, 2))
        self.assertIsNone(ae_out["mu"])

        vae = model_module.G1MotionAutoencoder(
            motion_format="g1_yaw_delta",
            latent_dim=32,
            hidden_dim=64,
            temporal_downsample=2,
            prior_type="vae",
        )
        vae_out = vae(x, sample=False)
        self.assertEqual(tuple(vae_out["latent"].shape), (2, 75, 32))
        self.assertEqual(tuple(vae_out["mu"].shape), (2, 75, 32))
        self.assertEqual(tuple(vae_out["logvar"].shape), (2, 75, 32))

    def test_loss_decode_fk_path_is_finite(self):
        model_module = reload_module("model.g1_motion_prior")
        kinematics_module = reload_module("model.g1_torch_kinematics")
        model = model_module.G1MotionAutoencoder(
            motion_format="g1_yaw_delta",
            latent_dim=32,
            hidden_dim=64,
            temporal_downsample=2,
        )
        target = torch.zeros(2, 150, 34)
        target[..., 2] = 0.78
        target[..., 4] = 1.0
        contact = torch.ones(2, 150, 2)
        output = model(target)
        mean = torch.zeros(1, 1, 34)
        std = torch.ones(1, 1, 34)
        kinematics = kinematics_module.G1TorchKinematics(g1_model_path(), root_quat_order="xyzw")

        loss, stats = model_module.compute_g1_motion_prior_losses(
            output,
            target,
            contact,
            mean,
            std,
            kinematics=kinematics,
            motion_format="g1_yaw_delta",
            ground=torch.zeros(2),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(stats["loss/fk_mpjpe"]))
        self.assertIn("contact_f1", stats)


class G1MotionPriorEvalTests(unittest.TestCase):
    def test_eval_writes_reconstruction_metrics(self):
        dataset_module = reload_module("dataset.g1_motion_prior_dataset")
        model_module = reload_module("model.g1_motion_prior")
        eval_module = reload_module("eval.run_g1_motion_prior_eval")
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_root = tmp / "data"
            cache_root = tmp / "cache"
            output_root = tmp / "eval"
            run_root = tmp / "run"
            write_dataset(data_root)
            dataset = dataset_module.G1MotionPriorDataset(
                data_path=data_root,
                backup_path=cache_root,
                split="train",
                motion_format="g1_yaw_delta",
                g1_fk_model_path=g1_model_path(),
                cache_batch_size=2,
                cache_device="cpu",
            )
            model = model_module.G1MotionAutoencoder(
                motion_format="g1_yaw_delta",
                latent_dim=16,
                hidden_dim=64,
                temporal_downsample=2,
            )
            checkpoint_path = run_root / "weights" / "train-1.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "epoch": 1,
                    "model": model.state_dict(),
                    "config": {
                        "latent_dim": 16,
                        "hidden_dim": 64,
                        "temporal_downsample": 2,
                        "prior_type": "ae",
                        "dropout": 0.0,
                        "motion_format": "g1_yaw_delta",
                    },
                    "normalizer": dataset.normalizer.state_dict(),
                    "metadata": dataset.metadata,
                },
                checkpoint_path,
            )
            args = argparse.Namespace(
                checkpoint=str(checkpoint_path),
                data_path=str(data_root),
                processed_data_dir=str(cache_root),
                output_dir=str(output_root),
                split="test",
                motion_format="g1_yaw_delta",
                batch_size=1,
                num_workers=0,
                max_eval_clips=1,
                cache_limit_per_split=0,
                rebuild_cache=False,
                diagnostic_count=0,
                render_count=0,
                enable_fk_metrics=False,
                g1_fk_model_path=str(g1_model_path()),
                g1_root_quat_order="xyzw",
                g1_render_backend="stick",
                g1_render_width=320,
                g1_render_height=240,
                g1_mujoco_gl="egl",
            )

            summary = eval_module.run_g1_motion_prior_eval(args)

            self.assertTrue((output_root / "reconstruction_metrics.json").is_file())
            self.assertTrue((output_root / "metrics.json").is_file())
            self.assertTrue((output_root / "gt_baseline" / "metrics.json").is_file())
            self.assertEqual(summary["num_eval_clips"], 1)


if __name__ == "__main__":
    unittest.main()
