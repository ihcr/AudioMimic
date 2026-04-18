import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.beat_estimator import BeatDistanceEstimator
from model.diffusion import (GaussianDiffusion, cond_batch_size, move_cond_to_device,
                             slice_cond)
from model.model import BeatDanceDecoder, DanceDecoder
from vis import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


def tensor_dataset_cache_name(split, feature_type, use_beats, beat_rep):
    beat_tag = "beat" if use_beats else "nobeat"
    return f"{split}_tensor_dataset_{feature_type}_{beat_tag}_{beat_rep}.pkl"


DEFAULT_MODEL_CONFIG = {
    "feature_type": "jukebox",
    "use_beats": False,
    "beat_rep": "distance",
}


def wandb_disabled_by_env():
    disabled = os.environ.get("WANDB_DISABLED", "").strip().lower()
    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    return disabled in {"1", "true", "yes", "on"} or mode == "disabled"


def safe_wandb_init(project, name):
    if wandb_disabled_by_env():
        print("wandb disabled via environment; skipping wandb.init()")
        return None
    try:
        return wandb.init(project=project, name=name)
    except Exception as exc:
        print(f"wandb init failed; continuing without wandb logging: {exc}")
        return None


def load_trusted_checkpoint(checkpoint_path, map_location):
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def resolve_model_config(feature_type, use_beats, beat_rep, checkpoint_config=None):
    resolved = {
        "feature_type": feature_type,
        "use_beats": use_beats,
        "beat_rep": beat_rep,
    }
    if checkpoint_config is None:
        return resolved

    for key in ("feature_type", "use_beats", "beat_rep"):
        if key not in checkpoint_config:
            continue
        checkpoint_value = checkpoint_config[key]
        runtime_value = resolved[key]
        if runtime_value != DEFAULT_MODEL_CONFIG[key] and runtime_value != checkpoint_value:
            raise ValueError(
                f"Runtime {key}={runtime_value!r} conflicts with checkpoint {key}={checkpoint_value!r}"
            )
        resolved[key] = checkpoint_value
    return resolved


def build_beat_estimator_from_checkpoint(checkpoint_path, device):
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_kwargs = {
        key: config[key]
        for key in ("input_dim", "hidden_dim", "num_heads", "num_layers", "ff_dim", "dropout")
        if key in config
    }
    model = BeatDistanceEstimator(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        use_beats=False,
        beat_rep="distance",
        lambda_acc=0.1,
        lambda_beat=0.5,
        beat_a=10.0,
        beat_c=0.1,
        beat_estimator_ckpt="",
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        self.lambda_acc = lambda_acc
        self.lambda_beat = lambda_beat
        self.beat_a = beat_a
        self.beat_c = beat_c
        self.beat_estimator_ckpt = beat_estimator_ckpt

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = load_trusted_checkpoint(
                checkpoint_path, map_location=self.accelerator.device
            )
            resolved = resolve_model_config(
                feature_type=feature_type,
                use_beats=use_beats,
                beat_rep=beat_rep,
                checkpoint_config=checkpoint.get("config"),
            )
            feature_type = resolved["feature_type"]
            use_beats = resolved["use_beats"]
            beat_rep = resolved["beat_rep"]
            self.normalizer = checkpoint["normalizer"]

        self.feature_type = feature_type
        self.use_beats = use_beats
        self.beat_rep = beat_rep
        use_baseline_feats = self.feature_type == "baseline"
        feature_dim = 35 if use_baseline_feats else 4800

        model_cls = BeatDanceDecoder if self.use_beats else DanceDecoder
        model_kwargs = dict(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )
        if self.use_beats:
            model_kwargs["beat_rep"] = self.beat_rep
        model = model_cls(**model_kwargs)

        smpl = SMPLSkeleton(self.accelerator.device)
        beat_estimator = None
        if self.use_beats and self.lambda_beat > 0:
            if not self.beat_estimator_ckpt:
                raise ValueError("Beat-enabled training with lambda_beat > 0 requires --beat_estimator_ckpt")
            beat_estimator = build_beat_estimator_from_checkpoint(
                self.beat_estimator_ckpt, self.accelerator.device
            )
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
            beat_estimator=beat_estimator,
            lambda_acc=self.lambda_acc,
            lambda_beat=self.lambda_beat,
            beat_a=self.beat_a,
            beat_c=self.beat_c,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir,
            tensor_dataset_cache_name("train", self.feature_type, self.use_beats, self.beat_rep),
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir,
            tensor_dataset_cache_name("test", self.feature_type, self.use_beats, self.beat_rep),
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                feature_type=self.feature_type,
                force_reload=opt.force_reload,
                use_beats=self.use_beats,
                beat_rep=self.beat_rep,
            )
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                feature_type=self.feature_type,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
                use_beats=self.use_beats,
                beat_rep=self.beat_rep,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        pin_memory = self.accelerator.device.type == "cuda"
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.train_num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.test_num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        wandb_run = None
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb_run = safe_wandb_init(opt.wandb_pj_name, opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            avg_accloss = 0
            avg_beatloss = 0
            # train
            self.train()
            train_loop = train_data_loader
            if self.accelerator.is_main_process:
                train_loop = tqdm(
                    train_data_loader,
                    total=len(train_data_loader),
                    position=1,
                    desc=f"Train {epoch}/{opt.epochs}",
                    unit="batch",
                )

            for step, (x, cond, filename, wavnames) in enumerate(train_loop):
                total_loss, (loss, v_loss, fk_loss, foot_loss, acc_loss, beat_loss) = self.diffusion(
                    x, cond, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    avg_accloss += acc_loss.detach().cpu().numpy()
                    avg_beatloss += beat_loss.detach().cpu().numpy()
                    if hasattr(train_loop, "set_postfix"):
                        train_loop.set_postfix(
                            loss=f"{loss.detach().cpu().item():.4f}",
                            beat=f"{beat_loss.detach().cpu().item():.4f}",
                        )
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    avg_accloss /= len(train_data_loader)
                    avg_beatloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                        "Acc Loss": avg_accloss,
                        "Beat Loss": avg_beatloss,
                    }
                    if wandb_run is not None:
                        wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                        "config": {
                            "feature_type": self.feature_type,
                            "use_beats": self.use_beats,
                            "beat_rep": self.beat_rep,
                            "lambda_acc": self.lambda_acc,
                            "lambda_beat": self.lambda_beat,
                        },
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename, wavnames) = next(iter(test_data_loader))
                    cond = move_cond_to_device(cond, self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        slice_cond(cond, slice(None, render_count)),
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process and wandb_run is not None:
            wandb_run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        if render_count < 0:
            render_count = cond_batch_size(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = move_cond_to_device(cond, self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            slice_cond(cond, slice(None, render_count)),
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
