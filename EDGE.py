import os
import pickle
import time
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

TENSOR_DATASET_CACHE_VERSION = "v5"


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


def atomic_pickle_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def tensor_dataset_cache_name(split, feature_type, use_beats, beat_rep):
    beat_tag = "beat" if use_beats else "nobeat"
    return (
        f"{split}_tensor_dataset_{feature_type}_{beat_tag}_{beat_rep}_"
        f"{TENSOR_DATASET_CACHE_VERSION}.pkl"
    )


def prune_legacy_tensor_dataset_caches(processed_data_dir, split, feature_type, use_beats, beat_rep):
    processed_data_dir = Path(processed_data_dir)
    beat_tag = "beat" if use_beats else "nobeat"
    current_name = tensor_dataset_cache_name(split, feature_type, use_beats, beat_rep)
    pattern = f"{split}_tensor_dataset_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
    for cache_path in processed_data_dir.glob(pattern):
        if cache_path.name == current_name:
            continue
        cache_path.unlink(missing_ok=True)


def build_dataloader_kwargs(num_workers, pin_memory):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return kwargs


def build_sample_dataloader(dataset, batch_size, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **build_dataloader_kwargs(num_workers, pin_memory),
    )


DEFAULT_MODEL_CONFIG = {
    "feature_type": "jukebox",
    "use_beats": False,
    "beat_rep": "distance",
}
DEFAULT_NON_BEAT_LEARNING_RATE = 2e-4
DEFAULT_BEAT_LEARNING_RATE = 2e-4


def resolve_runtime_mixed_precision(requested_mode, device):
    device_obj = torch.device(device)
    return requested_mode if device_obj.type == "cuda" else "no"


def configure_cuda_math():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


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


def default_learning_rate(use_beats):
    return DEFAULT_BEAT_LEARNING_RATE if use_beats else DEFAULT_NON_BEAT_LEARNING_RATE


def default_lambda_acc(use_beats):
    return 0.0


def effective_batch_size(batch_size, gradient_accumulation_steps, num_processes):
    return int(batch_size) * int(gradient_accumulation_steps) * max(int(num_processes), 1)


def build_checkpoint_config(
    feature_type,
    use_beats,
    beat_rep,
    batch_size,
    gradient_accumulation_steps,
    num_processes,
    learning_rate,
    weight_decay,
    lambda_acc,
    lambda_beat,
):
    return {
        "feature_type": feature_type,
        "use_beats": use_beats,
        "beat_rep": beat_rep,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size(
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_processes=num_processes,
        ),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lambda_acc": lambda_acc,
        "lambda_beat": lambda_beat,
    }


def maybe_apply_optimization_step(
    accelerator,
    model,
    optim,
    diffusion,
    grad_clip_norm,
    global_step,
    ema_interval,
):
    if not accelerator.sync_gradients:
        return False
    if grad_clip_norm is not None:
        accelerator.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optim.step()
    if global_step % ema_interval == 0:
        diffusion.ema.update_model_average(
            diffusion.master_model, diffusion.model
        )
    return True


def resolve_runtime_training_config(
    feature_type,
    use_beats,
    beat_rep,
    learning_rate,
    learning_rate_was_explicit=False,
    checkpoint_config=None,
):
    resolved = resolve_model_config(
        feature_type=feature_type,
        use_beats=use_beats,
        beat_rep=beat_rep,
        checkpoint_config=checkpoint_config,
    )
    if not learning_rate_was_explicit:
        learning_rate = default_learning_rate(resolved["use_beats"])
    resolved["learning_rate"] = learning_rate
    resolved["grad_clip_norm"] = 1.0 if resolved["use_beats"] else None
    return resolved


def restore_checkpoint_state(
    model,
    diffusion,
    optim,
    checkpoint,
    use_ema_weights,
    num_processes,
    restore_optimizer=False,
):
    state_key = "ema_state_dict" if use_ema_weights else "model_state_dict"
    model.load_state_dict(maybe_wrap(checkpoint[state_key], num_processes))
    ema_state_dict = checkpoint.get("ema_state_dict")
    if ema_state_dict is not None:
        diffusion.master_model.load_state_dict(ema_state_dict)
    if restore_optimizer and "optimizer_state_dict" in checkpoint:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])


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


def validate_loss_terms(total_loss, losses, filenames=None, wavnames=None):
    named_losses = {
        "recon_loss": losses[0],
        "velocity_loss": losses[1],
        "fk_loss": losses[2],
        "foot_loss": losses[3],
        "acc_loss": losses[4],
        "beat_loss": losses[5],
    }
    tensors = {"total_loss": total_loss, **named_losses}
    bad_terms = {
        name: float(value.detach().cpu())
        for name, value in tensors.items()
        if not torch.isfinite(value).all()
    }
    if bad_terms:
        context = []
        if filenames:
            context.append(f"features={list(filenames)[:2]}")
        if wavnames:
            context.append(f"wavs={list(wavnames)[:2]}")
        context_str = f" ({', '.join(context)})" if context else ""
        raise FloatingPointError(
            f"Non-finite loss detected{context_str}: {bad_terms}"
        )


def _format_loss_value(value):
    return f"{float(value.detach().cpu() if torch.is_tensor(value) else value):.4f}"


def build_train_postfix(loss, acc_loss, beat_loss, use_beats, beat_rep, lambda_beat):
    if not use_beats:
        beat_mode = "none"
        beat_loss_value = "n/a"
    elif lambda_beat > 0:
        beat_mode = f"{beat_rep}:cond+lbeat"
        beat_loss_value = _format_loss_value(beat_loss)
    else:
        beat_mode = f"{beat_rep}:cond"
        beat_loss_value = "off"

    return {
        "loss": _format_loss_value(loss),
        "acc_loss": _format_loss_value(acc_loss),
        "beat_mode": beat_mode,
        "beat_loss": beat_loss_value,
    }


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=DEFAULT_NON_BEAT_LEARNING_RATE,
        learning_rate_was_explicit=False,
        weight_decay=0.02,
        use_beats=False,
        beat_rep="distance",
        lambda_acc=0.1,
        lambda_beat=0.5,
        beat_a=10.0,
        beat_c=0.1,
        beat_estimator_ckpt="",
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        resume_training_state=False,
    ):
        configure_cuda_math()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = resolve_runtime_mixed_precision(
            mixed_precision, requested_device
        )
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        state = AcceleratorState()
        num_processes = state.num_processes
        self.num_processes = num_processes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lambda_acc = lambda_acc
        self.lambda_beat = lambda_beat
        self.beat_a = beat_a
        self.beat_c = beat_c
        self.beat_estimator_ckpt = beat_estimator_ckpt
        self.resume_training_state = resume_training_state

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
            self.normalizer = checkpoint["normalizer"]

        resolved = resolve_runtime_training_config(
            feature_type=feature_type,
            use_beats=use_beats,
            beat_rep=beat_rep,
            learning_rate=learning_rate,
            learning_rate_was_explicit=learning_rate_was_explicit,
            checkpoint_config=checkpoint.get("config") if checkpoint is not None else None,
        )
        feature_type = resolved["feature_type"]
        use_beats = resolved["use_beats"]
        beat_rep = resolved["beat_rep"]
        learning_rate = resolved["learning_rate"]
        self.grad_clip_norm = resolved["grad_clip_norm"]

        if checkpoint is not None and not learning_rate_was_explicit:
            checkpoint_uses_beats = checkpoint.get("config", {}).get("use_beats")
            if checkpoint_uses_beats is True:
                print(f"Using checkpoint-inferred beat defaults: learning_rate={learning_rate}")

        self.feature_type = feature_type
        self.use_beats = use_beats
        self.beat_rep = beat_rep
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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

        if checkpoint is not None:
            restore_checkpoint_state(
                self.model,
                self.diffusion,
                self.optim,
                checkpoint,
                use_ema_weights=EMA,
                num_processes=num_processes,
                restore_optimizer=self.resume_training_state,
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        training_recipe = build_checkpoint_config(
            feature_type=self.feature_type,
            use_beats=self.use_beats,
            beat_rep=self.beat_rep,
            batch_size=opt.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_processes=self.num_processes,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            lambda_acc=self.lambda_acc,
            lambda_beat=self.lambda_beat,
        )
        if self.accelerator.is_main_process:
            beat_mode = (
                "none"
                if not self.use_beats
                else f"{self.beat_rep}:cond+lbeat"
                if self.lambda_beat > 0
                else f"{self.beat_rep}:cond"
            )
            print(
                "Training config: "
                f"feature_type={self.feature_type} "
                f"use_beats={self.use_beats} "
                f"beat_mode={beat_mode} "
                f"batch_size={training_recipe['batch_size']} "
                f"gradient_accumulation_steps={training_recipe['gradient_accumulation_steps']} "
                f"effective_batch_size={training_recipe['effective_batch_size']} "
                f"learning_rate={training_recipe['learning_rate']} "
                f"weight_decay={training_recipe['weight_decay']} "
                f"lambda_beat={self.lambda_beat} "
                f"lambda_acc={self.lambda_acc}"
            )
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir,
            tensor_dataset_cache_name("train", self.feature_type, self.use_beats, self.beat_rep),
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir,
            tensor_dataset_cache_name("test", self.feature_type, self.use_beats, self.beat_rep),
        )
        processed_cache_dir = Path(opt.processed_data_dir)
        processed_cache_dir.mkdir(parents=True, exist_ok=True)
        prune_legacy_tensor_dataset_caches(
            processed_cache_dir, "train", self.feature_type, self.use_beats, self.beat_rep
        )
        prune_legacy_tensor_dataset_caches(
            processed_cache_dir, "test", self.feature_type, self.use_beats, self.beat_rep
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            tensor_cache_reused = True
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            tensor_cache_reused = False
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
                atomic_pickle_dump(train_dataset, train_tensor_dataset_path)
                atomic_pickle_dump(test_dataset, test_tensor_dataset_path)

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        pin_memory = self.accelerator.device.type == "cuda"
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            **build_dataloader_kwargs(opt.train_num_workers, pin_memory),
        )
        test_data_loader = build_sample_dataloader(
            test_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.test_num_workers,
            pin_memory=pin_memory,
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
            print(
                "Training config: "
                f"batch_size={opt.batch_size} "
                f"train_num_workers={opt.train_num_workers} "
                f"test_num_workers={opt.test_num_workers} "
                f"mixed_precision={self.mixed_precision} "
                f"tensor_cache_reused={tensor_cache_reused}"
            )

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            epoch_start = time.perf_counter()
            if self.accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.accelerator.device)
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

            optimizer_step_count = 0
            self.optim.zero_grad()
            for step, (x, cond, filename, wavnames) in enumerate(train_loop):
                with self.accelerator.accumulate(self.model):
                    total_loss, (loss, v_loss, fk_loss, foot_loss, acc_loss, beat_loss) = self.diffusion(
                        x, cond, t_override=None
                    )
                    validate_loss_terms(
                        total_loss,
                        (loss, v_loss, fk_loss, foot_loss, acc_loss, beat_loss),
                        filenames=filename,
                        wavnames=wavnames,
                    )
                    self.accelerator.backward(total_loss)
                    optimizer_step_completed = maybe_apply_optimization_step(
                        accelerator=self.accelerator,
                        model=self.model,
                        optim=self.optim,
                        diffusion=self.diffusion,
                        grad_clip_norm=self.grad_clip_norm,
                        global_step=optimizer_step_count,
                        ema_interval=opt.ema_interval,
                    )
                    if optimizer_step_completed:
                        optimizer_step_count += 1
                        self.optim.zero_grad()

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
                            **build_train_postfix(
                                loss=loss,
                                acc_loss=acc_loss,
                                beat_loss=beat_loss,
                                use_beats=self.use_beats,
                                beat_rep=self.beat_rep,
                                lambda_beat=self.lambda_beat,
                            )
                        )
            if self.accelerator.is_main_process:
                epoch_duration = max(time.perf_counter() - epoch_start, 1e-6)
                peak_cuda_memory_mb = (
                    torch.cuda.max_memory_allocated(self.accelerator.device) / (1024 ** 2)
                    if self.accelerator.device.type == "cuda"
                    else 0.0
                )
                print(
                    f"train_epoch={epoch} seconds={epoch_duration:.2f} "
                    f"batches_per_second={len(train_data_loader) / epoch_duration:.2f} "
                    f"samples_per_second={len(train_dataset) / epoch_duration:.2f} "
                    f"peak_cuda_memory_mb={peak_cuda_memory_mb:.2f}"
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
                        "config": training_recipe,
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
