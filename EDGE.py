import os
import pickle
import time
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.motion_representation import (
    G1_MOTION_FORMAT,
    SMPL_MOTION_FORMAT,
    motion_repr_dim,
    validate_motion_format,
)
from dataset.preprocess import increment_path
from model.adan import Adan
from model.beat_estimator import BeatDistanceEstimator, G1BeatDistanceEstimator
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


def tensor_dataset_cache_name(
    split,
    feature_type,
    use_beats,
    beat_rep,
    motion_format=SMPL_MOTION_FORMAT,
    feature_cache_mode="off",
    feature_cache_dtype="float32",
):
    validate_motion_format(motion_format)
    beat_tag = "beat" if use_beats else "nobeat"
    feature_cache_tag = (
        ""
        if feature_cache_mode == "off"
        else f"featcache_{feature_cache_mode}_{feature_cache_dtype}_"
    )
    if motion_format == SMPL_MOTION_FORMAT:
        return (
            f"{split}_tensor_dataset_{feature_type}_{beat_tag}_{beat_rep}_"
            f"{feature_cache_tag}{TENSOR_DATASET_CACHE_VERSION}.pkl"
        )
    return (
        f"{split}_tensor_dataset_{motion_format}_{feature_type}_{beat_tag}_{beat_rep}_"
        f"{feature_cache_tag}{TENSOR_DATASET_CACHE_VERSION}.pkl"
    )


def prune_legacy_tensor_dataset_caches(
    processed_data_dir,
    split,
    feature_type,
    use_beats,
    beat_rep,
    motion_format=SMPL_MOTION_FORMAT,
    feature_cache_mode="off",
    feature_cache_dtype="float32",
):
    validate_motion_format(motion_format)
    processed_data_dir = Path(processed_data_dir)
    beat_tag = "beat" if use_beats else "nobeat"
    current_name = tensor_dataset_cache_name(
        split,
        feature_type,
        use_beats,
        beat_rep,
        motion_format=motion_format,
        feature_cache_mode=feature_cache_mode,
        feature_cache_dtype=feature_cache_dtype,
    )
    if motion_format == SMPL_MOTION_FORMAT:
        pattern = f"{split}_tensor_dataset_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
    else:
        pattern = f"{split}_tensor_dataset_{motion_format}_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
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
    "motion_format": SMPL_MOTION_FORMAT,
}
FEATURE_DIMS = {
    "baseline": 35,
    "baseline34": 34,
    "jukebox": 4800,
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


def resolve_model_config(
    feature_type,
    use_beats,
    beat_rep,
    motion_format=SMPL_MOTION_FORMAT,
    checkpoint_config=None,
):
    motion_format = validate_motion_format(motion_format)
    resolved = {
        "feature_type": feature_type,
        "use_beats": use_beats,
        "beat_rep": beat_rep,
        "motion_format": motion_format,
    }
    if checkpoint_config is None:
        return resolved

    for key in ("feature_type", "use_beats", "beat_rep", "motion_format"):
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
    beat_loss_start_epoch=0,
    beat_loss_warmup_epochs=0,
    beat_loss_max_fraction=0.0,
    beat_loss_cap_mode="hard",
    beat_estimator_ckpt="",
    beat_estimator_config=None,
    motion_format=SMPL_MOTION_FORMAT,
    repr_dim=None,
    feature_cache_mode="off",
    feature_cache_dtype="float32",
    lambda_g1_fk=0.0,
    lambda_g1_fk_vel=0.0,
    lambda_g1_fk_acc=0.0,
    lambda_g1_foot=0.0,
    lambda_g1_kin=1.0,
    g1_kin_loss_warmup_epochs=0,
    g1_kin_loss_max_fraction=0.0,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
    epoch_offset=0,
):
    motion_format = validate_motion_format(motion_format)
    return {
        "feature_type": feature_type,
        "use_beats": use_beats,
        "beat_rep": beat_rep,
        "motion_format": motion_format,
        "repr_dim": motion_repr_dim(motion_format) if repr_dim is None else repr_dim,
        "feature_cache_mode": feature_cache_mode,
        "feature_cache_dtype": feature_cache_dtype,
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
        "beat_loss_start_epoch": beat_loss_start_epoch,
        "beat_loss_warmup_epochs": beat_loss_warmup_epochs,
        "beat_loss_max_fraction": beat_loss_max_fraction,
        "beat_loss_cap_mode": beat_loss_cap_mode,
        "beat_estimator_ckpt": beat_estimator_ckpt,
        "beat_estimator_config": beat_estimator_config or {},
        "lambda_g1_fk": lambda_g1_fk,
        "lambda_g1_fk_vel": lambda_g1_fk_vel,
        "lambda_g1_fk_acc": lambda_g1_fk_acc,
        "lambda_g1_foot": lambda_g1_foot,
        "lambda_g1_kin": lambda_g1_kin,
        "g1_kin_loss_warmup_epochs": g1_kin_loss_warmup_epochs,
        "g1_kin_loss_max_fraction": g1_kin_loss_max_fraction,
        "g1_fk_model_path": g1_fk_model_path,
        "g1_root_quat_order": g1_root_quat_order,
        "epoch_offset": epoch_offset,
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
    motion_format=SMPL_MOTION_FORMAT,
    learning_rate_was_explicit=False,
    checkpoint_config=None,
):
    resolved = resolve_model_config(
        feature_type=feature_type,
        use_beats=use_beats,
        beat_rep=beat_rep,
        motion_format=motion_format,
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


def validate_beat_estimator_config(
    config,
    max_val_loss=8.0,
    expected_motion_format=SMPL_MOTION_FORMAT,
):
    expected_motion_format = validate_motion_format(expected_motion_format)
    checkpoint_motion_format = validate_motion_format(
        config.get("motion_format", SMPL_MOTION_FORMAT)
    )
    if checkpoint_motion_format != expected_motion_format:
        raise ValueError(
            "Beat estimator checkpoint motion_format "
            f"{checkpoint_motion_format!r} does not match training motion_format "
            f"{expected_motion_format!r}."
        )
    if "best_val_loss" not in config:
        raise ValueError("Beat estimator checkpoint is missing validation loss.")
    try:
        best_val_loss = float(config["best_val_loss"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Beat estimator checkpoint is missing validation loss.") from exc
    if not math.isfinite(best_val_loss):
        raise ValueError("Beat estimator validation loss must be finite.")
    if best_val_loss > max_val_loss:
        raise ValueError(
            f"Beat estimator validation loss {best_val_loss} exceeds limit {max_val_loss}."
        )
    return {
        "motion_format": checkpoint_motion_format,
        "best_val_loss": best_val_loss,
        "best_epoch": config.get("best_epoch"),
        "val_split": config.get("val_split"),
        "max_val_loss": max_val_loss,
        "target_transform": config.get("target_transform", "raw_distance"),
        "output_activation": config.get("output_activation", "softplus"),
    }


def build_beat_estimator_from_checkpoint(
    checkpoint_path,
    device,
    max_val_loss=8.0,
    expected_motion_format=SMPL_MOTION_FORMAT,
):
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    checkpoint_summary = validate_beat_estimator_config(
        config,
        max_val_loss=max_val_loss,
        expected_motion_format=expected_motion_format,
    )
    model_kwargs = {
        key: config[key]
        for key in (
            "input_dim",
            "hidden_dim",
            "num_heads",
            "num_layers",
            "ff_dim",
            "dropout",
            "output_activation",
        )
        if key in config
    }
    if checkpoint_summary["motion_format"] != G1_MOTION_FORMAT:
        model_kwargs.pop("output_activation", None)
    if checkpoint_summary["motion_format"] == G1_MOTION_FORMAT:
        model = G1BeatDistanceEstimator(**model_kwargs)
    else:
        model = BeatDistanceEstimator(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.checkpoint_config_summary = checkpoint_summary
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


def build_train_postfix(
    loss,
    acc_loss,
    beat_loss,
    use_beats,
    beat_rep,
    lambda_beat,
    beat_weight=None,
    beat_contribution=None,
    beat_capped=None,
    g1_stats=None,
):
    if not use_beats:
        beat_mode = "none"
        beat_loss_value = "n/a"
    elif lambda_beat > 0:
        beat_mode = f"{beat_rep}:cond+lbeat"
        beat_loss_value = _format_loss_value(beat_loss)
    else:
        beat_mode = f"{beat_rep}:cond"
        beat_loss_value = "off"

    postfix = {
        "loss": _format_loss_value(loss),
        "acc_loss": _format_loss_value(acc_loss),
        "beat_mode": beat_mode,
        "beat_loss": beat_loss_value,
    }
    if lambda_beat > 0 and beat_weight is not None:
        postfix["beat_weight"] = f"{float(beat_weight):.4f}"
    if lambda_beat > 0 and beat_contribution is not None:
        postfix["beat_contrib"] = _format_loss_value(beat_contribution)
    if lambda_beat > 0 and beat_capped is not None:
        capped_value = bool(
            beat_capped.detach().cpu().item() if torch.is_tensor(beat_capped) else beat_capped
        )
        postfix["beat_capped"] = "yes" if capped_value else "no"
    if g1_stats:
        postfix["g1_fk"] = _format_loss_value(g1_stats.get("g1_fk_loss", 0.0))
        postfix["g1_fk_vel"] = _format_loss_value(g1_stats.get("g1_fk_vel_loss", 0.0))
        postfix["g1_fk_acc"] = _format_loss_value(g1_stats.get("g1_fk_acc_loss", 0.0))
        postfix["g1_foot"] = _format_loss_value(g1_stats.get("g1_foot_loss", 0.0))
        postfix["g1_kin"] = _format_loss_value(g1_stats.get("g1_kin_contribution", 0.0))
        if "g1_kin_capped" in g1_stats:
            capped_value = g1_stats["g1_kin_capped"]
            capped_value = bool(
                capped_value.detach().cpu().item()
                if torch.is_tensor(capped_value)
                else capped_value
            )
            postfix["g1_cap"] = "yes" if capped_value else "no"
    return postfix


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
        beat_estimator_max_val_loss=8.0,
        beat_loss_start_epoch=0,
        beat_loss_warmup_epochs=0,
        beat_loss_max_fraction=0.0,
        beat_loss_cap_mode="hard",
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        resume_training_state=False,
        motion_format=SMPL_MOTION_FORMAT,
        lambda_g1_fk=0.0,
        lambda_g1_fk_vel=0.0,
        lambda_g1_fk_acc=0.0,
        lambda_g1_foot=0.0,
        lambda_g1_kin=1.0,
        g1_kin_loss_warmup_epochs=0,
        g1_kin_loss_max_fraction=0.0,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
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
        self.beat_estimator_max_val_loss = beat_estimator_max_val_loss
        self.beat_loss_start_epoch = beat_loss_start_epoch
        self.beat_loss_warmup_epochs = beat_loss_warmup_epochs
        self.beat_loss_max_fraction = beat_loss_max_fraction
        self.beat_loss_cap_mode = beat_loss_cap_mode
        self.beat_estimator_config = {}
        self.resume_training_state = resume_training_state
        self.motion_format = validate_motion_format(motion_format)
        self.lambda_g1_fk = lambda_g1_fk
        self.lambda_g1_fk_vel = lambda_g1_fk_vel
        self.lambda_g1_fk_acc = lambda_g1_fk_acc
        self.lambda_g1_foot = lambda_g1_foot
        self.lambda_g1_kin = lambda_g1_kin
        self.g1_kin_loss_warmup_epochs = g1_kin_loss_warmup_epochs
        self.g1_kin_loss_max_fraction = g1_kin_loss_max_fraction
        self.g1_fk_model_path = g1_fk_model_path
        self.g1_root_quat_order = g1_root_quat_order

        self.repr_dim = repr_dim = motion_repr_dim(self.motion_format)

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        self.normalizer = None
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
            motion_format=self.motion_format,
            learning_rate=learning_rate,
            learning_rate_was_explicit=learning_rate_was_explicit,
            checkpoint_config=checkpoint.get("config") if checkpoint is not None else None,
        )
        feature_type = resolved["feature_type"]
        use_beats = resolved["use_beats"]
        beat_rep = resolved["beat_rep"]
        self.motion_format = resolved["motion_format"]
        repr_dim = self.repr_dim = motion_repr_dim(self.motion_format)
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
        if self.feature_type not in FEATURE_DIMS:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        feature_dim = FEATURE_DIMS[self.feature_type]

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

        smpl = (
            None
            if self.motion_format == G1_MOTION_FORMAT
            else SMPLSkeleton(self.accelerator.device)
        )
        beat_estimator = None
        if self.use_beats and self.lambda_beat > 0:
            if not self.beat_estimator_ckpt:
                raise ValueError("Beat-enabled training with lambda_beat > 0 requires --beat_estimator_ckpt")
            beat_estimator = build_beat_estimator_from_checkpoint(
                self.beat_estimator_ckpt,
                self.accelerator.device,
                max_val_loss=self.beat_estimator_max_val_loss,
                expected_motion_format=self.motion_format,
            )
            self.beat_estimator_config = getattr(
                beat_estimator, "checkpoint_config_summary", {}
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
            beat_loss_start_epoch=self.beat_loss_start_epoch,
            beat_loss_warmup_epochs=self.beat_loss_warmup_epochs,
            beat_loss_max_fraction=self.beat_loss_max_fraction,
            beat_loss_cap_mode=self.beat_loss_cap_mode,
            motion_format=self.motion_format,
            normalizer=self.normalizer,
            lambda_g1_fk=self.lambda_g1_fk,
            lambda_g1_fk_vel=self.lambda_g1_fk_vel,
            lambda_g1_fk_acc=self.lambda_g1_fk_acc,
            lambda_g1_foot=self.lambda_g1_foot,
            lambda_g1_kin=self.lambda_g1_kin,
            g1_kin_loss_warmup_epochs=self.g1_kin_loss_warmup_epochs,
            g1_kin_loss_max_fraction=self.g1_kin_loss_max_fraction,
            g1_fk_model_path=self.g1_fk_model_path,
            g1_root_quat_order=self.g1_root_quat_order,
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
        feature_cache_mode = getattr(opt, "feature_cache_mode", "off")
        feature_cache_dtype = getattr(opt, "feature_cache_dtype", "float32")
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
            beat_loss_start_epoch=self.beat_loss_start_epoch,
            beat_loss_warmup_epochs=self.beat_loss_warmup_epochs,
            beat_loss_max_fraction=self.beat_loss_max_fraction,
            beat_loss_cap_mode=self.beat_loss_cap_mode,
            beat_estimator_ckpt=self.beat_estimator_ckpt,
            beat_estimator_config=self.beat_estimator_config,
            motion_format=self.motion_format,
            repr_dim=self.repr_dim,
            feature_cache_mode=feature_cache_mode,
            feature_cache_dtype=feature_cache_dtype,
            lambda_g1_fk=self.lambda_g1_fk,
            lambda_g1_fk_vel=self.lambda_g1_fk_vel,
            lambda_g1_fk_acc=self.lambda_g1_fk_acc,
            lambda_g1_foot=self.lambda_g1_foot,
            lambda_g1_kin=self.lambda_g1_kin,
            g1_kin_loss_warmup_epochs=self.g1_kin_loss_warmup_epochs,
            g1_kin_loss_max_fraction=self.g1_kin_loss_max_fraction,
            g1_fk_model_path=self.g1_fk_model_path,
            g1_root_quat_order=self.g1_root_quat_order,
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
                f"motion_format={self.motion_format} "
                f"use_beats={self.use_beats} "
                f"beat_mode={beat_mode} "
                f"batch_size={training_recipe['batch_size']} "
                f"gradient_accumulation_steps={training_recipe['gradient_accumulation_steps']} "
                f"effective_batch_size={training_recipe['effective_batch_size']} "
                f"learning_rate={training_recipe['learning_rate']} "
                f"weight_decay={training_recipe['weight_decay']} "
                f"lambda_beat={self.lambda_beat} "
                f"lambda_acc={self.lambda_acc} "
                f"lambda_g1_fk={self.lambda_g1_fk} "
                f"lambda_g1_fk_vel={self.lambda_g1_fk_vel} "
                f"lambda_g1_fk_acc={self.lambda_g1_fk_acc} "
                f"lambda_g1_foot={self.lambda_g1_foot} "
                f"lambda_g1_kin={self.lambda_g1_kin}"
            )
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir,
            tensor_dataset_cache_name(
                "train",
                self.feature_type,
                self.use_beats,
                self.beat_rep,
                motion_format=self.motion_format,
                feature_cache_mode=feature_cache_mode,
                feature_cache_dtype=feature_cache_dtype,
            ),
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir,
            tensor_dataset_cache_name(
                "test",
                self.feature_type,
                self.use_beats,
                self.beat_rep,
                motion_format=self.motion_format,
                feature_cache_mode=feature_cache_mode,
                feature_cache_dtype=feature_cache_dtype,
            ),
        )
        processed_cache_dir = Path(opt.processed_data_dir)
        processed_cache_dir.mkdir(parents=True, exist_ok=True)
        prune_legacy_tensor_dataset_caches(
            processed_cache_dir,
            "train",
            self.feature_type,
            self.use_beats,
            self.beat_rep,
            motion_format=self.motion_format,
            feature_cache_mode=feature_cache_mode,
            feature_cache_dtype=feature_cache_dtype,
        )
        prune_legacy_tensor_dataset_caches(
            processed_cache_dir,
            "test",
            self.feature_type,
            self.use_beats,
            self.beat_rep,
            motion_format=self.motion_format,
            feature_cache_mode=feature_cache_mode,
            feature_cache_dtype=feature_cache_dtype,
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
                motion_format=self.motion_format,
                feature_cache_mode=feature_cache_mode,
                feature_cache_dtype=feature_cache_dtype,
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
                motion_format=self.motion_format,
                feature_cache_mode=feature_cache_mode,
                feature_cache_dtype=feature_cache_dtype,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                atomic_pickle_dump(train_dataset, train_tensor_dataset_path)
                atomic_pickle_dump(test_dataset, test_tensor_dataset_path)

        # set normalizer
        self.normalizer = test_dataset.normalizer
        self.diffusion.set_normalizer(self.normalizer)

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
                f"feature_cache_mode={feature_cache_mode} "
                f"feature_cache_dtype={feature_cache_dtype} "
                f"tensor_cache_reused={tensor_cache_reused}"
            )

        epoch_offset = int(getattr(opt, "epoch_offset", 0))
        if epoch_offset < 0:
            raise ValueError("--epoch_offset must be non-negative.")

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            global_epoch = epoch_offset + epoch
            self.diffusion.set_training_epoch(global_epoch)
            epoch_start = time.perf_counter()
            if self.accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.accelerator.device)
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            avg_accloss = 0
            avg_beatloss = 0
            avg_beatcontrib = 0
            avg_beatcap_hits = 0
            avg_g1_fk_vel_loss = 0
            avg_g1_fk_acc_loss = 0
            avg_g1_kin_contrib = 0
            avg_g1_kin_cap_hits = 0
            # train
            self.train()
            train_loop = train_data_loader
            if self.accelerator.is_main_process:
                train_loop = tqdm(
                    train_data_loader,
                    total=len(train_data_loader),
                    position=1,
                    desc=(
                        f"Train {global_epoch}/{epoch_offset + opt.epochs}"
                        if epoch_offset
                        else f"Train {epoch}/{opt.epochs}"
                    ),
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
                    beat_stats = getattr(self.diffusion, "last_beat_loss_stats", {})
                    beat_contribution = beat_stats.get(
                        "beat_contribution",
                        torch.zeros((), device=beat_loss.device),
                    )
                    beat_capped = beat_stats.get(
                        "beat_capped",
                        torch.zeros((), dtype=torch.bool, device=beat_loss.device),
                    )
                    avg_beatcontrib += float(beat_contribution.detach().cpu())
                    avg_beatcap_hits += float(beat_capped.detach().cpu())
                    g1_stats = getattr(self.diffusion, "last_g1_kin_loss_stats", {})
                    avg_g1_fk_vel_loss += float(
                        g1_stats.get("g1_fk_vel_loss", torch.zeros(())).detach().cpu()
                        if torch.is_tensor(g1_stats.get("g1_fk_vel_loss", None))
                        else g1_stats.get("g1_fk_vel_loss", 0.0)
                    )
                    avg_g1_fk_acc_loss += float(
                        g1_stats.get("g1_fk_acc_loss", torch.zeros(())).detach().cpu()
                        if torch.is_tensor(g1_stats.get("g1_fk_acc_loss", None))
                        else g1_stats.get("g1_fk_acc_loss", 0.0)
                    )
                    avg_g1_kin_contrib += float(
                        g1_stats.get("g1_kin_contribution", torch.zeros(())).detach().cpu()
                        if torch.is_tensor(g1_stats.get("g1_kin_contribution", None))
                        else g1_stats.get("g1_kin_contribution", 0.0)
                    )
                    avg_g1_kin_cap_hits += float(
                        g1_stats.get("g1_kin_capped", torch.zeros(())).detach().cpu()
                        if torch.is_tensor(g1_stats.get("g1_kin_capped", None))
                        else g1_stats.get("g1_kin_capped", 0.0)
                    )
                    if hasattr(train_loop, "set_postfix"):
                        train_loop.set_postfix(
                            **build_train_postfix(
                                loss=loss,
                                acc_loss=acc_loss,
                                beat_loss=beat_loss,
                                use_beats=self.use_beats,
                                beat_rep=self.beat_rep,
                                lambda_beat=self.lambda_beat,
                                beat_weight=beat_stats.get("effective_lambda_beat"),
                                beat_contribution=beat_contribution,
                                beat_capped=beat_capped,
                                g1_stats=g1_stats,
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
                    f"train_epoch={global_epoch} seconds={epoch_duration:.2f} "
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
                    avg_beatcontrib /= len(train_data_loader)
                    avg_beatcap_hits /= len(train_data_loader)
                    avg_g1_fk_vel_loss /= len(train_data_loader)
                    avg_g1_fk_acc_loss /= len(train_data_loader)
                    avg_g1_kin_contrib /= len(train_data_loader)
                    avg_g1_kin_cap_hits /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                        "Acc Loss": avg_accloss,
                        "Beat Loss": avg_beatloss,
                        "Beat Weight": self.diffusion.last_beat_loss_stats.get(
                            "effective_lambda_beat", 0.0
                        ),
                        "Beat Contribution": avg_beatcontrib,
                        "Beat Cap Hit Rate": avg_beatcap_hits,
                        "G1 FK Vel Loss": avg_g1_fk_vel_loss,
                        "G1 FK Acc Loss": avg_g1_fk_acc_loss,
                        "G1 Kin Contribution": avg_g1_kin_contrib,
                        "G1 Kin Cap Hit Rate": avg_g1_kin_cap_hits,
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
                    torch.save(ckpt, os.path.join(wdir, f"train-{global_epoch}.pt"))
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
                        global_epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {global_epoch}]")
        if self.accelerator.is_main_process and wandb_run is not None:
            wandb_run.finish()

    def render_sample(
        self,
        data_tuple,
        label,
        render_dir,
        render_count=-1,
        fk_out=None,
        render=True,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
        g1_render_backend="mujoco",
        g1_render_width=960,
        g1_render_height=720,
        g1_mujoco_gl="egl",
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
            render=render,
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
            g1_render_backend=g1_render_backend,
            g1_render_width=g1_render_width,
            g1_render_height=g1_render_height,
            g1_mujoco_gl=g1_mujoco_gl,
        )
