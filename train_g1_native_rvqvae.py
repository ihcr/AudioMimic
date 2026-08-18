import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.g1_native_rvqvae_dataset import G1NativeRVQVAEDataset
from dataset.motion_representation import G1_YAW_DELTA_MOTION_FORMAT, motion_repr_dim
from model.g1_native_rvqvae import (
    G1NativeRVQVAE,
    NATIVE_RVQVAE_VARIANTS,
    compute_g1_native_rvqvae_losses,
    loss_weights_for_objective_family,
)
from model.g1_torch_kinematics import G1TorchKinematics


EXPERIMENT_ID = "EXP-20260702-v6f-a-native-g1-rvqvae-tokenizer"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default=EXPERIMENT_ID, type=str)
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats", type=str)
    parser.add_argument(
        "--processed_data_dir",
        default="data/finedance_g1_v6f_native_rvqvae_dataset_backups",
        type=str,
    )
    parser.add_argument("--exp_name", default="", type=str)
    parser.add_argument("--project", default="runs/train", type=str)
    parser.add_argument("--motion_format", default=G1_YAW_DELTA_MOTION_FORMAT, type=str)
    parser.add_argument("--objective_family", choices=NATIVE_RVQVAE_VARIANTS, default="R0_minimal_rvqvae")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--code_dim", default=256, type=int)
    parser.add_argument("--temporal_downsample", choices=(2, 4), default=2, type=int)
    parser.add_argument("--num_codebooks", default=4, type=int)
    parser.add_argument("--codebook_size", default=512, type=int)
    parser.add_argument("--commitment_weight", default=0.25, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--quantizer_type", choices=("rvq", "fsq"), default="rvq")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--decoder_extra_blocks", default=0, type=int)
    parser.add_argument("--decoder_attention", action="store_true")
    parser.add_argument("--fsq_dim", default=16, type=int)
    parser.add_argument("--fsq_levels", default=8, type=int)
    parser.add_argument("--state_conditioning", action="store_true")
    parser.add_argument("--state_context_dropout", default=0.0, type=float)
    parser.add_argument("--streaming_loss_weight", default=0.0, type=float)
    parser.add_argument("--stream_token_horizon", default=8, type=int)
    parser.add_argument("--stream_commit_tokens", default=8, type=int)
    parser.add_argument("--stream_prefix_consistency_weight", default=0.0, type=float)
    parser.add_argument("--streaming_robot_loss_scale", default=0.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.02, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--data_len", default=0, type=int)
    parser.add_argument("--eval_data_len", default=0, type=int)
    parser.add_argument("--cache_limit_per_split", default=0, type=int)
    parser.add_argument("--cache_batch_size", default=512, type=int)
    parser.add_argument("--cache_device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--eval_interval", default=25, type=int)
    parser.add_argument("--eval_max_clips", default=512, type=int)
    parser.add_argument("--full_eval_interval", default=0, type=int)
    parser.add_argument("--full_eval_epochs", default="100,300,500,1000,1500,2000", type=str)
    parser.add_argument("--full_eval_render_epochs", default="500,1000,2000", type=str)
    parser.add_argument("--full_eval_max_clips", default=0, type=int)
    parser.add_argument("--full_eval_diagnostic_count", default=8, type=int)
    parser.add_argument("--full_eval_render_count", default=0, type=int)
    parser.add_argument("--full_eval_metric_workers", default=8, type=int)
    parser.add_argument("--full_eval_fail_policy", choices=("warn", "raise"), default="warn")
    parser.add_argument("--lambda_motion", default=1.0, type=float)
    parser.add_argument("--lambda_velocity", default=0.5, type=float)
    parser.add_argument("--lambda_acceleration", default=0.1, type=float)
    parser.add_argument("--lambda_fk", default=0.5, type=float)
    parser.add_argument("--lambda_contact_bce", default=0.1, type=float)
    parser.add_argument("--lambda_contact_height", default=0.2, type=float)
    parser.add_argument("--lambda_contact_slide", default=0.1, type=float)
    parser.add_argument("--lambda_rvq", default=1.0, type=float)
    parser.add_argument("--lambda_amplitude", default=0.2, type=float)
    parser.add_argument("--amplitude_ratio_floor", default=0.85, type=float)
    parser.add_argument("--amplitude_ratio_ceiling", default=1.5, type=float)
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--g1_mujoco_gl", default="egl", type=str)
    parser.add_argument("--wandb_pj_name", default="Musics2Dance", type=str)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def save_checkpoint_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            usage = shutil.disk_usage(path.parent)
            raise RuntimeError(
                f"failed to save checkpoint to {path}; free={usage.free} total={usage.total}"
            ) from exc


def parse_epoch_list(value):
    epochs = set()
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            epochs.add(int(item))
    return epochs


def should_run_full_eval(args, epoch):
    explicit_epochs = parse_epoch_list(args.full_eval_epochs)
    if epoch in explicit_epochs:
        return True
    return args.full_eval_interval > 0 and epoch % args.full_eval_interval == 0


def build_model(args):
    return G1NativeRVQVAE(
        input_dim=motion_repr_dim(args.motion_format),
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        temporal_downsample=args.temporal_downsample,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
        dropout=args.dropout,
        motion_format=args.motion_format,
        quantizer_type=args.quantizer_type,
        use_attention=args.use_attention,
        decoder_extra_blocks=args.decoder_extra_blocks,
        decoder_attention=args.decoder_attention,
        fsq_dim=args.fsq_dim,
        fsq_levels=args.fsq_levels,
        state_conditioning=args.state_conditioning,
        state_context_dropout=args.state_context_dropout,
    )


def move_batch_to_device(batch, device):
    moved = {
        "motion": batch["motion"].to(device, non_blocking=True).float(),
        "contact": batch["contact"].to(device, non_blocking=True).float(),
        "ground": batch["ground"].to(device, non_blocking=True).float(),
    }
    for key in ("target_fk_keypoints", "target_fk_feet"):
        if key in batch:
            moved[key] = batch[key].to(device, non_blocking=True).float()
    return moved


def average_stats(total, stats, batch_size):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    total["_count"] = total.get("_count", 0) + batch_size


def finalize_stats(total, prefix=""):
    count = max(int(total.pop("_count", 0)), 1)
    return {f"{prefix}{key}": value / count for key, value in sorted(total.items())}


def build_loss_weights(args):
    return loss_weights_for_objective_family(
        args.objective_family,
        motion=args.lambda_motion,
        velocity=args.lambda_velocity,
        acceleration=args.lambda_acceleration,
        fk=args.lambda_fk,
        contact_bce=args.lambda_contact_bce,
        contact_height=args.lambda_contact_height,
        contact_slide=args.lambda_contact_slide,
        rvq=args.lambda_rvq,
        amplitude=args.lambda_amplitude,
        amplitude_ratio_floor=args.amplitude_ratio_floor,
        amplitude_ratio_ceiling=args.amplitude_ratio_ceiling,
    )


def build_streaming_boundary_context(motion, frame_start):
    frame_start = int(frame_start)
    if frame_start <= 0:
        previous = motion[:, 0]
        velocity = torch.zeros_like(previous)
    else:
        previous = motion[:, frame_start - 1]
        if frame_start >= 2:
            velocity = previous - motion[:, frame_start - 2]
        else:
            velocity = torch.zeros_like(previous)
    return torch.cat((previous, velocity), dim=-1)


def _slice_tensor(value, start, length):
    if value is None:
        return None
    return value[:, start : start + length]


def _chunk_loss_output(recon, contact_logits, latent, zero):
    return {
        "recon": recon,
        "contact_logits": contact_logits,
        "latent": latent,
        "rvq_loss": zero,
        "codebook_loss": zero,
        "commitment_loss": zero,
        "quantization_error_by_layer": recon.new_empty(0),
        "final_residual_mse": zero,
        "mu": None,
        "logvar": None,
    }


def compute_streaming_decode_loss(model, output, moved, mean, std, kinematics, args, weights):
    if args.streaming_loss_weight <= 0.0:
        zero = output["recon"].sum() * 0.0
        return zero, {}
    latent = output["latent"]
    token_count = int(latent.shape[1])
    token_horizon = min(max(int(args.stream_token_horizon), 1), token_count)
    commit_tokens = min(max(int(args.stream_commit_tokens), 1), token_horizon)
    if token_count <= 0:
        zero = output["recon"].sum() * 0.0
        return zero, {}

    max_start = max(token_count - token_horizon, 0)
    token_start = random.randint(0, max_start) if max_start > 0 else 0
    frame_start = token_start * int(model.temporal_downsample)
    plan_frames = min(token_horizon * int(model.temporal_downsample), moved["motion"].shape[1] - frame_start)
    commit_frames = min(commit_tokens * int(model.temporal_downsample), plan_frames)
    if commit_frames <= 0:
        zero = output["recon"].sum() * 0.0
        return zero, {}

    state_context = build_streaming_boundary_context(moved["motion"], frame_start)
    plan_latent = latent[:, token_start : token_start + token_horizon]
    plan_recon, plan_contact = model.decode(
        plan_latent,
        state_context=state_context,
        target_frames=plan_frames,
    )
    plan_recon = plan_recon[:, :commit_frames]
    plan_contact = plan_contact[:, :commit_frames]
    target_motion = moved["motion"][:, frame_start : frame_start + commit_frames]
    target_contact = moved["contact"][:, frame_start : frame_start + commit_frames]
    zero = plan_recon.sum() * 0.0
    stream_weights = type(weights)(
        motion=weights.motion,
        velocity=weights.velocity,
        acceleration=weights.acceleration,
        fk=weights.fk * float(args.streaming_robot_loss_scale),
        contact_bce=weights.contact_bce,
        contact_height=weights.contact_height * float(args.streaming_robot_loss_scale),
        contact_slide=weights.contact_slide * float(args.streaming_robot_loss_scale),
        rvq=0.0,
        amplitude=weights.amplitude,
        amplitude_ratio_floor=weights.amplitude_ratio_floor,
        amplitude_ratio_ceiling=weights.amplitude_ratio_ceiling,
    )
    chunk_output = _chunk_loss_output(plan_recon, plan_contact, plan_latent, zero)
    chunk_loss, chunk_stats = compute_g1_native_rvqvae_losses(
        chunk_output,
        target_motion,
        target_contact,
        mean,
        std,
        kinematics=kinematics,
        motion_format=args.motion_format,
        weights=stream_weights,
        ground=moved["ground"],
        target_fk_keypoints=_slice_tensor(moved.get("target_fk_keypoints"), frame_start, commit_frames),
        target_fk_feet=_slice_tensor(moved.get("target_fk_feet"), frame_start, commit_frames),
    )

    consistency_loss = zero
    if args.stream_prefix_consistency_weight > 0.0 and commit_tokens < token_horizon:
        commit_latent = latent[:, token_start : token_start + commit_tokens]
        commit_recon, _ = model.decode(
            commit_latent,
            state_context=state_context,
            target_frames=commit_frames,
        )
        consistency_loss = F.mse_loss(commit_recon, plan_recon.detach())
        if commit_frames > 1:
            consistency_loss = consistency_loss + F.mse_loss(
                commit_recon[:, 1:] - commit_recon[:, :-1],
                (plan_recon[:, 1:] - plan_recon[:, :-1]).detach(),
            )

    total = float(args.streaming_loss_weight) * chunk_loss
    total = total + float(args.stream_prefix_consistency_weight) * consistency_loss
    stats = {
        f"stream/{key}": value for key, value in chunk_stats.items()
    }
    stats.update(
        {
            "stream/loss/weighted_total": total.detach(),
            "stream/loss/prefix_consistency": consistency_loss.detach(),
            "stream/token_start": torch.as_tensor(float(token_start), device=plan_recon.device),
            "stream/token_horizon": torch.as_tensor(float(token_horizon), device=plan_recon.device),
            "stream/commit_tokens": torch.as_tensor(float(commit_tokens), device=plan_recon.device),
            "stream/robot_loss_scale": torch.as_tensor(
                float(args.streaming_robot_loss_scale),
                device=plan_recon.device,
            ),
        }
    )
    return total, stats


@torch.inference_mode()
def evaluate_loss(model, loader, device, mean, std, kinematics, args, max_clips=0):
    model.eval()
    totals = {}
    emitted = 0
    weights = build_loss_weights(args)
    for batch in loader:
        batch_size = int(batch["motion"].shape[0])
        if max_clips and emitted >= max_clips:
            break
        if max_clips and emitted + batch_size > max_clips:
            keep = max_clips - emitted
            batch = {
                key: value[:keep] if torch.is_tensor(value) else value[:keep]
                for key, value in batch.items()
            }
            batch_size = keep
        emitted += batch_size
        moved = move_batch_to_device(batch, device)
        output = model(moved["motion"])
        _, stats = compute_g1_native_rvqvae_losses(
            output,
            moved["motion"],
            moved["contact"],
            mean,
            std,
            kinematics=kinematics,
            motion_format=args.motion_format,
            weights=weights,
            ground=moved["ground"],
            target_fk_keypoints=moved.get("target_fk_keypoints"),
            target_fk_feet=moved.get("target_fk_feet"),
        )
        average_stats(totals, stats, batch_size)
    model.train()
    return finalize_stats(totals, prefix="val/")


def load_checkpoint_if_requested(args, model, optimizer, scaler, device):
    if not args.checkpoint:
        return 0
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if args.resume_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0))


def init_wandb(args, run_dir):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        if args.wandb_mode == "disabled":
            return None
        raise
    return wandb.init(
        project=args.wandb_pj_name,
        name=args.exp_name,
        dir=str(run_dir),
        mode=args.wandb_mode,
        config=vars(args),
    )


def run_full_eval_subprocess(args, checkpoint_path, epoch):
    if not should_run_full_eval(args, epoch):
        return None
    render_epochs = parse_epoch_list(args.full_eval_render_epochs)
    render_count = args.full_eval_render_count if epoch in render_epochs else 0
    output_dir = Path("eval") / args.exp_name / f"ckpt{epoch:04d}_{args.objective_family}"
    command = [
        sys.executable,
        "-m",
        "eval.run_g1_native_rvqvae_eval",
        "--checkpoint",
        str(checkpoint_path),
        "--data_path",
        args.data_path,
        "--processed_data_dir",
        args.processed_data_dir,
        "--output_dir",
        str(output_dir),
        "--motion_format",
        args.motion_format,
        "--batch_size",
        str(args.batch_size),
        "--max_eval_clips",
        str(args.full_eval_max_clips),
        "--diagnostic_count",
        str(args.full_eval_diagnostic_count),
        "--render_count",
        str(render_count),
        "--metric_workers",
        str(args.full_eval_metric_workers),
        "--enable_fk_metrics",
        "--cache_limit_per_split",
        str(args.cache_limit_per_split),
        "--g1_fk_model_path",
        args.g1_fk_model_path,
        "--g1_root_quat_order",
        args.g1_root_quat_order,
    ]
    print("Launching full native G1 RVQ-VAE eval:", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
        return {"full_eval/status": 1.0, "full_eval/epoch": float(epoch)}
    except subprocess.CalledProcessError as exc:
        message = (
            f"Full native G1 RVQ-VAE eval failed at epoch {epoch} with return code "
            f"{exc.returncode}. Rerun command: {' '.join(command)}"
        )
        print(message, flush=True)
        if args.full_eval_fail_policy == "raise":
            raise
        return {"full_eval/status": 0.0, "full_eval/epoch": float(epoch)}


def main():
    args = parse_args()
    args.experiment_id = str(args.experiment_id)
    args.model_type = "g1_native_rvqvae"
    args.variant = args.objective_family
    if not args.exp_name:
        args.exp_name = f"{args.experiment_id}_{args.objective_family}"

    os.environ.setdefault("MUJOCO_GL", args.g1_mujoco_gl)
    os.environ.setdefault("PYOPENGL_PLATFORM", args.g1_mujoco_gl)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_device = device.type if args.cache_device == "auto" else args.cache_device
    run_dir = Path(args.project) / args.exp_name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    dataset_kwargs = dict(
        data_path=args.data_path,
        backup_path=args.processed_data_dir,
        motion_format=args.motion_format,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        cache_batch_size=args.cache_batch_size,
        cache_device=cache_device,
        cache_limit_per_split=args.cache_limit_per_split,
    )
    train_dataset = G1NativeRVQVAEDataset(
        split="train",
        rebuild_cache=args.rebuild_cache,
        data_len=args.data_len,
        **dataset_kwargs,
    )
    test_dataset = G1NativeRVQVAEDataset(
        split="test",
        rebuild_cache=False,
        data_len=args.eval_data_len,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        drop_last=False,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = device.type == "cuda" and args.mixed_precision != "no"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    scaler = GradScaler("cuda", enabled=amp_enabled and args.mixed_precision == "fp16")
    start_epoch = load_checkpoint_if_requested(args, model, optimizer, scaler, device)

    mean, std = train_dataset.normalizer.tensors(device=device)
    kinematics = G1TorchKinematics(
        args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
    ).to(device)
    wandb_run = init_wandb(args, run_dir)

    print(
        f"V6f-A native G1 RVQ-VAE training: {args.exp_name} "
        f"objective={args.objective_family} code_dim={args.code_dim} "
        f"quantizer={args.quantizer_type} downsample={args.temporal_downsample} "
        f"codebooks={args.num_codebooks} size={args.codebook_size} "
        f"fsq_dim={args.fsq_dim} fsq_levels={args.fsq_levels} "
        f"train={len(train_dataset)} test={len(test_dataset)} "
        f"device={device} start_epoch={start_epoch}",
        flush=True,
    )
    started_at = time.time()
    weights = build_loss_weights(args)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.time()
        totals = {}
        model.train()
        for batch in tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            unit="batch",
            disable=args.disable_progress_bar,
        ):
            moved = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
                output = model(moved["motion"])
                loss, stats = compute_g1_native_rvqvae_losses(
                    output,
                    moved["motion"],
                    moved["contact"],
                    mean,
                    std,
                    kinematics=kinematics,
                    motion_format=args.motion_format,
                    weights=weights,
                    ground=moved["ground"],
                    target_fk_keypoints=moved.get("target_fk_keypoints"),
                    target_fk_feet=moved.get("target_fk_feet"),
                )
                stream_loss, stream_stats = compute_streaming_decode_loss(
                    model,
                    output,
                    moved,
                    mean,
                    std,
                    kinematics,
                    args,
                    weights,
                )
                loss = loss + stream_loss
                stats.update(stream_stats)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            average_stats(totals, stats, int(moved["motion"].shape[0]))

        train_stats = finalize_stats(totals, prefix="train/")
        elapsed = time.time() - epoch_started
        epochs_done = epoch - start_epoch
        avg_epoch = (time.time() - started_at) / max(epochs_done, 1)
        eta = avg_epoch * max(args.epochs - epoch, 0)
        clips_per_second = len(train_dataset) / max(elapsed, 1e-8)
        log_payload = {
            "epoch": epoch,
            "progress/epoch": epoch,
            "progress/epochs_total": args.epochs,
            "progress/percent": epoch / max(args.epochs, 1) * 100.0,
            "progress/eta_seconds": eta,
            "throughput/clips_per_second": clips_per_second,
            "time/epoch_seconds": elapsed,
            **train_stats,
        }
        if args.eval_interval > 0 and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            val_stats = evaluate_loss(
                model,
                test_loader,
                device,
                mean,
                std,
                kinematics,
                args,
                max_clips=args.eval_max_clips,
            )
            log_payload.update(val_stats)

        should_save = (
            (args.save_interval > 0 and epoch % args.save_interval == 0)
            or epoch == args.epochs
            or should_run_full_eval(args, epoch)
        )
        if should_save:
            checkpoint_path = weights_dir / f"train-{epoch}.pt"
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "config": vars(args),
                    "normalizer": train_dataset.normalizer.state_dict(),
                    "metadata": train_dataset.metadata,
                },
                checkpoint_path,
            )
            log_payload["checkpoint/epoch"] = epoch
            log_payload["checkpoint/path"] = str(checkpoint_path)
            full_eval_payload = run_full_eval_subprocess(args, checkpoint_path, epoch)
            if full_eval_payload:
                log_payload.update(full_eval_payload)

        status = (
            f"epoch {epoch}/{args.epochs} "
            f"loss={log_payload.get('train/loss/total', 0.0):.6f} "
            f"val={log_payload.get('val/loss/total', float('nan')):.6f} "
            f"time={format_duration(elapsed)} "
            f"eta={format_duration(eta)} "
            f"clips/s={clips_per_second:.1f}"
        )
        print(status, flush=True)
        if wandb_run is not None:
            wandb_run.log(log_payload, step=epoch)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
