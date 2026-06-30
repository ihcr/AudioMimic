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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.g1_latent_beat_dataset import G1MusicControlLatentDataset
from dataset.motion_representation import G1_YAW_DELTA_MOTION_FORMAT
from model.g1_latent_diffusion import G1LatentDiffusion, G1MusicControlLatentDenoiser


EXPERIMENT_ID = "EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion"
DEFAULT_PRIOR_CHECKPOINT = (
    "runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/"
    "weights/train-500.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats", type=str)
    parser.add_argument(
        "--motion_prior_processed_data_dir",
        default="data/finedance_g1_v6b_motion_prior_dataset_backups",
        type=str,
    )
    parser.add_argument(
        "--latent_processed_data_dir",
        default="data/finedance_g1_v6bc_music_control_latent_dataset_backups",
        type=str,
    )
    parser.add_argument("--prior_checkpoint", default=DEFAULT_PRIOR_CHECKPOINT, type=str)
    parser.add_argument("--exp_name", default=f"{EXPERIMENT_ID}_r01_control_only", type=str)
    parser.add_argument("--project", default="runs/train", type=str)
    parser.add_argument("--motion_format", default=G1_YAW_DELTA_MOTION_FORMAT, type=str)
    parser.add_argument("--latent_dim", default=128, type=int)
    parser.add_argument("--latent_frames", default=75, type=int)
    parser.add_argument("--beat_frames", default=150, type=int)
    parser.add_argument("--use_wav2clip_semantic", action="store_true")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--ff_size", default=1024, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--diffusion_steps", default=1000, type=int)
    parser.add_argument("--beta_start", default=1e-4, type=float)
    parser.add_argument("--beta_end", default=0.02, type=float)
    parser.add_argument("--cond_drop_prob", default=0.1, type=float)
    parser.add_argument("--control_rank_weight", default=0.10, type=float)
    parser.add_argument("--control_rank_margin", default=0.02, type=float)
    parser.add_argument("--semantic_rank_weight", default=0.03, type=float)
    parser.add_argument("--semantic_rank_margin", default=0.01, type=float)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.02, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--data_len", default=0, type=int)
    parser.add_argument("--eval_data_len", default=0, type=int)
    parser.add_argument("--cache_limit_per_split", default=0, type=int)
    parser.add_argument("--cache_batch_size", default=1024, type=int)
    parser.add_argument("--cache_device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--eval_interval", default=25, type=int)
    parser.add_argument("--eval_max_clips", default=1024, type=int)
    parser.add_argument("--full_eval_interval", default=500, type=int)
    parser.add_argument("--full_eval_max_clips", default=0, type=int)
    parser.add_argument("--full_eval_sampling_steps", default=50, type=int)
    parser.add_argument("--full_eval_guidance_weight", default=1.0, type=float)
    parser.add_argument("--full_eval_diagnostic_count", default=8, type=int)
    parser.add_argument("--full_eval_metric_workers", default=8, type=int)
    parser.add_argument("--full_eval_fail_policy", choices=("warn", "raise"), default="warn")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--g1_mujoco_gl", default="egl", type=str)
    parser.add_argument("--wandb_pj_name", default="Musics2Dance", type=str)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="offline")
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


def average_stats(total, stats, batch_size):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    total["_count"] = total.get("_count", 0) + batch_size


def finalize_stats(total, prefix=""):
    count = max(int(total.pop("_count", 0)), 1)
    return {f"{prefix}{key}": value / count for key, value in sorted(total.items())}


def build_diffusion(args):
    denoiser = G1MusicControlLatentDenoiser(
        latent_dim=args.latent_dim,
        control_dim=8,
        semantic_dim=512,
        latent_frames=args.latent_frames,
        control_frames=args.beat_frames,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        use_wav2clip_semantic=args.use_wav2clip_semantic,
    )
    return G1LatentDiffusion(
        denoiser,
        timesteps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        cond_drop_prob=args.cond_drop_prob,
    )


def move_batch_to_device(batch, device):
    moved = {
        "latent": batch["latent"].to(device, non_blocking=True).float(),
        "control_features": batch["control_features"].to(device, non_blocking=True).float(),
    }
    if "semantic_features" in batch:
        moved["semantic_features"] = batch["semantic_features"].to(device, non_blocking=True).float()
    return moved


def loss_kwargs(args):
    return {
        "control_rank_weight": args.control_rank_weight,
        "control_rank_margin": args.control_rank_margin,
        "semantic_rank_weight": args.semantic_rank_weight if args.use_wav2clip_semantic else 0.0,
        "semantic_rank_margin": args.semantic_rank_margin,
    }


@torch.inference_mode()
def evaluate_loss(diffusion, loader, device, loss_config, max_clips=0):
    diffusion.eval()
    totals = {}
    emitted = 0
    for batch in loader:
        batch_size = int(batch["latent"].shape[0])
        if max_clips and emitted >= max_clips:
            break
        if max_clips and emitted + batch_size > max_clips:
            keep = max_clips - emitted
            batch = {
                key: value[:keep] if torch.is_tensor(value) else value[:keep]
                for key, value in batch.items()
            }
            batch_size = keep
        moved = move_batch_to_device(batch, device)
        loss, stats = diffusion.p_losses(
            moved["latent"],
            moved["control_features"],
            semantic_features=moved.get("semantic_features"),
            **loss_config,
        )
        average_stats(totals, stats, batch_size)
        emitted += batch_size
    diffusion.train()
    return finalize_stats(totals, prefix="val/")


def load_checkpoint_if_requested(args, diffusion, optimizer, scaler, device):
    if not args.checkpoint:
        return 0
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    diffusion.load_state_dict(checkpoint["diffusion"])
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
    if args.full_eval_interval <= 0 or epoch % args.full_eval_interval != 0:
        return None
    output_dir = Path("eval") / args.exp_name / f"ckpt{epoch:04d}_v6bc_variants"
    command = [
        sys.executable,
        "-m",
        "eval.run_g1_latent_diffusion_eval",
        "--checkpoint",
        str(checkpoint_path),
        "--prior_checkpoint",
        args.prior_checkpoint,
        "--data_path",
        args.data_path,
        "--motion_prior_processed_data_dir",
        args.motion_prior_processed_data_dir,
        "--latent_processed_data_dir",
        args.latent_processed_data_dir,
        "--output_dir",
        str(output_dir),
        "--motion_format",
        args.motion_format,
        "--batch_size",
        str(args.batch_size),
        "--max_eval_clips",
        str(args.full_eval_max_clips),
        "--sampling_steps",
        str(args.full_eval_sampling_steps),
        "--guidance_weight",
        str(args.full_eval_guidance_weight),
        "--diagnostic_count",
        str(args.full_eval_diagnostic_count),
        "--metric_workers",
        str(args.full_eval_metric_workers),
        "--enable_fk_metrics",
        "--cache_limit_per_split",
        str(args.cache_limit_per_split),
        "--g1_fk_model_path",
        args.g1_fk_model_path,
        "--g1_root_quat_order",
        args.g1_root_quat_order,
        "--resume_existing_metrics",
    ]
    print("Launching full latent eval:", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
        return {"full_eval/status": 1.0, "full_eval/epoch": float(epoch)}
    except subprocess.CalledProcessError as exc:
        message = (
            f"Full latent eval failed at epoch {epoch} with return code {exc.returncode}. "
            f"Rerun command: {' '.join(command)}"
        )
        print(message, flush=True)
        if args.full_eval_fail_policy == "raise":
            raise
        return {"full_eval/status": 0.0, "full_eval/epoch": float(epoch)}


def main():
    args = parse_args()
    args.experiment_id = EXPERIMENT_ID
    args.model_type = "g1_music_control_latent"
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

    train_dataset = G1MusicControlLatentDataset(
        data_path=args.data_path,
        motion_prior_processed_data_dir=args.motion_prior_processed_data_dir,
        latent_processed_data_dir=args.latent_processed_data_dir,
        prior_checkpoint=args.prior_checkpoint,
        split="train",
        use_wav2clip_semantic=args.use_wav2clip_semantic,
        motion_format=args.motion_format,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        cache_batch_size=args.cache_batch_size,
        cache_device=cache_device,
        cache_limit_per_split=args.cache_limit_per_split,
        rebuild_cache=args.rebuild_cache,
        data_len=args.data_len,
    )
    test_dataset = G1MusicControlLatentDataset(
        data_path=args.data_path,
        motion_prior_processed_data_dir=args.motion_prior_processed_data_dir,
        latent_processed_data_dir=args.latent_processed_data_dir,
        prior_checkpoint=args.prior_checkpoint,
        split="test",
        use_wav2clip_semantic=args.use_wav2clip_semantic,
        motion_format=args.motion_format,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        cache_batch_size=args.cache_batch_size,
        cache_device=cache_device,
        cache_limit_per_split=args.cache_limit_per_split,
        rebuild_cache=False,
        data_len=args.eval_data_len,
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

    diffusion = build_diffusion(args).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = device.type == "cuda" and args.mixed_precision != "no"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    scaler = GradScaler("cuda", enabled=amp_enabled and args.mixed_precision == "fp16")
    start_epoch = load_checkpoint_if_requested(args, diffusion, optimizer, scaler, device)
    wandb_run = init_wandb(args, run_dir)

    print(
        f"V6b-C music-control latent diffusion training: {args.exp_name} "
        f"use_wav2clip_semantic={args.use_wav2clip_semantic} "
        f"train={len(train_dataset)} test={len(test_dataset)} device={device} "
        f"start_epoch={start_epoch}",
        flush=True,
    )
    started_at = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.time()
        totals = {}
        diffusion.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch"):
            moved = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
                loss, stats = diffusion.p_losses(
                    moved["latent"],
                    moved["control_features"],
                    semantic_features=moved.get("semantic_features"),
                    **loss_kwargs(args),
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            average_stats(totals, stats, int(moved["latent"].shape[0]))

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
            log_payload.update(
                evaluate_loss(
                    diffusion,
                    test_loader,
                    device,
                    loss_kwargs(args),
                    max_clips=args.eval_max_clips,
                )
            )

        checkpoint_path = None
        should_save = (
            (args.save_interval > 0 and epoch % args.save_interval == 0)
            or epoch == args.epochs
            or (args.full_eval_interval > 0 and epoch % args.full_eval_interval == 0)
        )
        if should_save:
            checkpoint_path = weights_dir / f"train-{epoch}.pt"
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "diffusion": diffusion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "config": vars(args),
                    "latent_normalizer": train_dataset.normalizer.state_dict(),
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
