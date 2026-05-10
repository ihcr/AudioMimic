import argparse
import glob
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from args import resolve_worker_count
from dataset.dance_dataset import AISTPPDataset
from dataset.motion_representation import (
    G1_MOTION_FORMAT,
    G1_REPR_DIM,
    SMPL_MOTION_FORMAT,
    validate_motion_format,
)
from model.beat_estimator import BeatDistanceEstimator, G1BeatDistanceEstimator
from rotation_transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                 quaternion_multiply,
                                 quaternion_to_axis_angle)
from vis import SMPLSkeleton

BEAT_ESTIMATOR_CACHE_VERSION = "v2"
RAW_DISTANCE_TARGET = "raw_distance"
RELATIVE_DISTANCE_TARGET = "relative_distance"
TARGET_TRANSFORMS = (RAW_DISTANCE_TARGET, RELATIVE_DISTANCE_TARGET)
G1_TARGET_TRANSFORMS = TARGET_TRANSFORMS


def beat_estimator_cache_name(
    motion_dir,
    fps=30,
    seq_len=150,
    target_transform=RAW_DISTANCE_TARGET,
):
    split_name = Path(motion_dir).resolve().parent.name
    return (
        f"beat_estimator_{split_name}_fps{fps}_seq{seq_len}_"
        f"{target_transform}_{BEAT_ESTIMATOR_CACHE_VERSION}.pt"
    )


def default_cache_dir(motion_dir):
    motion_path = Path(motion_dir).resolve()
    if len(motion_path.parents) >= 2:
        return motion_path.parents[1] / "dataset_backups"
    return motion_path.parent / "dataset_backups"


def build_dataloader_kwargs(num_workers, pin_memory):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return kwargs


def transform_beat_targets(motion_dist, motion_spacing, target_transform):
    if target_transform == RAW_DISTANCE_TARGET:
        return motion_dist.float()
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported beat target transform: {target_transform}")
    if motion_spacing is None:
        raise ValueError("relative_distance targets require motion_spacing")
    spacing = torch.clamp(motion_spacing.float(), min=1.0)
    return torch.clamp(motion_dist.float() / spacing, min=0.0, max=1.0)


def transform_g1_beat_targets(motion_dist, motion_spacing, target_transform):
    return transform_beat_targets(motion_dist, motion_spacing, target_transform)


def resolve_runtime_mixed_precision(mixed_precision, device):
    device_obj = torch.device(device)
    return mixed_precision if device_obj.type == "cuda" else "no"


def configure_cuda_math():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def rotate_motion_to_z_up(local_q, root_pos):
    root_q = local_q[:, :1, :]
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.tensor(
        [0.7071068, 0.7071068, 0.0, 0.0],
        dtype=local_q.dtype,
        device=local_q.device,
    ).view(1, 1, 4)
    root_q = quaternion_to_axis_angle(quaternion_multiply(rotation, root_q_quat))
    local_q = local_q.clone()
    local_q[:, :1, :] = root_q

    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(root_pos.clone())
    return local_q, root_pos


def load_motion_joints(motion_path, fps=30, seq_len=150, device="cpu"):
    with open(motion_path, "rb") as handle:
        motion = pickle.load(handle)

    raw_fps = 60
    stride = max(raw_fps // fps, 1)
    root_pos = torch.tensor(motion["pos"], dtype=torch.float32, device=device)[
        ::stride
    ][:seq_len]
    local_q = torch.tensor(motion["q"], dtype=torch.float32, device=device)[
        ::stride
    ][:seq_len].reshape(-1, 24, 3)

    local_q, root_pos = rotate_motion_to_z_up(local_q, root_pos)
    joints = SMPLSkeleton(device=device).forward(
        local_q.unsqueeze(0), root_pos.unsqueeze(0)
    )
    return joints.squeeze(0).cpu()


class MotionBeatDataset(Dataset):
    def __init__(
        self,
        motion_dir,
        beat_dir,
        fps=30,
        seq_len=150,
        cache_dir=None,
        force_rebuild_cache=False,
        target_transform=RAW_DISTANCE_TARGET,
    ):
        if target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"target_transform must be one of {TARGET_TRANSFORMS}")
        self.target_transform = target_transform
        motion_map = {
            Path(path).stem: path for path in sorted(glob.glob(str(Path(motion_dir) / "*.pkl")))
        }
        beat_map = {
            Path(path).stem: path for path in sorted(glob.glob(str(Path(beat_dir) / "*.npz")))
        }
        if set(motion_map) != set(beat_map):
            raise ValueError("motions_sliced and beat_feats must match by basename")

        self.samples = [
            (motion_map[name], beat_map[name]) for name in sorted(motion_map.keys())
        ]
        self.fps = fps
        self.seq_len = seq_len
        cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir(motion_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_root / beat_estimator_cache_name(
            motion_dir,
            fps=fps,
            seq_len=seq_len,
            target_transform=target_transform,
        )

        if not force_rebuild_cache and self.cache_path.is_file():
            payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            self.joints = payload["joints"].float()
            self.targets = payload["targets"].float()
            return

        joints = []
        targets = []
        spacings = []
        for motion_path, beat_path in self.samples:
            joints.append(
                load_motion_joints(motion_path, fps=self.fps, seq_len=self.seq_len)
            )
            with np.load(beat_path) as beat_data:
                targets.append(
                    torch.from_numpy(beat_data["motion_dist"].astype(np.float32))
                )
                spacings.append(
                    torch.from_numpy(beat_data["motion_spacing"].astype(np.float32))
                    if "motion_spacing" in beat_data
                    else None
                )

        if joints:
            self.joints = torch.stack(joints, dim=0).float()
            raw_targets = torch.stack(targets, dim=0).float()
            spacing = (
                None
                if any(item is None for item in spacings)
                else torch.stack(spacings, dim=0).float()
            )
            self.targets = transform_beat_targets(
                raw_targets,
                spacing,
                target_transform,
            )
        else:
            self.joints = torch.empty((0, seq_len, 24, 3), dtype=torch.float32)
            self.targets = torch.empty((0, seq_len), dtype=torch.float32)

        torch.save(
            {
                "joints": self.joints,
                "targets": self.targets,
                "fps": fps,
                "seq_len": seq_len,
                "target_transform": target_transform,
            },
            self.cache_path,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.joints[idx], self.targets[idx]


class G1MotionBeatDataset(Dataset):
    def __init__(
        self,
        data_path,
        backup_path,
        feature_type="jukebox",
        force_reload=False,
        target_transform=RAW_DISTANCE_TARGET,
    ):
        if target_transform not in G1_TARGET_TRANSFORMS:
            raise ValueError(f"target_transform must be one of {G1_TARGET_TRANSFORMS}")
        self.target_transform = target_transform
        self.source_dataset = AISTPPDataset(
            data_path=data_path,
            backup_path=backup_path,
            train=True,
            feature_type=feature_type,
            use_beats=True,
            beat_rep="distance",
            motion_format=G1_MOTION_FORMAT,
            force_reload=force_reload,
        )
        self.pose = self.source_dataset.data["pose"].float()
        self.targets = transform_g1_beat_targets(
            self.source_dataset.data["motion_dist"],
            self.source_dataset.data.get("motion_spacing"),
            target_transform,
        )
        if self.pose.ndim != 3 or self.pose.shape[-1] != G1_REPR_DIM:
            raise ValueError(f"G1 estimator dataset expected pose [N, T, {G1_REPR_DIM}]")
        if self.targets.shape != self.pose.shape[:2]:
            raise ValueError("G1 beat targets must match pose batch and frame dimensions")

    def __len__(self):
        return self.pose.shape[0]

    def __getitem__(self, idx):
        return self.pose[idx], self.targets[idx]


def save_checkpoint(model, path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
        },
        path,
    )


def split_train_val_dataset(dataset, val_split=0.1, split_seed=42):
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in the range [0.0, 1.0).")

    dataset_size = len(dataset)
    if dataset_size < 2 or val_split == 0.0:
        return dataset, None

    val_size = max(1, int(round(dataset_size * val_split)))
    val_size = min(val_size, dataset_size - 1)
    train_size = dataset_size - val_size
    generator = torch.Generator().manual_seed(split_seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(model, dataloader, device, desc, optimizer=None, mixed_precision="no"):
    device_obj = torch.device(device)
    autocast_enabled = mixed_precision == "bf16" and device_obj.type == "cuda"
    if optimizer is None:
        model.eval()
    else:
        model.train()

    epoch_loss = 0.0
    batch_loop = tqdm(
        dataloader,
        desc=desc,
        unit="batch",
    )
    with torch.set_grad_enabled(optimizer is not None):
        for motion, target in batch_loop:
            motion = motion.to(device_obj, non_blocking=autocast_enabled)
            target = target.to(device_obj, non_blocking=autocast_enabled)

            with torch.autocast(
                device_type=device_obj.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                pred_dist = model(motion)
                loss = F.mse_loss(pred_dist, target)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_loop.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / max(len(dataloader), 1)


def train(
    motion_dir,
    beat_dir,
    data_path=None,
    processed_data_dir=None,
    feature_type="jukebox",
    motion_format=SMPL_MOTION_FORMAT,
    output_path="weights/beat_estimator.pt",
    epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=0.0,
    fps=30,
    seq_len=150,
    val_split=0.1,
    split_seed=42,
    num_workers=None,
    mixed_precision="bf16",
    force_rebuild_cache=False,
    beat_target_transform=RAW_DISTANCE_TARGET,
    g1_target_transform=RAW_DISTANCE_TARGET,
    device=None,
):
    motion_format = validate_motion_format(motion_format)
    target_transform = beat_target_transform
    if (
        motion_format == G1_MOTION_FORMAT
        and beat_target_transform == RAW_DISTANCE_TARGET
        and g1_target_transform != RAW_DISTANCE_TARGET
    ):
        target_transform = g1_target_transform
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_math()
    mixed_precision = resolve_runtime_mixed_precision(mixed_precision, device)
    device_obj = torch.device(device)
    resolved_num_workers = (
        num_workers
        if num_workers is not None
        else (resolve_worker_count() if device_obj.type == "cuda" else 0)
    )
    if motion_format == G1_MOTION_FORMAT:
        resolved_data_path = (
            Path(data_path)
            if data_path is not None
            else Path(motion_dir).resolve().parents[1]
        )
        resolved_processed_dir = (
            Path(processed_data_dir)
            if processed_data_dir is not None
            else default_cache_dir(motion_dir)
        )
        dataset = G1MotionBeatDataset(
            data_path=resolved_data_path,
            backup_path=resolved_processed_dir,
            feature_type=feature_type,
            force_reload=force_rebuild_cache,
            target_transform=target_transform,
        )
    else:
        dataset = MotionBeatDataset(
            motion_dir,
            beat_dir,
            fps=fps,
            seq_len=seq_len,
            force_rebuild_cache=force_rebuild_cache,
            target_transform=target_transform,
        )
    train_dataset, val_dataset = split_train_val_dataset(
        dataset, val_split=val_split, split_seed=split_seed
    )
    pin_memory = device_obj.type == "cuda"
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **build_dataloader_kwargs(resolved_num_workers, pin_memory),
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **build_dataloader_kwargs(resolved_num_workers, pin_memory),
        )

    print(
        f"Beat estimator config: batch_size={batch_size} num_workers={resolved_num_workers} "
        f"mixed_precision={mixed_precision} cache_path={getattr(dataset, 'cache_path', '<none>')}"
    )

    if motion_format == G1_MOTION_FORMAT:
        output_activation = (
            "sigmoid" if target_transform == RELATIVE_DISTANCE_TARGET else "softplus"
        )
        model = G1BeatDistanceEstimator(
            input_dim=G1_REPR_DIM,
            output_activation=output_activation,
        ).to(device)
        input_dim = G1_REPR_DIM
    else:
        input_dim = 24 * 3
        output_activation = (
            "sigmoid" if target_transform == RELATIVE_DISTANCE_TARGET else "softplus"
        )
        model = BeatDistanceEstimator(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    config = {
        "motion_format": motion_format,
        "input_dim": input_dim,
        "feature_type": feature_type,
        "fps": fps,
        "seq_len": seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "split_seed": split_seed,
        "num_workers": resolved_num_workers,
        "mixed_precision": mixed_precision,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 6,
        "ff_dim": 512,
        "target_transform": target_transform,
        "output_activation": output_activation,
    }

    best_epoch = None
    best_val_loss = None
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(
            model,
            train_dataloader,
            device=device,
            desc=f"Beat estimator train {epoch + 1}/{epochs}",
            optimizer=optimizer,
            mixed_precision=mixed_precision,
        )
        train_duration = max(time.perf_counter() - epoch_start, 1e-6)
        peak_cuda_memory_mb = (
            torch.cuda.max_memory_allocated(device=torch.device(device)) / (1024 ** 2)
            if torch.device(device).type == "cuda"
            else 0.0
        )
        print(
            f"train_epoch={epoch + 1} seconds={train_duration:.2f} "
            f"batches_per_second={len(train_dataloader) / train_duration:.2f} "
            f"samples_per_second={len(train_dataset) / train_duration:.2f} "
            f"peak_cuda_memory_mb={peak_cuda_memory_mb:.2f}"
        )
        if val_dataloader is None:
            best_epoch = epoch + 1
            print(f"epoch={epoch + 1} train_loss={train_loss:.6f}")
            continue

        val_start = time.perf_counter()
        val_loss = run_epoch(
            model,
            val_dataloader,
            device=device,
            desc=f"Beat estimator val {epoch + 1}/{epochs}",
            mixed_precision=mixed_precision,
        )
        val_duration = max(time.perf_counter() - val_start, 1e-6)
        print(
            f"val_epoch={epoch + 1} seconds={val_duration:.2f} "
            f"batches_per_second={len(val_dataloader) / val_duration:.2f} "
            f"samples_per_second={len(val_dataset) / val_duration:.2f}"
        )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if best_val_loss is None or val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            save_checkpoint(
                model,
                output_path,
                {
                    **config,
                    "best_epoch": best_epoch,
                    "best_train_loss": train_loss,
                    "best_val_loss": best_val_loss,
                },
            )

    if val_dataloader is None:
        save_checkpoint(
            model,
            output_path,
            {
                **config,
                "best_epoch": best_epoch,
                "best_train_loss": train_loss,
                "best_val_loss": None,
            },
        )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the beat distance estimator.")
    parser.add_argument("--motion_dir", required=True, help="Path to motions_sliced directory.")
    parser.add_argument("--beat_dir", required=True, help="Path to beat_feats directory.")
    parser.add_argument("--motion_format", choices=("smpl", "g1"), default="smpl")
    parser.add_argument(
        "--data_path",
        default=None,
        help="Prepared dataset root; required for exact G1 normalization if motion_dir is not canonical.",
    )
    parser.add_argument(
        "--processed_data_dir",
        default=None,
        help="Processed dataset cache root used by AISTPPDataset for G1 estimator inputs.",
    )
    parser.add_argument(
        "--feature_type",
        choices=("baseline", "jukebox"),
        default="jukebox",
    )
    parser.add_argument(
        "--output_path",
        default="weights/beat_estimator.pt",
        help="Checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "bf16"),
        default="bf16",
    )
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument(
        "--beat_target_transform",
        choices=TARGET_TRANSFORMS,
        default=RAW_DISTANCE_TARGET,
    )
    parser.add_argument(
        "--g1_target_transform",
        choices=G1_TARGET_TRANSFORMS,
        default=RAW_DISTANCE_TARGET,
    )
    parser.add_argument("--device", default=None, help="Explicit torch device override.")
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        motion_dir=args.motion_dir,
        beat_dir=args.beat_dir,
        data_path=args.data_path,
        processed_data_dir=args.processed_data_dir,
        feature_type=args.feature_type,
        motion_format=args.motion_format,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fps=args.fps,
        seq_len=args.seq_len,
        val_split=args.val_split,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        force_rebuild_cache=args.force_rebuild_cache,
        beat_target_transform=args.beat_target_transform,
        g1_target_transform=args.g1_target_transform,
        device=args.device,
    )


if __name__ == "__main__":
    main()
