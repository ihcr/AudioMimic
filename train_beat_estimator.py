import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model.beat_estimator import BeatDistanceEstimator
from rotation_transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                 quaternion_multiply,
                                 quaternion_to_axis_angle)
from vis import SMPLSkeleton


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
    def __init__(self, motion_dir, beat_dir, fps=30, seq_len=150):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        motion_path, beat_path = self.samples[idx]
        joints = load_motion_joints(motion_path, fps=self.fps, seq_len=self.seq_len)
        with np.load(beat_path) as beat_data:
            target = torch.from_numpy(beat_data["motion_dist"].astype(np.float32))
        return joints.float(), target.float()


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


def run_epoch(model, dataloader, device, desc, optimizer=None):
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
        for joints, target in batch_loop:
            joints = joints.to(device)
            target = target.to(device)

            pred_dist = model(joints)
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
    output_path="weights/beat_estimator.pt",
    epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=0.0,
    fps=30,
    seq_len=150,
    val_split=0.1,
    split_seed=42,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MotionBeatDataset(motion_dir, beat_dir, fps=fps, seq_len=seq_len)
    train_dataset, val_dataset = split_train_val_dataset(
        dataset, val_split=val_split, split_seed=split_seed
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BeatDistanceEstimator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    config = {
        "fps": fps,
        "seq_len": seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "split_seed": split_seed,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 6,
        "ff_dim": 512,
    }

    best_epoch = None
    best_val_loss = None
    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_dataloader,
            device=device,
            desc=f"Beat estimator train {epoch + 1}/{epochs}",
            optimizer=optimizer,
        )
        if val_dataloader is None:
            best_epoch = epoch + 1
            print(f"epoch={epoch + 1} train_loss={train_loss:.6f}")
            continue

        val_loss = run_epoch(
            model,
            val_dataloader,
            device=device,
            desc=f"Beat estimator val {epoch + 1}/{epochs}",
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
    parser.add_argument("--device", default=None, help="Explicit torch device override.")
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        motion_dir=args.motion_dir,
        beat_dir=args.beat_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fps=args.fps,
        seq_len=args.seq_len,
        val_split=args.val_split,
        split_seed=args.split_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
