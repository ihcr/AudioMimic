import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.motion_representation import (
    G1_YAW_DELTA_MOTION_FORMAT,
    decode_g1_motion,
    encode_g1_motion_for_format,
    motion_repr_dim,
    validate_motion_format,
)
from model.g1_torch_kinematics import G1TorchKinematics


MOTION_PRIOR_CACHE_VERSION = "motion_prior_v1"
DEFAULT_FRAMES = 150
DEFAULT_FPS = 30.0
DEFAULT_CONTACT_HEIGHT_MARGIN = 0.03
DEFAULT_CONTACT_VERTICAL_SPEED_THRESHOLD = 0.2
DEFAULT_NEAR_SUPPORT_MARGIN = 0.08
NORMALIZER_EPS = 1e-6


@dataclass(frozen=True)
class G1MotionPriorNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize_np(self, motion):
        return (motion - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)

    def unnormalize_np(self, motion):
        return motion * self.std.reshape(1, 1, -1) + self.mean.reshape(1, 1, -1)

    def tensors(self, device=None, dtype=torch.float32):
        mean = torch.as_tensor(self.mean, device=device, dtype=dtype).view(1, 1, -1)
        std = torch.as_tensor(self.std, device=device, dtype=dtype).view(1, 1, -1)
        return mean, std

    def state_dict(self):
        return {
            "mean": np.asarray(self.mean, dtype=np.float32),
            "std": np.asarray(self.std, dtype=np.float32),
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


def _cache_tag(motion_format, cache_limit_per_split=0):
    tag = f"{motion_format}_{MOTION_PRIOR_CACHE_VERSION}"
    if cache_limit_per_split:
        tag = f"{tag}_limit{int(cache_limit_per_split)}"
    return tag


def motion_prior_cache_paths(backup_path, motion_format, cache_limit_per_split=0):
    backup_path = Path(backup_path)
    tag = _cache_tag(motion_format, cache_limit_per_split=cache_limit_per_split)
    return {
        "train": backup_path / f"train_motion_prior_{tag}.pkl",
        "test": backup_path / f"test_motion_prior_{tag}.pkl",
        "normalizer": backup_path / f"normalizer_motion_prior_{tag}.pkl",
        "metadata": backup_path / f"metadata_motion_prior_{tag}.json",
    }


def _split_motion_paths(data_path, split, cache_limit_per_split=0):
    paths = sorted((Path(data_path) / split / "motions_sliced").glob("*.pkl"))
    if cache_limit_per_split:
        paths = paths[: int(cache_limit_per_split)]
    return paths


def _load_g1_motion_payload(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    root_pos = payload.get("root_pos", payload.get("pos"))
    root_rot = payload.get("root_rot")
    dof_pos = payload.get("dof_pos")
    if root_rot is None or dof_pos is None:
        q = np.asarray(payload["q"], dtype=np.float32)
        root_rot = q[:, :4]
        dof_pos = q[:, 4:33]
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot = np.asarray(root_rot, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    if root_pos.shape != (DEFAULT_FRAMES, 3):
        raise ValueError(f"{path} root_pos expected {(DEFAULT_FRAMES, 3)}, got {root_pos.shape}")
    if root_rot.shape != (DEFAULT_FRAMES, 4):
        raise ValueError(f"{path} root_rot expected {(DEFAULT_FRAMES, 4)}, got {root_rot.shape}")
    if dof_pos.shape != (DEFAULT_FRAMES, 29):
        raise ValueError(f"{path} dof_pos expected {(DEFAULT_FRAMES, 29)}, got {dof_pos.shape}")
    if not np.isfinite(root_pos).all() or not np.isfinite(root_rot).all() or not np.isfinite(dof_pos).all():
        raise ValueError(f"{path} contains non-finite G1 motion values")
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "fps": float(payload.get("fps", DEFAULT_FPS) or DEFAULT_FPS),
        "audio_path": payload.get("audio_path", ""),
        "designated_beat_frames": payload.get("designated_beat_frames"),
    }


def encode_g1_motion_payload(payload, motion_format=G1_YAW_DELTA_MOTION_FORMAT):
    motion_format = validate_motion_format(motion_format)
    encoded = encode_g1_motion_for_format(
        torch.from_numpy(payload["root_pos"]).unsqueeze(0),
        torch.from_numpy(payload["root_rot"]).unsqueeze(0),
        torch.from_numpy(payload["dof_pos"]).unsqueeze(0),
        motion_format=motion_format,
    )
    encoded = encoded.squeeze(0).detach().cpu().numpy().astype(np.float32)
    expected = (DEFAULT_FRAMES, motion_repr_dim(motion_format))
    if encoded.shape != expected:
        raise ValueError(f"encoded motion expected {expected}, got {encoded.shape}")
    if not np.isfinite(encoded).all():
        raise ValueError("encoded motion contains non-finite values")
    return encoded


def _support_contact_from_feet(
    feet,
    fps=DEFAULT_FPS,
    contact_height_margin=DEFAULT_CONTACT_HEIGHT_MARGIN,
    contact_vertical_speed_threshold=DEFAULT_CONTACT_VERTICAL_SPEED_THRESHOLD,
    near_support_margin=DEFAULT_NEAR_SUPPORT_MARGIN,
):
    feet = np.asarray(feet, dtype=np.float32)
    support_contact = np.zeros(feet.shape[:3], dtype=np.float32)
    near_support = np.zeros(feet.shape[:3], dtype=np.float32)
    lowest_foot_heights = feet[:, :, :, 2].astype(np.float32)
    ground = np.zeros((feet.shape[0],), dtype=np.float32)
    for index in range(feet.shape[0]):
        heights = lowest_foot_heights[index]
        sample_ground = float(np.percentile(heights, 1.0))
        velocity = np.zeros_like(feet[index], dtype=np.float32)
        if feet.shape[1] > 1:
            velocity[1:] = (feet[index, 1:] - feet[index, :-1]) * float(fps)
        vertical_speed = np.abs(velocity[:, :, 2])
        support_contact[index] = (
            (heights <= sample_ground + float(contact_height_margin))
            & (vertical_speed < float(contact_vertical_speed_threshold))
        ).astype(np.float32)
        near_support[index] = (heights <= sample_ground + float(near_support_margin)).astype(np.float32)
        ground[index] = sample_ground
    return {
        "support_contact": support_contact,
        "near_support": near_support,
        "lowest_foot_heights": lowest_foot_heights,
        "ground": ground,
    }


@torch.inference_mode()
def compute_contact_labels_for_encoded_motions(
    encoded_motions,
    kinematics,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    batch_size=256,
    device="cpu",
    contact_height_margin=DEFAULT_CONTACT_HEIGHT_MARGIN,
    contact_vertical_speed_threshold=DEFAULT_CONTACT_VERTICAL_SPEED_THRESHOLD,
    near_support_margin=DEFAULT_NEAR_SUPPORT_MARGIN,
):
    motion_format = validate_motion_format(motion_format)
    encoded_motions = np.asarray(encoded_motions, dtype=np.float32)
    labels = {
        "support_contact": [],
        "near_support": [],
        "lowest_foot_heights": [],
        "ground": [],
    }
    kinematics = kinematics.to(device)
    for start in tqdm(
        range(0, encoded_motions.shape[0], int(batch_size)),
        desc="Build G1 prior contact labels",
        unit="batch",
    ):
        batch = torch.from_numpy(encoded_motions[start : start + int(batch_size)]).to(device)
        decoded = decode_g1_motion(batch, motion_format=motion_format)
        fk = kinematics(
            decoded["root_pos"],
            decoded["root_rot"],
            decoded["dof_pos"],
        )
        contact_payload = _support_contact_from_feet(
            fk["feet"].detach().cpu().numpy(),
            contact_height_margin=contact_height_margin,
            contact_vertical_speed_threshold=contact_vertical_speed_threshold,
            near_support_margin=near_support_margin,
        )
        for key in labels:
            labels[key].append(contact_payload[key])
    return {
        key: np.concatenate(parts, axis=0).astype(np.float32)
        for key, parts in labels.items()
    }


def _encode_split(data_path, split, motion_format, cache_limit_per_split=0):
    paths = _split_motion_paths(
        data_path,
        split,
        cache_limit_per_split=cache_limit_per_split,
    )
    if not paths:
        raise FileNotFoundError(f"No {split} motions found under {Path(data_path) / split / 'motions_sliced'}")
    encoded = []
    stems = []
    source_paths = []
    for path in tqdm(paths, desc=f"Encode G1 prior {split}", unit="clip"):
        payload = _load_g1_motion_payload(path)
        encoded.append(encode_g1_motion_payload(payload, motion_format=motion_format))
        stems.append(path.stem)
        source_paths.append(str(path))
    return {
        "motion_raw": np.stack(encoded, axis=0).astype(np.float32),
        "stems": stems,
        "source_paths": source_paths,
    }


def _write_pickle(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _metadata_matches(metadata, motion_format, cache_limit_per_split):
    return (
        metadata.get("cache_version") == MOTION_PRIOR_CACHE_VERSION
        and metadata.get("motion_format") == motion_format
        and int(metadata.get("frames", 0)) == DEFAULT_FRAMES
        and int(metadata.get("repr_dim", 0)) == motion_repr_dim(motion_format)
        and int(metadata.get("cache_limit_per_split", 0)) == int(cache_limit_per_split)
    )


def build_g1_motion_prior_cache(
    data_path,
    backup_path,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
    batch_size=256,
    device="cpu",
    cache_limit_per_split=0,
):
    motion_format = validate_motion_format(motion_format)
    paths = motion_prior_cache_paths(
        backup_path,
        motion_format,
        cache_limit_per_split=cache_limit_per_split,
    )
    backup_path = Path(backup_path)
    backup_path.mkdir(parents=True, exist_ok=True)

    split_payloads = {
        split: _encode_split(
            data_path,
            split,
            motion_format,
            cache_limit_per_split=cache_limit_per_split,
        )
        for split in ("train", "test")
    }
    train_motion = split_payloads["train"]["motion_raw"]
    mean = train_motion.reshape(-1, train_motion.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_motion.reshape(-1, train_motion.shape[-1]).std(axis=0).astype(np.float32)
    std = np.maximum(std, NORMALIZER_EPS).astype(np.float32)
    normalizer = G1MotionPriorNormalizer(mean=mean, std=std)

    kinematics = G1TorchKinematics(g1_fk_model_path, root_quat_order=g1_root_quat_order)
    for split, payload in split_payloads.items():
        contacts = compute_contact_labels_for_encoded_motions(
            payload["motion_raw"],
            kinematics,
            motion_format=motion_format,
            batch_size=batch_size,
            device=device,
        )
        normalized = normalizer.normalize_np(payload["motion_raw"]).astype(np.float32)
        split_cache = {
            "motion": normalized,
            "contact": contacts["support_contact"],
            "near_support": contacts["near_support"],
            "lowest_foot_heights": contacts["lowest_foot_heights"],
            "ground": contacts["ground"],
            "source_paths": payload["source_paths"],
            "stems": payload["stems"],
        }
        _write_pickle(split_cache, paths[split])

    metadata = {
        "cache_version": MOTION_PRIOR_CACHE_VERSION,
        "data_path": str(data_path),
        "motion_format": motion_format,
        "frames": DEFAULT_FRAMES,
        "repr_dim": motion_repr_dim(motion_format),
        "train_count": int(split_payloads["train"]["motion_raw"].shape[0]),
        "test_count": int(split_payloads["test"]["motion_raw"].shape[0]),
        "cache_limit_per_split": int(cache_limit_per_split),
        "normalizer_eps": NORMALIZER_EPS,
        "contact": {
            "height_margin": DEFAULT_CONTACT_HEIGHT_MARGIN,
            "vertical_speed_threshold": DEFAULT_CONTACT_VERTICAL_SPEED_THRESHOLD,
            "near_support_margin": DEFAULT_NEAR_SUPPORT_MARGIN,
            "fps": DEFAULT_FPS,
            "source": "decoded_g1_yaw_delta_fk_feet",
        },
        "g1_fk_model_path": str(g1_fk_model_path),
        "g1_root_quat_order": g1_root_quat_order,
    }
    _write_pickle(normalizer.state_dict(), paths["normalizer"])
    _write_json(metadata, paths["metadata"])
    return paths


def ensure_g1_motion_prior_cache(
    data_path,
    backup_path,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
    batch_size=256,
    device="cpu",
    cache_limit_per_split=0,
    rebuild_cache=False,
):
    motion_format = validate_motion_format(motion_format)
    paths = motion_prior_cache_paths(
        backup_path,
        motion_format,
        cache_limit_per_split=cache_limit_per_split,
    )
    expected_files = [paths["train"], paths["test"], paths["normalizer"], paths["metadata"]]
    needs_build = rebuild_cache or any(not path.is_file() for path in expected_files)
    if not needs_build:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if not _metadata_matches(metadata, motion_format, cache_limit_per_split):
            raise ValueError(
                "G1 motion prior cache metadata does not match the requested configuration. "
                f"Delete {Path(backup_path)} or rerun with --rebuild_cache."
            )
        return paths
    return build_g1_motion_prior_cache(
        data_path=data_path,
        backup_path=backup_path,
        motion_format=motion_format,
        g1_fk_model_path=g1_fk_model_path,
        g1_root_quat_order=g1_root_quat_order,
        batch_size=batch_size,
        device=device,
        cache_limit_per_split=cache_limit_per_split,
    )


class G1MotionPriorDataset(Dataset):
    def __init__(
        self,
        data_path,
        backup_path,
        split,
        motion_format=G1_YAW_DELTA_MOTION_FORMAT,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
        cache_batch_size=256,
        cache_device="cpu",
        cache_limit_per_split=0,
        rebuild_cache=False,
        data_len=0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.motion_format = validate_motion_format(motion_format)
        paths = ensure_g1_motion_prior_cache(
            data_path=data_path,
            backup_path=backup_path,
            motion_format=self.motion_format,
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
            batch_size=cache_batch_size,
            device=cache_device,
            cache_limit_per_split=cache_limit_per_split,
            rebuild_cache=rebuild_cache,
        )
        with open(paths[split], "rb") as handle:
            payload = pickle.load(handle)
        with open(paths["normalizer"], "rb") as handle:
            self.normalizer = G1MotionPriorNormalizer.from_state_dict(pickle.load(handle))
        self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        count = int(data_len) if data_len else len(payload["stems"])
        self.motion = np.asarray(payload["motion"][:count], dtype=np.float32)
        self.contact = np.asarray(payload["contact"][:count], dtype=np.float32)
        self.near_support = np.asarray(payload["near_support"][:count], dtype=np.float32)
        self.lowest_foot_heights = np.asarray(payload["lowest_foot_heights"][:count], dtype=np.float32)
        self.ground = np.asarray(payload["ground"][:count], dtype=np.float32)
        self.source_paths = list(payload["source_paths"][:count])
        self.stems = list(payload["stems"][:count])
        if self.motion.shape[1:] != (DEFAULT_FRAMES, motion_repr_dim(self.motion_format)):
            raise ValueError(
                f"cached motion expected {(DEFAULT_FRAMES, motion_repr_dim(self.motion_format))}, "
                f"got {self.motion.shape[1:]}"
            )

    def __len__(self):
        return int(self.motion.shape[0])

    def __getitem__(self, index):
        return {
            "motion": torch.from_numpy(self.motion[index]),
            "contact": torch.from_numpy(self.contact[index]),
            "near_support": torch.from_numpy(self.near_support[index]),
            "lowest_foot_heights": torch.from_numpy(self.lowest_foot_heights[index]),
            "ground": torch.tensor(self.ground[index], dtype=torch.float32),
            "source_path": self.source_paths[index],
            "stem": self.stems[index],
            "index": torch.tensor(index, dtype=torch.long),
        }
