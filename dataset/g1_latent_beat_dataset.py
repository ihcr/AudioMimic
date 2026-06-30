import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.g1_motion_prior_dataset import (
    G1MotionPriorDataset,
    G1MotionPriorNormalizer,
)
from dataset.motion_representation import G1_YAW_DELTA_MOTION_FORMAT
from eval.run_g1_motion_prior_eval import build_model_from_checkpoint
from feature_config import BEAT_FEATURES_8D_DIM, WAV2CLIP_DIM, WAV2CLIP_STFT_BEAT_DIM


LATENT_BEAT_CACHE_VERSION = "g1_latent_beat8d_v1"
LATENT_MUSIC_CONTROL_CACHE_VERSION = "g1_music_control_latent_v1"
DEFAULT_LATENT_FRAMES = 75
DEFAULT_LATENT_DIM = 128
DEFAULT_BEAT_FRAMES = 150
NORMALIZER_EPS = 1e-6


@dataclass(frozen=True)
class G1LatentNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize_np(self, latent):
        return (latent - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)

    def unnormalize_tensor(self, latent):
        mean = torch.as_tensor(self.mean, device=latent.device, dtype=latent.dtype).view(1, 1, -1)
        std = torch.as_tensor(self.std, device=latent.device, dtype=latent.dtype).view(1, 1, -1)
        return latent * std + mean

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


def _checkpoint_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def latent_beat_cache_paths(backup_path, prior_checkpoint, cache_limit_per_split=0):
    backup_path = Path(backup_path)
    tag = f"{LATENT_BEAT_CACHE_VERSION}_{_checkpoint_fingerprint(prior_checkpoint)}"
    if cache_limit_per_split:
        tag = f"{tag}_limit{int(cache_limit_per_split)}"
    return {
        "train": backup_path / f"train_{tag}.pkl",
        "test": backup_path / f"test_{tag}.pkl",
        "normalizer": backup_path / f"normalizer_{tag}.pkl",
        "metadata": backup_path / f"metadata_{tag}.json",
    }


def latent_music_control_cache_paths(
    backup_path,
    prior_checkpoint,
    use_wav2clip_semantic=False,
    cache_limit_per_split=0,
):
    backup_path = Path(backup_path)
    semantic_tag = "wav2clip" if use_wav2clip_semantic else "control"
    tag = (
        f"{LATENT_MUSIC_CONTROL_CACHE_VERSION}_{semantic_tag}_"
        f"{_checkpoint_fingerprint(prior_checkpoint)}"
    )
    if cache_limit_per_split:
        tag = f"{tag}_limit{int(cache_limit_per_split)}"
    return {
        "train": backup_path / f"train_{tag}.pkl",
        "test": backup_path / f"test_{tag}.pkl",
        "normalizer": backup_path / f"normalizer_{tag}.pkl",
        "metadata": backup_path / f"metadata_{tag}.json",
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


def _load_beat_features(data_path, split, stems):
    beat_dir = Path(data_path) / split / "beat_features_8d_feats"
    beats = []
    for stem in stems:
        path = beat_dir / f"{stem}.npy"
        feature = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if feature.shape != (DEFAULT_BEAT_FRAMES, BEAT_FEATURES_8D_DIM):
            raise ValueError(
                f"{path} expected {(DEFAULT_BEAT_FRAMES, BEAT_FEATURES_8D_DIM)}, "
                f"got {feature.shape}"
            )
        if not np.isfinite(feature).all():
            raise ValueError(f"{path} contains non-finite beat features")
        beats.append(np.array(feature, dtype=np.float32, copy=True))
    return np.stack(beats, axis=0).astype(np.float32)


def _load_wav2clip_semantic(data_path, split, stems):
    wav2clip_dir = Path(data_path) / split / "wav2clip_stft_beat_feats"
    semantics = []
    for stem in stems:
        path = wav2clip_dir / f"{stem}.npy"
        feature = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if feature.shape != (DEFAULT_BEAT_FRAMES, WAV2CLIP_STFT_BEAT_DIM):
            raise ValueError(
                f"{path} expected {(DEFAULT_BEAT_FRAMES, WAV2CLIP_STFT_BEAT_DIM)}, "
                f"got {feature.shape}"
            )
        semantic = np.array(feature[:, :WAV2CLIP_DIM], dtype=np.float32, copy=True)
        if not np.isfinite(semantic).all():
            raise ValueError(f"{path} contains non-finite Wav2CLIP semantic features")
        semantics.append(semantic)
    return np.stack(semantics, axis=0).astype(np.float32)


def _motion_for_prior(dataset, prior_normalizer):
    dataset_mean = dataset.normalizer.mean
    dataset_std = dataset.normalizer.std
    if np.allclose(dataset_mean, prior_normalizer.mean) and np.allclose(dataset_std, prior_normalizer.std):
        return np.asarray(dataset.motion, dtype=np.float32)
    raw = dataset.normalizer.unnormalize_np(np.asarray(dataset.motion, dtype=np.float32))
    return prior_normalizer.normalize_np(raw).astype(np.float32)


@torch.inference_mode()
def _encode_dataset_latents(dataset, model, prior_normalizer, device, batch_size):
    model.eval()
    motion = _motion_for_prior(dataset, prior_normalizer)
    latents = []
    for start in tqdm(
        range(0, motion.shape[0], int(batch_size)),
        desc="Encode V6b-A latents",
        unit="batch",
    ):
        batch = torch.from_numpy(motion[start : start + int(batch_size)]).to(device=device)
        latent = model.encode(batch.float(), sample=False)["latent"]
        latents.append(latent.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0).astype(np.float32)


def _metadata_matches(metadata, prior_checkpoint, cache_limit_per_split):
    return (
        metadata.get("cache_version") == LATENT_BEAT_CACHE_VERSION
        and metadata.get("prior_checkpoint_fingerprint") == _checkpoint_fingerprint(prior_checkpoint)
        and int(metadata.get("cache_limit_per_split", 0)) == int(cache_limit_per_split)
        and int(metadata.get("latent_frames", 0)) == DEFAULT_LATENT_FRAMES
        and int(metadata.get("latent_dim", 0)) == DEFAULT_LATENT_DIM
        and int(metadata.get("beat_frames", 0)) == DEFAULT_BEAT_FRAMES
        and int(metadata.get("beat_dim", 0)) == BEAT_FEATURES_8D_DIM
    )


def _music_control_metadata_matches(
    metadata,
    prior_checkpoint,
    use_wav2clip_semantic,
    cache_limit_per_split,
):
    return (
        metadata.get("cache_version") == LATENT_MUSIC_CONTROL_CACHE_VERSION
        and metadata.get("prior_checkpoint_fingerprint") == _checkpoint_fingerprint(prior_checkpoint)
        and bool(metadata.get("use_wav2clip_semantic", False)) == bool(use_wav2clip_semantic)
        and int(metadata.get("cache_limit_per_split", 0)) == int(cache_limit_per_split)
        and int(metadata.get("latent_frames", 0)) == DEFAULT_LATENT_FRAMES
        and int(metadata.get("latent_dim", 0)) == DEFAULT_LATENT_DIM
        and int(metadata.get("control_frames", 0)) == DEFAULT_BEAT_FRAMES
        and int(metadata.get("control_dim", 0)) == BEAT_FEATURES_8D_DIM
        and int(metadata.get("semantic_dim", 0)) == (
            WAV2CLIP_DIM if use_wav2clip_semantic else 0
        )
    )


def build_g1_latent_beat_cache(
    data_path,
    motion_prior_processed_data_dir,
    latent_processed_data_dir,
    prior_checkpoint,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
    batch_size=512,
    device="cpu",
    cache_limit_per_split=0,
):
    paths = latent_beat_cache_paths(
        latent_processed_data_dir,
        prior_checkpoint,
        cache_limit_per_split=cache_limit_per_split,
    )
    checkpoint = torch.load(prior_checkpoint, map_location=device, weights_only=False)
    prior_normalizer = G1MotionPriorNormalizer.from_state_dict(checkpoint["normalizer"])
    model = build_model_from_checkpoint(checkpoint, motion_format).to(device)

    split_payloads = {}
    train_latent_raw = None
    for split in ("train", "test"):
        dataset = G1MotionPriorDataset(
            data_path=data_path,
            backup_path=motion_prior_processed_data_dir,
            split=split,
            motion_format=motion_format,
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
            cache_batch_size=batch_size,
            cache_device=device,
            cache_limit_per_split=cache_limit_per_split,
            rebuild_cache=False,
        )
        latent_raw = _encode_dataset_latents(dataset, model, prior_normalizer, device, batch_size)
        beat_features = _load_beat_features(data_path, split, dataset.stems)
        split_payloads[split] = {
            "latent_raw": latent_raw,
            "beat_features": beat_features,
            "source_paths": dataset.source_paths,
            "stems": dataset.stems,
        }
        if split == "train":
            train_latent_raw = latent_raw

    mean = train_latent_raw.reshape(-1, train_latent_raw.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_latent_raw.reshape(-1, train_latent_raw.shape[-1]).std(axis=0).astype(np.float32)
    std = np.maximum(std, NORMALIZER_EPS).astype(np.float32)
    normalizer = G1LatentNormalizer(mean=mean, std=std)

    for split, payload in split_payloads.items():
        split_cache = {
            "latent": normalizer.normalize_np(payload["latent_raw"]).astype(np.float32),
            "beat_features": payload["beat_features"],
            "source_paths": payload["source_paths"],
            "stems": payload["stems"],
        }
        _write_pickle(split_cache, paths[split])

    metadata = {
        "cache_version": LATENT_BEAT_CACHE_VERSION,
        "data_path": str(data_path),
        "motion_format": motion_format,
        "prior_checkpoint": str(prior_checkpoint),
        "prior_checkpoint_fingerprint": _checkpoint_fingerprint(prior_checkpoint),
        "latent_frames": DEFAULT_LATENT_FRAMES,
        "latent_dim": DEFAULT_LATENT_DIM,
        "beat_frames": DEFAULT_BEAT_FRAMES,
        "beat_dim": BEAT_FEATURES_8D_DIM,
        "train_count": int(split_payloads["train"]["latent_raw"].shape[0]),
        "test_count": int(split_payloads["test"]["latent_raw"].shape[0]),
        "cache_limit_per_split": int(cache_limit_per_split),
        "normalizer_eps": NORMALIZER_EPS,
        "condition": "beat_features_8d",
    }
    _write_pickle(normalizer.state_dict(), paths["normalizer"])
    _write_json(metadata, paths["metadata"])
    return paths


def ensure_g1_latent_beat_cache(
    data_path,
    motion_prior_processed_data_dir,
    latent_processed_data_dir,
    prior_checkpoint,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
    batch_size=512,
    device="cpu",
    cache_limit_per_split=0,
    rebuild_cache=False,
):
    paths = latent_beat_cache_paths(
        latent_processed_data_dir,
        prior_checkpoint,
        cache_limit_per_split=cache_limit_per_split,
    )
    expected_files = [paths["train"], paths["test"], paths["normalizer"], paths["metadata"]]
    needs_build = rebuild_cache or any(not path.is_file() for path in expected_files)
    if not needs_build:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if not _metadata_matches(metadata, prior_checkpoint, cache_limit_per_split):
            raise ValueError(
                "G1 latent beat cache metadata does not match the requested configuration. "
                f"Delete {Path(latent_processed_data_dir)} or rerun with --rebuild_cache."
            )
        return paths
    return build_g1_latent_beat_cache(
        data_path=data_path,
        motion_prior_processed_data_dir=motion_prior_processed_data_dir,
        latent_processed_data_dir=latent_processed_data_dir,
        prior_checkpoint=prior_checkpoint,
        motion_format=motion_format,
        g1_fk_model_path=g1_fk_model_path,
        g1_root_quat_order=g1_root_quat_order,
        batch_size=batch_size,
        device=device,
        cache_limit_per_split=cache_limit_per_split,
    )


def build_g1_music_control_latent_cache(
    data_path,
    motion_prior_processed_data_dir,
    latent_processed_data_dir,
    prior_checkpoint,
    use_wav2clip_semantic=False,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
    batch_size=512,
    device="cpu",
    cache_limit_per_split=0,
):
    paths = latent_music_control_cache_paths(
        latent_processed_data_dir,
        prior_checkpoint,
        use_wav2clip_semantic=use_wav2clip_semantic,
        cache_limit_per_split=cache_limit_per_split,
    )
    checkpoint = torch.load(prior_checkpoint, map_location=device, weights_only=False)
    prior_normalizer = G1MotionPriorNormalizer.from_state_dict(checkpoint["normalizer"])
    model = build_model_from_checkpoint(checkpoint, motion_format).to(device)

    split_payloads = {}
    train_latent_raw = None
    for split in ("train", "test"):
        dataset = G1MotionPriorDataset(
            data_path=data_path,
            backup_path=motion_prior_processed_data_dir,
            split=split,
            motion_format=motion_format,
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
            cache_batch_size=batch_size,
            cache_device=device,
            cache_limit_per_split=cache_limit_per_split,
            rebuild_cache=False,
        )
        latent_raw = _encode_dataset_latents(dataset, model, prior_normalizer, device, batch_size)
        control_features = _load_beat_features(data_path, split, dataset.stems)
        payload = {
            "latent_raw": latent_raw,
            "control_features": control_features,
            "source_paths": dataset.source_paths,
            "stems": dataset.stems,
        }
        if use_wav2clip_semantic:
            payload["semantic_features"] = _load_wav2clip_semantic(data_path, split, dataset.stems)
        split_payloads[split] = payload
        if split == "train":
            train_latent_raw = latent_raw

    mean = train_latent_raw.reshape(-1, train_latent_raw.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_latent_raw.reshape(-1, train_latent_raw.shape[-1]).std(axis=0).astype(np.float32)
    std = np.maximum(std, NORMALIZER_EPS).astype(np.float32)
    normalizer = G1LatentNormalizer(mean=mean, std=std)

    for split, payload in split_payloads.items():
        split_cache = {
            "latent": normalizer.normalize_np(payload["latent_raw"]).astype(np.float32),
            "control_features": payload["control_features"],
            "source_paths": payload["source_paths"],
            "stems": payload["stems"],
        }
        if use_wav2clip_semantic:
            split_cache["semantic_features"] = payload["semantic_features"]
        _write_pickle(split_cache, paths[split])

    metadata = {
        "cache_version": LATENT_MUSIC_CONTROL_CACHE_VERSION,
        "data_path": str(data_path),
        "motion_format": motion_format,
        "prior_checkpoint": str(prior_checkpoint),
        "prior_checkpoint_fingerprint": _checkpoint_fingerprint(prior_checkpoint),
        "latent_frames": DEFAULT_LATENT_FRAMES,
        "latent_dim": DEFAULT_LATENT_DIM,
        "control_frames": DEFAULT_BEAT_FRAMES,
        "control_dim": BEAT_FEATURES_8D_DIM,
        "semantic_frames": DEFAULT_BEAT_FRAMES if use_wav2clip_semantic else 0,
        "semantic_dim": WAV2CLIP_DIM if use_wav2clip_semantic else 0,
        "train_count": int(split_payloads["train"]["latent_raw"].shape[0]),
        "test_count": int(split_payloads["test"]["latent_raw"].shape[0]),
        "cache_limit_per_split": int(cache_limit_per_split),
        "normalizer_eps": NORMALIZER_EPS,
        "condition": "beat_features_8d+wav2clip_semantic"
        if use_wav2clip_semantic
        else "beat_features_8d",
        "use_wav2clip_semantic": bool(use_wav2clip_semantic),
        "semantic_source": "wav2clip_stft_beat_feats[:, :512]"
        if use_wav2clip_semantic
        else "",
    }
    _write_pickle(normalizer.state_dict(), paths["normalizer"])
    _write_json(metadata, paths["metadata"])
    return paths


def ensure_g1_music_control_latent_cache(
    data_path,
    motion_prior_processed_data_dir,
    latent_processed_data_dir,
    prior_checkpoint,
    use_wav2clip_semantic=False,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
    batch_size=512,
    device="cpu",
    cache_limit_per_split=0,
    rebuild_cache=False,
):
    paths = latent_music_control_cache_paths(
        latent_processed_data_dir,
        prior_checkpoint,
        use_wav2clip_semantic=use_wav2clip_semantic,
        cache_limit_per_split=cache_limit_per_split,
    )
    expected_files = [paths["train"], paths["test"], paths["normalizer"], paths["metadata"]]
    needs_build = rebuild_cache or any(not path.is_file() for path in expected_files)
    if not needs_build:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if not _music_control_metadata_matches(
            metadata,
            prior_checkpoint,
            use_wav2clip_semantic,
            cache_limit_per_split,
        ):
            raise ValueError(
                "G1 music-control latent cache metadata does not match the requested "
                f"configuration. Delete {Path(latent_processed_data_dir)} or rerun "
                "with --rebuild_cache."
            )
        return paths
    return build_g1_music_control_latent_cache(
        data_path=data_path,
        motion_prior_processed_data_dir=motion_prior_processed_data_dir,
        latent_processed_data_dir=latent_processed_data_dir,
        prior_checkpoint=prior_checkpoint,
        use_wav2clip_semantic=use_wav2clip_semantic,
        motion_format=motion_format,
        g1_fk_model_path=g1_fk_model_path,
        g1_root_quat_order=g1_root_quat_order,
        batch_size=batch_size,
        device=device,
        cache_limit_per_split=cache_limit_per_split,
    )


class G1LatentBeatDataset(Dataset):
    def __init__(
        self,
        data_path,
        motion_prior_processed_data_dir,
        latent_processed_data_dir,
        prior_checkpoint,
        split,
        motion_format=G1_YAW_DELTA_MOTION_FORMAT,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
        cache_batch_size=512,
        cache_device="cpu",
        cache_limit_per_split=0,
        rebuild_cache=False,
        data_len=0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        paths = ensure_g1_latent_beat_cache(
            data_path=data_path,
            motion_prior_processed_data_dir=motion_prior_processed_data_dir,
            latent_processed_data_dir=latent_processed_data_dir,
            prior_checkpoint=prior_checkpoint,
            motion_format=motion_format,
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
            self.normalizer = G1LatentNormalizer.from_state_dict(pickle.load(handle))
        self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        count = int(data_len) if data_len else len(payload["stems"])
        self.latent = np.asarray(payload["latent"][:count], dtype=np.float32)
        self.beat_features = np.asarray(payload["beat_features"][:count], dtype=np.float32)
        self.source_paths = list(payload["source_paths"][:count])
        self.stems = list(payload["stems"][:count])
        if self.latent.shape[1:] != (DEFAULT_LATENT_FRAMES, DEFAULT_LATENT_DIM):
            raise ValueError(
                f"latent expected {(DEFAULT_LATENT_FRAMES, DEFAULT_LATENT_DIM)}, "
                f"got {self.latent.shape[1:]}"
            )
        if self.beat_features.shape[1:] != (DEFAULT_BEAT_FRAMES, BEAT_FEATURES_8D_DIM):
            raise ValueError(
                f"beat_features expected {(DEFAULT_BEAT_FRAMES, BEAT_FEATURES_8D_DIM)}, "
                f"got {self.beat_features.shape[1:]}"
            )

    def __len__(self):
        return int(self.latent.shape[0])

    def __getitem__(self, index):
        return {
            "latent": torch.from_numpy(self.latent[index]),
            "beat_features": torch.from_numpy(self.beat_features[index]),
            "source_path": self.source_paths[index],
            "stem": self.stems[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


class G1MusicControlLatentDataset(Dataset):
    def __init__(
        self,
        data_path,
        motion_prior_processed_data_dir,
        latent_processed_data_dir,
        prior_checkpoint,
        split,
        use_wav2clip_semantic=False,
        motion_format=G1_YAW_DELTA_MOTION_FORMAT,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
        cache_batch_size=512,
        cache_device="cpu",
        cache_limit_per_split=0,
        rebuild_cache=False,
        data_len=0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.use_wav2clip_semantic = bool(use_wav2clip_semantic)
        paths = ensure_g1_music_control_latent_cache(
            data_path=data_path,
            motion_prior_processed_data_dir=motion_prior_processed_data_dir,
            latent_processed_data_dir=latent_processed_data_dir,
            prior_checkpoint=prior_checkpoint,
            use_wav2clip_semantic=self.use_wav2clip_semantic,
            motion_format=motion_format,
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
            self.normalizer = G1LatentNormalizer.from_state_dict(pickle.load(handle))
        self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        count = int(data_len) if data_len else len(payload["stems"])
        self.latent = np.asarray(payload["latent"][:count], dtype=np.float32)
        self.control_features = np.asarray(payload["control_features"][:count], dtype=np.float32)
        self.semantic_features = None
        if self.use_wav2clip_semantic:
            self.semantic_features = np.asarray(
                payload["semantic_features"][:count],
                dtype=np.float32,
            )
        self.source_paths = list(payload["source_paths"][:count])
        self.stems = list(payload["stems"][:count])
        if self.latent.shape[1:] != (DEFAULT_LATENT_FRAMES, DEFAULT_LATENT_DIM):
            raise ValueError(
                f"latent expected {(DEFAULT_LATENT_FRAMES, DEFAULT_LATENT_DIM)}, "
                f"got {self.latent.shape[1:]}"
            )
        if self.control_features.shape[1:] != (DEFAULT_BEAT_FRAMES, BEAT_FEATURES_8D_DIM):
            raise ValueError(
                f"control_features expected {(DEFAULT_BEAT_FRAMES, BEAT_FEATURES_8D_DIM)}, "
                f"got {self.control_features.shape[1:]}"
            )
        if self.use_wav2clip_semantic and self.semantic_features.shape[1:] != (
            DEFAULT_BEAT_FRAMES,
            WAV2CLIP_DIM,
        ):
            raise ValueError(
                f"semantic_features expected {(DEFAULT_BEAT_FRAMES, WAV2CLIP_DIM)}, "
                f"got {self.semantic_features.shape[1:]}"
            )

    def __len__(self):
        return int(self.latent.shape[0])

    def __getitem__(self, index):
        sample = {
            "latent": torch.from_numpy(self.latent[index]),
            "control_features": torch.from_numpy(self.control_features[index]),
            "source_path": self.source_paths[index],
            "stem": self.stems[index],
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.semantic_features is not None:
            sample["semantic_features"] = torch.from_numpy(self.semantic_features[index])
        return sample
