import glob
import os
import pickle
import random
import zipfile
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.motion_representation import (
    SMPL_MOTION_FORMAT,
    encode_g1_motion_for_format,
    is_g1_motion_format,
    validate_motion_format,
)
from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from feature_config import (
    BEAT_FEATURES_8D_DIM,
    BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE,
    BODY_INTENSITY_DIM,
    GAUSSIAN_BEAT_DIM,
    MOTION_BEATNESS_DIM,
    MOTION_ENERGY_DIM,
    MOTION_INTENSITY_DIM,
    SUPPORT_BEATNESS_DIM,
    SUPPORT_CONTACT_DIM,
    UPPER_BEATNESS_DIM,
    WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_DIM,
    WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE,
    WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_STFT_BEAT_DIM,
)
from rotation_transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                 quaternion_multiply,
                                 quaternion_to_axis_angle)
from vis import SMPLSkeleton

DATASET_CACHE_VERSION = "v4"
FEATURE_STORE_CACHE_VERSION = "v1"
MOTION_CONTROL_STORE_CACHE_VERSION = "v1"
FEATURE_CACHE_OFF = "off"
FEATURE_CACHE_MEMMAP = "memmap"
FEATURE_CACHE_MODES = (FEATURE_CACHE_OFF, FEATURE_CACHE_MEMMAP)
FEATURE_CACHE_DTYPES = ("float32", "float16")


def is_structured_motion_energy_feature(feature_type):
    return feature_type in (
        BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE,
        WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    )


def is_motion_intensity_beatness_feature(feature_type):
    return feature_type in (
        WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    )


def is_motion_beatness_only_feature(feature_type):
    return feature_type == BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE


def is_body_support_beatness_feature(feature_type):
    return feature_type == WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE


def motion_control_feature_dir(feature_type):
    if feature_type == BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE:
        return "motion_control_v3_local_feats"
    if feature_type == WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE:
        return "motion_control_v3_local_feats"
    if feature_type == WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE:
        return "motion_control_v4_support_feats"
    if feature_type == WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE:
        return "motion_control_v2_feats"
    return "motion_energy_feats"


def atomic_pickle_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def resolve_feature_cache_dtype(dtype_name):
    if dtype_name not in FEATURE_CACHE_DTYPES:
        raise ValueError(f"Unsupported feature cache dtype: {dtype_name}")
    return np.dtype(dtype_name)


def feature_store_cache_name(split_name, feature_type, dtype_name):
    return (
        f"{split_name}_{feature_type}_features_{FEATURE_CACHE_MEMMAP}_"
        f"{dtype_name}_{FEATURE_STORE_CACHE_VERSION}.npy"
    )


def feature_store_index_name(split_name, feature_type, dtype_name):
    return (
        f"{split_name}_{feature_type}_features_{FEATURE_CACHE_MEMMAP}_"
        f"{dtype_name}_{FEATURE_STORE_CACHE_VERSION}.pkl"
    )


def motion_control_store_cache_name(split_name, feature_type, dtype_name):
    return (
        f"{split_name}_{feature_type}_motion_control_{FEATURE_CACHE_MEMMAP}_"
        f"{dtype_name}_{MOTION_CONTROL_STORE_CACHE_VERSION}.npy"
    )


def motion_control_store_index_name(split_name, feature_type, dtype_name):
    return (
        f"{split_name}_{feature_type}_motion_control_{FEATURE_CACHE_MEMMAP}_"
        f"{dtype_name}_{MOTION_CONTROL_STORE_CACHE_VERSION}.pkl"
    )


def _feature_store_metadata_matches(metadata, feature_paths, dtype_name):
    if not metadata:
        return False
    if metadata.get("cache_version") != FEATURE_STORE_CACHE_VERSION:
        return False
    if metadata.get("dtype") != dtype_name:
        return False
    return metadata.get("source_files") == [str(path) for path in feature_paths]


def _read_feature_store_metadata(index_path):
    if not index_path.is_file():
        return None
    with open(index_path, "rb") as handle:
        return pickle.load(handle)


def build_or_reuse_feature_store(
    feature_paths,
    cache_dir,
    split_name,
    feature_type,
    dtype_name="float32",
    force_rebuild=False,
):
    feature_paths = [str(path) for path in feature_paths]
    if not feature_paths:
        raise ValueError("Cannot build feature cache without feature files.")

    dtype = resolve_feature_cache_dtype(dtype_name)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    store_path = cache_dir / feature_store_cache_name(split_name, feature_type, dtype_name)
    index_path = cache_dir / feature_store_index_name(split_name, feature_type, dtype_name)

    if not force_rebuild and store_path.is_file() and index_path.is_file():
        metadata = _read_feature_store_metadata(index_path)
        if _feature_store_metadata_matches(metadata, feature_paths, dtype_name):
            print(f"Using cached feature store: {store_path}")
            return metadata

    print(f"Building feature store: {store_path}")
    sample = np.load(feature_paths[0], mmap_mode="r")
    feature_shape = tuple(sample.shape)
    tmp_store_path = store_path.with_name(f"{store_path.stem}.{os.getpid()}.tmp.npy")
    feature_store = np.lib.format.open_memmap(
        tmp_store_path,
        mode="w+",
        dtype=dtype,
        shape=(len(feature_paths), *feature_shape),
    )
    for idx, feature_path in enumerate(
        tqdm(
            feature_paths,
            desc=f"Packing {split_name} {feature_type} features",
            unit="clip",
            mininterval=10,
        )
    ):
        feature = np.load(feature_path, mmap_mode="r")
        if tuple(feature.shape) != feature_shape:
            raise ValueError(
                f"Feature shape mismatch for {feature_path}: "
                f"expected {feature_shape}, got {tuple(feature.shape)}"
            )
        feature_store[idx] = feature.astype(dtype, copy=False)
    feature_store.flush()
    del feature_store
    tmp_store_path.replace(store_path)

    metadata = {
        "cache_version": FEATURE_STORE_CACHE_VERSION,
        "mode": FEATURE_CACHE_MEMMAP,
        "dtype": dtype_name,
        "store_path": str(store_path),
        "source_files": feature_paths,
        "shape": (len(feature_paths), *feature_shape),
    }
    atomic_pickle_dump(metadata, index_path)
    return metadata


def _load_motion_control_pair(path):
    try:
        with np.load(path) as motion_control:
            motion_intensity = np.array(
                motion_control["motion_intensity_envelope"],
                dtype=np.float32,
                copy=True,
            )
            motion_beatness = np.array(
                motion_control["motion_beatness_envelope"],
                dtype=np.float32,
                copy=True,
            )
    except (KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Failed to read motion-control feature cache {path}. "
            "Rebuild the motion-control feature cache and processed dataset cache."
        ) from exc
    if motion_intensity.shape != (150, MOTION_INTENSITY_DIM):
        raise ValueError(
            f"{path} motion_intensity_envelope expected "
            f"{(150, MOTION_INTENSITY_DIM)}, got {motion_intensity.shape}"
        )
    if motion_beatness.shape != (150, MOTION_BEATNESS_DIM):
        raise ValueError(
            f"{path} motion_beatness_envelope expected "
            f"{(150, MOTION_BEATNESS_DIM)}, got {motion_beatness.shape}"
        )
    return motion_intensity, motion_beatness


def _load_body_support_control(path):
    try:
        with np.load(path) as motion_control:
            body_intensity = np.array(
                motion_control["body_intensity_envelope"],
                dtype=np.float32,
                copy=True,
            )
            support_beatness = np.array(
                motion_control["support_beatness_envelope"],
                dtype=np.float32,
                copy=True,
            )
            upper_beatness = np.array(
                motion_control["upper_beatness_envelope"],
                dtype=np.float32,
                copy=True,
            )
            support_contact = np.array(
                motion_control["support_contact"],
                dtype=np.float32,
                copy=True,
            )
    except (KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Failed to read V6a body/support feature cache {path}. "
            "Rebuild motion_control_v4_support_feats and the processed dataset cache."
        ) from exc
    expected = (
        ("body_intensity_envelope", body_intensity, (150, BODY_INTENSITY_DIM)),
        ("support_beatness_envelope", support_beatness, (150, SUPPORT_BEATNESS_DIM)),
        ("upper_beatness_envelope", upper_beatness, (150, UPPER_BEATNESS_DIM)),
        ("support_contact", support_contact, (150, SUPPORT_CONTACT_DIM)),
    )
    for name, value, shape in expected:
        if value.shape != shape:
            raise ValueError(f"{path} {name} expected {shape}, got {value.shape}")
    return body_intensity, support_beatness, upper_beatness, support_contact


def motion_control_store_fields(feature_type):
    if is_body_support_beatness_feature(feature_type):
        return (
            "body_intensity",
            "support_beatness",
            "upper_beatness",
            "support_contact",
        )
    return ("motion_intensity", "motion_beatness")


def build_or_reuse_motion_control_store(
    motion_control_paths,
    cache_dir,
    split_name,
    feature_type,
    dtype_name="float32",
    force_rebuild=False,
):
    motion_control_paths = [str(path) for path in motion_control_paths]
    if not motion_control_paths:
        raise ValueError("Cannot build motion-control store without feature files.")

    dtype = resolve_feature_cache_dtype(dtype_name)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    store_path = cache_dir / motion_control_store_cache_name(split_name, feature_type, dtype_name)
    index_path = cache_dir / motion_control_store_index_name(split_name, feature_type, dtype_name)

    if not force_rebuild and store_path.is_file() and index_path.is_file():
        metadata = _read_feature_store_metadata(index_path)
        if (
            metadata
            and metadata.get("cache_version") == MOTION_CONTROL_STORE_CACHE_VERSION
            and metadata.get("mode") == FEATURE_CACHE_MEMMAP
            and metadata.get("dtype") == dtype_name
            and metadata.get("source_files") == motion_control_paths
            and Path(metadata.get("store_path", "")) == store_path
        ):
            print(f"Using cached motion-control store: {store_path}")
            return metadata

    print(f"Building motion-control store: {store_path}")
    if is_body_support_beatness_feature(feature_type):
        control_dim = (
            BODY_INTENSITY_DIM
            + SUPPORT_BEATNESS_DIM
            + UPPER_BEATNESS_DIM
            + SUPPORT_CONTACT_DIM
        )
    else:
        control_dim = MOTION_INTENSITY_DIM + MOTION_BEATNESS_DIM
    tmp_store_path = store_path.with_name(f"{store_path.stem}.{os.getpid()}.tmp.npy")
    motion_control_store = np.lib.format.open_memmap(
        tmp_store_path,
        mode="w+",
        dtype=dtype,
        shape=(len(motion_control_paths), 150, control_dim),
    )
    for idx, path in enumerate(
        tqdm(
            motion_control_paths,
            desc=f"Packing {split_name} {feature_type} motion controls",
            unit="clip",
            mininterval=10,
        )
    ):
        if is_body_support_beatness_feature(feature_type):
            (
                body_intensity,
                support_beatness,
                upper_beatness,
                support_contact,
            ) = _load_body_support_control(path)
            motion_control_store[idx] = np.concatenate(
                (body_intensity, support_beatness, upper_beatness, support_contact),
                axis=-1,
            ).astype(dtype, copy=False)
        else:
            motion_intensity, motion_beatness = _load_motion_control_pair(path)
            motion_control_store[idx] = np.concatenate(
                (motion_intensity, motion_beatness),
                axis=-1,
            ).astype(dtype, copy=False)
    motion_control_store.flush()
    del motion_control_store
    tmp_store_path.replace(store_path)

    metadata = {
        "cache_version": MOTION_CONTROL_STORE_CACHE_VERSION,
        "mode": FEATURE_CACHE_MEMMAP,
        "dtype": dtype_name,
        "store_path": str(store_path),
        "source_files": motion_control_paths,
        "shape": (len(motion_control_paths), 150, control_dim),
        "fields": motion_control_store_fields(feature_type),
    }
    atomic_pickle_dump(metadata, index_path)
    return metadata


def processed_dataset_cache_name(
    split_name,
    feature_type,
    use_beats,
    beat_rep,
    motion_format=SMPL_MOTION_FORMAT,
):
    validate_motion_format(motion_format)
    beat_tag = "beat" if use_beats else "nobeat"
    if motion_format == SMPL_MOTION_FORMAT:
        return f"processed_{split_name}_{feature_type}_{beat_tag}_{beat_rep}_{DATASET_CACHE_VERSION}.pkl"
    return (
        f"processed_{split_name}_{motion_format}_{feature_type}_{beat_tag}_{beat_rep}_"
        f"{DATASET_CACHE_VERSION}.pkl"
    )


def prune_legacy_processed_dataset_caches(
    backup_path,
    split_name,
    feature_type,
    use_beats,
    beat_rep,
    motion_format=SMPL_MOTION_FORMAT,
):
    validate_motion_format(motion_format)
    backup_path = Path(backup_path)
    beat_tag = "beat" if use_beats else "nobeat"
    current_name = processed_dataset_cache_name(
        split_name,
        feature_type,
        use_beats,
        beat_rep,
        motion_format=motion_format,
    )
    if motion_format == SMPL_MOTION_FORMAT:
        pattern = f"processed_{split_name}_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
    else:
        pattern = f"processed_{split_name}_{motion_format}_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
    for cache_path in backup_path.glob(pattern):
        if cache_path.name == current_name:
            continue
        cache_path.unlink(missing_ok=True)


def _stack_loaded_beats(beatnames):
    payload = {
        "motion_dist": [],
        "motion_spacing": [],
        "motion_mask": [],
        "audio_dist": [],
        "audio_mask": [],
    }
    for beat_path in beatnames:
        with np.load(beat_path) as beat_data:
            payload["motion_dist"].append(beat_data["motion_dist"].astype(np.int64))
            payload["motion_spacing"].append(beat_data["motion_spacing"].astype(np.float32))
            payload["motion_mask"].append(beat_data["motion_mask"].astype(np.float32))
            payload["audio_dist"].append(beat_data["audio_dist"].astype(np.int64))
            payload["audio_mask"].append(beat_data["audio_mask"].astype(np.float32))
    return {key: np.stack(values, axis=0) for key, values in payload.items()}


class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "jukebox",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
        use_beats: bool = False,
        beat_rep: str = "distance",
        motion_format: str = SMPL_MOTION_FORMAT,
        feature_cache_mode: str = FEATURE_CACHE_OFF,
        feature_cache_dtype: str = "float32",
    ):
        self.data_path = data_path
        self.motion_format = validate_motion_format(motion_format)
        if feature_cache_mode not in FEATURE_CACHE_MODES:
            raise ValueError(f"Unsupported feature cache mode: {feature_cache_mode}")
        resolve_feature_cache_dtype(feature_cache_dtype)
        self.feature_cache_mode = feature_cache_mode
        self.feature_cache_dtype = feature_cache_dtype
        self.feature_store_path = None
        self.feature_store_shape = None
        self._feature_store = None
        self.motion_control_store_path = None
        self.motion_control_store_shape = None
        self._motion_control_store = None
        self.raw_fps = 30 if is_g1_motion_format(self.motion_format) else 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type
        self.use_beats = use_beats
        self.beat_rep = beat_rep
        self.structured_motion_energy = is_structured_motion_energy_feature(feature_type)
        self.structured_motion_intensity_beatness = is_motion_intensity_beatness_feature(
            feature_type
        )
        self.structured_motion_beatness_only = is_motion_beatness_only_feature(feature_type)
        self.structured_body_support_beatness = is_body_support_beatness_feature(feature_type)
        if self.structured_motion_energy and self.use_beats:
            raise ValueError(f"{feature_type} uses structured control and does not support --use_beats")
        if self.structured_motion_energy and self.feature_cache_mode != FEATURE_CACHE_OFF:
            raise ValueError(f"{feature_type} does not support feature_cache_mode=memmap")

        self.normalizer = normalizer
        self.data_len = data_len

        split_name = "train" if train else "test"
        pickle_name = processed_dataset_cache_name(
            split_name,
            feature_type,
            use_beats,
            beat_rep,
            motion_format=self.motion_format,
        )

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        prune_legacy_processed_dataset_caches(
            backup_path,
            split_name,
            feature_type,
            use_beats,
            beat_rep,
            motion_format=self.motion_format,
        )
        # save normalizer
        if not train:
            atomic_pickle_dump(normalizer, backup_path / "normalizer.pkl")
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_aistpp()  # Call this last
            atomic_pickle_dump(data, backup_path / pickle_name)

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )

        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"])
        self.data = {
            "pose": pose_input,
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        if "structured_condition_paths" in data:
            self.data["structured_condition_paths"] = data["structured_condition_paths"]
            if (
                self.structured_motion_intensity_beatness
                or self.structured_motion_beatness_only
                or self.structured_body_support_beatness
            ):
                metadata = build_or_reuse_motion_control_store(
                    [paths["motion_control"] for paths in data["structured_condition_paths"]],
                    backup_path / "feature_stores",
                    split_name,
                    self.feature_type,
                    dtype_name="float32",
                    force_rebuild=force_reload,
                )
                self.motion_control_store_path = metadata["store_path"]
                self.motion_control_store_shape = tuple(metadata["shape"])
        if self.feature_cache_mode == FEATURE_CACHE_MEMMAP:
            metadata = build_or_reuse_feature_store(
                data["filenames"],
                backup_path / "feature_stores",
                split_name,
                self.feature_type,
                dtype_name=self.feature_cache_dtype,
                force_rebuild=force_reload,
            )
            self.feature_store_path = metadata["store_path"]
            self.feature_store_shape = tuple(metadata["shape"])
        if self.use_beats:
            if all(
                key in data
                for key in ("motion_dist", "motion_spacing", "motion_mask", "audio_dist", "audio_mask")
            ):
                beat_payload = {
                    key: data[key]
                    for key in ("motion_dist", "motion_spacing", "motion_mask", "audio_dist", "audio_mask")
                }
            else:
                beat_payload = _stack_loaded_beats(data["beatnames"])
            self.data["motion_dist"] = torch.from_numpy(beat_payload["motion_dist"]).long()
            self.data["motion_spacing"] = torch.from_numpy(beat_payload["motion_spacing"]).float()
            self.data["motion_mask"] = torch.from_numpy(beat_payload["motion_mask"]).float()
            self.data["audio_dist"] = torch.from_numpy(beat_payload["audio_dist"]).long()
            self.data["audio_mask"] = torch.from_numpy(beat_payload["audio_mask"]).float()
            if "beatnames" in data:
                self.data["beatnames"] = data["beatnames"]
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_feature_store"] = None
        state["_motion_control_store"] = None
        return state

    def __len__(self):
        return self.length

    def _open_feature_store(self):
        if not hasattr(self, "_feature_store"):
            self._feature_store = None
        if self._feature_store is None:
            self._feature_store = np.load(self.feature_store_path, mmap_mode="r")
        return self._feature_store

    def _open_motion_control_store(self):
        if not hasattr(self, "_motion_control_store"):
            self._motion_control_store = None
        if self._motion_control_store is None:
            self._motion_control_store = np.load(self.motion_control_store_path, mmap_mode="r")
        return self._motion_control_store

    def _load_feature(self, idx):
        if self.structured_motion_energy:
            return self._load_structured_motion_energy_feature(idx)
        feature_store_path = getattr(self, "feature_store_path", None)
        if feature_store_path:
            feature = self._open_feature_store()[idx]
        else:
            feature = np.load(self.data["filenames"][idx], mmap_mode="r")
        return torch.from_numpy(np.array(feature, dtype=np.float32, copy=True))

    def _load_structured_motion_energy_feature(self, idx):
        paths = self.data["structured_condition_paths"][idx]
        if self.structured_motion_beatness_only:
            beat_features = np.array(
                np.load(paths["beat_features_8d"], mmap_mode="r"),
                dtype=np.float32,
                copy=True,
            )
            if beat_features.shape != (150, BEAT_FEATURES_8D_DIM):
                raise ValueError(
                    f"{paths['beat_features_8d']} expected {(150, BEAT_FEATURES_8D_DIM)}, "
                    f"got {beat_features.shape}"
                )
            motion_control_store_path = getattr(self, "motion_control_store_path", None)
            if motion_control_store_path:
                motion_control = np.array(
                    self._open_motion_control_store()[idx],
                    dtype=np.float32,
                    copy=True,
                )
                motion_beatness = motion_control[:, MOTION_INTENSITY_DIM:]
            else:
                _, motion_beatness = _load_motion_control_pair(paths["motion_control"])
            return {
                "semantic": {
                    "beat_features_8d": torch.from_numpy(beat_features),
                },
                "control": {
                    "motion_beatness": torch.from_numpy(motion_beatness),
                },
            }

        combined = np.array(
            np.load(paths["wav2clip_stft_beat"], mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        )
        if combined.shape != (150, WAV2CLIP_STFT_BEAT_DIM):
            raise ValueError(
                f"{paths['wav2clip_stft_beat']} expected {(150, WAV2CLIP_STFT_BEAT_DIM)}, "
                f"got {combined.shape}"
            )
        gaussian_beat = np.array(
            np.load(paths["gaussian_beat"], mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        )
        if gaussian_beat.shape != (150, GAUSSIAN_BEAT_DIM):
            raise ValueError(
                f"{paths['gaussian_beat']} expected {(150, GAUSSIAN_BEAT_DIM)}, got {gaussian_beat.shape}"
            )
        control = {
            "gaussian_beat": torch.from_numpy(
                np.array(gaussian_beat, dtype=np.float32, copy=True)
            ),
        }
        if self.structured_body_support_beatness:
            motion_control_store_path = getattr(self, "motion_control_store_path", None)
            if motion_control_store_path:
                motion_control = np.array(
                    self._open_motion_control_store()[idx],
                    dtype=np.float32,
                    copy=True,
                )
                start = 0
                body_intensity = motion_control[:, start : start + BODY_INTENSITY_DIM]
                start += BODY_INTENSITY_DIM
                support_beatness = motion_control[:, start : start + SUPPORT_BEATNESS_DIM]
                start += SUPPORT_BEATNESS_DIM
                upper_beatness = motion_control[:, start : start + UPPER_BEATNESS_DIM]
                start += UPPER_BEATNESS_DIM
                support_contact = motion_control[:, start : start + SUPPORT_CONTACT_DIM]
            else:
                (
                    body_intensity,
                    support_beatness,
                    upper_beatness,
                    support_contact,
                ) = _load_body_support_control(paths["motion_control"])
            control["body_intensity"] = torch.from_numpy(body_intensity)
            control["support_beatness"] = torch.from_numpy(support_beatness)
            control["upper_beatness"] = torch.from_numpy(upper_beatness)
            control["support_contact"] = torch.from_numpy(support_contact)
        elif self.structured_motion_intensity_beatness:
            motion_control_store_path = getattr(self, "motion_control_store_path", None)
            if motion_control_store_path:
                motion_control = np.array(
                    self._open_motion_control_store()[idx],
                    dtype=np.float32,
                    copy=True,
                )
                motion_intensity = motion_control[:, :MOTION_INTENSITY_DIM]
                motion_beatness = motion_control[:, MOTION_INTENSITY_DIM:]
            else:
                motion_intensity, motion_beatness = _load_motion_control_pair(
                    paths["motion_control"]
                )
            control["motion_intensity"] = torch.from_numpy(motion_intensity)
            control["motion_beatness"] = torch.from_numpy(motion_beatness)
        else:
            with np.load(paths["motion_energy"]) as motion_energy:
                beat_energy = motion_energy["beat_energy_envelope"]
            if beat_energy.shape != (150, MOTION_ENERGY_DIM):
                raise ValueError(
                    f"{paths['motion_energy']} beat_energy_envelope expected "
                    f"{(150, MOTION_ENERGY_DIM)}, got {beat_energy.shape}"
                )
            control["beat_energy_envelope"] = torch.from_numpy(
                np.array(beat_energy, dtype=np.float32, copy=True)
            )
        return {
            "semantic": {
                "wav2clip": torch.from_numpy(
                    np.array(combined[:, :WAV2CLIP_DIM], dtype=np.float32, copy=True)
                ),
            },
            "control": control,
        }

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = type(self)._load_feature(self, idx)
        wavname = self.data["wavs"][idx]
        if not self.use_beats:
            return self.data["pose"][idx], feature, filename_, wavname

        if self.beat_rep == "distance":
            beat = self.data["motion_dist"][idx] if self.train else self.data["audio_dist"][idx]
        elif self.beat_rep == "pulse":
            beat_source = self.data["motion_mask"][idx] if self.train else self.data["audio_mask"][idx]
            beat = beat_source.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported beat representation: {self.beat_rep}")

        cond = {
            "music": feature,
            "beat": beat,
            "beat_target": self.data["motion_dist"][idx].float(),
            "beat_spacing": self.data["motion_spacing"][idx],
            "audio_mask": self.data["audio_mask"][idx],
        }
        return self.data["pose"][idx], cond, filename_, wavname

    def load_aistpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        # Structure:
        # data
        #   |- train
        #   |    |- motion_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_features
        #   |    |- jukebox_features
        #   |    |- motions
        #   |    |- wavs

        motion_path = os.path.join(split_data_path, "motions_sliced")
        if self.structured_motion_energy:
            if self.structured_motion_beatness_only:
                sound_path = os.path.join(split_data_path, "beat_features_8d_feats")
                gaussian_beat_path = None
            else:
                sound_path = os.path.join(split_data_path, "wav2clip_stft_beat_feats")
                gaussian_beat_path = os.path.join(split_data_path, "gaussian_beat_feats")
            motion_energy_path = os.path.join(
                split_data_path,
                motion_control_feature_dir(self.feature_type),
            )
        else:
            sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        beat_path = os.path.join(split_data_path, "beat_feats")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        gaussian_features = (
            sorted(glob.glob(os.path.join(gaussian_beat_path, "*.npy")))
            if self.structured_motion_energy and gaussian_beat_path is not None
            else []
        )
        motion_energy_features = (
            sorted(glob.glob(os.path.join(motion_energy_path, "*.npz")))
            if self.structured_motion_energy
            else []
        )
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))
        beats = sorted(glob.glob(os.path.join(beat_path, "*.npz"))) if self.use_beats else []

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        all_structured_condition_paths = []
        all_beats = []
        all_motion_dist = []
        all_motion_spacing = []
        all_motion_mask = []
        all_audio_dist = []
        all_audio_mask = []
        if self.structured_motion_beatness_only:
            assert len(motions) == len(features) == len(motion_energy_features) == len(wavs)
            pairs = zip(motions, features, motion_energy_features, wavs)
        elif self.structured_motion_energy:
            assert len(motions) == len(features) == len(gaussian_features) == len(motion_energy_features) == len(wavs)
            pairs = zip(motions, features, gaussian_features, motion_energy_features, wavs)
        elif self.use_beats:
            assert len(motions) == len(features) == len(wavs) == len(beats)
            pairs = zip(motions, features, wavs, beats)
        else:
            assert len(motions) == len(features) == len(wavs)
            pairs = zip(motions, features, wavs)

        for items in pairs:
            if self.structured_motion_beatness_only:
                motion, feature, motion_energy_feature, wav = items
            elif self.structured_motion_energy:
                motion, feature, gaussian_feature, motion_energy_feature, wav = items
            elif self.use_beats:
                motion, feature, wav, beat = items
            else:
                motion, feature, wav = items
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            if self.structured_motion_beatness_only:
                e_name = os.path.splitext(os.path.basename(motion_energy_feature))[0]
                assert m_name == f_name == e_name == w_name, str(
                    (motion, feature, motion_energy_feature, wav)
                )
            elif self.structured_motion_energy:
                g_name = os.path.splitext(os.path.basename(gaussian_feature))[0]
                e_name = os.path.splitext(os.path.basename(motion_energy_feature))[0]
                assert m_name == f_name == g_name == e_name == w_name, str(
                    (motion, feature, gaussian_feature, motion_energy_feature, wav)
                )
            elif self.use_beats:
                b_name = os.path.splitext(os.path.basename(beat))[0]
                assert m_name == f_name == w_name == b_name, str((motion, feature, wav, beat))
            else:
                assert m_name == f_name == w_name, str((motion, feature, wav))
            # load motion
            with open(motion, "rb") as handle:
                data = pickle.load(handle)
            if is_g1_motion_format(self.motion_format):
                pos = data.get("root_pos", data.get("pos"))
                root_rot = data.get("root_rot")
                dof_pos = data.get("dof_pos")
                if root_rot is None or dof_pos is None:
                    q = np.asarray(data["q"], dtype=np.float32)
                    root_rot = q[:, :4]
                    dof_pos = q[:, 4:]
                q = np.concatenate(
                    (
                        np.asarray(root_rot, dtype=np.float32),
                        np.asarray(dof_pos, dtype=np.float32),
                    ),
                    axis=-1,
                )
            else:
                pos = data["pos"]
                q = data["q"]
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature)
            all_wavs.append(wav)
            if self.structured_motion_beatness_only:
                all_structured_condition_paths.append(
                    {
                        "beat_features_8d": feature,
                        "motion_control": motion_energy_feature,
                    }
                )
            elif self.structured_motion_energy:
                all_structured_condition_paths.append(
                    {
                        "wav2clip_stft_beat": feature,
                        "gaussian_beat": gaussian_feature,
                        (
                            "motion_control"
                            if (
                                self.structured_motion_intensity_beatness
                                or self.structured_body_support_beatness
                            )
                            else "motion_energy"
                        ): motion_energy_feature,
                    }
                )
            if self.use_beats:
                all_beats.append(beat)
                with np.load(beat) as beat_meta:
                    all_motion_dist.append(beat_meta["motion_dist"].astype(np.int64))
                    all_motion_spacing.append(beat_meta["motion_spacing"].astype(np.float32))
                    all_motion_mask.append(beat_meta["motion_mask"].astype(np.float32))
                    all_audio_dist.append(beat_meta["audio_dist"].astype(np.int64))
                    all_audio_mask.append(beat_meta["audio_mask"].astype(np.float32))

        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        print(all_pos.shape)
        all_pos = all_pos[:, :: self.data_stride, :]
        all_q = all_q[:, :: self.data_stride, :]
        data = {
            "pos": all_pos,
            "q": all_q,
            "filenames": all_names,
            "wavs": all_wavs,
        }
        if self.structured_motion_energy:
            data["structured_condition_paths"] = all_structured_condition_paths
        if self.use_beats:
            data["beatnames"] = all_beats
            data["motion_dist"] = np.stack(all_motion_dist, axis=0)
            data["motion_spacing"] = np.stack(all_motion_spacing, axis=0)
            data["motion_mask"] = np.stack(all_motion_mask, axis=0)
            data["audio_dist"] = np.stack(all_audio_dist, axis=0)
            data["audio_mask"] = np.stack(all_audio_mask, axis=0)
        return data

    def process_dataset(self, root_pos, local_q):
        if is_g1_motion_format(self.motion_format):
            return self.process_g1_dataset(root_pos, local_q)
        return self.process_smpl_dataset(root_pos, local_q)

    def process_g1_dataset(self, root_pos, local_q):
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        if local_q.shape[-1] != 33:
            raise ValueError(f"G1 motion q expected 33 channels, got {local_q.shape[-1]}")
        global_pose_vec_input = encode_g1_motion_for_format(
            root_pos,
            local_q[:, :, :4],
            local_q[:, :, 4:],
            motion_format=self.motion_format,
        ).float().detach()

        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")
        return global_pose_vec_input

    def process_smpl_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, local_q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input


class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "baseline",
        data_name: str = "aist",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.feature_type = feature_type
        self.test_list = set(
            [
                "mLH4",
                "mKR2",
                "mBR0",
                "mLO2",
                "mJB5",
                "mWA0",
                "mJS3",
                "mMH3",
                "mHO5",
                "mPO1",
            ]
        )
        self.train = train

        # if not aist, then set train to true to ignore test split logic
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True

        self.data = self.load_music()  # Call this last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            max_start = len(seq) - batch_size
            start = random.randint(0, max_start)
            seq_slice = seq[start : start + batch_size]

        # now we have a batch of filenames
        filenames = [os.path.join(self.music_path, x + ".npy") for x in seq_slice]
        # get the features
        features = np.array([np.load(x) for x in filenames])

        return torch.Tensor(features), seq_slice

    def load_music(self):
        # open data path
        split_data_path = os.path.join(self.data_path)
        music_path = os.path.join(
            split_data_path,
            f"{self.data_name}_baseline_feats"
            if self.feature_type == "baseline"
            else f"{self.data_name}_juke_feats/juke_66",
        )
        self.music_path = music_path
        # get the music filenames strided, with each subsequent item 5 slices (2.5 seconds) apart
        all_names = []

        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
            aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
            ka, kb = key_func(a), key_func(b)
            if aa < bb:
                return -1
            if aa > bb:
                return 1
            if ka < kb:
                return -1
            if ka > kb:
                return 1
            return 0

        for features in glob.glob(os.path.join(music_path, "*.npy")):
            fname = os.path.splitext(os.path.basename(features))[0]
            all_names.append(fname)
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))
        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or (
                (not self.train) and k not in self.test_list
            ):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
        self.keys = sorted(list(data_dict.keys()))
        return data_dict
