import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.motion_representation import (
    G1_MOTION_FORMAT,
    SMPL_MOTION_FORMAT,
    encode_g1_motion,
    validate_motion_format,
)
from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from rotation_transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                 quaternion_multiply,
                                 quaternion_to_axis_angle)
from vis import SMPLSkeleton

DATASET_CACHE_VERSION = "v4"
FEATURE_STORE_CACHE_VERSION = "v1"
FEATURE_CACHE_OFF = "off"
FEATURE_CACHE_MEMMAP = "memmap"
FEATURE_CACHE_MODES = (FEATURE_CACHE_OFF, FEATURE_CACHE_MEMMAP)
FEATURE_CACHE_DTYPES = ("float32", "float16")


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
        self.raw_fps = 30 if self.motion_format == G1_MOTION_FORMAT else 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type
        self.use_beats = use_beats
        self.beat_rep = beat_rep

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
        return state

    def __len__(self):
        return self.length

    def _open_feature_store(self):
        if not hasattr(self, "_feature_store"):
            self._feature_store = None
        if self._feature_store is None:
            self._feature_store = np.load(self.feature_store_path, mmap_mode="r")
        return self._feature_store

    def _load_feature(self, idx):
        feature_store_path = getattr(self, "feature_store_path", None)
        if feature_store_path:
            feature = self._open_feature_store()[idx]
        else:
            feature = np.load(self.data["filenames"][idx], mmap_mode="r")
        return torch.from_numpy(np.array(feature, dtype=np.float32, copy=True))

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = self._load_feature(idx)
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
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        beat_path = os.path.join(split_data_path, "beat_feats")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))
        beats = sorted(glob.glob(os.path.join(beat_path, "*.npz"))) if self.use_beats else []

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        all_beats = []
        all_motion_dist = []
        all_motion_spacing = []
        all_motion_mask = []
        all_audio_dist = []
        all_audio_mask = []
        if self.use_beats:
            assert len(motions) == len(features) == len(wavs) == len(beats)
            pairs = zip(motions, features, wavs, beats)
        else:
            assert len(motions) == len(features) == len(wavs)
            pairs = zip(motions, features, wavs)

        for items in pairs:
            if self.use_beats:
                motion, feature, wav, beat = items
            else:
                motion, feature, wav = items
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            if self.use_beats:
                b_name = os.path.splitext(os.path.basename(beat))[0]
                assert m_name == f_name == w_name == b_name, str((motion, feature, wav, beat))
            else:
                assert m_name == f_name == w_name, str((motion, feature, wav))
            # load motion
            with open(motion, "rb") as handle:
                data = pickle.load(handle)
            if self.motion_format == G1_MOTION_FORMAT:
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
        if self.use_beats:
            data["beatnames"] = all_beats
            data["motion_dist"] = np.stack(all_motion_dist, axis=0)
            data["motion_spacing"] = np.stack(all_motion_spacing, axis=0)
            data["motion_mask"] = np.stack(all_motion_mask, axis=0)
            data["audio_dist"] = np.stack(all_audio_dist, axis=0)
            data["audio_mask"] = np.stack(all_audio_mask, axis=0)
        return data

    def process_dataset(self, root_pos, local_q):
        if self.motion_format == G1_MOTION_FORMAT:
            return self.process_g1_dataset(root_pos, local_q)
        return self.process_smpl_dataset(root_pos, local_q)

    def process_g1_dataset(self, root_pos, local_q):
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        if local_q.shape[-1] != 33:
            raise ValueError(f"G1 motion q expected 33 channels, got {local_q.shape[-1]}")
        global_pose_vec_input = encode_g1_motion(
            root_pos,
            local_q[:, :, :4],
            local_q[:, :, 4:],
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
