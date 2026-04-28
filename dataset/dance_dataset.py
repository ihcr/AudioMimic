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

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from rotation_transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                 quaternion_multiply,
                                 quaternion_to_axis_angle)
from vis import SMPLSkeleton

DATASET_CACHE_VERSION = "v4"


def atomic_pickle_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def processed_dataset_cache_name(split_name, feature_type, use_beats, beat_rep):
    beat_tag = "beat" if use_beats else "nobeat"
    return f"processed_{split_name}_{feature_type}_{beat_tag}_{beat_rep}_{DATASET_CACHE_VERSION}.pkl"


def prune_legacy_processed_dataset_caches(backup_path, split_name, feature_type, use_beats, beat_rep):
    backup_path = Path(backup_path)
    beat_tag = "beat" if use_beats else "nobeat"
    current_name = processed_dataset_cache_name(split_name, feature_type, use_beats, beat_rep)
    pattern = f"processed_{split_name}_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
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
    ):
        self.data_path = data_path
        self.raw_fps = 60
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
            split_name, feature_type, use_beats, beat_rep
        )

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        prune_legacy_processed_dataset_caches(
            backup_path, split_name, feature_type, use_beats, beat_rep
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(
            np.array(np.load(filename_, mmap_mode="r"), dtype=np.float32, copy=True)
        )
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
