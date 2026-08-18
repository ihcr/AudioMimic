import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.g1_native_rvqvae_dataset import G1NativeRVQVAEDataset
from dataset.g1_streaming_state import (
    G1StreamingStateStatistics,
    boundary_state_from_motion,
)
from model.g1_streaming_rvqvae import build_g1_streaming_rvqvae_from_checkpoint


STREAMING_GENERATOR_CACHE_VERSION = "g1_streaming_generator_v1"
HISTORY_TOKENS = 64
HORIZON_TOKENS = 8
COMMIT_TOKENS = 4
COMMIT_FRAMES = 8
LOCK_STALE_SECONDS = 12 * 60 * 60
LOCK_POLL_SECONDS = 30


def checkpoint_fingerprint(path, block_size=1024 * 1024):
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(int(block_size)), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "short_sha256": digest.hexdigest()[:16],
        "size_bytes": int(stat.st_size),
    }


def _array_dict_digest(payload):
    digest = hashlib.sha256()
    for key in sorted(payload):
        value = payload[key]
        digest.update(str(key).encode("utf-8"))
        if isinstance(value, (str, int, float, bool)):
            digest.update(str(value).encode("utf-8"))
        else:
            array = np.asarray(value)
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def streaming_generator_cache_paths(
    cache_dir,
    codec_fingerprint,
    cache_limit_per_split=0,
    cache_data_len=0,
):
    tag = (
        f"{STREAMING_GENERATOR_CACHE_VERSION}_{codec_fingerprint['short_sha256']}"
        f"_k{HISTORY_TOKENS}_h{HORIZON_TOKENS}_c{COMMIT_TOKENS}"
    )
    if cache_limit_per_split:
        tag = f"{tag}_limit{int(cache_limit_per_split)}"
    if cache_data_len:
        tag = f"{tag}_n{int(cache_data_len)}"
    cache_dir = Path(cache_dir)
    return {
        "train": cache_dir / f"train_{tag}.pkl",
        "test": cache_dir / f"test_{tag}.pkl",
        "metadata": cache_dir / f"metadata_{tag}.json",
    }


def _write_pickle(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def cache_manifest_digest(metadata):
    payload = dict(metadata)
    payload.pop("manifest_digest", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_matches(
    metadata,
    codec_fingerprint,
    cache_limit_per_split,
    cache_data_len,
):
    return (
        metadata.get("cache_version") == STREAMING_GENERATOR_CACHE_VERSION
        and metadata.get("codec", {}).get("sha256") == codec_fingerprint["sha256"]
        and int(metadata.get("cache_limit_per_split", 0)) == int(cache_limit_per_split)
        and int(metadata.get("cache_data_len", 0)) == int(cache_data_len)
        and metadata.get("stream_contract")
        == {
            "history_tokens": HISTORY_TOKENS,
            "horizon_tokens": HORIZON_TOKENS,
            "commit_tokens": COMMIT_TOKENS,
            "commit_frames": COMMIT_FRAMES,
        }
        and metadata.get("manifest_digest") == cache_manifest_digest(metadata)
    )


def _cache_ready(paths, codec_fingerprint, cache_limit_per_split, cache_data_len):
    if any(not paths[key].is_file() for key in ("train", "test", "metadata")):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not _metadata_matches(
        metadata,
        codec_fingerprint,
        cache_limit_per_split,
        cache_data_len,
    ):
        raise ValueError(
            "streaming generator cache manifest does not match the frozen codec/contract. "
            f"Delete {paths['metadata'].parent} or rerun with --rebuild_generator_cache."
        )
    return True


def _acquire_cache_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            print(f"Waiting for streaming generator cache lock: {path}", flush=True)
            time.sleep(LOCK_POLL_SECONDS)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} created={time.time()}\n")
        return path


def _gather_frames(value, starts, length):
    indices = starts[:, None] + torch.arange(int(length), device=starts.device)[None]
    batch = torch.arange(value.shape[0], device=value.device)[:, None]
    return value[batch, indices]


@torch.inference_mode()
def encode_canonical_streaming_plans(
    codec,
    codec_motion,
    raw_motion,
    statistics,
    commit_frames=COMMIT_FRAMES,
):
    starts = torch.arange(
        0,
        codec_motion.shape[1] - int(codec.config.plan_frames) + 1,
        int(commit_frames),
        device=codec_motion.device,
    )
    batch = int(codec_motion.shape[0])
    plan_count = int(starts.numel())
    expanded_starts = starts[None].expand(batch, -1).reshape(-1)
    plans = _gather_frames(
        codec_motion[:, None]
        .expand(-1, plan_count, -1, -1)
        .reshape(batch * plan_count, codec_motion.shape[1], codec_motion.shape[2]),
        expanded_starts,
        codec.config.plan_frames,
    )
    states_physical = boundary_state_from_motion(
        raw_motion[:, None]
        .expand(-1, plan_count, -1, -1)
        .reshape(batch * plan_count, raw_motion.shape[1], raw_motion.shape[2]),
        expanded_starts,
        fps=statistics.fps,
    )
    pre_quant = codec.encode(plans, state=statistics.normalize_state(states_physical))
    tokens = codec.quantizer(pre_quant)["indices"]
    return (
        tokens.view(batch, plan_count, HORIZON_TOKENS, codec.config.num_codebooks),
        states_physical.view(batch, plan_count, -1),
        starts,
    )


@torch.inference_mode()
def _encode_split(
    split,
    codec,
    statistics,
    codec_normalizer,
    *,
    data_path,
    motion_prior_cache_dir,
    g1_fk_model_path,
    g1_root_quat_order,
    cache_batch_size,
    cache_device,
    cache_limit_per_split,
    cache_data_len,
):
    dataset = G1NativeRVQVAEDataset(
        split=split,
        data_path=data_path,
        backup_path=motion_prior_cache_dir,
        motion_format="g1_yaw_delta",
        g1_fk_model_path=g1_fk_model_path,
        g1_root_quat_order=g1_root_quat_order,
        cache_limit_per_split=cache_limit_per_split,
        data_len=cache_data_len,
    )
    dataset_mean, dataset_std = dataset.normalizer.tensors(device=cache_device)
    codec_mean = torch.as_tensor(
        codec_normalizer["mean"], device=cache_device, dtype=torch.float32
    ).view(1, 1, -1)
    codec_std = torch.as_tensor(
        codec_normalizer["std"], device=cache_device, dtype=torch.float32
    ).view(1, 1, -1)
    if not torch.equal(dataset_mean, codec_mean) or not torch.equal(dataset_std, codec_std):
        raise ValueError("motion cache normalizer does not exactly match the frozen codec")
    loader = DataLoader(
        dataset,
        batch_size=int(cache_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    token_batches = []
    state_batches = []
    plan_starts = None
    stems = []
    source_paths = []
    for batch in tqdm(loader, desc=f"Build streaming generator cache {split}", unit="batch"):
        normalized = batch["motion"].to(cache_device).float()
        raw = normalized * dataset_std + dataset_mean
        codec_motion = (raw - codec_mean) / codec_std
        tokens, states, starts = encode_canonical_streaming_plans(
            codec,
            codec_motion,
            raw,
            statistics,
        )
        if plan_starts is None:
            plan_starts = starts.detach().cpu()
        elif not torch.equal(plan_starts, starts.detach().cpu()):
            raise RuntimeError("canonical plan starts changed within one cache build")
        token_batches.append(tokens.detach().cpu().to(torch.int16))
        state_batches.append(states.detach().cpu().to(torch.float32))
        stems.extend(list(batch["stem"]))
        source_paths.extend(list(batch["source_path"]))
    return {
        "canonical_tokens": torch.cat(token_batches).numpy(),
        "canonical_states": torch.cat(state_batches).numpy(),
        "plan_starts": plan_starts.numpy(),
        "stems": stems,
        "source_paths": source_paths,
    }, dataset.metadata, dataset.normalizer.state_dict()


def build_g1_streaming_generator_cache(
    *,
    codec_checkpoint,
    data_path,
    motion_prior_cache_dir,
    generator_cache_dir,
    cache_batch_size=64,
    cache_device="cuda",
    cache_limit_per_split=0,
    cache_data_len=0,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
):
    fingerprint = checkpoint_fingerprint(codec_checkpoint)
    paths = streaming_generator_cache_paths(
        generator_cache_dir,
        fingerprint,
        cache_limit_per_split=cache_limit_per_split,
        cache_data_len=cache_data_len,
    )
    device = torch.device(cache_device)
    checkpoint = torch.load(codec_checkpoint, map_location=device, weights_only=False)
    codec = build_g1_streaming_rvqvae_from_checkpoint(checkpoint).to(device).eval()
    codec.requires_grad_(False)
    if codec.config.plan_frames != 16 or codec.config.num_codebooks != 8:
        raise ValueError("streaming generator cache requires the formal H8/RVQ8 codec")
    statistics = G1StreamingStateStatistics.from_state_dict(checkpoint["streaming_statistics"])
    split_metadata = {}
    split_counts = {}
    source_normalizer = None
    for split in ("train", "test"):
        payload, metadata, normalizer = _encode_split(
            split,
            codec,
            statistics,
            checkpoint["normalizer"],
            data_path=data_path,
            motion_prior_cache_dir=motion_prior_cache_dir,
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
            cache_batch_size=cache_batch_size,
            cache_device=device,
            cache_limit_per_split=cache_limit_per_split,
            cache_data_len=cache_data_len,
        )
        _write_pickle(payload, paths[split])
        split_metadata[split] = metadata
        split_counts[split] = int(payload["canonical_tokens"].shape[0])
        source_normalizer = normalizer
    metadata = {
        "cache_version": STREAMING_GENERATOR_CACHE_VERSION,
        "codec": {**fingerprint, "config": codec.config.asdict()},
        "codec_normalizer_digest": _array_dict_digest(checkpoint["normalizer"]),
        "source_normalizer_digest": _array_dict_digest(source_normalizer),
        "streaming_statistics_digest": _array_dict_digest(statistics.state_dict()),
        "stream_contract": {
            "history_tokens": HISTORY_TOKENS,
            "horizon_tokens": HORIZON_TOKENS,
            "commit_tokens": COMMIT_TOKENS,
            "commit_frames": COMMIT_FRAMES,
        },
        "motion_format": "g1_yaw_delta",
        "frames": 150,
        "plan_count": 17,
        "train_count": split_counts["train"],
        "test_count": split_counts["test"],
        "cache_limit_per_split": int(cache_limit_per_split),
        "cache_data_len": int(cache_data_len),
        "source_motion_cache": split_metadata,
    }
    metadata["manifest_digest"] = cache_manifest_digest(metadata)
    _write_json(metadata, paths["metadata"])
    return paths


def ensure_g1_streaming_generator_cache(
    *,
    codec_checkpoint,
    data_path,
    motion_prior_cache_dir,
    generator_cache_dir,
    cache_batch_size=64,
    cache_device="cuda",
    cache_limit_per_split=0,
    cache_data_len=0,
    rebuild_cache=False,
    g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    g1_root_quat_order="xyzw",
):
    fingerprint = checkpoint_fingerprint(codec_checkpoint)
    paths = streaming_generator_cache_paths(
        generator_cache_dir,
        fingerprint,
        cache_limit_per_split=cache_limit_per_split,
        cache_data_len=cache_data_len,
    )
    if not rebuild_cache and _cache_ready(
        paths,
        fingerprint,
        cache_limit_per_split,
        cache_data_len,
    ):
        return paths
    lock = _acquire_cache_lock(paths["metadata"].with_suffix(".json.lock"))
    try:
        if not rebuild_cache and _cache_ready(
            paths,
            fingerprint,
            cache_limit_per_split,
            cache_data_len,
        ):
            return paths
        return build_g1_streaming_generator_cache(
            codec_checkpoint=codec_checkpoint,
            data_path=data_path,
            motion_prior_cache_dir=motion_prior_cache_dir,
            generator_cache_dir=generator_cache_dir,
            cache_batch_size=cache_batch_size,
            cache_device=cache_device,
            cache_limit_per_split=cache_limit_per_split,
            cache_data_len=cache_data_len,
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
        )
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


class G1StreamingGeneratorDataset(Dataset):
    def __init__(
        self,
        *,
        codec_checkpoint,
        data_path,
        motion_prior_cache_dir,
        generator_cache_dir,
        split,
        cache_batch_size=64,
        cache_device="cuda",
        cache_limit_per_split=0,
        cache_data_len=0,
        rebuild_cache=False,
        data_len=0,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        effective_data_len = int(data_len) if data_len else int(cache_data_len)
        paths = ensure_g1_streaming_generator_cache(
            codec_checkpoint=codec_checkpoint,
            data_path=data_path,
            motion_prior_cache_dir=motion_prior_cache_dir,
            generator_cache_dir=generator_cache_dir,
            cache_batch_size=cache_batch_size,
            cache_device=cache_device,
            cache_limit_per_split=cache_limit_per_split,
            cache_data_len=cache_data_len,
            rebuild_cache=rebuild_cache,
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
        )
        self.base = G1NativeRVQVAEDataset(
            split=split,
            data_path=data_path,
            backup_path=motion_prior_cache_dir,
            motion_format="g1_yaw_delta",
            g1_fk_model_path=g1_fk_model_path,
            g1_root_quat_order=g1_root_quat_order,
            cache_limit_per_split=cache_limit_per_split,
            data_len=effective_data_len,
        )
        with open(paths[split], "rb") as handle:
            payload = pickle.load(handle)
        self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        count = effective_data_len if effective_data_len else len(payload["stems"])
        self.canonical_tokens = np.asarray(payload["canonical_tokens"][:count], dtype=np.int64)
        self.canonical_states = np.asarray(payload["canonical_states"][:count], dtype=np.float32)
        self.plan_starts = np.asarray(payload["plan_starts"], dtype=np.int64)
        if len(self.base) != count:
            raise ValueError("motion and streaming generator cache counts do not match")
        if list(self.base.stems) != list(payload["stems"][:count]):
            raise ValueError("motion and streaming generator cache order does not match")
        if self.canonical_tokens.shape[1:] != (17, HORIZON_TOKENS, 8):
            raise ValueError("canonical tokens must have shape [N,17,8,8]")
        if self.canonical_states.shape[1:] != (17, 66):
            raise ValueError("canonical states must have shape [N,17,66]")

    @property
    def normalizer(self):
        return self.base.normalizer

    @property
    def manifest_digest(self):
        return self.metadata["manifest_digest"]

    def __len__(self):
        return int(self.canonical_tokens.shape[0])

    def __getitem__(self, index):
        return {
            "motion": torch.from_numpy(self.base.motion[index]),
            "canonical_tokens": torch.from_numpy(self.canonical_tokens[index]),
            "canonical_states": torch.from_numpy(self.canonical_states[index]),
            "index": torch.tensor(index, dtype=torch.long),
        }
