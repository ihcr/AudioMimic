import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.g1_motion_prior_dataset import G1MotionPriorNormalizer
from dataset.g1_streaming_generator_dataset import checkpoint_fingerprint
from dataset.g1_streaming_state import (
    G1StreamingStateStatistics,
    boundary_state_from_motion,
)
from dataset.motion_representation import encode_g1_motion_for_format
from model.g1_hybrid_streaming_codec import (
    build_g1_hybrid_streaming_codec_from_checkpoint,
)
from model.g1_paper_faithful_dc_streaming import (
    COMMIT_TOKENS,
    HISTORY_TOKENS,
    PLAN_TOKENS,
    TRAINING_ROLLOUT_TOKENS,
)


EXPERIMENT_ID = "EXP-20260723-v6f-w-paper-faithful-dc-streaming"
PAPER_DC_CACHE_VERSION = "g1_paper_faithful_dc_v3_causal_history"
MINIMUM_LATENT_WIDTH_CACHE_VERSION = (
    "g1_paper_faithful_dc_v4_latent_width"
)
STRUCTURAL_QUANTIZER_CACHE_VERSION = (
    "g1_paper_faithful_dc_v5_structural_quantizer"
)
REPRESENTATION_BOUND_CACHE_VERSIONS = (
    MINIMUM_LATENT_WIDTH_CACHE_VERSION,
    STRUCTURAL_QUANTIZER_CACHE_VERSION,
)
MOTION_FRAMES_PER_TOKEN = 2
COMMIT_FRAMES = COMMIT_TOKENS * MOTION_FRAMES_PER_TOKEN
CAUSAL_HISTORY_VALID_TOKENS = HISTORY_TOKENS - COMMIT_TOKENS
TOTAL_TOKENS = HISTORY_TOKENS + TRAINING_ROLLOUT_TOKENS
TOTAL_COMMITS = TOTAL_TOKENS // COMMIT_TOKENS
ROLLOUT_COMMITS = TRAINING_ROLLOUT_TOKENS // COMMIT_TOKENS
ENCODE_WINDOW_FRAMES = (TOTAL_COMMITS - 1) * COMMIT_FRAMES + PLAN_TOKENS * 2
COMMITTED_WINDOW_FRAMES = TOTAL_TOKENS * MOTION_FRAMES_PER_TOKEN
LOCK_STALE_SECONDS = 12 * 60 * 60
LOCK_POLL_SECONDS = 30


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _manifest_digest(payload):
    payload = _jsonable(dict(payload))
    payload.pop("manifest_digest", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stream_contract():
    return {
        "history_tokens": HISTORY_TOKENS,
        "causal_history_valid_tokens": CAUSAL_HISTORY_VALID_TOKENS,
        "causal_history_padding": "left",
        "training_rollout_tokens": TRAINING_ROLLOUT_TOKENS,
        "plan_tokens": PLAN_TOKENS,
        "commit_tokens": COMMIT_TOKENS,
        "encode_window_frames": ENCODE_WINDOW_FRAMES,
        "committed_window_frames": COMMITTED_WINDOW_FRAMES,
        "rollout_plan_targets": ROLLOUT_COMMITS,
    }


def paper_dc_cache_paths(cache_dir, provenance):
    tag = (
        f"{provenance['experiment_id']}_"
        f"{provenance['cache_schema_version']}_"
        f"{provenance['provenance_digest']}"
    )
    root = Path(cache_dir) / tag
    paths = {
        "root": root,
        "metadata": root / "metadata.json",
        "lock": root.with_suffix(".lock"),
    }
    for split in ("train", "test"):
        paths[f"{split}_q0"] = root / f"{split}_q0.npy"
        paths[f"{split}_residual"] = root / f"{split}_residual.npy"
        paths[f"{split}_states"] = root / f"{split}_states.npy"
        paths[f"{split}_motion"] = root / f"{split}_motion.npy"
        paths[f"{split}_plan_q0"] = root / f"{split}_plan_q0.npy"
        paths[f"{split}_plan_residual"] = root / f"{split}_plan_residual.npy"
    return paths


def _write_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
                path.unlink()
                continue
            print(f"Waiting for paper D+C cache lock: {path}", flush=True)
            time.sleep(LOCK_POLL_SECONDS)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} created={time.time()}\n")
        return path


def _parse_slice_stem(stem):
    sequence_id, slice_index = stem.rsplit("_slice", 1)
    return sequence_id, int(slice_index)


def _load_full_g1_sequence(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    root_pos = np.asarray(payload["root_pos"], dtype=np.float32)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError(f"{path}: invalid root_pos shape {root_pos.shape}")
    if root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(f"{path}: invalid root_rot shape {root_rot.shape}")
    if dof_pos.shape != (root_pos.shape[0], 29):
        raise ValueError(f"{path}: invalid dof_pos shape {dof_pos.shape}")
    encoded = encode_g1_motion_for_format(
        torch.from_numpy(root_pos).unsqueeze(0),
        torch.from_numpy(root_rot).unsqueeze(0),
        torch.from_numpy(dof_pos).unsqueeze(0),
        motion_format="g1_yaw_delta",
    ).squeeze(0)
    if not torch.isfinite(encoded).all():
        raise ValueError(f"{path}: encoded full sequence contains non-finite values")
    return encoded.float()


def _split_descriptors(data_path, raw_root, split, stride_frames, cache_data_len):
    slice_paths = sorted(
        (Path(data_path) / split / "motions_sliced").glob("*.pkl")
    )
    by_sequence = {}
    for path in slice_paths:
        sequence_id, slice_index = _parse_slice_stem(path.stem)
        by_sequence.setdefault(sequence_id, []).append(slice_index)
    descriptors = []
    source_lengths = {}
    for sequence_id in sorted(by_sequence):
        raw_path = Path(raw_root) / f"{sequence_id}.pkl"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        with open(raw_path, "rb") as handle:
            payload = pickle.load(handle)
        frame_count = int(np.asarray(payload["root_pos"]).shape[0])
        source_lengths[sequence_id] = frame_count
        for slice_index in sorted(set(by_sequence[sequence_id])):
            start = int(slice_index) * int(stride_frames)
            if start + ENCODE_WINDOW_FRAMES <= frame_count:
                descriptors.append((sequence_id, start))
    if cache_data_len:
        descriptors = descriptors[: int(cache_data_len)]
    if not descriptors:
        raise ValueError(f"{split}: no {ENCODE_WINDOW_FRAMES}-frame D+C windows")
    return descriptors, source_lengths


def _current_cache_provenance(
    codec_checkpoint,
    data_path,
    cache_data_len,
    *,
    experiment_id=EXPERIMENT_ID,
    cache_schema_version=PAPER_DC_CACHE_VERSION,
):
    fingerprint = checkpoint_fingerprint(codec_checkpoint)
    checkpoint = torch.load(
        codec_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    data_path = Path(data_path).resolve()
    data_metadata = json.loads(
        (data_path / "metadata.json").read_text(encoding="utf-8")
    )
    raw_root = Path(data_metadata["source_motion_dir"])
    if not raw_root.is_absolute():
        raw_root = data_path.parents[1] / raw_root
    raw_root = raw_root.resolve()
    stride_frames = int(
        round(
            float(data_metadata["stride_seconds"])
            * float(data_metadata["target_fps"])
        )
    )
    split_inputs = {}
    source_split = {
        "data_path": str(data_path),
        "data_metadata_digest": _manifest_digest(data_metadata),
        "raw_source_root": str(raw_root),
        "source_stride_frames": stride_frames,
        "cache_data_len": int(cache_data_len),
        "splits": {},
    }
    for split in ("train", "test"):
        descriptors, source_lengths = _split_descriptors(
            data_path,
            raw_root,
            split,
            stride_frames,
            cache_data_len,
        )
        descriptor_payload = [
            {"sequence_id": sequence_id, "start_frame": int(start)}
            for sequence_id, start in descriptors
        ]
        split_inputs[split] = {
            "descriptors": descriptors,
            "source_lengths": source_lengths,
        }
        source_split["splits"][split] = {
            "count": len(descriptors),
            "descriptors_digest": _manifest_digest(
                {"descriptors": descriptor_payload}
            ),
            "source_lengths_digest": _manifest_digest(
                {"source_lengths": source_lengths}
            ),
        }
    codec_config = checkpoint.get("config", {})
    code_dim = int(codec_config.get("code_dim", 0))
    if code_dim <= 0:
        raise ValueError("codec checkpoint is missing a positive code_dim")
    codec_training_seed = checkpoint.get("training_config", {}).get("seed")
    if cache_schema_version in REPRESENTATION_BOUND_CACHE_VERSIONS:
        if codec_training_seed is None:
            raise ValueError(
                "minimum-latent-width cache requires codec training seed"
            )
        if checkpoint.get("training_config", {}).get("experiment_id") != str(
            experiment_id
        ):
            raise ValueError(
                "minimum-latent-width codec experiment provenance mismatch"
            )
    normalization_sha256 = _manifest_digest(
        {"normalizer": checkpoint["normalizer"]}
    )
    statistics_sha256 = _manifest_digest(
        {"streaming_statistics": checkpoint["streaming_statistics"]}
    )
    provenance = {
        "experiment_id": str(experiment_id),
        "cache_schema_version": str(cache_schema_version),
        "codec_sha256": fingerprint["sha256"],
        "code_dim": code_dim,
        "codebook_size": int(codec_config.get("codebook_size", 0)),
        "num_codebooks": int(codec_config.get("num_codebooks", 0)),
        "structural_quantizer": codec_config.get("structural_quantizer", "vq"),
        "quantizer_manifest": codec_config.get("quantizer", {
            "type": codec_config.get("structural_quantizer", "vq"),
            "vocabulary_size": int(codec_config.get("codebook_size", 0)),
            "num_token_streams": int(codec_config.get("num_codebooks", 0)),
        }),
        "codec_training_seed": (
            None if codec_training_seed is None else int(codec_training_seed)
        ),
        "stream_contract": _stream_contract(),
        "source_split": source_split,
        "normalization_digest": normalization_sha256,
        "normalizer_sha256": normalization_sha256,
        "statistics_digest": statistics_sha256,
        "streaming_statistics_sha256": statistics_sha256,
    }
    provenance["provenance_digest"] = _manifest_digest(provenance)
    return provenance, split_inputs, fingerprint, data_metadata


@torch.inference_mode()
def _encode_window_batch(
    codec,
    statistics,
    normalized_windows,
    raw_windows,
):
    batch = normalized_windows.shape[0]
    starts = (
        torch.arange(
            TOTAL_COMMITS,
            device=normalized_windows.device,
            dtype=torch.long,
        )
        * COMMIT_FRAMES
    )
    expanded_starts = starts.unsqueeze(0).expand(batch, -1).reshape(-1)
    expanded_normalized = normalized_windows[:, None].expand(
        -1,
        TOTAL_COMMITS,
        -1,
        -1,
    ).reshape(
        batch * TOTAL_COMMITS,
        normalized_windows.shape[1],
        normalized_windows.shape[2],
    )
    expanded_raw = raw_windows[:, None].expand(
        -1,
        TOTAL_COMMITS,
        -1,
        -1,
    ).reshape(
        batch * TOTAL_COMMITS,
        raw_windows.shape[1],
        raw_windows.shape[2],
    )
    frame_offsets = torch.arange(
        codec.config.plan_frames,
        device=normalized_windows.device,
    )
    gather = expanded_starts[:, None] + frame_offsets[None]
    plans = expanded_normalized.gather(
        1,
        gather[..., None].expand(-1, -1, normalized_windows.shape[-1]),
    )
    state_physical = boundary_state_from_motion(
        expanded_raw,
        expanded_starts,
        fps=statistics.fps,
        state_spec=statistics.state_spec,
    )
    components = codec.encode_components(
        plans,
        statistics.normalize_state(state_physical),
    )
    q0 = components["q0_ids"].view(
        batch,
        TOTAL_COMMITS,
        PLAN_TOKENS,
        1,
    )[..., 0]
    residual = components["residual"].view(
        batch,
        TOTAL_COMMITS,
        PLAN_TOKENS,
        codec.config.code_dim,
    )
    states = state_physical.view(batch, TOTAL_COMMITS, -1)
    committed_q0 = q0[:, :, :COMMIT_TOKENS].reshape(batch, TOTAL_TOKENS)
    committed_residual = residual[:, :, :COMMIT_TOKENS].reshape(
        batch,
        TOTAL_TOKENS,
        codec.config.code_dim,
    )
    return (
        committed_q0,
        committed_residual,
        states,
        q0[:, -ROLLOUT_COMMITS:],
        residual[:, -ROLLOUT_COMMITS:],
    )


def _cache_ready(paths, provenance, cache_data_len):
    required = ["metadata"]
    for split in ("train", "test"):
        required.extend(
            (
                f"{split}_q0",
                f"{split}_residual",
                f"{split}_states",
                f"{split}_motion",
                f"{split}_plan_q0",
                f"{split}_plan_residual",
            )
        )
    if any(not paths[key].is_file() for key in required):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    valid = (
        metadata.get("cache_version") == provenance["cache_schema_version"]
        and metadata.get("experiment_id") == provenance["experiment_id"]
        and metadata.get("cache_schema_version")
        == provenance["cache_schema_version"]
        and metadata.get("codec", {}).get("sha256")
        == provenance["codec_sha256"]
        and int(metadata.get("cache_data_len", 0)) == int(cache_data_len)
        and metadata.get("stream_contract")
        == provenance["stream_contract"]
        and metadata.get("source_split") == provenance["source_split"]
        and metadata.get("normalization_digest")
        == provenance["normalization_digest"]
        and metadata.get("statistics_digest")
        == provenance["statistics_digest"]
        and metadata.get("provenance_digest")
        == provenance["provenance_digest"]
        and metadata.get("manifest_digest") == _manifest_digest(metadata)
    )
    if provenance["cache_schema_version"] in REPRESENTATION_BOUND_CACHE_VERSIONS:
        valid = valid and (
            int(metadata.get("code_dim", 0)) == provenance["code_dim"]
            and int(metadata.get("codebook_size", 0))
            == provenance["codebook_size"]
            and int(metadata.get("num_codebooks", 0))
            == provenance["num_codebooks"]
            and metadata.get("structural_quantizer")
            == provenance["structural_quantizer"]
            and metadata.get("quantizer_manifest")
            == provenance["quantizer_manifest"]
            and metadata.get("codec_training_seed")
            == provenance["codec_training_seed"]
            and metadata.get("normalizer_sha256")
            == provenance["normalizer_sha256"]
            and metadata.get("streaming_statistics_sha256")
            == provenance["streaming_statistics_sha256"]
        )
    if not valid:
        raise ValueError(
            "paper D+C cache provenance mismatch; rebuild the exact namespace "
            f"at {paths['root']}"
        )
    return True


def _build_split(
    *,
    split,
    descriptors,
    source_lengths,
    raw_root,
    codec,
    codec_checkpoint,
    statistics,
    paths,
    batch_size,
    device,
):
    count = len(descriptors)
    code_dim = int(codec.config.code_dim)
    q0_memmap = np.lib.format.open_memmap(
        paths[f"{split}_q0"],
        mode="w+",
        dtype=np.int16,
        shape=(count, TOTAL_TOKENS),
    )
    residual_memmap = np.lib.format.open_memmap(
        paths[f"{split}_residual"],
        mode="w+",
        dtype=np.float16,
        shape=(count, TOTAL_TOKENS, code_dim),
    )
    states_memmap = np.lib.format.open_memmap(
        paths[f"{split}_states"],
        mode="w+",
        dtype=np.float16,
        shape=(count, TOTAL_COMMITS, statistics.state_mean.shape[0]),
    )
    motion_memmap = np.lib.format.open_memmap(
        paths[f"{split}_motion"],
        mode="w+",
        dtype=np.float16,
        shape=(count, ENCODE_WINDOW_FRAMES, 34),
    )
    plan_q0_memmap = np.lib.format.open_memmap(
        paths[f"{split}_plan_q0"],
        mode="w+",
        dtype=np.int16,
        shape=(count, ROLLOUT_COMMITS, PLAN_TOKENS),
    )
    plan_residual_memmap = np.lib.format.open_memmap(
        paths[f"{split}_plan_residual"],
        mode="w+",
        dtype=np.float16,
        shape=(count, ROLLOUT_COMMITS, PLAN_TOKENS, code_dim),
    )
    normalizer = G1MotionPriorNormalizer.from_state_dict(
        codec_checkpoint["normalizer"]
    )
    mean = torch.as_tensor(normalizer.mean, device=device).view(1, 1, -1)
    std = torch.as_tensor(normalizer.std, device=device).view(1, 1, -1)
    total = torch.zeros(code_dim, dtype=torch.float64)
    total_square = torch.zeros(code_dim, dtype=torch.float64)
    residual_count = 0
    active_counts = torch.zeros(codec.config.codebook_size, dtype=torch.long)
    current_sequence_id = None
    current_encoded = None
    for cursor in tqdm(
        range(0, count, int(batch_size)),
        desc=f"Build paper D+C cache {split}",
        unit="batch",
    ):
        batch_descriptors = descriptors[cursor : cursor + int(batch_size)]
        windows = []
        for sequence_id, start in batch_descriptors:
            if sequence_id != current_sequence_id:
                current_encoded = _load_full_g1_sequence(
                    Path(raw_root) / f"{sequence_id}.pkl"
                )
                current_sequence_id = sequence_id
            window = current_encoded[start : start + ENCODE_WINDOW_FRAMES].clone()
            if window.shape != (ENCODE_WINDOW_FRAMES, 34):
                raise ValueError(
                    f"{sequence_id}:{start} produced incomplete encoded window"
                )
            # Each training window has its own cold boundary; the first local
            # root displacement and yaw delta therefore match the codec's
            # original five-second slice convention.
            window[0, :2] = 0.0
            window[0, 3] = 0.0
            window[0, 4] = 1.0
            windows.append(window)
        raw_windows = torch.stack(windows).to(device)
        normalized_windows = (raw_windows - mean) / std
        (
            batch_q0,
            batch_residual,
            batch_states,
            batch_plan_q0,
            batch_plan_residual,
        ) = _encode_window_batch(
            codec,
            statistics,
            normalized_windows,
            raw_windows,
        )
        end = cursor + len(batch_descriptors)
        q0_cpu = batch_q0.cpu().to(torch.int64)
        residual_cpu = batch_residual.cpu().float()
        q0_memmap[cursor:end] = q0_cpu.numpy().astype(np.int16)
        residual_memmap[cursor:end] = residual_cpu.numpy().astype(np.float16)
        states_memmap[cursor:end] = (
            batch_states.cpu().float().numpy().astype(np.float16)
        )
        plan_q0_memmap[cursor:end] = (
            batch_plan_q0.cpu().to(torch.int64).numpy().astype(np.int16)
        )
        plan_residual_memmap[cursor:end] = (
            batch_plan_residual.cpu().float().numpy().astype(np.float16)
        )
        motion_memmap[cursor:end] = (
            normalized_windows
            .cpu()
            .float()
            .numpy()
            .astype(np.float16)
        )
        flat = torch.cat(
            (
                residual_cpu.reshape(-1, code_dim),
                batch_plan_residual.cpu().float().reshape(-1, code_dim),
            ),
            dim=0,
        ).double()
        total += flat.sum(dim=0)
        total_square += flat.square().sum(dim=0)
        residual_count += int(flat.shape[0])
        active_counts += torch.bincount(
            q0_cpu.reshape(-1),
            minlength=codec.config.codebook_size,
        )
    for memmap in (
        q0_memmap,
        residual_memmap,
        states_memmap,
        motion_memmap,
        plan_q0_memmap,
        plan_residual_memmap,
    ):
        memmap.flush()
    residual_mean = total / max(residual_count, 1)
    variance = total_square / max(residual_count, 1) - residual_mean.square()
    probability = active_counts.double() / active_counts.sum().clamp_min(1)
    nonzero = probability > 0
    perplexity = torch.exp(
        -(probability[nonzero] * probability[nonzero].log()).sum()
    )
    return {
        "count": count,
        "descriptors": [
            {"sequence_id": sequence_id, "start_frame": int(start)}
            for sequence_id, start in descriptors
        ],
        "source_lengths": source_lengths,
        "residual_mean": residual_mean.float().tolist(),
        "residual_std": variance.clamp_min(1e-12).sqrt().clamp_min(1e-6).float().tolist(),
        "residual_count": residual_count,
        "q0_active_codes": int(active_counts.gt(0).sum()),
        "q0_perplexity": float(perplexity),
    }


def build_g1_paper_faithful_dc_cache(
    *,
    codec_checkpoint,
    data_path,
    generator_cache_dir,
    experiment_id=EXPERIMENT_ID,
    cache_schema_version=PAPER_DC_CACHE_VERSION,
    cache_batch_size=8,
    cache_device="cuda",
    cache_data_len=0,
    _provenance=None,
    _split_inputs=None,
    _fingerprint=None,
    _data_metadata=None,
):
    if (
        _provenance is None
        or _split_inputs is None
        or _fingerprint is None
        or _data_metadata is None
    ):
        (
            _provenance,
            _split_inputs,
            _fingerprint,
            _data_metadata,
        ) = _current_cache_provenance(
            codec_checkpoint,
            data_path,
            cache_data_len,
            experiment_id=experiment_id,
            cache_schema_version=cache_schema_version,
        )
    paths = paper_dc_cache_paths(generator_cache_dir, _provenance)
    paths["root"].mkdir(parents=True, exist_ok=True)
    device = torch.device(cache_device)
    checkpoint = torch.load(codec_checkpoint, map_location=device, weights_only=False)
    codec = build_g1_hybrid_streaming_codec_from_checkpoint(checkpoint).to(device).eval()
    codec.requires_grad_(False)
    if codec.config.plan_frames != 16 or codec.config.num_codebooks != 1:
        raise ValueError("paper D+C cache requires the formal H8 single-q0 codec")
    statistics = G1StreamingStateStatistics.from_state_dict(
        checkpoint["streaming_statistics"]
    )
    data_metadata = _data_metadata
    raw_root = Path(_provenance["source_split"]["raw_source_root"])
    stride_frames = int(_provenance["source_split"]["source_stride_frames"])
    split_metadata = {}
    for split in ("train", "test"):
        descriptors = _split_inputs[split]["descriptors"]
        source_lengths = _split_inputs[split]["source_lengths"]
        split_metadata[split] = _build_split(
            split=split,
            descriptors=descriptors,
            source_lengths=source_lengths,
            raw_root=raw_root,
            codec=codec,
            codec_checkpoint=checkpoint,
            statistics=statistics,
            paths=paths,
            batch_size=cache_batch_size,
            device=device,
        )
    metadata = {
        "cache_version": _provenance["cache_schema_version"],
        "experiment_id": _provenance["experiment_id"],
        "cache_schema_version": _provenance["cache_schema_version"],
        "code_dim": _provenance["code_dim"],
        "codebook_size": _provenance["codebook_size"],
        "num_codebooks": _provenance["num_codebooks"],
        "structural_quantizer": _provenance["structural_quantizer"],
        "quantizer_manifest": _provenance["quantizer_manifest"],
        "codec_training_seed": _provenance["codec_training_seed"],
        "codec": {
            **_fingerprint,
            "config": codec.manifest(),
        },
        "normalization_digest": _provenance["normalization_digest"],
        "normalizer_sha256": _provenance["normalizer_sha256"],
        "statistics_digest": _provenance["statistics_digest"],
        "streaming_statistics_sha256": _provenance[
            "streaming_statistics_sha256"
        ],
        "source_split": _provenance["source_split"],
        "provenance_digest": _provenance["provenance_digest"],
        "cache_data_len": int(cache_data_len),
        "data_path": str(Path(data_path).resolve()),
        "raw_source_root": str(raw_root),
        "source_stride_frames": stride_frames,
        "motion_format": "g1_yaw_delta",
        "residual_target": "encoder_pre_quant_minus_frozen_q0_raw",
        "residual_storage_dtype": "float16",
        "state_storage_dtype": "float16",
        "motion_storage_dtype": "float16",
        "stream_contract": _provenance["stream_contract"],
        "splits": split_metadata,
    }
    metadata["manifest_digest"] = _manifest_digest(metadata)
    _write_json_atomic(metadata, paths["metadata"])
    return paths


def ensure_g1_paper_faithful_dc_cache(
    *,
    codec_checkpoint,
    data_path,
    generator_cache_dir,
    experiment_id=EXPERIMENT_ID,
    cache_schema_version=PAPER_DC_CACHE_VERSION,
    cache_batch_size=8,
    cache_device="cuda",
    cache_data_len=0,
    rebuild_cache=False,
):
    (
        provenance,
        split_inputs,
        fingerprint,
        data_metadata,
    ) = _current_cache_provenance(
        codec_checkpoint,
        data_path,
        cache_data_len,
        experiment_id=experiment_id,
        cache_schema_version=cache_schema_version,
    )
    paths = paper_dc_cache_paths(generator_cache_dir, provenance)
    if not rebuild_cache and _cache_ready(paths, provenance, cache_data_len):
        return paths
    lock = _acquire_lock(paths["lock"])
    try:
        if not rebuild_cache and _cache_ready(
            paths,
            provenance,
            cache_data_len,
        ):
            return paths
        return build_g1_paper_faithful_dc_cache(
            codec_checkpoint=codec_checkpoint,
            data_path=data_path,
            generator_cache_dir=generator_cache_dir,
            experiment_id=experiment_id,
            cache_schema_version=cache_schema_version,
            cache_batch_size=cache_batch_size,
            cache_device=cache_device,
            cache_data_len=cache_data_len,
            _provenance=provenance,
            _split_inputs=split_inputs,
            _fingerprint=fingerprint,
            _data_metadata=data_metadata,
        )
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


class G1PaperFaithfulDCDataset(Dataset):
    def __init__(
        self,
        *,
        codec_checkpoint,
        data_path,
        generator_cache_dir,
        experiment_id=EXPERIMENT_ID,
        cache_schema_version=PAPER_DC_CACHE_VERSION,
        split,
        cache_batch_size=8,
        cache_device="cuda",
        cache_data_len=0,
        rebuild_cache=False,
        data_len=0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        paths = ensure_g1_paper_faithful_dc_cache(
            codec_checkpoint=codec_checkpoint,
            data_path=data_path,
            generator_cache_dir=generator_cache_dir,
            experiment_id=experiment_id,
            cache_schema_version=cache_schema_version,
            cache_batch_size=cache_batch_size,
            cache_device=cache_device,
            cache_data_len=cache_data_len,
            rebuild_cache=rebuild_cache,
        )
        self.split = split
        self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        available = int(self.metadata["splits"][split]["count"])
        count = min(int(data_len), available) if data_len else available
        self.q0 = np.load(paths[f"{split}_q0"], mmap_mode="r")[:count]
        self.residual = np.load(paths[f"{split}_residual"], mmap_mode="r")[:count]
        self.states = np.load(paths[f"{split}_states"], mmap_mode="r")[:count]
        self.motion = np.load(paths[f"{split}_motion"], mmap_mode="r")[:count]
        self.plan_q0 = np.load(paths[f"{split}_plan_q0"], mmap_mode="r")[:count]
        self.plan_residual = np.load(
            paths[f"{split}_plan_residual"],
            mmap_mode="r",
        )[:count]
        if self.q0.shape[1:] != (TOTAL_TOKENS,):
            raise ValueError("paper D+C q0 cache shape mismatch")
        if self.residual.shape[1:2] != (TOTAL_TOKENS,):
            raise ValueError("paper D+C residual cache shape mismatch")
        state_dim = int(self.metadata["codec"]["config"]["state_dim"])
        if self.states.shape[1:] != (TOTAL_COMMITS, state_dim):
            raise ValueError("paper D+C state cache shape mismatch")
        if self.motion.shape[1:] != (ENCODE_WINDOW_FRAMES, 34):
            raise ValueError("paper D+C motion cache shape mismatch")
        if self.plan_q0.shape[1:] != (ROLLOUT_COMMITS, PLAN_TOKENS):
            raise ValueError("paper D+C plan-q0 cache shape mismatch")
        if self.plan_residual.shape[1:3] != (
            ROLLOUT_COMMITS,
            PLAN_TOKENS,
        ):
            raise ValueError("paper D+C plan-residual cache shape mismatch")

    def __len__(self):
        return int(self.q0.shape[0])

    @property
    def manifest_digest(self):
        return self.metadata["manifest_digest"]

    @property
    def codec_sha256(self):
        return self.metadata["codec"]["sha256"]

    @property
    def residual_statistics(self):
        split = self.metadata["splits"]["train"]
        return (
            np.asarray(split["residual_mean"], dtype=np.float32),
            np.asarray(split["residual_std"], dtype=np.float32),
            int(split["residual_count"]),
        )

    def __getitem__(self, index):
        q0 = np.asarray(self.q0[index], dtype=np.int64)
        residual = np.asarray(self.residual[index], dtype=np.float32)
        states = np.asarray(self.states[index], dtype=np.float32)
        motion = np.asarray(self.motion[index], dtype=np.float32)
        target_start_commit = HISTORY_TOKENS // COMMIT_TOKENS
        history_q0 = np.zeros(HISTORY_TOKENS, dtype=np.int64)
        history_residual = np.zeros(
            (HISTORY_TOKENS, residual.shape[-1]),
            dtype=np.float32,
        )
        history_q0[COMMIT_TOKENS:] = q0[:CAUSAL_HISTORY_VALID_TOKENS]
        history_residual[COMMIT_TOKENS:] = residual[
            :CAUSAL_HISTORY_VALID_TOKENS
        ]
        history_valid = torch.zeros(HISTORY_TOKENS, dtype=torch.bool)
        history_valid[COMMIT_TOKENS:] = True
        return {
            "history_q0": torch.from_numpy(history_q0),
            "history_residual": torch.from_numpy(history_residual),
            "history_valid": history_valid,
            "committed_q0": torch.from_numpy(q0.copy()),
            "committed_residual": torch.from_numpy(residual.copy()),
            "target_q0": torch.from_numpy(
                q0[
                    HISTORY_TOKENS : HISTORY_TOKENS
                    + TRAINING_ROLLOUT_TOKENS
                ].copy()
            ),
            "target_residual": torch.from_numpy(
                residual[
                    HISTORY_TOKENS : HISTORY_TOKENS
                    + TRAINING_ROLLOUT_TOKENS
                ].copy()
            ),
            "rollout_plan_q0": torch.from_numpy(
                np.asarray(self.plan_q0[index], dtype=np.int64).copy()
            ),
            "rollout_plan_residual": torch.from_numpy(
                np.asarray(self.plan_residual[index], dtype=np.float32).copy()
            ),
            "initial_state": torch.from_numpy(
                states[target_start_commit].copy()
            ),
            "window_start_state": torch.from_numpy(states[0].copy()),
            "commit_states": torch.from_numpy(
                states[target_start_commit:].copy()
            ),
            "target_motion": torch.from_numpy(
                motion[HISTORY_TOKENS * MOTION_FRAMES_PER_TOKEN :].copy()
            ),
            "index": torch.tensor(index, dtype=torch.long),
        }
