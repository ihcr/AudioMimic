import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_CACHE_VERSION = "v4"
TENSOR_DATASET_CACHE_VERSION = "v5"
SMPL_MOTION_FORMAT = "smpl"
G1_MOTION_FORMAT = "g1"
VALID_MOTION_FORMATS = (SMPL_MOTION_FORMAT, G1_MOTION_FORMAT)
G1_DOF_DIM = 29


def validate_motion_format(motion_format):
    if motion_format not in VALID_MOTION_FORMATS:
        raise ValueError(
            f"Unsupported motion_format {motion_format!r}; expected one of {VALID_MOTION_FORMATS}"
        )
    return motion_format


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


def tensor_dataset_cache_name(
    split,
    feature_type,
    use_beats,
    beat_rep,
    motion_format=SMPL_MOTION_FORMAT,
):
    validate_motion_format(motion_format)
    beat_tag = "beat" if use_beats else "nobeat"
    if motion_format == SMPL_MOTION_FORMAT:
        return (
            f"{split}_tensor_dataset_{feature_type}_{beat_tag}_{beat_rep}_"
            f"{TENSOR_DATASET_CACHE_VERSION}.pkl"
        )
    return (
        f"{split}_tensor_dataset_{motion_format}_{feature_type}_{beat_tag}_{beat_rep}_"
        f"{TENSOR_DATASET_CACHE_VERSION}.pkl"
    )


EXPECTED_RAW_FRAMES = 300
EXPECTED_MODEL_FRAMES = 150
EXPECTED_FEATURE_DIMS = {
    "baseline": 35,
    "jukebox": 4800,
}
ROOT_HEIGHT_MIN = 0.0
ROOT_HEIGHT_MAX = 4.0
G1_ROOT_HEIGHT_MIN = -5.0
G1_ROOT_HEIGHT_MAX = 5.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate preprocessed EDGE data before training starts."
    )
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--processed_data_dir", default="data/dataset_backups")
    parser.add_argument("--feature_type", choices=("baseline", "jukebox"), required=True)
    parser.add_argument("--motion_format", choices=("smpl", "g1"), default="smpl")
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--sample_count", type=int, default=64)
    return parser.parse_args(argv)


def list_stems(directory, suffix):
    paths = sorted(directory.glob(f"*{suffix}"))
    return [path.stem for path in paths]


def evenly_sample_stems(stems, sample_count):
    if sample_count <= 0:
        raise ValueError("sample_count must be at least 1")
    if len(stems) <= sample_count:
        return list(stems)
    indices = np.linspace(0, len(stems) - 1, num=sample_count, dtype=int)
    return [stems[index] for index in sorted(set(indices.tolist()))]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_smpl_motion_file(path, payload):
    require(set(payload) == {"pos", "q"}, f"{path}: expected pos/q motion payload.")
    pos = np.asarray(payload["pos"], dtype=np.float32)
    q = np.asarray(payload["q"], dtype=np.float32)
    require(pos.shape == (EXPECTED_RAW_FRAMES, 3), f"{path}: expected pos shape {(EXPECTED_RAW_FRAMES, 3)}, got {pos.shape}.")
    require(q.shape == (EXPECTED_RAW_FRAMES, 72), f"{path}: expected q shape {(EXPECTED_RAW_FRAMES, 72)}, got {q.shape}.")
    require(np.isfinite(pos).all(), f"{path}: motion pos contains non-finite values.")
    require(np.isfinite(q).all(), f"{path}: motion q contains non-finite values.")
    root_height = pos[:, 1]
    return root_height


def validate_g1_motion_file(path, payload):
    required_keys = {"root_pos", "root_rot", "dof_pos"}
    require(required_keys.issubset(payload), f"{path}: expected G1 root_pos/root_rot/dof_pos payload.")
    root_pos = np.asarray(payload["root_pos"], dtype=np.float32)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)
    require(root_pos.shape == (EXPECTED_MODEL_FRAMES, 3), f"{path}: expected root_pos shape {(EXPECTED_MODEL_FRAMES, 3)}, got {root_pos.shape}.")
    require(root_rot.shape == (EXPECTED_MODEL_FRAMES, 4), f"{path}: expected root_rot shape {(EXPECTED_MODEL_FRAMES, 4)}, got {root_rot.shape}.")
    require(dof_pos.shape == (EXPECTED_MODEL_FRAMES, G1_DOF_DIM), f"{path}: expected dof_pos shape {(EXPECTED_MODEL_FRAMES, G1_DOF_DIM)}, got {dof_pos.shape}.")
    require(np.isfinite(root_pos).all(), f"{path}: root_pos contains non-finite values.")
    require(np.isfinite(root_rot).all(), f"{path}: root_rot contains non-finite values.")
    require(np.isfinite(dof_pos).all(), f"{path}: dof_pos contains non-finite values.")
    quat_norm = np.linalg.norm(root_rot, axis=-1)
    require(np.allclose(quat_norm, 1.0, atol=1e-3), f"{path}: root_rot quaternions are not unit length.")
    if "q" in payload:
        q = np.asarray(payload["q"], dtype=np.float32)
        require(q.shape == (EXPECTED_MODEL_FRAMES, 4 + G1_DOF_DIM), f"{path}: expected q shape {(EXPECTED_MODEL_FRAMES, 4 + G1_DOF_DIM)}, got {q.shape}.")
    return root_pos[:, 2]


def validate_motion_file(path, motion_format=SMPL_MOTION_FORMAT):
    motion_format = validate_motion_format(motion_format)
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"{path}: could not read motion pickle ({type(exc).__name__}: {exc}).") from exc

    if motion_format == G1_MOTION_FORMAT:
        root_height_values = validate_g1_motion_file(path, payload)
    else:
        root_height_values = validate_smpl_motion_file(path, payload)

    root_height_min = float(root_height_values.min())
    root_height_max = float(root_height_values.max())
    min_bound = G1_ROOT_HEIGHT_MIN if motion_format == G1_MOTION_FORMAT else ROOT_HEIGHT_MIN
    max_bound = G1_ROOT_HEIGHT_MAX if motion_format == G1_MOTION_FORMAT else ROOT_HEIGHT_MAX
    require(
        min_bound <= root_height_min <= max_bound,
        f"{path}: root height min {root_height_min:.4f} is outside [{min_bound}, {max_bound}].",
    )
    require(
        min_bound <= root_height_max <= max_bound,
        f"{path}: root height max {root_height_max:.4f} is outside [{min_bound}, {max_bound}].",
    )
    return {
        "root_height_min": root_height_min,
        "root_height_max": root_height_max,
    }


def validate_feature_file(path, feature_type):
    feature = np.load(path)
    expected_dim = EXPECTED_FEATURE_DIMS[feature_type]
    require(feature.ndim == 2, f"{path}: expected 2D feature array, got shape {feature.shape}.")
    require(
        feature.shape == (EXPECTED_MODEL_FRAMES, expected_dim),
        f"{path}: expected feature shape {(EXPECTED_MODEL_FRAMES, expected_dim)}, got {feature.shape}.",
    )
    require(np.isfinite(feature).all(), f"{path}: feature array contains non-finite values.")


def validate_beat_file(path):
    required_keys = {
        "motion_beats",
        "motion_mask",
        "motion_dist",
        "motion_spacing",
        "audio_beats",
        "audio_mask",
        "audio_dist",
        "audio_spacing",
    }
    with np.load(path) as beat_data:
        keys = set(beat_data.files)
        require(keys == required_keys, f"{path}: expected beat keys {sorted(required_keys)}, got {sorted(keys)}.")
        for key in ("motion_mask", "motion_dist", "motion_spacing", "audio_mask", "audio_dist", "audio_spacing"):
            values = np.asarray(beat_data[key])
            require(values.shape == (EXPECTED_MODEL_FRAMES,), f"{path}: {key} expected shape {(EXPECTED_MODEL_FRAMES,)}, got {values.shape}.")
            require(np.isfinite(values).all(), f"{path}: {key} contains non-finite values.")


def find_legacy_caches(
    processed_data_dir,
    feature_type,
    use_beats,
    beat_rep,
    motion_format=SMPL_MOTION_FORMAT,
):
    motion_format = validate_motion_format(motion_format)
    beat_tag = "beat" if use_beats else "nobeat"
    processed_data_dir = Path(processed_data_dir)
    stale = []
    for split in ("train", "test"):
        current_processed_name = processed_dataset_cache_name(
            split,
            feature_type,
            use_beats,
            beat_rep,
            motion_format=motion_format,
        )
        if motion_format == SMPL_MOTION_FORMAT:
            processed_pattern = f"processed_{split}_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
        else:
            processed_pattern = f"processed_{split}_{motion_format}_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
        for cache_path in processed_data_dir.glob(processed_pattern):
            if cache_path.name != current_processed_name:
                stale.append(cache_path.name)

        current_tensor_name = tensor_dataset_cache_name(
            split,
            feature_type,
            use_beats,
            beat_rep,
            motion_format=motion_format,
        )
        if motion_format == SMPL_MOTION_FORMAT:
            tensor_pattern = f"{split}_tensor_dataset_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
        else:
            tensor_pattern = f"{split}_tensor_dataset_{motion_format}_{feature_type}_{beat_tag}_{beat_rep}*.pkl"
        for cache_path in processed_data_dir.glob(tensor_pattern):
            if cache_path.name != current_tensor_name:
                stale.append(cache_path.name)
    return sorted(set(stale))


def validate_split(
    split_name,
    data_path,
    feature_type,
    use_beats,
    sample_count,
    motion_format=SMPL_MOTION_FORMAT,
):
    split_dir = Path(data_path) / split_name
    motion_dir = split_dir / "motions_sliced"
    wav_dir = split_dir / "wavs_sliced"
    feature_dir = split_dir / f"{feature_type}_feats"
    beat_dir = split_dir / "beat_feats"

    require(motion_dir.is_dir(), f"{motion_dir}: missing motions_sliced directory.")
    require(wav_dir.is_dir(), f"{wav_dir}: missing wavs_sliced directory.")
    require(feature_dir.is_dir(), f"{feature_dir}: missing {feature_type}_feats directory.")
    if use_beats:
        require(beat_dir.is_dir(), f"{beat_dir}: missing beat_feats directory.")

    motion_stems = list_stems(motion_dir, ".pkl")
    wav_stems = list_stems(wav_dir, ".wav")
    feature_stems = list_stems(feature_dir, ".npy")
    beat_stems = list_stems(beat_dir, ".npz") if use_beats else []

    require(motion_stems, f"{motion_dir}: no sliced motions found.")
    require(motion_stems == wav_stems == feature_stems, f"{split_name}: motion, wav, and feature file names do not match.")
    if use_beats:
        require(motion_stems == beat_stems, f"{split_name}: beat file names do not match motion files.")

    height_mins = []
    height_maxes = []
    for stem in motion_stems:
        motion_stats = validate_motion_file(
            motion_dir / f"{stem}.pkl",
            motion_format=motion_format,
        )
        height_mins.append(motion_stats["root_height_min"])
        height_maxes.append(motion_stats["root_height_max"])

    sampled_stems = evenly_sample_stems(motion_stems, sample_count)
    for stem in sampled_stems:
        validate_feature_file(feature_dir / f"{stem}.npy", feature_type)
        if use_beats:
            validate_beat_file(beat_dir / f"{stem}.npz")
        wav_path = wav_dir / f"{stem}.wav"
        require(wav_path.is_file(), f"{wav_path}: missing wav file.")
        require(wav_path.stat().st_size > 0, f"{wav_path}: wav file is empty.")

    return {
        "count": len(motion_stems),
        "sampled": len(sampled_stems),
        "feature_dir": feature_dir.name,
        "beat_count": len(beat_stems),
        "root_height_min": min(height_mins),
        "root_height_max": max(height_maxes),
    }


def validate_preprocessed_dataset(
    data_path,
    processed_data_dir,
    feature_type,
    use_beats=False,
    beat_rep="distance",
    sample_count=64,
    motion_format=SMPL_MOTION_FORMAT,
):
    motion_format = validate_motion_format(motion_format)
    data_path = Path(data_path)
    processed_data_dir = Path(processed_data_dir)
    require(data_path.is_dir(), f"{data_path}: data path does not exist.")
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    stale_caches = find_legacy_caches(
        processed_data_dir=processed_data_dir,
        feature_type=feature_type,
        use_beats=use_beats,
        beat_rep=beat_rep,
        motion_format=motion_format,
    )
    require(
        not stale_caches,
        "Found legacy cache files for the active training setup. Remove them or bump the cache version before training: "
        + ", ".join(stale_caches),
    )

    summary = {
        "cache_versions": {
            "processed": DATASET_CACHE_VERSION,
            "tensor": tensor_dataset_cache_name(
                "train",
                feature_type,
                use_beats,
                beat_rep,
                motion_format=motion_format,
            ).rsplit("_", 1)[-1].removesuffix(".pkl"),
            "motion_format": motion_format,
        }
    }
    for split in ("train", "test"):
        summary[split] = validate_split(
            split_name=split,
            data_path=data_path,
            feature_type=feature_type,
            use_beats=use_beats,
            sample_count=sample_count,
            motion_format=motion_format,
        )
    return summary


def format_summary(summary):
    lines = [
        "Preprocessed data validation passed.",
        f"Motion format: {summary['cache_versions']['motion_format']}",
        f"Processed cache version: {summary['cache_versions']['processed']}",
        f"Tensor cache version: {summary['cache_versions']['tensor']}",
    ]
    for split in ("train", "test"):
        info = summary[split]
        lines.append(
            f"{split}: count={info['count']} sampled={info['sampled']} "
            f"feature_dir={info['feature_dir']} beat_count={info['beat_count']} "
            f"root_height=[{info['root_height_min']:.4f}, {info['root_height_max']:.4f}]"
        )
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    summary = validate_preprocessed_dataset(
        data_path=args.data_path,
        processed_data_dir=args.processed_data_dir,
        feature_type=args.feature_type,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        sample_count=args.sample_count,
        motion_format=args.motion_format,
    )
    print(format_summary(summary))


if __name__ == "__main__":
    main()
