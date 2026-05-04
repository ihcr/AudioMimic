import argparse
import json
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FINE_DANCE_SEQUENCE_IDS = tuple(f"{idx:03d}" for idx in range(1, 212))
FINE_DANCE_CROSS_GENRE_TEST = (
    "063",
    "132",
    "143",
    "036",
    "098",
    "198",
    "130",
    "012",
    "211",
    "193",
    "179",
    "065",
    "137",
    "161",
    "092",
    "120",
    "037",
    "109",
    "204",
    "144",
)
FINE_DANCE_CROSS_GENRE_IGNORE = (
    "116",
    "117",
    "118",
    "119",
    "120",
    "121",
    "122",
    "123",
    "202",
    "130",
)
FINE_DANCE_CROSS_DANCER_TEST = (
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "124",
    "126",
    "128",
    "130",
    "132",
)
FINE_DANCE_CROSS_DANCER_IGNORE = (
    "115",
    "117",
    "119",
    "121",
    "122",
    "135",
    "137",
    "139",
    "141",
    "143",
    "145",
    "147",
    "116",
    "118",
    "120",
    "123",
    "202",
    "159",
    "130",
)
DEFAULT_LENGTH_SECONDS = 5.0
DEFAULT_STRIDE_SECONDS = 0.5
DEFAULT_FPS = 30
EDGE_RAW_FPS = 60
DEFAULT_MIN_MOTION_STD = 0.07
MOTION_DIR_NAME = "motion"
WAV_DIR_NAME = "music_wav"
PREPARED_DIRS = ("motions_sliced", "wavs_sliced")
SMPLH_AXIS_ANGLE_COLUMNS = 159
SMPLH_ROT6D_COLUMNS = 315
SMPLH_BODY_JOINT_COUNT = 52
EDGE_BODY_JOINT_INDICES = tuple(range(22)) + (22, 37)


@dataclass(frozen=True)
class FineDanceSplit:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    ignore_ids: tuple[str, ...]


def finedance_split_ids(split_name):
    if split_name == "cross_genre":
        test_ids = FINE_DANCE_CROSS_GENRE_TEST
        ignore_ids = FINE_DANCE_CROSS_GENRE_IGNORE
    elif split_name == "cross_dancer":
        test_ids = FINE_DANCE_CROSS_DANCER_TEST
        ignore_ids = FINE_DANCE_CROSS_DANCER_IGNORE
    else:
        raise ValueError("split_name must be 'cross_genre' or 'cross_dancer'")

    test_set = set(test_ids)
    ignore_set = set(ignore_ids)
    train_ids = tuple(
        sequence_id
        for sequence_id in FINE_DANCE_SEQUENCE_IDS
        if sequence_id not in test_set and sequence_id not in ignore_set
    )
    usable_test_ids = tuple(sequence_id for sequence_id in test_ids if sequence_id not in ignore_set)
    return FineDanceSplit(
        train_ids=train_ids,
        test_ids=usable_test_ids,
        ignore_ids=tuple(sorted(ignore_set)),
    )


def convert_finedance_body_motion_to_edge(raw_motion, root_height_offset=0.0):
    raw_motion = np.asarray(raw_motion, dtype=np.float32)
    if raw_motion.ndim != 2:
        raise ValueError(f"FineDance motion expected a 2D array, got shape {raw_motion.shape}")
    if raw_motion.shape[1] not in (SMPLH_AXIS_ANGLE_COLUMNS, SMPLH_ROT6D_COLUMNS):
        raise ValueError(
            "FineDance raw SMPLH motion expected 159 axis-angle columns or "
            f"315 6D-rotation columns, got {raw_motion.shape[1]}"
        )
    if not np.isfinite(raw_motion).all():
        raise ValueError("FineDance motion contains non-finite values")

    pos = raw_motion[:, :3].copy()
    pos[:, 1] += float(root_height_offset)
    if raw_motion.shape[1] == SMPLH_AXIS_ANGLE_COLUMNS:
        q = np.concatenate(
            (
                raw_motion[:, 3:69],
                raw_motion[:, 69:72],
                raw_motion[:, 114:117],
            ),
            axis=1,
        ).copy()
    else:
        local_q_rot6d = raw_motion[:, 3:].reshape(
            raw_motion.shape[0],
            SMPLH_BODY_JOINT_COUNT,
            6,
        )
        body_q_rot6d = local_q_rot6d[:, EDGE_BODY_JOINT_INDICES, :]
        q = _rotation_6d_to_axis_angle(body_q_rot6d).reshape(raw_motion.shape[0], -1)
    _validate_edge_body_motion(pos, q, "converted FineDance motion")
    return pos, q


def _rotation_6d_to_axis_angle(rot6d):
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"expected 6D rotations, got shape {rot6d.shape}")
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    b1 = _normalize_vectors(a1)
    b2 = _normalize_vectors(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    matrix = np.stack((b1, b2, b3), axis=-2)
    return _matrix_to_axis_angle(matrix).astype(np.float32)


def _normalize_vectors(vectors):
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("FineDance 6D rotation contains a degenerate vector")
    return vectors / norm


def _matrix_to_axis_angle(matrix):
    quaternion = _matrix_to_quaternion(matrix)
    xyz = quaternion[..., 1:]
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternion[..., :1])
    angles = 2.0 * half_angles
    scale = np.empty_like(norms)
    small = norms < 1e-8
    scale[~small] = angles[~small] / norms[~small]
    scale[small] = 2.0
    return xyz * scale


def _matrix_to_quaternion(matrix):
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotation matrices with shape (..., 3, 3), got {matrix.shape}")
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = np.sqrt(
        np.maximum(
            0.0,
            np.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                axis=-1,
            ),
        )
    )
    quat_by_rijk = np.stack(
        (
            np.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), axis=-1),
            np.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), axis=-1),
            np.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), axis=-1),
            np.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), axis=-1),
        ),
        axis=-2,
    )
    denom = np.maximum(2.0 * q_abs[..., None], 0.1)
    quat_candidates = quat_by_rijk / denom
    best = np.argmax(q_abs, axis=-1)
    quaternion = np.take_along_axis(
        quat_candidates,
        best[..., None, None],
        axis=-2,
    )[..., 0, :]
    quaternion = _normalize_vectors(quaternion)
    return np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def _validate_edge_body_motion(pos, q, label):
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{label}: expected pos shape [frames, 3], got {pos.shape}")
    if q.ndim != 2 or q.shape[1] != 72:
        raise ValueError(f"{label}: expected q shape [frames, 72], got {q.shape}")
    if pos.shape[0] != q.shape[0]:
        raise ValueError(f"{label}: pos and q frame counts do not match")
    if not np.isfinite(pos).all() or not np.isfinite(q).all():
        raise ValueError(f"{label}: converted motion contains non-finite values")
    if pos.shape[0] == 0:
        raise ValueError(f"{label}: no frames found")


def _motion_activity_score(pos, q):
    motion = np.concatenate((pos, q), axis=1)
    return float(motion.std(axis=0).mean())


def _repeat_to_edge_raw_fps(pos, q, source_fps=DEFAULT_FPS):
    if EDGE_RAW_FPS % source_fps != 0:
        raise ValueError(
            f"FineDance source fps {source_fps} must divide EDGE raw fps {EDGE_RAW_FPS}"
        )
    repeat_factor = EDGE_RAW_FPS // source_fps
    return (
        np.repeat(pos, repeat_factor, axis=0),
        np.repeat(q, repeat_factor, axis=0),
    )


def _load_finedance_sequence(finedance_root, sequence_id, root_height_offset):
    motion_path = Path(finedance_root) / MOTION_DIR_NAME / f"{sequence_id}.npy"
    wav_path = Path(finedance_root) / WAV_DIR_NAME / f"{sequence_id}.wav"
    if not motion_path.is_file():
        raise FileNotFoundError(f"Missing FineDance motion file: {motion_path}")
    if not wav_path.is_file():
        raise FileNotFoundError(f"Missing FineDance wav file: {wav_path}")

    raw_motion = np.load(motion_path)
    pos, q = convert_finedance_body_motion_to_edge(
        raw_motion,
        root_height_offset=root_height_offset,
    )
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"{wav_path}: wav file is empty")
    return pos, q, audio, sample_rate


def _slice_count(frame_count, sample_count, fps, sample_rate, length_seconds, stride_seconds):
    motion_window = int(round(length_seconds * fps))
    motion_stride = int(round(stride_seconds * fps))
    audio_window = int(round(length_seconds * sample_rate))
    audio_stride = int(round(stride_seconds * sample_rate))
    if min(motion_window, motion_stride, audio_window, audio_stride) <= 0:
        raise ValueError("length_seconds and stride_seconds must produce positive windows")
    motion_slices = 0 if frame_count < motion_window else (frame_count - motion_window) // motion_stride + 1
    audio_slices = 0 if sample_count < audio_window else (sample_count - audio_window) // audio_stride + 1
    return min(motion_slices, audio_slices), motion_window, motion_stride, audio_window, audio_stride


def _write_motion_slice(path, pos, q):
    _validate_edge_body_motion(pos, q, str(path))
    with open(path, "wb") as handle:
        pickle.dump({"pos": pos.astype(np.float32), "q": q.astype(np.float32)}, handle)


def _write_audio_slice(path, audio, sample_rate):
    sf.write(path, audio, sample_rate)


def _prepare_split(
    finedance_root,
    output_root,
    split_name,
    sequence_ids,
    length_seconds,
    stride_seconds,
    fps,
    min_motion_std,
    root_height_offset,
):
    split_root = Path(output_root) / split_name
    motion_out = split_root / "motions_sliced"
    wav_out = split_root / "wavs_sliced"
    motion_out.mkdir(parents=True, exist_ok=True)
    wav_out.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_sequences": 0,
        "missing_sequences": [],
        "clips": 0,
        "dropped_static_clips": 0,
        "dropped_short_sequences": 0,
        "motion_format": "smpl_body_24",
    }
    for sequence_id in tqdm(sequence_ids, desc=f"FineDance {split_name}", unit="seq"):
        motion_path = Path(finedance_root) / MOTION_DIR_NAME / f"{sequence_id}.npy"
        wav_path = Path(finedance_root) / WAV_DIR_NAME / f"{sequence_id}.wav"
        if not motion_path.exists() or not wav_path.exists():
            summary["missing_sequences"].append(sequence_id)
            continue
        pos, q, audio, sample_rate = _load_finedance_sequence(
            finedance_root,
            sequence_id,
            root_height_offset=root_height_offset,
        )
        summary["source_sequences"] += 1
        num_slices, motion_window, motion_stride, audio_window, audio_stride = _slice_count(
            len(pos),
            len(audio),
            fps=fps,
            sample_rate=sample_rate,
            length_seconds=length_seconds,
            stride_seconds=stride_seconds,
        )
        if num_slices == 0:
            summary["dropped_short_sequences"] += 1
            continue

        for slice_idx in range(num_slices):
            motion_start = slice_idx * motion_stride
            audio_start = slice_idx * audio_stride
            pos_slice = pos[motion_start : motion_start + motion_window]
            q_slice = q[motion_start : motion_start + motion_window]
            if split_name == "train" and min_motion_std > 0:
                if _motion_activity_score(pos_slice, q_slice) <= min_motion_std:
                    summary["dropped_static_clips"] += 1
                    continue
            saved_pos, saved_q = _repeat_to_edge_raw_fps(pos_slice, q_slice, source_fps=fps)
            stem = f"{sequence_id}_slice{slice_idx}"
            _write_motion_slice(motion_out / f"{stem}.pkl", saved_pos, saved_q)
            _write_audio_slice(
                wav_out / f"{stem}.wav",
                audio[audio_start : audio_start + audio_window],
                sample_rate,
            )
            summary["clips"] += 1
    return summary


def _load_feature_extractors(feature_type, use_beats):
    baseline_extract = None
    jukebox_extract = None
    beat_extract = None
    if feature_type == "baseline":
        from data.audio_extraction.baseline_features import extract_folder as baseline_extract
    elif feature_type == "jukebox":
        from data.audio_extraction.jukebox_features import extract_folder as jukebox_extract
    else:
        raise ValueError("feature_type must be 'baseline' or 'jukebox'")

    if use_beats:
        from data.audio_extraction.beat_features import extract_folder as beat_extract
    return baseline_extract, jukebox_extract, beat_extract


def extract_prepared_features(output_root, feature_type="jukebox", use_beats=True):
    baseline_extract = None
    jukebox_extract = None
    beat_extract = None
    if feature_type is not None:
        baseline_extract, jukebox_extract, _ = _load_feature_extractors(
            feature_type,
            False,
        )
    if use_beats:
        from data.audio_extraction.beat_features import extract_folder as beat_extract

    for split_name in ("train", "test"):
        split_root = Path(output_root) / split_name
        wav_dir = split_root / "wavs_sliced"
        if feature_type == "baseline":
            baseline_extract(str(wav_dir), str(split_root / "baseline_feats"))
        elif feature_type == "jukebox":
            jukebox_extract(str(wav_dir), str(split_root / "jukebox_feats"))
        if use_beats:
            beat_extract(
                str(split_root / "motions_sliced"),
                str(wav_dir),
                str(split_root / "beat_feats"),
            )


def prepare_finedance_dataset(
    finedance_root,
    output_root,
    split_name="cross_genre",
    length_seconds=DEFAULT_LENGTH_SECONDS,
    stride_seconds=DEFAULT_STRIDE_SECONDS,
    fps=DEFAULT_FPS,
    min_motion_std=DEFAULT_MIN_MOTION_STD,
    root_height_offset=0.0,
    feature_type=None,
    use_beats=False,
):
    finedance_root = Path(finedance_root)
    output_root = Path(output_root)
    if not finedance_root.is_dir():
        raise FileNotFoundError(f"FineDance root does not exist: {finedance_root}")
    split = finedance_split_ids(split_name)
    output_root.mkdir(parents=True, exist_ok=True)
    reset_dir_names = list(PREPARED_DIRS)
    if feature_type is not None:
        reset_dir_names.append(f"{feature_type}_feats")
    if use_beats:
        reset_dir_names.append("beat_feats")
    _reset_dirs(output_root, ("train", "test"), reset_dir_names)

    summary = {
        "source": "FineDance",
        "source_root": str(finedance_root),
        "split": split_name,
        "ignored_sequences": list(split.ignore_ids),
        "length_seconds": length_seconds,
        "stride_seconds": stride_seconds,
        "fps": fps,
        "saved_raw_fps": EDGE_RAW_FPS,
        "min_motion_std": min_motion_std,
        "root_height_offset": root_height_offset,
        "conversion": {
            "input": "FineDance raw SMPLH 315-column 6D rotations or 159-column axis-angle rotations",
            "output": "EDGE body-only pos (3) + q (72)",
            "q_mapping": ["joints 0:21", "left hand base joint 22", "right hand base joint 37"],
            "hands": "dropped",
            "fps": "FineDance 30 fps frames are duplicated into EDGE's 60 fps container",
        },
    }
    summary["train"] = _prepare_split(
        finedance_root,
        output_root,
        "train",
        split.train_ids,
        length_seconds,
        stride_seconds,
        fps,
        min_motion_std,
        root_height_offset,
    )
    summary["test"] = _prepare_split(
        finedance_root,
        output_root,
        "test",
        split.test_ids,
        length_seconds,
        stride_seconds,
        fps,
        min_motion_std,
        root_height_offset,
    )
    if feature_type is not None or use_beats:
        extract_prepared_features(output_root, feature_type=feature_type, use_beats=use_beats)
    if feature_type is not None:
        summary["feature_type"] = feature_type
    summary["use_beats"] = use_beats
    _write_metadata(output_root / "metadata.json", summary)
    return summary


def _write_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reset_dirs(root, split_names, dir_names):
    root = Path(root)
    for split_name in split_names:
        for dir_name in dir_names:
            path = root / split_name / dir_name
            if path.exists():
                shutil.rmtree(path)


def _link_or_copy(src, dst, copy_files=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copyfile(src, dst)
    else:
        os.symlink(Path(src).resolve(), dst)


def _required_split_dirs(split_root, feature_type, use_beats):
    dirs = ["motions_sliced", "wavs_sliced", f"{feature_type}_feats"]
    if use_beats:
        dirs.append("beat_feats")
    return dirs


def _copy_prepared_split(src_root, dst_root, output_split, prefix, feature_type, use_beats, copy_files=False):
    src_root = Path(src_root)
    dst_split_root = Path(dst_root) / output_split
    required_dirs = _required_split_dirs(src_root, feature_type, use_beats)
    motion_dir = Path(src_root) / "motions_sliced"
    stems = sorted(path.stem for path in motion_dir.glob("*.pkl"))
    for dirname in required_dirs:
        src_dir = Path(src_root) / dirname
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing prepared directory: {src_dir}")
        (dst_split_root / dirname).mkdir(parents=True, exist_ok=True)

    suffixes = {
        "motions_sliced": ".pkl",
        "wavs_sliced": ".wav",
        f"{feature_type}_feats": ".npy",
    }
    if use_beats:
        suffixes["beat_feats"] = ".npz"

    for stem in stems:
        for dirname, suffix in suffixes.items():
            src = Path(src_root) / dirname / f"{stem}{suffix}"
            if not src.is_file():
                raise FileNotFoundError(f"Missing prepared file: {src}")
            dst = dst_split_root / dirname / f"{prefix}__{stem}{suffix}"
            _link_or_copy(src, dst, copy_files=copy_files)
    return len(stems)


def build_mixed_aist_finedance_dataset(
    aist_root,
    finedance_root,
    output_root,
    feature_type="jukebox",
    use_beats=True,
    copy_files=False,
):
    aist_root = Path(aist_root)
    finedance_root = Path(finedance_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _reset_dirs(
        output_root,
        ("train", "test", "finedance_test"),
        _required_split_dirs(output_root, feature_type, use_beats),
    )

    train_count = 0
    train_count += _copy_prepared_split(
        aist_root / "train",
        output_root,
        "train",
        "aist",
        feature_type,
        use_beats,
        copy_files=copy_files,
    )
    train_count += _copy_prepared_split(
        finedance_root / "train",
        output_root,
        "train",
        "finedance",
        feature_type,
        use_beats,
        copy_files=copy_files,
    )
    aist_test_count = _copy_prepared_split(
        aist_root / "test",
        output_root,
        "test",
        "aist",
        feature_type,
        use_beats,
        copy_files=copy_files,
    )
    finedance_test_count = _copy_prepared_split(
        finedance_root / "test",
        output_root,
        "finedance_test",
        "finedance",
        feature_type,
        use_beats,
        copy_files=copy_files,
    )
    summary = {
        "source": "AIST + FineDance",
        "feature_type": feature_type,
        "use_beats": use_beats,
        "train": {"source": "aist+finedance", "clips": train_count},
        "test": {"source": "aist", "clips": aist_test_count},
        "finedance_test": {"source": "finedance", "clips": finedance_test_count},
        "link_mode": "copy" if copy_files else "symlink",
    }
    _write_metadata(output_root / "metadata.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert FineDance to EDGE body-only data and optionally build an AIST+FineDance tree."
    )
    parser.add_argument("--finedance_root", required=True)
    parser.add_argument("--output_root", default="data/finedance_aistpp")
    parser.add_argument("--split", choices=("cross_genre", "cross_dancer"), default="cross_genre")
    parser.add_argument("--length", type=float, default=DEFAULT_LENGTH_SECONDS)
    parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--min_motion_std", type=float, default=DEFAULT_MIN_MOTION_STD)
    parser.add_argument("--root_height_offset", type=float, default=0.0)
    parser.add_argument("--feature_type", choices=("baseline", "jukebox"), default=None)
    parser.add_argument("--extract_beats", action="store_true")
    parser.add_argument("--build_mixed", action="store_true")
    parser.add_argument("--aist_root", default="data")
    parser.add_argument("--mixed_output_root", default="data/aist_finedance")
    parser.add_argument("--copy_files", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = prepare_finedance_dataset(
        finedance_root=args.finedance_root,
        output_root=args.output_root,
        split_name=args.split,
        length_seconds=args.length,
        stride_seconds=args.stride,
        fps=args.fps,
        min_motion_std=args.min_motion_std,
        root_height_offset=args.root_height_offset,
        feature_type=args.feature_type,
        use_beats=args.extract_beats,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.build_mixed:
        if args.feature_type is None:
            raise ValueError("--build_mixed requires --feature_type so feature directories are known")
        mixed_summary = build_mixed_aist_finedance_dataset(
            aist_root=args.aist_root,
            finedance_root=args.output_root,
            output_root=args.mixed_output_root,
            feature_type=args.feature_type,
            use_beats=args.extract_beats,
            copy_files=args.copy_files,
        )
        print(json.dumps(mixed_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
