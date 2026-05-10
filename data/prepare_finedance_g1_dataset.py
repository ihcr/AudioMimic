import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FPS = 30
DEFAULT_LENGTH_SECONDS = 5.0
DEFAULT_STRIDE_SECONDS = 0.5
G1_DOF_DIM = 29


def extract_beat_metadata(*args, **kwargs):
    from data.audio_extraction.beat_features import extract_folder

    return extract_folder(*args, **kwargs)


def load_g1_payload(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    root_pos = np.asarray(payload["root_pos"], dtype=np.float32)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError(f"{path}: expected root_pos [T, 3], got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[-1] != 4:
        raise ValueError(f"{path}: expected root_rot [T, 4], got {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[-1] != G1_DOF_DIM:
        raise ValueError(f"{path}: expected dof_pos [T, {G1_DOF_DIM}], got {dof_pos.shape}")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError(f"{path}: G1 arrays have mismatched frame counts")
    if not (np.isfinite(root_pos).all() and np.isfinite(root_rot).all() and np.isfinite(dof_pos).all()):
        raise ValueError(f"{path}: G1 motion contains non-finite values")
    quat_norm = np.linalg.norm(root_rot, axis=-1, keepdims=True)
    root_rot = root_rot / np.maximum(quat_norm, 1e-8)
    return root_pos, root_rot.astype(np.float32), dof_pos


def write_pickle(path, root_pos, root_rot, dof_pos, fps=DEFAULT_FPS):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "motion_format": "g1",
        "motion_rep": "g1",
        "fps": float(fps),
        "root_pos": root_pos.astype(np.float32, copy=False),
        "root_rot": root_rot.astype(np.float32, copy=False),
        "dof_pos": dof_pos.astype(np.float32, copy=False),
        "pos": root_pos.astype(np.float32, copy=False),
        "q": np.concatenate(
            (
                root_rot.astype(np.float32, copy=False),
                dof_pos.astype(np.float32, copy=False),
            ),
            axis=-1,
        ),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def link_file(src, dst, copy_files=False):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copyfile(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def reset_split_dirs(output_root, feature_type, use_beats):
    dir_names = ["motions_sliced", "wavs_sliced", f"{feature_type}_feats"]
    if use_beats:
        dir_names.append("beat_feats")
    for split_name in ("train", "test"):
        for dirname in dir_names:
            path = Path(output_root) / split_name / dirname
            if path.exists():
                shutil.rmtree(path)


def parse_slice_stem(stem):
    sequence_id, slice_part = stem.rsplit("_slice", 1)
    return sequence_id, int(slice_part)


def prepare_split(
    split_name,
    source_split,
    output_root,
    g1_map,
    feature_type,
    length_seconds,
    stride_seconds,
    fps,
    copy_files=False,
):
    source_split = Path(source_split)
    output_split = Path(output_root) / split_name
    motion_out = output_split / "motions_sliced"
    wav_out = output_split / "wavs_sliced"
    feature_out = output_split / f"{feature_type}_feats"
    for directory in (motion_out, wav_out, feature_out):
        directory.mkdir(parents=True, exist_ok=True)

    stems = sorted(path.stem for path in (source_split / "motions_sliced").glob("*.pkl"))
    window_frames = int(round(length_seconds * fps))
    stride_frames = int(round(stride_seconds * fps))
    cache = {}
    missing_sequences = set()
    short_clips = []
    padded_clips = []
    padded_frames = 0

    for stem in tqdm(stems, desc=f"FineDance-G1 {split_name}", unit="clip"):
        sequence_id, slice_idx = parse_slice_stem(stem)
        if sequence_id not in g1_map:
            missing_sequences.add(sequence_id)
            continue
        if sequence_id not in cache:
            cache[sequence_id] = load_g1_payload(g1_map[sequence_id])
        root_pos, root_rot, dof_pos = cache[sequence_id]
        start = slice_idx * stride_frames
        end = start + window_frames
        if end > root_pos.shape[0]:
            missing_frames = end - root_pos.shape[0]
            if missing_frames > 1:
                short_clips.append(stem)
                continue
            root_pos = np.concatenate((root_pos, root_pos[-1:]), axis=0)
            root_rot = np.concatenate((root_rot, root_rot[-1:]), axis=0)
            dof_pos = np.concatenate((dof_pos, dof_pos[-1:]), axis=0)
            cache[sequence_id] = (root_pos, root_rot, dof_pos)
            padded_clips.append(stem)
            padded_frames += missing_frames
        write_pickle(
            motion_out / f"{stem}.pkl",
            root_pos[start:end],
            root_rot[start:end],
            dof_pos[start:end],
            fps=fps,
        )
        link_file(source_split / "wavs_sliced" / f"{stem}.wav", wav_out / f"{stem}.wav", copy_files)
        link_file(
            source_split / f"{feature_type}_feats" / f"{stem}.npy",
            feature_out / f"{stem}.npy",
            copy_files,
        )

    if missing_sequences:
        raise ValueError(f"{split_name}: missing G1 retargeted sequences {sorted(missing_sequences)[:20]}")
    if short_clips:
        raise ValueError(f"{split_name}: G1 sequences are too short for clips {short_clips[:20]}")

    return {
        "source_clips": len(stems),
        "clips": len(list(motion_out.glob("*.pkl"))),
        "sequences": len(cache),
        "padded_clips": len(padded_clips),
        "padded_frames": padded_frames,
    }


def prepare_finedance_g1_dataset(
    g1_motion_dir,
    source_prepared_root,
    output_root,
    feature_type="jukebox",
    length_seconds=DEFAULT_LENGTH_SECONDS,
    stride_seconds=DEFAULT_STRIDE_SECONDS,
    fps=DEFAULT_FPS,
    extract_beats=False,
    g1_motion_beat_source="fk",
    g1_fk_model_path=None,
    g1_root_quat_order="xyzw",
    clean=False,
    copy_files=False,
):
    g1_motion_dir = Path(g1_motion_dir)
    source_prepared_root = Path(source_prepared_root)
    output_root = Path(output_root)
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reset_split_dirs(output_root, feature_type, extract_beats)

    g1_map = {path.stem: path for path in sorted(g1_motion_dir.glob("*.pkl"))}
    summary = {
        "source": "FineDance retargeted to G1",
        "source_motion_dir": str(g1_motion_dir),
        "source_prepared_root": str(source_prepared_root),
        "feature_type": feature_type,
        "link_mode": "copy" if copy_files else "symlink",
        "target_fps": fps,
        "window_seconds": length_seconds,
        "stride_seconds": stride_seconds,
        "raw_g1_sequences": len(g1_map),
        "motion_payload": "G1 root_pos/root_rot/dof_pos plus compatibility pos and q=root_rot+dof_pos.",
        "beat_metadata": {
            "extracted": bool(extract_beats),
            "motion_beat_source": g1_motion_beat_source if extract_beats else "",
            "g1_fk_model_path": str(g1_fk_model_path) if extract_beats and g1_fk_model_path else "",
            "g1_root_quat_order": g1_root_quat_order if extract_beats else "",
        },
    }
    for split_name in ("train", "test"):
        summary[split_name] = prepare_split(
            split_name,
            source_prepared_root / split_name,
            output_root,
            g1_map,
            feature_type,
            length_seconds,
            stride_seconds,
            fps,
            copy_files=copy_files,
        )

    if extract_beats:
        if g1_motion_beat_source == "fk" and not g1_fk_model_path:
            raise ValueError("--g1_fk_model_path is required for FK beat extraction")
        for split_name in ("train", "test"):
            split_root = output_root / split_name
            extract_beat_metadata(
                str(split_root / "motions_sliced"),
                str(split_root / "wavs_sliced"),
                str(split_root / "beat_feats"),
                fps=fps,
                seq_len=int(round(length_seconds * fps)),
                g1_motion_beat_source=g1_motion_beat_source,
                g1_fk_model_path=g1_fk_model_path,
                g1_root_quat_order=g1_root_quat_order,
            )

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare FineDance retargeted G1 data for EDGE.")
    parser.add_argument("--g1_motion_dir", default="/projects/u6ed/yukun/EDGE/data/finedance-g1-retargeted")
    parser.add_argument("--source_prepared_root", default="data/finedance_aistpp")
    parser.add_argument("--output_root", default="data/finedance_g1_fkbeats")
    parser.add_argument("--feature_type", choices=("baseline", "jukebox"), default="jukebox")
    parser.add_argument("--length", type=float, default=DEFAULT_LENGTH_SECONDS)
    parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--extract_beats", action="store_true")
    parser.add_argument("--g1_motion_beat_source", choices=("proxy", "fk"), default="fk")
    parser.add_argument(
        "--g1_fk_model_path",
        default=str(REPO_ROOT / "third_party" / "unitree_g1_description" / "g1_29dof_rev_1_0.xml"),
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy_files", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = prepare_finedance_g1_dataset(
        g1_motion_dir=args.g1_motion_dir,
        source_prepared_root=args.source_prepared_root,
        output_root=args.output_root,
        feature_type=args.feature_type,
        length_seconds=args.length,
        stride_seconds=args.stride,
        fps=args.fps,
        extract_beats=args.extract_beats,
        g1_motion_beat_source=args.g1_motion_beat_source,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        clean=args.clean,
        copy_files=args.copy_files,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
