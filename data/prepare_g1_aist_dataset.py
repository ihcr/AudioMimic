import argparse
import json
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
SHARED_ROOT = next(
    (parent.parent for parent in [REPO_ROOT, *REPO_ROOT.parents] if parent.name == ".worktrees"),
    REPO_ROOT,
)
DEFAULT_G1_MOTION_DIR = SHARED_ROOT / "aist-g1-retargeted"
DEFAULT_AIST_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_SPLIT_DIR = SCRIPT_ROOT / "splits"
WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 0.5
FPS = 30.0
WINDOW_FRAMES = int(round(WINDOW_SECONDS * FPS))
STRIDE_FRAMES = int(round(STRIDE_SECONDS * FPS))
G1_DOF_DIM = 29


def extract_beat_metadata(*args, **kwargs):
    from data.audio_extraction.beat_features import extract_folder

    return extract_folder(*args, **kwargs)


def read_split_file(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_g1_payload(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    root_pos = np.asarray(payload["root_pos"], dtype=np.float32)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError(f"{path}: expected root_pos shape [T, 3], got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[-1] != 4:
        raise ValueError(f"{path}: expected root_rot shape [T, 4], got {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[-1] != G1_DOF_DIM:
        raise ValueError(f"{path}: expected dof_pos shape [T, {G1_DOF_DIM}], got {dof_pos.shape}")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError(f"{path}: root_pos/root_rot/dof_pos frame counts do not match")
    if not (np.isfinite(root_pos).all() and np.isfinite(root_rot).all() and np.isfinite(dof_pos).all()):
        raise ValueError(f"{path}: G1 motion contains non-finite values")

    quat_norm = np.linalg.norm(root_rot, axis=-1, keepdims=True)
    root_rot = root_rot / np.maximum(quat_norm, 1e-8)
    return {
        "motion_format": "g1",
        "motion_rep": "g1",
        "fps": float(payload.get("fps", FPS)),
        "root_pos": root_pos,
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": dof_pos,
    }


def compatible_payload(root_pos, root_rot, dof_pos, fps=FPS):
    return {
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


def write_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def replace_symlink(source, target):
    source = Path(source).resolve()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def sorted_slice_paths(directory, stem, suffix):
    return sorted(Path(directory).glob(f"{stem}_slice*{suffix}"))


def expected_slice_count(frame_count, window_frames=WINDOW_FRAMES, stride_frames=STRIDE_FRAMES):
    if frame_count < window_frames:
        return 0
    return ((frame_count - window_frames) // stride_frames) + 1


def prepare_split(
    split_name,
    sequence_names,
    g1_map,
    aist_data_root,
    output_root,
    feature_type,
):
    split_root = output_root / split_name
    motion_dir = split_root / "motions"
    motion_slice_dir = split_root / "motions_sliced"
    wav_slice_dir = split_root / "wavs_sliced"
    feature_dir = split_root / f"{feature_type}_feats"
    for directory in (motion_dir, motion_slice_dir, wav_slice_dir, feature_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_split = Path(aist_data_root) / split_name
    source_wav_dir = source_split / "wavs_sliced"
    source_feature_dir = source_split / f"{feature_type}_feats"
    if not source_wav_dir.is_dir():
        raise FileNotFoundError(f"Missing AIST sliced wav directory: {source_wav_dir}")
    if not source_feature_dir.is_dir():
        raise FileNotFoundError(f"Missing AIST feature directory: {source_feature_dir}")

    total_slices = 0
    padded_sequences = 0
    padded_frames = 0
    for sequence_name in tqdm(
        sorted(sequence_names),
        total=len(sequence_names),
        desc=f"G1 {split_name}",
        unit="seq",
    ):
        payload = load_g1_payload(g1_map[sequence_name])
        root_pos = payload["root_pos"]
        root_rot = payload["root_rot"]
        dof_pos = payload["dof_pos"]
        possible_slices = expected_slice_count(root_pos.shape[0])
        wav_slices = sorted_slice_paths(source_wav_dir, sequence_name, ".wav")
        feature_slices = sorted_slice_paths(source_feature_dir, sequence_name, ".npy")
        if len(wav_slices) != len(feature_slices):
            raise ValueError(
                f"{sequence_name}: AIST wav/feature slice count mismatch "
                f"({len(wav_slices)} vs {len(feature_slices)})"
            )
        required_frames = WINDOW_FRAMES + STRIDE_FRAMES * (len(wav_slices) - 1)
        if root_pos.shape[0] < required_frames:
            missing_frames = required_frames - root_pos.shape[0]
            if missing_frames > 1:
                raise ValueError(
                    f"{sequence_name}: G1 motion is short by {missing_frames} frames "
                    f"for {len(wav_slices)} AIST audio slices"
                )
            root_pos = np.concatenate((root_pos, root_pos[-1:]), axis=0)
            root_rot = np.concatenate((root_rot, root_rot[-1:]), axis=0)
            dof_pos = np.concatenate((dof_pos, dof_pos[-1:]), axis=0)
            padded_sequences += 1
            padded_frames += missing_frames
            possible_slices = expected_slice_count(root_pos.shape[0])
        if possible_slices < len(wav_slices):
            raise ValueError(
                f"{sequence_name}: G1 motion produces {possible_slices} slices, "
                f"but AIST has {len(wav_slices)} audio slices"
            )

        write_pickle(
            motion_dir / f"{sequence_name}.pkl",
            compatible_payload(root_pos, root_rot, dof_pos, fps=payload["fps"]),
        )

        for slice_index in range(possible_slices):
            start = slice_index * STRIDE_FRAMES
            end = start + WINDOW_FRAMES
            slice_stem = f"{sequence_name}_slice{slice_index}"
            write_pickle(
                motion_slice_dir / f"{slice_stem}.pkl",
                compatible_payload(
                    root_pos[start:end],
                    root_rot[start:end],
                    dof_pos[start:end],
                    fps=payload["fps"],
                ),
            )
            replace_symlink(source_wav_dir / f"{slice_stem}.wav", wav_slice_dir / f"{slice_stem}.wav")
            replace_symlink(
                source_feature_dir / f"{slice_stem}.npy",
                feature_dir / f"{slice_stem}.npy",
            )
        total_slices += possible_slices

    return {
        "sequences": len(sequence_names),
        "slices": total_slices,
        "padded_sequences": padded_sequences,
        "padded_frames": padded_frames,
    }


def prepare_g1_aist_dataset(
    g1_motion_dir=DEFAULT_G1_MOTION_DIR,
    aist_data_root=DEFAULT_AIST_DATA_ROOT,
    output_root=REPO_ROOT / "data" / "g1_aistpp_full",
    split_dir=DEFAULT_SPLIT_DIR,
    feature_type="jukebox",
    clean=False,
    extract_beats=False,
    g1_motion_beat_source="proxy",
    g1_fk_model_path=None,
    g1_root_quat_order="xyzw",
):
    g1_motion_dir = Path(g1_motion_dir)
    aist_data_root = Path(aist_data_root)
    output_root = Path(output_root)
    split_dir = Path(split_dir)
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    g1_map = {path.stem: path for path in sorted(g1_motion_dir.glob("*.pkl"))}
    train_split = set(read_split_file(split_dir / "crossmodal_train.txt"))
    test_split = set(read_split_file(split_dir / "crossmodal_test.txt"))
    ignore = set(read_split_file(split_dir / "ignore_list.txt"))
    train_names = sorted((train_split - ignore) & set(g1_map))
    test_names = sorted(test_split & set(g1_map))

    missing_train = sorted((train_split - ignore) - set(g1_map))
    missing_test = sorted(test_split - set(g1_map))
    if missing_train or missing_test:
        raise ValueError(
            "Retargeted G1 folder is missing official split motions: "
            f"train={missing_train[:10]}, test={missing_test[:10]}"
        )

    summary = {
        "source": "AIST retargeted to G1",
        "source_motion_dir": str(g1_motion_dir),
        "aist_data_root": str(aist_data_root),
        "feature_type": feature_type,
        "motion_payload": "G1 root_pos/root_rot/dof_pos plus compatibility pos and q=root_rot+dof_pos.",
        "split_policy": "Official EDGE AIST crossmodal train/test; train ignores ignore_list.txt; FineDance excluded.",
        "beat_metadata": {
            "extracted": bool(extract_beats),
            "motion_beat_source": g1_motion_beat_source if extract_beats else "",
            "g1_fk_model_path": str(g1_fk_model_path) if extract_beats and g1_fk_model_path else "",
            "g1_root_quat_order": g1_root_quat_order if extract_beats else "",
        },
        "target_fps": FPS,
        "window_seconds": WINDOW_SECONDS,
        "window_frames": WINDOW_FRAMES,
        "stride_seconds": STRIDE_SECONDS,
        "stride_frames": STRIDE_FRAMES,
        "raw_g1_sequences": len(g1_map),
        "official_train_sequences": len(train_split),
        "official_test_sequences": len(test_split),
        "ignored_train_sequences": len(train_split & ignore),
        "unused_sequences_outside_official_split": len(set(g1_map) - train_split - test_split - ignore),
    }
    summary["train"] = prepare_split(
        "train",
        train_names,
        g1_map,
        aist_data_root,
        output_root,
        feature_type,
    )
    summary["test"] = prepare_split(
        "test",
        test_names,
        g1_map,
        aist_data_root,
        output_root,
        feature_type,
    )

    if extract_beats:
        if g1_motion_beat_source == "fk" and not g1_fk_model_path:
            raise ValueError("--g1_fk_model_path is required when --g1_motion_beat_source fk")
        for split_name in ("train", "test"):
            split_root = output_root / split_name
            extract_beat_metadata(
                str(split_root / "motions_sliced"),
                str(split_root / "wavs_sliced"),
                str(split_root / "beat_feats"),
                fps=int(round(FPS)),
                seq_len=WINDOW_FRAMES,
                g1_motion_beat_source=g1_motion_beat_source,
                g1_fk_model_path=g1_fk_model_path,
                g1_root_quat_order=g1_root_quat_order,
            )

    split_out = output_root / "splits"
    split_out.mkdir(parents=True, exist_ok=True)
    for name in ("crossmodal_train.txt", "crossmodal_test.txt", "ignore_list.txt"):
        shutil.copyfile(split_dir / name, split_out / name)

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare full AIST retargeted G1 data for EDGE.")
    parser.add_argument("--g1_motion_dir", default=str(DEFAULT_G1_MOTION_DIR))
    parser.add_argument("--aist_data_root", default=str(DEFAULT_AIST_DATA_ROOT))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "data" / "g1_aistpp_full"))
    parser.add_argument("--split_dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--feature_type", choices=("jukebox", "baseline"), default="jukebox")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--extract_beats", action="store_true")
    parser.add_argument(
        "--g1_motion_beat_source",
        choices=("proxy", "fk"),
        default="proxy",
    )
    parser.add_argument(
        "--g1_fk_model_path",
        default=str(REPO_ROOT / "third_party" / "unitree_g1_description" / "g1_29dof_rev_1_0.xml"),
    )
    parser.add_argument(
        "--g1_root_quat_order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = prepare_g1_aist_dataset(
        g1_motion_dir=args.g1_motion_dir,
        aist_data_root=args.aist_data_root,
        output_root=args.output_root,
        split_dir=args.split_dir,
        feature_type=args.feature_type,
        clean=args.clean,
        extract_beats=args.extract_beats,
        g1_motion_beat_source=args.g1_motion_beat_source,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
