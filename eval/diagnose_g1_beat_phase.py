import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from tqdm import tqdm

from data.audio_extraction.motion_energy_features import KEYPOINT_WEIGHTS
from eval.eval_bas_bap import (
    DEFAULT_BAP_TOLERANCE,
    DEFAULT_BAS_SIGMA_SQUARED,
    load_audio_beat_frames,
)
from model.g1_torch_kinematics import G1TorchKinematics


def _load_motion(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    root_pos = np.asarray(payload.get("root_pos", payload.get("pos")), dtype=np.float32)
    root_rot = payload.get("root_rot")
    dof_pos = payload.get("dof_pos")
    if root_rot is None or dof_pos is None:
        q = np.asarray(payload["q"], dtype=np.float32)
        root_rot = q[:, :4]
        dof_pos = q[:, 4:33]
    return {
        "root_pos": root_pos,
        "root_rot": np.asarray(root_rot, dtype=np.float32),
        "dof_pos": np.asarray(dof_pos, dtype=np.float32),
        "fps": float(payload.get("fps", 30.0)),
        "audio_path": payload.get("audio_path"),
    }


def _gt_audio_path(data_root, motion_path):
    return str(Path(data_root) / "test" / "wavs_sliced" / f"{motion_path.stem}.wav")


def _stack_motion_batch(paths):
    roots, rots, dofs = [], [], []
    for path in paths:
        motion = _load_motion(path)
        roots.append(motion["root_pos"])
        rots.append(motion["root_rot"])
        dofs.append(motion["dof_pos"])
    return (
        torch.from_numpy(np.stack(roots, axis=0)),
        torch.from_numpy(np.stack(rots, axis=0)),
        torch.from_numpy(np.stack(dofs, axis=0)),
    )


def _mean_fk_speed(keypoints, fps):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    velocity = np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=-1).mean(axis=-1)
    velocity = np.concatenate((velocity[:1], velocity), axis=0) * float(fps)
    return velocity.astype(np.float32)


def _weighted_fk_speed(keypoints, keypoint_names, fps):
    indices = [keypoint_names.index(name) for name in KEYPOINT_WEIGHTS]
    weights = np.asarray([KEYPOINT_WEIGHTS[name] for name in KEYPOINT_WEIGHTS], dtype=np.float32)
    selected = np.asarray(keypoints, dtype=np.float32)[:, indices]
    frame_speed = np.zeros(selected.shape[:2], dtype=np.float32)
    frame_speed[1:] = np.linalg.norm(selected[1:] - selected[:-1], axis=-1) * float(fps)
    return np.sum(frame_speed * weights[None, :], axis=-1).astype(np.float32)


def _nearest_offsets(events, beats):
    events = np.sort(np.asarray(events, dtype=np.int64).reshape(-1))
    beats = np.asarray(beats, dtype=np.int64).reshape(-1)
    if beats.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if events.size == 0:
        return np.full((beats.size,), np.nan, dtype=np.float32)
    right = np.searchsorted(events, beats, side="left")
    left = np.clip(right - 1, 0, events.size - 1)
    right = np.clip(right, 0, events.size - 1)
    left_offsets = events[left] - beats
    right_offsets = events[right] - beats
    use_right = np.abs(right_offsets) < np.abs(left_offsets)
    return np.where(use_right, right_offsets, left_offsets).astype(np.float32)


def _window_offsets(speed, beats, radius, mode):
    offsets = []
    speed = np.asarray(speed, dtype=np.float32)
    for beat in np.asarray(beats, dtype=np.int64).reshape(-1):
        start = max(int(beat) - radius, 0)
        end = min(int(beat) + radius + 1, speed.shape[0])
        if start >= end:
            continue
        window = speed[start:end]
        local = int(np.argmin(window) if mode == "min" else np.argmax(window))
        offsets.append(start + local - int(beat))
    return np.asarray(offsets, dtype=np.float32)


def _summarize_offsets(offsets, prefix, out, tolerance):
    offsets = np.asarray(offsets, dtype=np.float32)
    finite = offsets[np.isfinite(offsets)]
    out[f"{prefix}_valid_beats"] = int(finite.size)
    if finite.size == 0:
        for key in (
            "soft_score",
            "abs_mean",
            "abs_median",
            "signed_mean",
            "coverage_t2",
            "coverage_t3",
            "coverage_t6",
            "coverage_t12",
        ):
            out[f"{prefix}_{key}"] = 0.0
        return
    abs_offsets = np.abs(finite)
    out[f"{prefix}_soft_score"] = float(
        np.mean(np.exp(-(finite ** 2) / (2.0 * DEFAULT_BAS_SIGMA_SQUARED)))
    )
    out[f"{prefix}_abs_mean"] = float(np.mean(abs_offsets))
    out[f"{prefix}_abs_median"] = float(np.median(abs_offsets))
    out[f"{prefix}_signed_mean"] = float(np.mean(finite))
    for window in (2, 3, 6, 12):
        out[f"{prefix}_coverage_t{window}"] = float(np.mean(abs_offsets <= window))
    out[f"{prefix}_coverage_eval_tolerance"] = float(np.mean(abs_offsets <= tolerance))


def _empty_accumulator():
    return {
        "files": 0,
        "audio_beats": 0,
        "min_events": 0,
        "max_events": 0,
        "nearest_min_offsets": [],
        "nearest_max_offsets": [],
        "window_min_offsets": [],
        "window_max_offsets": [],
    }


def _add_speed_sample(acc, speed, audio_beats, extrema_sigma, window_radius):
    audio_beats = np.asarray(audio_beats, dtype=np.int64).reshape(-1)
    smoothed = gaussian_filter1d(np.asarray(speed, dtype=np.float32), sigma=extrema_sigma)
    min_events = np.asarray(argrelextrema(smoothed, np.less)[0], dtype=np.int64)
    max_events = np.asarray(argrelextrema(smoothed, np.greater)[0], dtype=np.int64)
    acc["files"] += 1
    acc["audio_beats"] += int(audio_beats.size)
    acc["min_events"] += int(min_events.size)
    acc["max_events"] += int(max_events.size)
    acc["nearest_min_offsets"].append(_nearest_offsets(min_events, audio_beats))
    acc["nearest_max_offsets"].append(_nearest_offsets(max_events, audio_beats))
    acc["window_min_offsets"].append(_window_offsets(smoothed, audio_beats, window_radius, "min"))
    acc["window_max_offsets"].append(_window_offsets(smoothed, audio_beats, window_radius, "max"))


def _finish_accumulator(label, acc, tolerance):
    summary = {
        "label": label,
        "files": int(acc["files"]),
        "audio_beats": int(acc["audio_beats"]),
        "audio_beats_per_clip": float(acc["audio_beats"] / max(acc["files"], 1)),
        "min_events": int(acc["min_events"]),
        "max_events": int(acc["max_events"]),
        "min_events_per_clip": float(acc["min_events"] / max(acc["files"], 1)),
        "max_events_per_clip": float(acc["max_events"] / max(acc["files"], 1)),
    }
    nearest_min = np.concatenate(acc["nearest_min_offsets"]) if acc["nearest_min_offsets"] else np.zeros(0)
    nearest_max = np.concatenate(acc["nearest_max_offsets"]) if acc["nearest_max_offsets"] else np.zeros(0)
    window_min = np.concatenate(acc["window_min_offsets"]) if acc["window_min_offsets"] else np.zeros(0)
    window_max = np.concatenate(acc["window_max_offsets"]) if acc["window_max_offsets"] else np.zeros(0)

    _summarize_offsets(nearest_min, "nearest_min", summary, tolerance)
    _summarize_offsets(nearest_max, "nearest_max", summary, tolerance)
    _summarize_offsets(window_min, "window_min", summary, tolerance)
    _summarize_offsets(window_max, "window_max", summary, tolerance)

    finite_min = nearest_min[np.isfinite(nearest_min)]
    finite_max = nearest_max[np.isfinite(nearest_max)]
    pair_count = min(finite_min.size, finite_max.size)
    if pair_count:
        abs_min = np.abs(finite_min[:pair_count])
        abs_max = np.abs(finite_max[:pair_count])
        summary["nearest_min_closer_fraction"] = float(np.mean(abs_min < abs_max))
        summary["nearest_max_closer_fraction"] = float(np.mean(abs_max < abs_min))
        summary["nearest_min_max_tie_fraction"] = float(np.mean(abs_min == abs_max))
    else:
        summary["nearest_min_closer_fraction"] = 0.0
        summary["nearest_max_closer_fraction"] = 0.0
        summary["nearest_min_max_tie_fraction"] = 0.0
    return summary


def summarize_motion_set(
    label,
    paths,
    *,
    data_root,
    kinematics,
    device,
    batch_size,
    extrema_sigma,
    window_radius,
    tolerance,
):
    paths = list(paths)
    mean_acc = _empty_accumulator()
    weighted_acc = _empty_accumulator()
    keypoint_names = list(kinematics.keypoint_names)
    for start in tqdm(range(0, len(paths), batch_size), desc=f"phase {label}", unit="batch"):
        batch_paths = paths[start : start + batch_size]
        root_pos, root_rot, dof_pos = _stack_motion_batch(batch_paths)
        with torch.inference_mode():
            fk = kinematics(
                root_pos.to(device=device, dtype=torch.float32),
                root_rot.to(device=device, dtype=torch.float32),
                dof_pos.to(device=device, dtype=torch.float32),
            )
        keypoints = fk["keypoints"].detach().cpu().numpy()
        for path, sample_keypoints in zip(batch_paths, keypoints):
            motion = _load_motion(path)
            audio_path = motion["audio_path"] or _gt_audio_path(data_root, path)
            audio_beats = load_audio_beat_frames(
                audio_path,
                fps=int(round(motion["fps"])),
                seq_len=motion["root_pos"].shape[0],
            )
            _add_speed_sample(
                mean_acc,
                _mean_fk_speed(sample_keypoints, motion["fps"]),
                audio_beats,
                extrema_sigma,
                window_radius,
            )
            _add_speed_sample(
                weighted_acc,
                _weighted_fk_speed(sample_keypoints, keypoint_names, motion["fps"]),
                audio_beats,
                extrema_sigma,
                window_radius,
            )
    return {
        "mean_fk_speed": _finish_accumulator(label, mean_acc, tolerance),
        "weighted_fk_speed": _finish_accumulator(label, weighted_acc, tolerance),
    }


def summarize_motion_energy_cache(label, paths, *, extrema_sigma, window_radius, tolerance):
    acc = _empty_accumulator()
    for path in tqdm(paths, desc=f"phase {label}", unit="file"):
        with np.load(path) as data:
            speed = np.asarray(data["weighted_fk_speed"], dtype=np.float32)
            audio_beats = np.asarray(data["audio_beat_frames"], dtype=np.int64)
        _add_speed_sample(acc, speed, audio_beats, extrema_sigma, window_radius)
    return _finish_accumulator(label, acc, tolerance)


def _parse_motion_set(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--motion-set must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def _motion_paths(path):
    if path.is_dir():
        return sorted(path.glob("*.pkl"))
    return sorted(Path().glob(str(path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-set", action="append", type=_parse_motion_set, default=[])
    parser.add_argument("--gt-motion-dir", type=Path)
    parser.add_argument("--motion-energy-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/finedance_g1_fkbeats"))
    parser.add_argument("--g1-fk-model-path", type=Path, default=Path("third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"))
    parser.add_argument("--root-quat-order", default="xyzw", choices=("xyzw", "wxyz"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--extrema-sigma", type=float, default=5.0)
    parser.add_argument("--window-radius", type=int, default=6)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_BAP_TOLERANCE)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    kinematics = G1TorchKinematics(
        args.g1_fk_model_path,
        root_quat_order=args.root_quat_order,
    ).to(device)
    kinematics.eval()

    summaries = {}
    for label, path in args.motion_set:
        paths = _motion_paths(path)
        if args.max_files:
            paths = paths[: args.max_files]
        summaries[label] = summarize_motion_set(
            label,
            paths,
            data_root=args.data_root,
            kinematics=kinematics,
            device=device,
            batch_size=args.batch_size,
            extrema_sigma=args.extrema_sigma,
            window_radius=args.window_radius,
            tolerance=args.tolerance,
        )

    if args.gt_motion_dir:
        paths = sorted(args.gt_motion_dir.glob("*.pkl"))
        if args.max_files:
            paths = paths[: args.max_files]
        summaries["GT_reference"] = summarize_motion_set(
            "GT_reference",
            paths,
            data_root=args.data_root,
            kinematics=kinematics,
            device=device,
            batch_size=args.batch_size,
            extrema_sigma=args.extrema_sigma,
            window_radius=args.window_radius,
            tolerance=args.tolerance,
        )

    if args.motion_energy_dir:
        paths = sorted(args.motion_energy_dir.glob("*.npz"))
        if args.max_files:
            paths = paths[: args.max_files]
        summaries["GT_motion_energy_cache_weighted_speed"] = {
            "weighted_fk_speed": summarize_motion_energy_cache(
                "GT_motion_energy_cache_weighted_speed",
                paths,
                extrema_sigma=args.extrema_sigma,
                window_radius=args.window_radius,
                tolerance=args.tolerance,
            )
        }

    result = {
        "config": {
            "data_root": str(args.data_root),
            "g1_fk_model_path": str(args.g1_fk_model_path),
            "root_quat_order": args.root_quat_order,
            "device": str(device),
            "batch_size": args.batch_size,
            "extrema_sigma": args.extrema_sigma,
            "window_radius": args.window_radius,
            "tolerance": args.tolerance,
            "bas_sigma_squared": DEFAULT_BAS_SIGMA_SQUARED,
            "max_files": args.max_files,
        },
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["config"], indent=2, sort_keys=True))
    for label, value in summaries.items():
        mean_summary = value.get("mean_fk_speed") or value.get("weighted_fk_speed")
        print(
            label,
            "nearest_min_soft",
            round(mean_summary["nearest_min_soft_score"], 4),
            "nearest_max_soft",
            round(mean_summary["nearest_max_soft_score"], 4),
            "min_cov_t2",
            round(mean_summary["nearest_min_coverage_t2"], 4),
            "max_cov_t2",
            round(mean_summary["nearest_max_coverage_t2"], 4),
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
