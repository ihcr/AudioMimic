import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.g1_motion_prior_dataset import (
    G1MotionPriorDataset,
    G1MotionPriorNormalizer,
)
from dataset.motion_representation import (
    G1_YAW_DELTA_MOTION_FORMAT,
    decode_g1_motion,
    motion_repr_dim,
)
from eval.g1_metrics import run_g1_motion_evaluation
from model.g1_motion_prior import (
    G1MotionAutoencoder,
    G1MotionPriorLossWeights,
    compute_g1_motion_prior_losses,
)
from model.g1_torch_kinematics import G1TorchKinematics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats", type=str)
    parser.add_argument(
        "--processed_data_dir",
        default="data/finedance_g1_v6b_motion_prior_dataset_backups",
        type=str,
    )
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--motion_format", default=G1_YAW_DELTA_MOTION_FORMAT, type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_eval_clips", default=0, type=int)
    parser.add_argument("--cache_limit_per_split", default=0, type=int)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--diagnostic_count", default=8, type=int)
    parser.add_argument("--render_count", default=0, type=int)
    parser.add_argument("--enable_fk_metrics", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--g1_render_backend", choices=("mujoco", "stick"), default="mujoco")
    parser.add_argument("--g1_render_width", default=960, type=int)
    parser.add_argument("--g1_render_height", default=720, type=int)
    parser.add_argument("--g1_mujoco_gl", default="egl", type=str)
    return parser.parse_args()


def _load_source_payload(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _save_pickle(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _decode_raw_motion(raw_motion, motion_format):
    decoded = decode_g1_motion(raw_motion.unsqueeze(0), motion_format=motion_format)
    return {key: value.squeeze(0).detach().cpu().numpy().astype(np.float32) for key, value in decoded.items()}


def save_decoded_g1_motion(raw_motion, source_path, output_path, motion_format):
    decoded = _decode_raw_motion(raw_motion, motion_format)
    source = _load_source_payload(source_path)
    designated = source.get("designated_beat_frames")
    if designated is not None:
        designated = np.asarray(designated, dtype=np.int64)
    payload = {
        "motion_rep": "g1",
        "motion_format": motion_format,
        "fps": float(source.get("fps", 30.0) or 30.0),
        "root_pos": decoded["root_pos"],
        "root_rot": decoded["root_rot"],
        "dof_pos": decoded["dof_pos"],
        "pos": decoded["root_pos"],
        "q": np.concatenate([decoded["root_rot"], decoded["dof_pos"]], axis=-1).astype(np.float32),
        "source_path": str(source_path),
        "audio_path": source.get("audio_path", ""),
        "designated_beat_frames": designated,
    }
    _save_pickle(payload, output_path)
    return payload


def build_model_from_checkpoint(checkpoint, motion_format):
    config = checkpoint.get("config", {})
    model = G1MotionAutoencoder(
        input_dim=motion_repr_dim(motion_format),
        latent_dim=int(config.get("latent_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        temporal_downsample=int(config.get("temporal_downsample", 2)),
        prior_type=config.get("prior_type", "ae"),
        dropout=float(config.get("dropout", 0.0)),
        motion_format=motion_format,
    )
    model.load_state_dict(checkpoint["model"])
    return model


def _average_stats(total, stats, batch_size):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    total["_count"] = total.get("_count", 0) + batch_size


def _finalize_stats(total):
    count = max(int(total.pop("_count", 0)), 1)
    return {key: value / count for key, value in sorted(total.items())}


def _render_pairs(records, output_dir, args):
    if args.render_count <= 0:
        return []
    from eval.g1_visualization import render_g1_motion

    pair_dir = Path(output_dir) / "render_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index, record in enumerate(records[: args.render_count]):
        with open(record["target_path"], "rb") as handle:
            target_motion = pickle.load(handle)
        with open(record["pred_path"], "rb") as handle:
            pred_motion = pickle.load(handle)
        target_video = render_g1_motion(
            target_motion,
            pair_dir,
            epoch=0,
            num=index,
            name=f"{record['stem']}_target",
            sound=False,
            model_path=args.g1_fk_model_path,
            root_quat_order=args.g1_root_quat_order,
            render_backend=args.g1_render_backend,
            width=args.g1_render_width,
            height=args.g1_render_height,
            mujoco_gl=args.g1_mujoco_gl,
        )
        pred_video = render_g1_motion(
            pred_motion,
            pair_dir,
            epoch=0,
            num=index,
            name=f"{record['stem']}_recon",
            sound=False,
            model_path=args.g1_fk_model_path,
            root_quat_order=args.g1_root_quat_order,
            render_backend=args.g1_render_backend,
            width=args.g1_render_width,
            height=args.g1_render_height,
            mujoco_gl=args.g1_mujoco_gl,
        )
        comparison = pair_dir / f"{index:03d}_{record['stem']}_target_recon_hstack.mp4"
        if shutil.which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    target_video,
                    "-i",
                    pred_video,
                    "-filter_complex",
                    "hstack=inputs=2",
                    str(comparison),
                ],
                check=True,
            )
            comparison_path = str(comparison)
        else:
            comparison_path = ""
        rendered.append(
            {
                "stem": record["stem"],
                "target_video": target_video,
                "recon_video": pred_video,
                "comparison_video": comparison_path,
                "layout": "left=target,right=reconstruction" if comparison_path else "separate",
            }
        )
    return rendered


def run_g1_motion_prior_eval(args):
    os.environ.setdefault("MUJOCO_GL", args.g1_mujoco_gl)
    os.environ.setdefault("PYOPENGL_PLATFORM", args.g1_mujoco_gl)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "recon_motions"
    target_dir = output_dir / "target_motions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    motion_format = args.motion_format or checkpoint.get("config", {}).get("motion_format", G1_YAW_DELTA_MOTION_FORMAT)
    normalizer = G1MotionPriorNormalizer.from_state_dict(checkpoint["normalizer"])
    mean, std = normalizer.tensors(device=device)
    model = build_model_from_checkpoint(checkpoint, motion_format).to(device)
    model.eval()
    dataset = G1MotionPriorDataset(
        data_path=args.data_path,
        backup_path=args.processed_data_dir,
        split=args.split,
        motion_format=motion_format,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
        cache_batch_size=args.batch_size,
        cache_device=device.type,
        cache_limit_per_split=args.cache_limit_per_split,
        rebuild_cache=args.rebuild_cache,
        data_len=args.max_eval_clips,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    kinematics = G1TorchKinematics(args.g1_fk_model_path, root_quat_order=args.g1_root_quat_order).to(device)
    weights = G1MotionPriorLossWeights()
    stats_total = {}
    records = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="V6b prior reconstruction eval", unit="batch"):
            motion = batch["motion"].to(device).float()
            contact = batch["contact"].to(device).float()
            ground = batch["ground"].to(device).float()
            output = model(motion, sample=False)
            _, stats = compute_g1_motion_prior_losses(
                output,
                motion,
                contact,
                mean,
                std,
                kinematics=kinematics,
                motion_format=motion_format,
                weights=weights,
                ground=ground,
            )
            _average_stats(stats_total, stats, int(motion.shape[0]))
            pred_raw = output["recon"] * std + mean
            target_raw = motion * std + mean
            for index in range(motion.shape[0]):
                stem = batch["stem"][index]
                source_path = batch["source_path"][index]
                pred_path = pred_dir / f"{stem}.pkl"
                target_path = target_dir / f"{stem}.pkl"
                save_decoded_g1_motion(pred_raw[index].detach().cpu(), source_path, pred_path, motion_format)
                save_decoded_g1_motion(target_raw[index].detach().cpu(), source_path, target_path, motion_format)
                records.append(
                    {
                        "stem": stem,
                        "source_path": source_path,
                        "pred_path": str(pred_path),
                        "target_path": str(target_path),
                    }
                )

    reconstruction_metrics = _finalize_stats(stats_total)
    reconstruction_metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "num_eval_clips": len(records),
            "motion_format": motion_format,
        }
    )
    _write_json(reconstruction_metrics, output_dir / "reconstruction_metrics.json")
    _write_json({"records": records}, output_dir / "manifest.json")
    rendered = _render_pairs(records, output_dir, args)
    _write_json({"rendered": rendered}, output_dir / "render_manifest.json")

    metrics = run_g1_motion_evaluation(
        motion_path=pred_dir,
        reference_motion_path=target_dir,
        metrics_path=output_dir / "metrics.json",
        g1_table_path=output_dir / "g1_table.json",
        motion_audit_path=output_dir / "motion_audit.json",
        paper_report_path=output_dir / "paper_report.md",
        render_dir=output_dir / "diagnostics",
        diagnostic_count=args.diagnostic_count,
        checkpoint=args.checkpoint,
        feature_type="g1_motion_prior_reconstruction",
        use_beats=False,
        beat_rep="none",
        seed=1234,
        sample_limit=None,
        enable_fk_metrics=args.enable_fk_metrics,
        fk_model_path=args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
        failure_panel_path=output_dir / "failure_panel.json",
    )
    gt_dir = output_dir / "gt_baseline"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_metrics = run_g1_motion_evaluation(
        motion_path=target_dir,
        reference_motion_path=target_dir,
        metrics_path=gt_dir / "metrics.json",
        g1_table_path=gt_dir / "g1_table.json",
        motion_audit_path=gt_dir / "motion_audit.json",
        paper_report_path=gt_dir / "paper_report.md",
        render_dir=gt_dir / "diagnostics",
        diagnostic_count=min(args.diagnostic_count, 4),
        checkpoint="target_yaw_delta",
        feature_type="g1_motion_prior_target_baseline",
        use_beats=False,
        beat_rep="none",
        seed=1234,
        sample_limit=None,
        enable_fk_metrics=args.enable_fk_metrics,
        fk_model_path=args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
        failure_panel_path=gt_dir / "failure_panel.json",
    )
    summary = {
        "metrics_path": str(output_dir / "metrics.json"),
        "gt_metrics_path": str(gt_dir / "metrics.json"),
        "reconstruction_metrics_path": str(output_dir / "reconstruction_metrics.json"),
        "render_manifest_path": str(output_dir / "render_manifest.json"),
        "num_eval_clips": len(records),
        "G1NoNearSupportRate_delta_vs_gt": metrics.get("G1NoNearSupportRate", 0.0)
        - gt_metrics.get("G1NoNearSupportRate", 0.0),
        "G1FootHighLiftRate_delta_vs_gt": metrics.get("G1FootHighLiftRate", 0.0)
        - gt_metrics.get("G1FootHighLiftRate", 0.0),
        "G1GroundPenetration_delta_vs_gt": metrics.get("G1GroundPenetration", 0.0)
        - gt_metrics.get("G1GroundPenetration", 0.0),
    }
    _write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    run_g1_motion_prior_eval(parse_args())


if __name__ == "__main__":
    main()
