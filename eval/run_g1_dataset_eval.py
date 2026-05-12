import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EDGE import EDGE
from dataset.dance_dataset import AISTPPDataset
from eval.g1_metrics import run_g1_motion_evaluation
from model.diffusion import cond_batch_size, move_cond_to_device, slice_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--feature_type", default="jukebox", type=str)
    parser.add_argument("--data_path", default="data/g1_aistpp", type=str)
    parser.add_argument("--processed_data_dir", default="data/g1_dataset_backups", type=str)
    parser.add_argument("--render_dir", default="eval/g1/renders", type=str)
    parser.add_argument("--motion_save_dir", default="eval/g1/motions", type=str)
    parser.add_argument("--metrics_path", default="eval/g1/metrics.json", type=str)
    parser.add_argument("--g1_table_path", default="eval/g1/g1_table.json", type=str)
    parser.add_argument("--motion_audit_path", default="eval/g1/motion_audit.json", type=str)
    parser.add_argument("--paper_report_path", default="eval/g1/paper_report.md", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--max_eval_clips", default=0, type=int)
    parser.add_argument("--diagnostic_count", default=8, type=int)
    parser.add_argument("--enable_fk_metrics", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument(
        "--g1_root_quat_order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
    )
    parser.add_argument(
        "--g1_render_backend",
        choices=("mujoco", "stick"),
        default="mujoco",
    )
    parser.add_argument("--g1_render_width", default=960, type=int)
    parser.add_argument("--g1_render_height", default=720, type=int)
    parser.add_argument("--g1_mujoco_gl", default="egl", type=str)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_motion_dir(motion_dir):
    motion_dir = Path(motion_dir)
    if motion_dir.exists():
        shutil.rmtree(motion_dir)
    motion_dir.mkdir(parents=True, exist_ok=True)
    return motion_dir


def iter_limited_batches(loader, max_eval_clips=0):
    for idx, batch in enumerate(loader):
        if max_eval_clips and idx >= max_eval_clips:
            break
        yield batch


def render_g1_dataset_batch(model, batch, render_dir, motion_dir, label="g1_eval"):
    _, cond, _, wavname = batch
    render_count = cond_batch_size(cond)
    shape = (render_count, model.horizon, model.repr_dim)
    cond = move_cond_to_device(cond, model.accelerator.device)
    model.diffusion.render_sample(
        shape,
        slice_cond(cond, slice(None, render_count)),
        model.normalizer,
        label,
        render_dir,
        name=wavname[:render_count],
        sound=True,
        mode="normal",
        fk_out=str(motion_dir),
        render=False,
    )


def run_g1_dataset_evaluation(args):
    set_seed(args.seed)
    motion_dir = clear_motion_dir(args.motion_save_dir)
    Path(args.render_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)

    model = EDGE(
        args.feature_type,
        args.checkpoint,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        lambda_beat=0.0,
        motion_format="g1",
    )
    model.eval()

    dataset = AISTPPDataset(
        data_path=args.data_path,
        backup_path=args.processed_data_dir,
        train=False,
        feature_type=args.feature_type,
        normalizer=model.normalizer,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        motion_format="g1",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(
        iter_limited_batches(loader, max_eval_clips=args.max_eval_clips),
        desc="G1 dataset eval",
        unit="clip",
    ):
        render_g1_dataset_batch(model, batch, args.render_dir, motion_dir)

    return run_g1_motion_evaluation(
        motion_path=motion_dir,
        reference_motion_path=Path(args.data_path) / "test" / "motions_sliced",
        metrics_path=args.metrics_path,
        g1_table_path=args.g1_table_path,
        motion_audit_path=args.motion_audit_path,
        paper_report_path=args.paper_report_path,
        render_dir=args.render_dir,
        diagnostic_count=args.diagnostic_count,
        checkpoint=args.checkpoint,
        feature_type=args.feature_type,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        seed=args.seed,
        enable_fk_metrics=args.enable_fk_metrics,
        fk_model_path=args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
    )


if __name__ == "__main__":
    run_g1_dataset_evaluation(parse_args())
