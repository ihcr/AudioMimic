import argparse
import json
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
from eval.run_benchmark_eval import run_benchmark_evaluation
from model.diffusion import cond_batch_size, move_cond_to_device, slice_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--feature_type", default="jukebox", type=str)
    parser.add_argument("--data_path", default="data", type=str)
    parser.add_argument("--processed_data_dir", default="data/dataset_backups", type=str)
    parser.add_argument("--render_dir", default="renders/eval_dataset", type=str)
    parser.add_argument("--motion_save_dir", default="eval/motions_dataset", type=str)
    parser.add_argument("--metrics_path", default="eval/metrics_dataset.json", type=str)
    parser.add_argument("--edge_table_path", default="eval/edge_table.json", type=str)
    parser.add_argument("--beatit_table_path", default="eval/beatit_table.json", type=str)
    parser.add_argument("--paper_report_path", default="eval/paper_report.md", type=str)
    parser.add_argument("--pfc_audit_path", default="eval/pfc_audit.json", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--max_eval_clips", default=0, type=int)
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


def batch_to_render_tuple(batch):
    _, cond, _, wavname = batch
    return None, cond, wavname


def render_dataset_batch(model, batch, render_dir, motion_dir, label="eval_dataset"):
    _, cond, wavname = batch_to_render_tuple(batch)
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


def iter_limited_batches(loader, max_eval_clips=0):
    for idx, batch in enumerate(loader):
        if max_eval_clips and idx >= max_eval_clips:
            break
        yield batch


def run_dataset_evaluation(args):
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
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(
        iter_limited_batches(loader, max_eval_clips=args.max_eval_clips),
        desc="Dataset eval",
        unit="clip",
    ):
        render_dataset_batch(model, batch, args.render_dir, motion_dir)

    method_name = Path(args.checkpoint).stem
    return run_benchmark_evaluation(
        motion_path=str(motion_dir),
        metrics_path=args.metrics_path,
        edge_table_path=args.edge_table_path,
        beatit_table_path=args.beatit_table_path,
        paper_report_path=args.paper_report_path,
        pfc_audit_path=args.pfc_audit_path,
        method_name=method_name,
        feature_type=args.feature_type,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        reference_motion_path=str(Path(args.data_path) / "test" / "motions_sliced"),
        seed=args.seed,
        checkpoint=args.checkpoint,
        motion_source="generated",
        extra_metadata={"num_generated_clips": len(list(Path(motion_dir).glob("*.pkl")))},
    )


if __name__ == "__main__":
    run_dataset_evaluation(parse_args())
