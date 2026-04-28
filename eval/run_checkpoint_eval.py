import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.run_benchmark_eval import run_benchmark_evaluation
from test import test as run_generation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--feature_type", default="jukebox", type=str)
    parser.add_argument("--music_dir", default="data/test/wavs_sliced", type=str)
    parser.add_argument("--processed_data_dir", default="data/dataset_backups/", type=str)
    parser.add_argument("--render_dir", default="renders/eval", type=str)
    parser.add_argument("--motion_save_dir", default="eval/motions", type=str)
    parser.add_argument("--metrics_path", default="eval/metrics.json", type=str)
    parser.add_argument("--edge_table_path", default="eval/edge_table.json", type=str)
    parser.add_argument("--beatit_table_path", default="eval/beatit_table.json", type=str)
    parser.add_argument("--paper_report_path", default="eval/paper_report.md", type=str)
    parser.add_argument("--pfc_audit_path", default="eval/pfc_audit.json", type=str)
    parser.add_argument("--out_length", default=5.0, type=float)
    parser.add_argument("--feature_cache_dir", default="cached_features/", type=str)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--use_cached_features", action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--reference_motion_path", default="data/test/motions_sliced", type=str)
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--beat_source", choices=("none", "audio", "user"), default="none")
    parser.add_argument("--beat_file", default="", type=str)
    return parser.parse_args()


def build_test_options(args):
    return SimpleNamespace(
        feature_type=args.feature_type,
        out_length=args.out_length,
        processed_data_dir=args.processed_data_dir,
        render_dir=args.render_dir,
        checkpoint=args.checkpoint,
        music_dir=args.music_dir,
        save_motions=True,
        motion_save_dir=args.motion_save_dir,
        cache_features=args.cache_features,
        no_render=True,
        use_cached_features=args.use_cached_features,
        feature_cache_dir=args.feature_cache_dir,
        seed=args.seed,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        beat_source=args.beat_source,
        beat_file=args.beat_file,
    )


def clear_motion_dir(motion_dir):
    motion_dir = Path(motion_dir)
    if motion_dir.exists():
        shutil.rmtree(motion_dir)
    motion_dir.mkdir(parents=True, exist_ok=True)
    return motion_dir


def run_checkpoint_evaluation(args):
    motion_dir = clear_motion_dir(args.motion_save_dir)
    Path(args.render_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)

    run_generation(build_test_options(args))

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
        reference_motion_path=args.reference_motion_path,
        seed=args.seed,
        checkpoint=args.checkpoint,
        beat_source=args.beat_source,
        motion_source="generated",
    )


if __name__ == "__main__":
    run_checkpoint_evaluation(parse_args())
