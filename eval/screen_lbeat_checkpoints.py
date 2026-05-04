import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.lbeat_quality_gate import (evaluate_lbeat_acceptance,
                                     exit_code_for_lbeat_report,
                                     load_reference_metrics,
                                     select_best_checkpoint, write_json)
from eval.run_dataset_eval import run_dataset_evaluation


def parse_epoch_list(value):
    epochs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        epochs.append(int(item))
    if not epochs:
        raise ValueError("--checkpoint_epochs must include at least one epoch.")
    return epochs


def build_eval_args(args, checkpoint_path, eval_dir, max_eval_clips):
    return SimpleNamespace(
        checkpoint=str(checkpoint_path),
        feature_type=args.feature_type,
        data_path=args.data_path,
        processed_data_dir=args.processed_data_dir,
        render_dir=str(eval_dir / "renders"),
        motion_save_dir=str(eval_dir / "motions"),
        metrics_path=str(eval_dir / "metrics.json"),
        edge_table_path=str(eval_dir / "edge_table.json"),
        beatit_table_path=str(eval_dir / "beatit_table.json"),
        paper_report_path=str(eval_dir / "paper_report.md"),
        pfc_audit_path=str(eval_dir / "pfc_audit.json"),
        seed=args.seed,
        use_beats=True,
        beat_rep=args.beat_rep,
        max_eval_clips=max_eval_clips,
    )


def checkpoint_path(project, train_name, epoch):
    return Path(project) / train_name / "weights" / f"train-{epoch}.pt"


def screen_checkpoints(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_metrics = load_reference_metrics(args.reference_eval_dir)
    candidates = []

    for epoch in parse_epoch_list(args.checkpoint_epochs):
        ckpt = checkpoint_path(args.project, args.train_name, epoch)
        if not ckpt.is_file():
            continue
        eval_dir = output_dir / "screen" / f"epoch_{epoch}"
        metrics = run_dataset_evaluation(
            build_eval_args(
                args,
                checkpoint_path=ckpt,
                eval_dir=eval_dir,
                max_eval_clips=args.screen_eval_clips,
            )
        )
        candidates.append(
            {
                "epoch": epoch,
                "checkpoint": str(ckpt),
                "metrics_path": str(eval_dir / "metrics.json"),
                "metrics": metrics,
            }
        )

    if not candidates:
        raise FileNotFoundError("No requested lbeat checkpoint files were found.")

    selected = select_best_checkpoint(candidates, reference_metrics)
    full_eval_dir = output_dir / "full"
    full_metrics = run_dataset_evaluation(
        build_eval_args(
            args,
            checkpoint_path=selected["checkpoint"],
            eval_dir=full_eval_dir,
            max_eval_clips=0,
        )
    )
    full_acceptance = evaluate_lbeat_acceptance(full_metrics, reference_metrics)
    report = {
        "accepted": full_acceptance["accepted"],
        "failed_rules": full_acceptance["failed_rules"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_epoch": selected["epoch"],
        "screen_candidates": candidates,
        "screen_selection": selected,
        "full_metrics_path": str(full_eval_dir / "metrics.json"),
        "full_acceptance": full_acceptance,
        "reference_eval_dir": args.reference_eval_dir,
    }
    write_json(output_dir / "lbeat_selection.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--train_name", required=True)
    parser.add_argument("--checkpoint_epochs", default="50,100,200,300,400,500")
    parser.add_argument("--screen_eval_clips", type=int, default=16)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_eval_dir", required=True)
    parser.add_argument("--feature_type", default="jukebox")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--processed_data_dir", default="data/dataset_backups")
    parser.add_argument("--render_dir", default="renders")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument(
        "--fail_on_rejection",
        action="store_true",
        help="Exit non-zero when full eval completes but the lbeat quality gate rejects it.",
    )
    return parser.parse_args()


def main(args=None):
    args = args or parse_args()
    report = screen_checkpoints(args)
    return exit_code_for_lbeat_report(
        report,
        fail_on_rejection=getattr(args, "fail_on_rejection", False),
    )


if __name__ == "__main__":
    sys.exit(main())
