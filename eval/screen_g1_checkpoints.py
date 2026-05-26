import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.g1_metrics import write_json
from eval.run_g1_dataset_eval import run_g1_dataset_evaluation


DEFAULT_REFERENCE_EVAL_DIR = (
    "slurm/pipelines/20260504-g1-aist-beatdistance-fkbeats/eval"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_epoch_list(value):
    epochs = []
    for item in value.split(","):
        item = item.strip()
        if item:
            epochs.append(int(item))
    if not epochs:
        raise ValueError("--checkpoint_epochs must include at least one epoch")
    return epochs


def checkpoint_path(project, train_name, epoch):
    return Path(project) / train_name / "weights" / f"train-{epoch}.pt"


def metric_value(metrics, name, default=0.0):
    try:
        value = float(metrics.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def ratio_excess(candidate, reference, name, ratio, additive=0.0):
    value = metric_value(candidate, name)
    ref_value = max(metric_value(reference, name), 1e-8)
    allowed = max(ref_value * float(ratio), ref_value + float(additive))
    return max(0.0, value / allowed - 1.0)


def g1_beat_score(metrics):
    fk_bas = metric_value(metrics, "G1FKBAS")
    f1 = metric_value(metrics, "G1BeatF1")
    bas = metric_value(metrics, "G1BAS")
    if fk_bas > 0.0 or f1 > 0.0:
        return 0.45 * fk_bas + 0.45 * f1 + 0.10 * bas
    return bas


def g1_quality_penalty(metrics, reference):
    return (
        1.20 * ratio_excess(metrics, reference, "G1FootSliding", 1.35, additive=0.15)
        + 1.10 * ratio_excess(metrics, reference, "RootVelocityMean", 1.50, additive=0.20)
        + 1.00 * ratio_excess(metrics, reference, "RootSmoothnessJerkMean", 1.60, additive=250.0)
        + 0.90 * ratio_excess(metrics, reference, "G1Dist", 1.35, additive=1.0)
        + 0.75 * ratio_excess(metrics, reference, "RootHeightViolationRate", 1.50, additive=0.08)
        + 0.50 * ratio_excess(metrics, reference, "G1GroundPenetration", 1.30, additive=0.02)
    )


def evaluate_g1_acceptance(metrics, reference):
    failed_rules = []
    checks = (
        ("G1FootSliding", 1.50, 0.20),
        ("RootVelocityMean", 1.75, 0.25),
        ("RootSmoothnessJerkMean", 1.90, 300.0),
        ("G1Dist", 1.60, 1.5),
        ("RootHeightViolationRate", 1.75, 0.10),
        ("G1GroundPenetration", 1.40, 0.02),
    )
    for name, ratio, additive in checks:
        if ratio_excess(metrics, reference, name, ratio, additive=additive) > 0.0:
            failed_rules.append(name)
    if g1_beat_score(metrics) < g1_beat_score(reference) - 0.02:
        failed_rules.append("beat_score")
    return {
        "accepted": not failed_rules,
        "failed_rules": failed_rules,
    }


def g1_checkpoint_score(metrics, reference):
    acceptance = evaluate_g1_acceptance(metrics, reference)
    return (
        1 if acceptance["accepted"] else 0,
        g1_beat_score(metrics) - g1_quality_penalty(metrics, reference),
        metric_value(metrics, "G1FKBAS"),
        metric_value(metrics, "G1BeatF1"),
        -metric_value(metrics, "G1Dist"),
    )


def build_eval_args(args, checkpoint, eval_dir, max_eval_clips, diagnostic_count):
    return SimpleNamespace(
        checkpoint=str(checkpoint),
        feature_type=args.feature_type,
        data_path=args.data_path,
        processed_data_dir=args.processed_data_dir,
        render_dir=str(eval_dir / "renders"),
        motion_save_dir=str(eval_dir / "motions"),
        metrics_path=str(eval_dir / "metrics.json"),
        g1_table_path=str(eval_dir / "g1_table.json"),
        motion_audit_path=str(eval_dir / "motion_audit.json"),
        paper_report_path=str(eval_dir / "paper_report.md"),
        seed=args.seed,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        max_eval_clips=max_eval_clips,
        diagnostic_count=diagnostic_count,
        enable_fk_metrics=args.enable_fk_metrics,
        g1_fk_model_path=args.g1_fk_model_path,
        g1_root_quat_order=args.g1_root_quat_order,
    )


def screen_checkpoints(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_metrics = load_json(Path(args.reference_eval_dir) / "metrics.json")
    candidates = []
    for epoch in parse_epoch_list(args.checkpoint_epochs):
        ckpt = checkpoint_path(args.project, args.train_name, epoch)
        if not ckpt.is_file():
            continue
        eval_dir = output_dir / "screen" / f"epoch_{epoch}"
        metrics = run_g1_dataset_evaluation(
            build_eval_args(
                args,
                checkpoint=ckpt,
                eval_dir=eval_dir,
                max_eval_clips=args.screen_eval_clips,
                diagnostic_count=0,
            )
        )
        acceptance = evaluate_g1_acceptance(metrics, reference_metrics)
        candidates.append(
            {
                "epoch": epoch,
                "checkpoint": str(ckpt),
                "metrics_path": str(eval_dir / "metrics.json"),
                "metrics": metrics,
                "acceptance": acceptance,
                "beat_score": g1_beat_score(metrics),
                "quality_penalty": g1_quality_penalty(metrics, reference_metrics),
                "score_tuple": list(g1_checkpoint_score(metrics, reference_metrics)),
            }
        )
    if not candidates:
        raise FileNotFoundError("No requested G1 checkpoint files were found")

    selected = max(
        candidates,
        key=lambda candidate: tuple(candidate["score_tuple"]),
    )
    full_eval_dir = output_dir / "full"
    full_metrics = run_g1_dataset_evaluation(
        build_eval_args(
            args,
            checkpoint=selected["checkpoint"],
            eval_dir=full_eval_dir,
            max_eval_clips=0,
            diagnostic_count=args.diagnostic_count,
        )
    )
    full_acceptance = evaluate_g1_acceptance(full_metrics, reference_metrics)
    report = {
        "accepted": full_acceptance["accepted"],
        "failed_rules": full_acceptance["failed_rules"],
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_screen_metrics_path": selected["metrics_path"],
        "screen_eval_clips": args.screen_eval_clips,
        "screen_candidates": candidates,
        "full_metrics_path": str(full_eval_dir / "metrics.json"),
        "full_acceptance": full_acceptance,
        "reference_eval_dir": args.reference_eval_dir,
        "data_path": args.data_path,
        "processed_data_dir": args.processed_data_dir,
        "quality_note": (
            "This is global checkpoint selection within one train_name. "
            "It does not mix clips across methods."
        ),
    }
    write_json(output_dir / "g1_checkpoint_selection.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--train_name", required=True)
    parser.add_argument("--checkpoint_epochs", default="100,200,300,400,500")
    parser.add_argument("--screen_eval_clips", type=int, default=32)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_eval_dir", default=DEFAULT_REFERENCE_EVAL_DIR)
    parser.add_argument("--feature_type", default="jukebox")
    parser.add_argument("--data_path", default="data/g1_aistpp_full_fkbeats")
    parser.add_argument(
        "--processed_data_dir",
        default="data/g1_aistpp_full_fkbeats_dataset_backups",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--diagnostic_count", type=int, default=8)
    parser.add_argument("--enable_fk_metrics", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument(
        "--g1_root_quat_order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
    )
    return parser.parse_args()


def main(args=None):
    return screen_checkpoints(args or parse_args())


if __name__ == "__main__":
    main()
