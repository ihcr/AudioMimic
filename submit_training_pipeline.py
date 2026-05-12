import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from args import resolve_train_test_workers, resolve_worker_count


def resolve_shared_root(path):
    path = Path(path).resolve()
    for parent in [path, *path.parents]:
        if parent.name == ".worktrees":
            return parent.parent
    return path


SCRIPT_ROOT = Path(__file__).resolve().parent
SHARED_ROOT = resolve_shared_root(SCRIPT_ROOT)
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LBEAT_REFERENCE_EVAL_DIR = (
    "/lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/"
    "slurm/pipelines/20260424-090100-edge_beatdistance_20260424_shmfix/eval"
)
DEFAULT_G1_FKBEAT_CHECKPOINT = (
    "runs/train/g1_aist_beatdistance_fkbeats/weights/train-2000.pt"
)


def default_lambda_acc(use_beats):
    return 0.0


def resolve_venv_activate(repo_root):
    return resolve_shared_root(repo_root) / ".venv311" / "bin" / "activate"


@dataclass
class SubmissionResult:
    run_dir: Path
    job_ids: dict


PRESET_DEFAULTS = {
    "edge_baseline": {
        "feature_type": "jukebox",
        "use_beats": False,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 2000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
    },
    "edge_beatpulse": {
        "feature_type": "jukebox",
        "use_beats": True,
        "beat_rep": "pulse",
        "lambda_acc": 0.0,
        "lambda_beat": 0.0,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 2000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
    },
    "edge_beatdistance": {
        "feature_type": "jukebox",
        "use_beats": True,
        "beat_rep": "distance",
        "lambda_acc": 0.0,
        "lambda_beat": 0.0,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 2000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
    },
    "edge_beatdistance_lbeat": {
        "feature_type": "jukebox",
        "use_beats": True,
        "beat_rep": "distance",
        "lambda_acc": 0.1,
        "lambda_beat": 0.02,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 500,
        "save_interval": 50,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "beat_loss_start_epoch": 25,
        "beat_loss_warmup_epochs": 200,
        "beat_loss_max_fraction": 0.25,
        "finetune_from_checkpoint": True,
    },
    "g1_beatdistance": {
        "motion_format": "g1",
        "data_path": "data/g1_aistpp_full",
        "processed_data_dir": "data/g1_aistpp_full_dataset_backups",
        "feature_type": "jukebox",
        "use_beats": True,
        "beat_rep": "distance",
        "lambda_acc": 0.0,
        "lambda_beat": 0.0,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 2000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "feature_cache_mode": "memmap",
        "feature_cache_dtype": "float32",
        "skip_preprocess": True,
    },
    "g1_beatdistance_fkbeats": {
        "motion_format": "g1",
        "data_path": "data/g1_aistpp_full_fkbeats",
        "processed_data_dir": "data/g1_aistpp_full_fkbeats_dataset_backups",
        "feature_type": "jukebox",
        "use_beats": True,
        "beat_rep": "distance",
        "lambda_acc": 0.0,
        "lambda_beat": 0.0,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 2000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "feature_cache_mode": "memmap",
        "feature_cache_dtype": "float32",
        "skip_preprocess": True,
        "enable_g1_fk_metrics": True,
    },
    "g1_beatdistance_fkbeats_lbeat": {
        "motion_format": "g1",
        "data_path": "data/g1_aistpp_full_fkbeats",
        "processed_data_dir": "data/g1_aistpp_full_fkbeats_dataset_backups",
        "feature_type": "jukebox",
        "use_beats": True,
        "beat_rep": "distance",
        "lambda_acc": 0.0,
        "lambda_beat": 0.01,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 500,
        "save_interval": 50,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "beat_loss_start_epoch": 50,
        "beat_loss_warmup_epochs": 300,
        "beat_loss_max_fraction": 0.10,
        "beat_loss_cap_mode": "soft",
        "g1_target_transform": "relative_distance",
        "checkpoint": DEFAULT_G1_FKBEAT_CHECKPOINT,
        "finetune_from_checkpoint": True,
        "feature_cache_mode": "memmap",
        "feature_cache_dtype": "float32",
        "skip_preprocess": True,
        "enable_g1_fk_metrics": True,
    },
    "g1_baseline": {
        "motion_format": "g1",
        "data_path": "data/g1_aistpp_full",
        "processed_data_dir": "data/g1_aistpp_full_dataset_backups",
        "feature_type": "jukebox",
        "use_beats": False,
        "lambda_acc": 0.0,
        "lambda_beat": 0.0,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 2000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "feature_cache_mode": "memmap",
        "feature_cache_dtype": "float32",
        "skip_preprocess": True,
    },
    "g1_finedance_librosa35_lbeat_robotloss": {
        "motion_format": "g1",
        "data_path": "data/finedance_g1_fkbeats",
        "processed_data_dir": "data/finedance_g1_librosa35_lbeat_robotloss_dataset_backups",
        "feature_type": "baseline",
        "use_beats": True,
        "beat_rep": "distance",
        "lambda_acc": 0.0,
        "lambda_beat": 0.05,
        "lambda_g1_fk": 0.1,
        "lambda_g1_fk_vel": 0.5,
        "lambda_g1_fk_acc": 0.05,
        "lambda_g1_foot": 1.0,
        "lambda_g1_kin": 0.1,
        "g1_kin_loss_warmup_epochs": 50,
        "g1_kin_loss_max_fraction": 1.0,
        "batch_size": 512,
        "gradient_accumulation_steps": 1,
        "epochs": 1000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "beat_loss_start_epoch": 50,
        "beat_loss_warmup_epochs": 300,
        "beat_loss_max_fraction": 1.0,
        "beat_loss_cap_mode": "soft",
        "g1_target_transform": "relative_distance",
        "finetune_from_checkpoint": True,
        "feature_cache_mode": "memmap",
        "feature_cache_dtype": "float16",
        "skip_preprocess": True,
        "enable_g1_fk_metrics": True,
    },
    "aist_finedance_beatdistance": {
        "motion_format": "smpl",
        "data_path": "data/aist_finedance",
        "processed_data_dir": "data/aist_finedance_dataset_backups",
        "feature_type": "jukebox",
        "use_beats": True,
        "beat_rep": "distance",
        "lambda_acc": 0.0,
        "lambda_beat": 0.0,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "epochs": 2000,
        "save_interval": 100,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "skip_preprocess": True,
    },
}

PRESET_TRACKED_ARGS = (
    "motion_format",
    "data_path",
    "processed_data_dir",
    "feature_type",
    "use_beats",
    "beat_rep",
    "lambda_acc",
    "lambda_beat",
    "lambda_g1_fk",
    "lambda_g1_fk_vel",
    "lambda_g1_fk_acc",
    "lambda_g1_foot",
    "lambda_g1_kin",
    "g1_kin_loss_warmup_epochs",
    "g1_kin_loss_max_fraction",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "save_interval",
    "learning_rate",
    "beat_loss_start_epoch",
    "beat_loss_warmup_epochs",
    "beat_loss_max_fraction",
    "beat_loss_cap_mode",
    "beat_target_transform",
    "g1_target_transform",
    "checkpoint",
    "finetune_from_checkpoint",
    "g1_fk_model_path",
    "g1_root_quat_order",
    "feature_cache_mode",
    "feature_cache_dtype",
    "root_height_min",
    "root_height_max",
    "skip_preprocess",
    "skip_eval",
)


def _flag_was_explicit(argv, flag):
    return argv is not None and flag in argv


def _annotate_explicit_args(args, argv):
    argv = list(argv or [])
    for arg_name in PRESET_TRACKED_ARGS:
        flag = f"--{arg_name}"
        setattr(args, f"{arg_name}_was_explicit", _flag_was_explicit(argv, flag))
    return args


def apply_preset_defaults(args):
    preset = PRESET_DEFAULTS.get(args.preset)
    if preset is None:
        return args
    for key, value in preset.items():
        if not getattr(args, f"{key}_was_explicit", False):
            setattr(args, key, value)
    return args


def parse_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Submit the full EDGE training pipeline as chained Slurm jobs."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_DEFAULTS),
        default="",
        help="Apply one of the canonical experiment presets.",
    )
    parser.add_argument("--train_name", required=True, help="Logical name for the training run.")
    parser.add_argument(
        "--feature_type", choices=("baseline", "jukebox"), default="baseline"
    )
    parser.add_argument("--motion_format", choices=("smpl", "g1"), default="smpl")
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument(
        "--beat_rep", choices=("distance", "pulse"), default="distance"
    )
    parser.add_argument("--lambda_beat", type=float, default=0.5)
    parser.add_argument("--lambda_acc", type=float, default=None)
    parser.add_argument("--lambda_g1_fk", type=float, default=0.0)
    parser.add_argument("--lambda_g1_fk_vel", type=float, default=0.0)
    parser.add_argument("--lambda_g1_fk_acc", type=float, default=0.0)
    parser.add_argument("--lambda_g1_foot", type=float, default=0.0)
    parser.add_argument("--lambda_g1_kin", type=float, default=1.0)
    parser.add_argument("--g1_kin_loss_warmup_epochs", type=int, default=0)
    parser.add_argument("--g1_kin_loss_max_fraction", type=float, default=0.0)
    parser.add_argument("--beat_estimator_ckpt", default="", help="Reuse an existing estimator checkpoint.")
    parser.add_argument("--beat_estimator_max_val_loss", type=float, default=8.0)
    parser.add_argument("--beat_loss_start_epoch", type=int, default=0)
    parser.add_argument("--beat_loss_warmup_epochs", type=int, default=0)
    parser.add_argument("--beat_loss_max_fraction", type=float, default=0.0)
    parser.add_argument(
        "--beat_loss_cap_mode",
        choices=("hard", "soft"),
        default="hard",
    )
    parser.add_argument(
        "--g1_target_transform",
        choices=("raw_distance", "relative_distance"),
        default="raw_distance",
    )
    parser.add_argument(
        "--beat_target_transform",
        choices=("raw_distance", "relative_distance"),
        default="raw_distance",
    )
    parser.add_argument("--checkpoint", default="", help="Training checkpoint to load before this run.")
    parser.add_argument(
        "--finetune_from_checkpoint",
        action="store_true",
        help="Load checkpoint weights with a fresh optimizer state.",
    )
    parser.add_argument(
        "--allow_lbeat_from_scratch",
        action="store_true",
        help="Allow lbeat training without a base beat-distance checkpoint.",
    )
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--processed_data_dir", default="data/dataset_backups")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--render_dir", default="renders")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--epoch_offset", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--ema_interval", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--estimator_epochs", type=int, default=50)
    parser.add_argument("--estimator_batch_size", type=int, default=None)
    parser.add_argument("--estimator_learning_rate", type=float, default=1e-4)
    parser.add_argument("--estimator_weight_decay", type=float, default=0.0)
    parser.add_argument("--estimator_val_split", type=float, default=0.1)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--test_num_workers", type=int, default=0)
    parser.add_argument("--estimator_num_workers", type=int, default=0)
    parser.add_argument(
        "--train_mixed_precision",
        choices=("no", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--feature_cache_mode",
        choices=("off", "memmap"),
        default="off",
    )
    parser.add_argument(
        "--feature_cache_dtype",
        choices=("float32", "float16"),
        default="float32",
    )
    parser.add_argument(
        "--estimator_mixed_precision",
        choices=("no", "bf16"),
        default="bf16",
    )
    parser.add_argument("--partition", default="workq")
    parser.add_argument("--preprocess_time", default="04:00:00")
    parser.add_argument("--validate_time", default="01:00:00")
    parser.add_argument("--estimator_time", default="08:00:00")
    parser.add_argument("--train_time", default="24:00:00")
    parser.add_argument("--eval_time", default="04:00:00")
    parser.add_argument("--preprocess_cpus", type=int, default=8)
    parser.add_argument("--validate_cpus", type=int, default=4)
    parser.add_argument("--estimator_cpus", type=int, default=8)
    parser.add_argument("--train_cpus", type=int, default=8)
    parser.add_argument("--eval_cpus", type=int, default=8)
    parser.add_argument("--preprocess_mem", default="32G")
    parser.add_argument("--validate_mem", default="16G")
    parser.add_argument("--estimator_mem", default="32G")
    parser.add_argument("--train_mem", default="64G")
    parser.add_argument("--eval_mem", default="32G")
    parser.add_argument("--preprocess_gpus", type=int, default=None)
    parser.add_argument("--validate_gpus", type=int, default=0)
    parser.add_argument("--estimator_gpus", type=int, default=1)
    parser.add_argument("--train_gpus", type=int, default=1)
    parser.add_argument("--eval_gpus", type=int, default=1)
    parser.add_argument("--validate_sample_count", type=int, default=64)
    parser.add_argument("--root_height_min", type=float, default=None)
    parser.add_argument("--root_height_max", type=float, default=None)
    parser.add_argument("--eval_music_dir", default="data/test/wavs")
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--enable_g1_fk_metrics", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument(
        "--g1_root_quat_order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
    )
    parser.add_argument(
        "--eval_mode",
        choices=("dataset", "full_song"),
        default="dataset",
        help="Evaluate sliced test clips or one stitched motion per full test song.",
    )
    parser.add_argument("--checkpoint_screen_epochs", default="50,100,200,300,400,500")
    parser.add_argument("--screen_eval_clips", type=int, default=16)
    parser.add_argument("--quality_reference_eval_dir", default=DEFAULT_LBEAT_REFERENCE_EVAL_DIR)
    parser.add_argument("--run_id", default="", help="Fixed Slurm pipeline run id.")
    parser.add_argument(
        "--initial_dependency",
        default="",
        help="Existing Slurm job id that the first submitted stage should wait for.",
    )
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_beat_estimator", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args(raw_argv)
    args = _annotate_explicit_args(args, raw_argv)
    args.train_num_workers_was_explicit = _flag_was_explicit(raw_argv, "--train_num_workers")
    args.test_num_workers_was_explicit = _flag_was_explicit(raw_argv, "--test_num_workers")
    args.estimator_num_workers_was_explicit = _flag_was_explicit(
        raw_argv, "--estimator_num_workers"
    )
    return args


def resolve_stage_mixed_precision(requested_mode, gpus):
    return requested_mode if gpus > 0 else "no"


def apply_dynamic_defaults(args):
    apply_preset_defaults(args)
    if args.preprocess_gpus is None:
        args.preprocess_gpus = 1 if args.feature_type == "jukebox" else 0
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATE
    if args.lambda_acc is None:
        args.lambda_acc = default_lambda_acc(args.use_beats)
    if args.estimator_batch_size is None:
        args.estimator_batch_size = min(args.batch_size, 16)
    args.train_num_workers, args.test_num_workers = resolve_train_test_workers(
        args.train_num_workers if getattr(args, "train_num_workers_was_explicit", False) else None,
        args.test_num_workers if getattr(args, "test_num_workers_was_explicit", False) else None,
        cpu_budget=args.train_cpus,
    )
    if not getattr(args, "estimator_num_workers_was_explicit", False):
        args.estimator_num_workers = resolve_worker_count(cpu_budget=args.estimator_cpus)
    args.train_mixed_precision = resolve_stage_mixed_precision(
        args.train_mixed_precision, args.train_gpus
    )
    args.estimator_mixed_precision = resolve_stage_mixed_precision(
        args.estimator_mixed_precision, args.estimator_gpus
    )
    return args


def is_lbeat_run(args):
    return args.preset == "edge_beatdistance_lbeat"


def validate_pipeline_config(args):
    if args.allow_lbeat_from_scratch and not args.checkpoint:
        args.finetune_from_checkpoint = False
    safe_lbeat_presets = (
        "edge_beatdistance_lbeat",
        "g1_beatdistance_fkbeats_lbeat",
        "g1_finedance_librosa35_lbeat_robotloss",
    )
    if (
        args.preset in safe_lbeat_presets
        and not args.checkpoint
        and not args.allow_lbeat_from_scratch
    ):
        raise ValueError(
            "Safe lbeat preset requires --checkpoint unless --allow_lbeat_from_scratch is set."
        )
    if args.finetune_from_checkpoint and not args.checkpoint:
        raise ValueError("--finetune_from_checkpoint requires --checkpoint.")
    if args.epoch_offset < 0:
        raise ValueError("--epoch_offset must be non-negative.")
    if args.beat_loss_max_fraction < 0:
        raise ValueError("--beat_loss_max_fraction must be non-negative.")
    g1_loss_weights = (
        args.lambda_g1_fk,
        args.lambda_g1_fk_vel,
        args.lambda_g1_fk_acc,
        args.lambda_g1_foot,
    )
    if any(weight < 0 for weight in g1_loss_weights):
        raise ValueError("G1 robot loss weights must be non-negative.")
    if args.lambda_g1_kin < 0:
        raise ValueError("--lambda_g1_kin must be non-negative.")
    if args.g1_kin_loss_warmup_epochs < 0:
        raise ValueError("--g1_kin_loss_warmup_epochs must be non-negative.")
    if args.g1_kin_loss_max_fraction < 0:
        raise ValueError("--g1_kin_loss_max_fraction must be non-negative.")
    if any(weight > 0 for weight in g1_loss_weights) and args.motion_format != "g1":
        raise ValueError("G1 robot losses require --motion_format g1.")
    if args.beat_estimator_max_val_loss <= 0:
        raise ValueError("--beat_estimator_max_val_loss must be positive.")
    if args.motion_format == "g1" and args.eval_mode != "dataset":
        raise ValueError("G1 evaluation currently supports --eval_mode dataset only.")
    return args


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_preprocess_command(args):
    command = ["python", "data/create_dataset.py"]
    if args.feature_type == "baseline":
        command.append("--extract-baseline")
    if args.feature_type == "jukebox":
        command.append("--extract-jukebox")
    if args.use_beats:
        command.append("--extract-beats")
    return shell_join(command)


def build_estimator_command(args, output_path):
    return shell_join(
        [
            "python",
            "train_beat_estimator.py",
            "--motion_dir",
            Path(args.data_path) / "train" / "motions_sliced",
            "--beat_dir",
            Path(args.data_path) / "train" / "beat_feats",
            "--output_path",
            output_path,
            "--epochs",
            args.estimator_epochs,
            "--batch_size",
            args.estimator_batch_size,
            "--learning_rate",
            args.estimator_learning_rate,
            "--weight_decay",
            args.estimator_weight_decay,
            "--val_split",
            args.estimator_val_split,
            "--num_workers",
            args.estimator_num_workers,
            "--mixed_precision",
            args.estimator_mixed_precision,
            "--device",
            "cuda" if args.estimator_gpus > 0 else "cpu",
            "--motion_format",
            args.motion_format,
            "--data_path",
            args.data_path,
            "--processed_data_dir",
            args.processed_data_dir,
            "--feature_type",
            args.feature_type,
            "--beat_target_transform",
            args.beat_target_transform,
            "--g1_target_transform",
            args.g1_target_transform,
        ]
    )


def build_validation_command(args):
    command = [
        "python",
        "data/validate_preprocessed_data.py",
        "--data_path",
        args.data_path,
        "--processed_data_dir",
        args.processed_data_dir,
        "--feature_type",
        args.feature_type,
        "--motion_format",
        args.motion_format,
        "--feature_cache_mode",
        args.feature_cache_mode,
        "--feature_cache_dtype",
        args.feature_cache_dtype,
        "--sample_count",
        args.validate_sample_count,
    ]
    if args.use_beats:
        command.extend(
            [
                "--use_beats",
                "--beat_rep",
                args.beat_rep,
            ]
        )
    if args.root_height_min is not None:
        command.extend(["--root_height_min", args.root_height_min])
    if args.root_height_max is not None:
        command.extend(["--root_height_max", args.root_height_max])
    return shell_join(command)


def build_train_command(args, beat_estimator_ckpt=None):
    command = [
        "python",
        "train.py",
        "--feature_type",
        args.feature_type,
        "--motion_format",
        args.motion_format,
    ]
    if args.use_beats:
        command.extend(
            [
                "--use_beats",
                "--beat_rep",
                args.beat_rep,
                "--lambda_beat",
                args.lambda_beat,
                "--lambda_acc",
                args.lambda_acc,
            ]
        )
        command.extend(
            [
                "--beat_loss_start_epoch",
                args.beat_loss_start_epoch,
                "--beat_loss_warmup_epochs",
                args.beat_loss_warmup_epochs,
                "--beat_loss_max_fraction",
                args.beat_loss_max_fraction,
                "--beat_loss_cap_mode",
                args.beat_loss_cap_mode,
                "--beat_estimator_max_val_loss",
                args.beat_estimator_max_val_loss,
            ]
        )
        if beat_estimator_ckpt:
            command.extend(["--beat_estimator_ckpt", beat_estimator_ckpt])
    elif args.motion_format == "g1":
        command.extend(["--lambda_beat", args.lambda_beat])
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.finetune_from_checkpoint:
        command.append("--finetune_from_checkpoint")
    command.extend(
        [
            "--data_path",
            args.data_path,
            "--processed_data_dir",
            args.processed_data_dir,
            "--project",
            args.project,
            "--exp_name",
            args.train_name,
            "--render_dir",
            args.render_dir,
            "--batch_size",
            args.batch_size,
            "--gradient_accumulation_steps",
            args.gradient_accumulation_steps,
            "--epochs",
            args.epochs,
            "--epoch_offset",
            args.epoch_offset,
            "--save_interval",
            args.save_interval,
            "--ema_interval",
            args.ema_interval,
            "--learning_rate",
            args.learning_rate,
            "--weight_decay",
            args.weight_decay,
            "--train_num_workers",
            args.train_num_workers,
            "--test_num_workers",
            args.test_num_workers,
            "--mixed_precision",
            args.train_mixed_precision,
            "--feature_cache_mode",
            args.feature_cache_mode,
            "--feature_cache_dtype",
            args.feature_cache_dtype,
            "--lambda_g1_fk",
            args.lambda_g1_fk,
            "--lambda_g1_fk_vel",
            args.lambda_g1_fk_vel,
            "--lambda_g1_fk_acc",
            args.lambda_g1_fk_acc,
            "--lambda_g1_foot",
            args.lambda_g1_foot,
            "--lambda_g1_kin",
            args.lambda_g1_kin,
            "--g1_kin_loss_warmup_epochs",
            args.g1_kin_loss_warmup_epochs,
            "--g1_kin_loss_max_fraction",
            args.g1_kin_loss_max_fraction,
            "--g1_fk_model_path",
            args.g1_fk_model_path,
            "--g1_root_quat_order",
            args.g1_root_quat_order,
        ]
    )
    return shell_join(command)


def final_checkpoint_path(args):
    return (
        Path(args.project)
        / args.train_name
        / "weights"
        / f"train-{args.epoch_offset + args.epochs}.pt"
    )


def build_eval_command(
    args,
    checkpoint_path,
    motion_dir,
    metrics_path,
    render_dir,
    edge_table_path=None,
    beatit_table_path=None,
    paper_report_path=None,
    pfc_audit_path=None,
):
    eval_script = (
        "eval/run_full_song_eval.py"
        if args.eval_mode == "full_song"
        else "eval/run_dataset_eval.py"
    )
    command = [
        "python",
        eval_script,
        "--checkpoint",
        checkpoint_path,
        "--feature_type",
        args.feature_type,
        "--data_path",
        args.data_path,
        "--render_dir",
        render_dir,
        "--motion_save_dir",
        motion_dir,
        "--metrics_path",
        metrics_path,
        "--edge_table_path",
        edge_table_path or (Path(metrics_path).parent / "edge_table.json"),
        "--beatit_table_path",
        beatit_table_path or (Path(metrics_path).parent / "beatit_table.json"),
        "--paper_report_path",
        paper_report_path or (Path(metrics_path).parent / "paper_report.md"),
        "--pfc_audit_path",
        pfc_audit_path or (Path(metrics_path).parent / "pfc_audit.json"),
        "--seed",
        args.eval_seed,
    ]
    if args.eval_mode == "dataset":
        command.extend(
            [
                "--processed_data_dir",
                args.processed_data_dir,
            ]
        )
    if args.use_beats:
        command.extend(
            [
                "--use_beats",
                "--beat_rep",
                args.beat_rep,
            ]
        )
        if args.eval_mode == "full_song":
            command.extend(["--beat_source", "audio"])
    return shell_join(command)


def build_g1_eval_command(
    args,
    checkpoint_path,
    motion_dir,
    metrics_path,
    render_dir,
    g1_table_path=None,
    paper_report_path=None,
    motion_audit_path=None,
):
    command = [
        "python",
        "eval/run_g1_dataset_eval.py",
        "--checkpoint",
        checkpoint_path,
        "--feature_type",
        args.feature_type,
        "--data_path",
        args.data_path,
        "--processed_data_dir",
        args.processed_data_dir,
        "--render_dir",
        render_dir,
        "--motion_save_dir",
        motion_dir,
        "--metrics_path",
        metrics_path,
        "--g1_table_path",
        g1_table_path or (Path(metrics_path).parent / "g1_table.json"),
        "--motion_audit_path",
        motion_audit_path or (Path(metrics_path).parent / "motion_audit.json"),
        "--paper_report_path",
        paper_report_path or (Path(metrics_path).parent / "paper_report.md"),
        "--seed",
        args.eval_seed,
    ]
    if args.use_beats:
        command.extend(
            [
                "--use_beats",
                "--beat_rep",
                args.beat_rep,
            ]
        )
    if args.enable_g1_fk_metrics:
        command.extend(
            [
                "--enable_fk_metrics",
                "--g1_fk_model_path",
                args.g1_fk_model_path,
                "--g1_root_quat_order",
                args.g1_root_quat_order,
            ]
        )
    return shell_join(command)


def build_lbeat_screen_command(args, eval_dir):
    command = [
        "python",
        "eval/screen_lbeat_checkpoints.py",
        "--project",
        args.project,
        "--train_name",
        args.train_name,
        "--checkpoint_epochs",
        args.checkpoint_screen_epochs,
        "--screen_eval_clips",
        args.screen_eval_clips,
        "--output_dir",
        eval_dir,
        "--reference_eval_dir",
        args.quality_reference_eval_dir,
        "--feature_type",
        args.feature_type,
        "--data_path",
        args.data_path,
        "--processed_data_dir",
        args.processed_data_dir,
        "--render_dir",
        eval_dir / "renders",
        "--seed",
        args.eval_seed,
        "--beat_rep",
        args.beat_rep,
    ]
    return shell_join(command)


def build_header(stage_name, log_path, partition, time_limit, cpus, mem, gpus):
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=edge_{stage_name}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    return lines


def build_stage_script(repo_root, stage_name, command, log_path, partition, time_limit, cpus, mem, gpus):
    venv_activate = resolve_venv_activate(repo_root)
    lines = build_header(stage_name, log_path, partition, time_limit, cpus, mem, gpus)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            f"source {shlex.quote(str(venv_activate))}",
            "export PYTHONUNBUFFERED=1",
            "export TERM=xterm-256color",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export MPLCONFIGDIR=/tmp/matplotlib",
            "export MUJOCO_GL=${MUJOCO_GL:-egl}",
            "export WANDB_DISABLED=true",
            'echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting stage\"',
            f"stdbuf -oL -eL {command}",
            'echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete\"',
            "",
        ]
    )
    return "\n".join(lines)


def parse_job_id(stdout):
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        raise ValueError(f"Could not parse sbatch job id from: {stdout!r}")
    return match.group(1)


def submit_sbatch(cmd):
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(
            part.strip()
            for part in (exc.stdout or "", exc.stderr or "")
            if part and part.strip()
        )
        raise RuntimeError(
            f"sbatch failed for {shell_join(cmd)}"
            + (f":\n{details}" if details else "")
        ) from exc
    return parse_job_id(completed.stdout)


def default_run_id(train_name):
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{train_name}"


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_submit_log(run_dir, job_ids):
    job_list = ",".join(job_ids.values())
    lines = [
        f"Pipeline directory: {run_dir}",
        "Monitor with:",
        f"squeue --jobs {job_list}",
        f"sacct -j {job_list} --format=JobID,JobName,State,Elapsed",
    ]
    for stage in ("preprocess", "validate_preprocessed_data", "beat_estimator", "train", "evaluate"):
        if stage in job_ids:
            lines.append(f"tail -f {run_dir / (stage + '.out')}")
    return "\n".join(lines) + "\n"


def submit_pipeline(args, repo_root=None, sbatch_submitter=submit_sbatch):
    args = apply_dynamic_defaults(args)
    validate_pipeline_config(args)
    repo_root = Path(repo_root or SCRIPT_ROOT).resolve()
    run_id = args.run_id or default_run_id(args.train_name)
    run_dir = repo_root / "slurm" / "pipelines" / run_id
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    job_ids = {}
    previous_job_id = args.initial_dependency or None

    def submit_stage(stage_name, command, time_limit, cpus, mem, gpus):
        nonlocal previous_job_id
        log_path = run_dir / f"{stage_name}.out"
        script_path = run_dir / f"{stage_name}.sbatch"
        script = build_stage_script(
            repo_root=repo_root,
            stage_name=stage_name,
            command=command,
            log_path=log_path,
            partition=args.partition,
            time_limit=time_limit,
            cpus=cpus,
            mem=mem,
            gpus=gpus,
        )
        write_text(script_path, script)
        cmd = ["sbatch"]
        if previous_job_id is not None:
            cmd.append(f"--dependency=afterok:{previous_job_id}")
        cmd.append(str(script_path))
        job_id = sbatch_submitter(cmd)
        job_ids[stage_name] = job_id
        previous_job_id = job_id

    if not args.skip_preprocess:
        submit_stage(
            "preprocess",
            build_preprocess_command(args),
            args.preprocess_time,
            args.preprocess_cpus,
            args.preprocess_mem,
            args.preprocess_gpus,
        )

    submit_stage(
        "validate_preprocessed_data",
        build_validation_command(args),
        args.validate_time,
        args.validate_cpus,
        args.validate_mem,
        args.validate_gpus,
    )

    estimator_ckpt = args.beat_estimator_ckpt
    needs_estimator = args.use_beats and args.lambda_beat > 0 and not estimator_ckpt
    if needs_estimator and not args.skip_beat_estimator:
        estimator_ckpt = str(weights_dir / "beat_estimator.pt")
        submit_stage(
            "beat_estimator",
            build_estimator_command(args, estimator_ckpt),
            args.estimator_time,
            args.estimator_cpus,
            args.estimator_mem,
            args.estimator_gpus,
        )

    submit_stage(
        "train",
        build_train_command(args, beat_estimator_ckpt=estimator_ckpt),
        args.train_time,
        args.train_cpus,
        args.train_mem,
        args.train_gpus,
    )

    if not args.skip_eval:
        eval_dir = run_dir / "eval"
        if args.motion_format == "g1":
            eval_command = build_g1_eval_command(
                args,
                checkpoint_path=final_checkpoint_path(args),
                motion_dir=eval_dir / "motions",
                metrics_path=eval_dir / "metrics.json",
                render_dir=eval_dir / "renders",
                g1_table_path=eval_dir / "g1_table.json",
                paper_report_path=eval_dir / "paper_report.md",
                motion_audit_path=eval_dir / "motion_audit.json",
            )
        else:
            eval_command = (
                build_lbeat_screen_command(args, eval_dir)
                if is_lbeat_run(args)
                else build_eval_command(
                    args,
                    checkpoint_path=final_checkpoint_path(args),
                    motion_dir=eval_dir / "motions",
                    metrics_path=eval_dir / "metrics.json",
                    render_dir=eval_dir / "renders",
                    edge_table_path=eval_dir / "edge_table.json",
                    beatit_table_path=eval_dir / "beatit_table.json",
                    paper_report_path=eval_dir / "paper_report.md",
                    pfc_audit_path=eval_dir / "pfc_audit.json",
                )
            )
        submit_stage(
            "evaluate",
            eval_command,
            args.eval_time,
            args.eval_cpus,
            args.eval_mem,
            args.eval_gpus,
        )

    write_text(run_dir / "submit.log", build_submit_log(run_dir, job_ids))
    return SubmissionResult(run_dir=run_dir, job_ids=job_ids)


def main(argv=None):
    args = apply_dynamic_defaults(parse_args(argv))
    result = submit_pipeline(args)
    print(f"Submitted pipeline in {result.run_dir}")
    for stage, job_id in result.job_ids.items():
        print(f"{stage}: {job_id}")
    print("Monitor with:")
    print((result.run_dir / "submit.log").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
