import argparse
import os


DEFAULT_DATALOADER_WORKER_CAP = 6
DEFAULT_TRAIN_DATALOADER_WORKER_CAP = 2
DEFAULT_TEST_WORKER_CAP = 2
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 2e-4


def default_lambda_acc(use_beats):
    return 0.0


def _flag_was_explicit(argv, flag):
    return flag in set(argv or [])


def resolve_cpu_budget(cpu_budget=None):
    if cpu_budget is not None:
        return max(int(cpu_budget), 1)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus.isdigit():
        return max(int(slurm_cpus), 1)
    detected = os.cpu_count() or 1
    return max(int(detected), 1)


def resolve_worker_count(cpu_budget=None, reserve=2, cap=DEFAULT_DATALOADER_WORKER_CAP):
    budget = resolve_cpu_budget(cpu_budget)
    return min(cap, max(1, budget - reserve))


def resolve_train_test_workers(train_num_workers, test_num_workers, cpu_budget=None):
    resolved_train = (
        train_num_workers
        if train_num_workers is not None
        else resolve_worker_count(
            cpu_budget=cpu_budget,
            cap=DEFAULT_TRAIN_DATALOADER_WORKER_CAP,
        )
    )
    resolved_test = (
        test_num_workers
        if test_num_workers is not None
        else min(DEFAULT_TEST_WORKER_CAP, resolved_train)
    )
    return resolved_train, resolved_test


def parse_train_opt(argv=None):
    raw_argv = list(argv or os.sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--motion_format", type=str, choices=("smpl", "g1"), default="smpl"
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="EDGE", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=0,
        help="DataLoader workers for the training split",
    )
    parser.add_argument(
        "--test_num_workers",
        type=int,
        default=0,
        help="DataLoader workers for the test split",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument(
        "--beat_rep", type=str, choices=("distance", "pulse"), default="distance"
    )
    parser.add_argument("--lambda_acc", type=float, default=None)
    parser.add_argument("--lambda_beat", type=float, default=0.5)
    parser.add_argument("--beat_a", type=float, default=10.0)
    parser.add_argument("--beat_c", type=float, default=0.1)
    parser.add_argument("--beat_estimator_ckpt", type=str, default="")
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
        "--finetune_from_checkpoint",
        action="store_true",
        help="Load checkpoint weights with a fresh optimizer state.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of microbatches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=("no", "bf16"),
        default="bf16",
        help="Autocast/accelerator mixed precision mode for training.",
    )
    parser.add_argument(
        "--feature_cache_mode",
        type=str,
        choices=("off", "memmap"),
        default="off",
        help="Disk-backed training feature cache mode.",
    )
    parser.add_argument(
        "--feature_cache_dtype",
        type=str,
        choices=("float32", "float16"),
        default="float32",
        help="Storage dtype for the disk-backed feature cache.",
    )
    opt = parser.parse_args(raw_argv)
    opt.learning_rate_was_explicit = opt.learning_rate is not None
    opt.lambda_acc_was_explicit = opt.lambda_acc is not None
    train_workers_explicit = _flag_was_explicit(raw_argv, "--train_num_workers")
    test_workers_explicit = _flag_was_explicit(raw_argv, "--test_num_workers")
    if opt.learning_rate is None:
        opt.learning_rate = DEFAULT_LEARNING_RATE
    if opt.lambda_acc is None:
        opt.lambda_acc = default_lambda_acc(opt.use_beats)
    opt.train_num_workers, opt.test_num_workers = resolve_train_test_workers(
        opt.train_num_workers if train_workers_explicit else None,
        opt.test_num_workers if test_workers_explicit else None,
    )
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--motion_format", type=str, choices=("smpl", "g1"), default="smpl"
    )
    parser.add_argument(
        "--g1_fk_model_path",
        type=str,
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument(
        "--g1_root_quat_order",
        type=str,
        choices=("wxyz", "xyzw"),
        default="xyzw",
    )
    parser.add_argument(
        "--g1_render_backend",
        type=str,
        choices=("mujoco", "stick"),
        default="mujoco",
    )
    parser.add_argument("--g1_render_width", type=int, default=960)
    parser.add_argument("--g1_render_height", type=int, default=720)
    parser.add_argument("--g1_mujoco_gl", type=str, default="egl")
    parser.add_argument(
        "--out_length",
        type=float,
        default=30,
        help="max. length of output, in seconds",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for deterministic slice selection (-1 disables seeding)",
    )
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument(
        "--beat_rep", type=str, choices=("distance", "pulse"), default="distance"
    )
    parser.add_argument(
        "--beat_source", type=str, choices=("none", "audio", "user"), default="none"
    )
    parser.add_argument("--beat_file", type=str, default="")
    opt = parser.parse_args()
    return opt
