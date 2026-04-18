import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def resolve_shared_root(path):
    path = Path(path).resolve()
    for parent in [path, *path.parents]:
        if parent.name == ".worktrees":
            return parent.parent
    return path


SCRIPT_ROOT = Path(__file__).resolve().parent
SHARED_ROOT = resolve_shared_root(SCRIPT_ROOT)


def resolve_venv_activate(repo_root):
    return resolve_shared_root(repo_root) / ".venv311" / "bin" / "activate"


@dataclass
class SubmissionResult:
    run_dir: Path
    job_ids: dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Submit the full EDGE training pipeline as chained Slurm jobs."
    )
    parser.add_argument("--train_name", required=True, help="Logical name for the training run.")
    parser.add_argument(
        "--feature_type", choices=("baseline", "jukebox"), default="baseline"
    )
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument(
        "--beat_rep", choices=("distance", "pulse"), default="distance"
    )
    parser.add_argument("--lambda_beat", type=float, default=0.5)
    parser.add_argument("--lambda_acc", type=float, default=0.1)
    parser.add_argument("--beat_estimator_ckpt", default="", help="Reuse an existing estimator checkpoint.")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--processed_data_dir", default="data/dataset_backups")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--render_dir", default="renders")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--ema_interval", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--test_num_workers", type=int, default=0)
    parser.add_argument("--partition", default="workq")
    parser.add_argument("--preprocess_time", default="04:00:00")
    parser.add_argument("--estimator_time", default="08:00:00")
    parser.add_argument("--train_time", default="24:00:00")
    parser.add_argument("--preprocess_cpus", type=int, default=8)
    parser.add_argument("--estimator_cpus", type=int, default=8)
    parser.add_argument("--train_cpus", type=int, default=8)
    parser.add_argument("--preprocess_mem", default="32G")
    parser.add_argument("--estimator_mem", default="32G")
    parser.add_argument("--train_mem", default="64G")
    parser.add_argument("--preprocess_gpus", type=int, default=None)
    parser.add_argument("--estimator_gpus", type=int, default=1)
    parser.add_argument("--train_gpus", type=int, default=1)
    parser.add_argument("--run_id", default="", help="Fixed Slurm pipeline run id.")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_beat_estimator", action="store_true")
    return parser.parse_args(argv)


def apply_dynamic_defaults(args):
    if args.preprocess_gpus is None:
        args.preprocess_gpus = 1 if args.feature_type == "jukebox" else 0
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
            "--batch_size",
            min(args.batch_size, 16),
            "--device",
            "cuda" if args.estimator_gpus > 0 else "cpu",
        ]
    )


def build_train_command(args, beat_estimator_ckpt=None):
    command = [
        "python",
        "train.py",
        "--feature_type",
        args.feature_type,
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
        if beat_estimator_ckpt:
            command.extend(["--beat_estimator_ckpt", beat_estimator_ckpt])
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
            "--epochs",
            args.epochs,
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
        ]
    )
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
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
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
    for stage in ("preprocess", "beat_estimator", "train"):
        if stage in job_ids:
            lines.append(f"tail -f {run_dir / (stage + '.out')}")
    return "\n".join(lines) + "\n"


def submit_pipeline(args, repo_root=None, sbatch_submitter=submit_sbatch):
    args = apply_dynamic_defaults(args)
    repo_root = Path(repo_root or SCRIPT_ROOT).resolve()
    run_id = args.run_id or default_run_id(args.train_name)
    run_dir = repo_root / "slurm" / "pipelines" / run_id
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    job_ids = {}
    previous_job_id = None

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
