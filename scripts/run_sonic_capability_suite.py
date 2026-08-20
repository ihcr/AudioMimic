"""Run repeatable headless SONIC capability trials end to end.

Each trial owns fresh MuJoCo and SONIC processes.  It enters CONTROL on the
bundled reference, releases the elastic band, verifies unassisted standing,
then enables ZMQ and invokes the fixed AudioMimic reference player.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pexpect
except ModuleNotFoundError:
    pexpect = None


NATIVE_MOTIONS = {
    "low": "walking_quip_360_R_002__A428_M",
    "medium": "macarena_001__A545_M",
    "high": "dance_in_da_party_001__A464_M",
}
RETARGETED_GT_MOTIONS = {
    "low": Path(
        "/home/tianhup/Downloads/edge_smpl_dataset_retargeted/unitree_g1/"
        "gLH_sBM_cAll_d16_mLH0_ch09.pkl"
    ),
    "medium": Path(
        "/home/tianhup/Downloads/edge_smpl_dataset_retargeted/unitree_g1/"
        "gKR_sBM_cAll_d28_mKR1_ch05.pkl"
    ),
    "high": Path(
        "/home/tianhup/Downloads/edge_smpl_dataset_retargeted/unitree_g1/"
        "gMH_sBM_cAll_d22_mMH0_ch01.pkl"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("sonic_native", "retargeted_gt"), default="sonic_native")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=tuple(NATIVE_MOTIONS),
        default=list(NATIVE_MOTIONS),
    )
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--groot_root", type=Path, default=Path("~/GR00T-WholeBodyControl"))
    parser.add_argument(
        "--audiomimic_python",
        type=Path,
        default=Path("~/anaconda3/envs/audiomimic/bin/python"),
    )
    parser.add_argument("--control_settle_seconds", type=float, default=3.0)
    parser.add_argument("--unassisted_settle_seconds", type=float, default=3.0)
    parser.add_argument("--minimum_standing_height", type=float, default=0.65)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _expanded(path: Path) -> Path:
    return path.expanduser().resolve()


def _expect_status(
    simulator: pexpect.spawn, minimum_height: float, *, enforce: bool = True
) -> float:
    simulator.sendline("status")
    simulator.expect(r"HEADLESS_SIM_STATUS height=([0-9.]+) elastic_band=(True|False)")
    height = float(simulator.match.group(1))
    if enforce and height < minimum_height:
        raise RuntimeError(
            f"unassisted standing check failed: {height:.4f} < {minimum_height:.4f} m"
        )
    return height


def _motion_spec(args: argparse.Namespace, repo_root: Path, level: str) -> tuple[str, Path, str, Path, Path]:
    if args.source == "sonic_native":
        motion_name = NATIVE_MOTIONS[level]
        motion_path = (
            repo_root
            / "eval/gt_sonic_capability/sonic_known_trackable_20260820"
            / f"{motion_name}.pkl"
        )
        run_id = f"sonic_native_{level}_{motion_name}"
        runs_root = repo_root / "eval/gt_sonic_capability/known_trackable_runs"
        logs_root = repo_root / "eval/gt_sonic_capability/suite_logs/sonic_native"
    else:
        motion_path = RETARGETED_GT_MOTIONS[level]
        motion_name = motion_path.stem
        run_id = f"gt_cap_{level}_{motion_name}"
        runs_root = repo_root / "eval/gt_sonic_capability/retargeted_gt_runs_v2"
        logs_root = repo_root / "eval/gt_sonic_capability/suite_logs/retargeted_gt_v2"
    return motion_name, motion_path, run_id, runs_root, logs_root


def _record_execution_outcome(run_dir: Path, result: dict, fall_height: float = 0.45) -> None:
    sim_path = run_dir / "sim_state.json"
    if not sim_path.is_file():
        return
    payload = json.loads(sim_path.read_text())
    records = payload.get("records", payload)
    heights = np.asarray(
        [record["base_position"][2] for record in records if "base_position" in record],
        dtype=np.float64,
    )
    if not len(heights):
        return
    result["minimum_base_height_m"] = float(np.min(heights))
    result["fell_below_height_threshold"] = bool(np.any(heights < fall_height))


def _close_child(child: pexpect.spawn | None, *, quit_command: str | None = None) -> None:
    if child is None or not child.isalive():
        return
    try:
        if quit_command is not None:
            child.sendline(quit_command)
        else:
            child.sendcontrol("c")
        child.expect(pexpect.EOF, timeout=5)
    except (pexpect.ExceptionPexpect, OSError):
        child.terminate(force=True)


def _sonic_command(groot_root: Path) -> tuple[str, list[str], Path, dict[str, str]]:
    deploy_root = groot_root / "gear_sonic_deploy"
    arguments = [
        "lo",
        "policy/release/model_decoder.onnx",
        "reference/example/",
        "--obs-config",
        "policy/release/observation_config.yaml",
        "--encoder-file",
        "policy/release/model_encoder.onnx",
        "--planner-file",
        "planner/target_vel/V2/planner_sonic.onnx",
        "--input-type",
        "zmq",
        "--zmq-host",
        "localhost",
        "--zmq-port",
        "5556",
        "--zmq-topic",
        "pose",
        "--zmq-verbose",
        "--output-type",
        "all",
        "--disable-crc-check",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "TensorRT_ROOT": "/usr",
            "CMAKE_PREFIX_PATH": "/opt/onnxruntime:/usr/lib/x86_64-linux-gnu/cmake",
            "COLCON_PREFIX_PATH": "",
        }
    )
    for name in ("ROS_DISTRO", "RMW_IMPLEMENTATION", "AMENT_PREFIX_PATH"):
        environment.pop(name, None)
    return str(deploy_root / "target/release/g1_deploy_onnx_ref"), arguments, deploy_root, environment


def run_trial(args: argparse.Namespace, level: str, repeat: int) -> dict:
    repo_root = _expanded(args.repo_root)
    groot_root = _expanded(args.groot_root)
    audiomimic_python = _expanded(args.audiomimic_python)
    motion_name, motion_path, run_prefix, runs_root, logs_root = _motion_spec(
        args, repo_root, level
    )
    with motion_path.open("rb") as handle:
        motion = pickle.load(handle)
    duration = len(np.asarray(motion["dof_pos"])) / float(motion["fps"]) + 4.0
    run_id = f"{run_prefix}_r0{repeat}"
    run_dir = runs_root / run_id
    log_dir = logs_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    simulator = sonic = streamer = None
    result = {
        "run_id": run_id,
        "source": args.source,
        "level": level,
        "repeat": repeat,
        "motion": motion_name,
        "status": "running",
        "reference_duration_seconds": duration - 4.0,
    }
    with (
        (log_dir / "mujoco.log").open("w") as sim_log,
        (log_dir / "sonic.log").open("w") as sonic_log,
        (log_dir / "streamer.log").open("w") as streamer_log,
    ):
        try:
            simulator = pexpect.spawn(
                str(groot_root / ".venv_sim/bin/python"),
                [str(repo_root / "scripts/run_sonic_headless_sim.py")],
                cwd=str(repo_root),
                encoding="utf-8",
                timeout=30,
            )
            simulator.logfile_read = sim_log
            simulator.expect("HEADLESS_SIM_READY elastic_band=enabled")

            command, command_args, cwd, environment = _sonic_command(groot_root)
            sonic = pexpect.spawn(
                command,
                command_args,
                cwd=str(cwd),
                env=environment,
                encoding="utf-8",
                timeout=90,
            )
            sonic.logfile_read = sonic_log
            sonic.expect("Init Done")
            sonic.send("]")
            sonic.expect("transitioning to CONTROL state")
            time.sleep(args.control_settle_seconds)

            simulator.sendline("release")
            simulator.expect("HEADLESS_SIM_RELEASED elastic_band=disabled")
            time.sleep(args.unassisted_settle_seconds)
            result["pre_stream_height_m"] = _expect_status(
                simulator, args.minimum_standing_height
            )

            sonic.sendline("")
            sonic.expect("ZMQ STREAMING MODE: ENABLED")
            streamer_args = [
                str(repo_root / "stream_to_sonic.py"),
                "--pkl",
                str(motion_path),
                "--root_quat_order",
                "xyzw",
                "--packet_mode",
                "full",
                "--playback_rate",
                "1.0",
                "--align_from_feedback_seconds",
                "3",
                "--align_hold_seconds",
                "1",
                "--output_dir",
                str(run_dir),
                "--record_feedback",
                "--feedback_port",
                "5557",
                "--sim_state_port",
                "5559",
                "--reference_safety",
                "none",
                "--sonic_reference_fps",
                "50",
                "--startup_wait",
                "2",
            ]
            if args.overwrite:
                streamer_args.append("--overwrite_output")
            streamer = pexpect.spawn(
                str(audiomimic_python),
                streamer_args,
                cwd=str(repo_root),
                encoding="utf-8",
                timeout=duration + 30,
            )
            streamer.logfile_read = streamer_log
            streamer.expect(pexpect.EOF)
            streamer.close()
            result["streamer_exit_status"] = streamer.exitstatus
            if streamer.exitstatus != 0:
                raise RuntimeError(f"streamer exited with status {streamer.exitstatus}")
            result["post_stream_height_m"] = _expect_status(
                simulator, args.minimum_standing_height, enforce=False
            )
            _record_execution_outcome(run_dir, result)
            result["status"] = "complete"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            _close_child(streamer)
            _close_child(sonic)
            _close_child(simulator, quit_command="quit")
            (log_dir / "trial.json").write_text(json.dumps(result, indent=2) + "\n")
            time.sleep(1.0)
    return result


def main(args: argparse.Namespace) -> None:
    if pexpect is None:
        raise RuntimeError(
            "run_sonic_capability_suite.py requires pexpect in the launcher environment"
        )
    if any(repeat not in (1, 2, 3) for repeat in args.repeats):
        raise ValueError("repeats must be selected from 1, 2, 3")
    summary_name = (
        "suite_sonic_native.json"
        if args.source == "sonic_native"
        else "suite_retargeted_gt_v2.json"
    )
    summary_path = (
        _expanded(args.repo_root) / "eval/gt_sonic_capability/suite_logs" / summary_name
    )
    if summary_path.is_file():
        results = json.loads(summary_path.read_text())
    else:
        results = []
    for level in args.levels:
        for repeat in args.repeats:
            print(f"Starting {level} repeat {repeat}", flush=True)
            try:
                result = run_trial(args, level, repeat)
            except Exception as error:
                print(f"FAILED {level} repeat {repeat}: {error}", file=sys.stderr, flush=True)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(results, indent=2) + "\n")
                raise
            results = [item for item in results if item.get("run_id") != result["run_id"]]
            results.append(result)
            print(
                f"Completed {result['run_id']}: "
                f"height {result['pre_stream_height_m']:.3f} -> "
                f"{result['post_stream_height_m']:.3f} m, "
                f"min={result.get('minimum_base_height_m', float('nan')):.3f}, "
                f"fell={result.get('fell_below_height_threshold')}",
                flush=True,
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main(parse_args())
