"""Run V6f-X D+C as an online plan-C4-execute-S66-replan loop over SONIC."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dataset.motion_representation import encode_g1_yaw_delta_motion
from dataset.g1_streaming_state import motion_to_streaming_state
from dataset.g1_streaming_state import G1StreamingStateStatistics
from model.g1_paper_faithful_dc_checkpoint import (
    DEFAULT_EXPERIMENT_ID,
    load_generator_bundle,
)
from model.g1_paper_faithful_dc_runtime import (
    COMMIT_FRAMES,
    PaperFaithfulDCRuntime,
    encode_k64_history_from_motion,
)
from model.g1_paper_faithful_dc_streaming import (
    DIFFUSION_TIMESTEPS,
    PredX0CosineDiffusion,
)
from sonic_bridge import (
    G1_FPS,
    SonicReferenceAdapter,
    SonicFeedbackSubscriber,
    SonicS66Synchronizer,
    g1_motion_from_yaw_delta_commit,
    jsonable_feedback,
    load_g1_motion,
    pack_zmq_message,
    reference_fields_from_commit,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator_checkpoint", required=True)
    parser.add_argument("--q0_checkpoint", default="")
    parser.add_argument("--codec_checkpoint", default="")
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--allow_historical_provenance", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--plan_cache_dir",
        default="",
        help="Optional online-run directory whose committed motion and H8 plans are replayed.",
    )
    parser.add_argument("--max_commits", default=120, type=int)
    parser.add_argument("--nfe", choices=(10, 20, 50, 100), default=50, type=int)
    parser.add_argument("--sampling_seed", default=1234, type=int)
    parser.add_argument("--q0_policy", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--seed_mode", choices=("cold", "k64_pkl"), default="cold")
    parser.add_argument("--seed_motion_pkl", default="")
    parser.add_argument("--seed_start_frame", default=0, type=int)
    parser.add_argument(
        "--seed_execution",
        choices=("latent_only", "replay"),
        default="latent_only",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", default=5556, type=int)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--startup_wait", default=1.0, type=float)
    parser.add_argument("--record_feedback", action="store_true")
    parser.add_argument(
        "--preview_mode",
        choices=("c4_hold", "h8_preview", "oracle_1s"),
        default="c4_hold",
    )
    parser.add_argument(
        "--feedback_source",
        choices=("measured", "synthetic", "delayed_residual", "jit_measured"),
        default="measured",
        help="S66 used to condition the next generator plan.",
    )
    parser.add_argument(
        "--feedback_alpha",
        default=0.0,
        type=float,
        help="one-C4 delayed measured-minus-synthetic residual gain",
    )
    parser.add_argument(
        "--jit_inference_budget_ms",
        default=80.0,
        type=float,
        help="time reserved before each C4 deadline for JIT measured-state inference",
    )
    parser.add_argument("--feedback_host", default="localhost")
    parser.add_argument("--feedback_port", default=5557, type=int)
    parser.add_argument("--feedback_topic", default="g1_debug")
    parser.add_argument("--sim_state_host", default="localhost")
    parser.add_argument("--sim_state_port", default=5559, type=int)
    parser.add_argument("--sim_state_topic", default="sonic_state")
    parser.add_argument("--initial_state_timeout", default=5.0, type=float)
    parser.add_argument("--execution_timeout", default=1.0, type=float)
    parser.add_argument(
        "--reference_safety",
        choices=("conservative", "none"),
        default="conservative",
    )
    parser.add_argument("--sonic_reference_fps", default=50.0, type=float)
    parser.add_argument("--max_joint_velocity_rad_s", default=1.5, type=float)
    parser.add_argument("--max_yaw_velocity_rad_s", default=0.5, type=float)
    parser.add_argument("--ramp_seconds", default=1.0, type=float)
    parser.add_argument("--joint_limit_margin_rad", default=0.05, type=float)
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _device_from_args(args) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def _build_runtime(args, device: torch.device) -> tuple[PaperFaithfulDCRuntime, dict]:
    bundle = load_generator_bundle(args, device)
    checkpoint = bundle["checkpoint"]
    codec_checkpoint = bundle["codec_checkpoint"]
    statistics = G1StreamingStateStatistics.from_state_dict(
        codec_checkpoint["streaming_statistics"]
    )
    residual_mean = checkpoint["residual_mean"].to(device).view(1, 1, -1)
    residual_std = checkpoint["residual_std"].to(device).view(1, 1, -1)
    motion_mean = torch.as_tensor(
        codec_checkpoint["normalizer"]["mean"], device=device
    ).view(1, 1, -1)
    motion_std = torch.as_tensor(
        codec_checkpoint["normalizer"]["std"], device=device
    ).view(1, 1, -1)
    generator = torch.Generator(device=device).manual_seed(int(args.sampling_seed))
    runtime = PaperFaithfulDCRuntime(
        q0_model=bundle["q0_model"],
        residual_model=bundle["residual_model"],
        stage=bundle["stage"],
        codec=bundle["codec"],
        diffusion=PredX0CosineDiffusion(
            timesteps=DIFFUSION_TIMESTEPS,
            eta=0.0,
        ).to(device),
        statistics=statistics,
        residual_mean=residual_mean,
        residual_std=residual_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        nfe=args.nfe,
        q0_policy=args.q0_policy,
        temperature=args.temperature,
        generator=generator,
    )
    return runtime, bundle


@torch.inference_mode()
def _encode_k64_pkl_seed(
    *,
    path: str,
    start_frame: int,
    runtime: PaperFaithfulDCRuntime,
    execution_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Encode an exported G1 pickle, but keep SONIC feedback as the condition."""
    motion = load_g1_motion(path)
    if motion.root_pos is None:
        raise ValueError("K64 bootstrap pickle must contain root_pos [T,3]")
    if not np.isclose(motion.fps, G1_FPS, rtol=0.0, atol=1e-4):
        raise ValueError(
            f"K64 bootstrap requires {G1_FPS:g} Hz motion, got {motion.fps:g} Hz"
        )
    seed_frames = 128
    start_frame = int(start_frame)
    stop_frame = start_frame + seed_frames
    if start_frame < 0 or stop_frame > motion.frames:
        raise ValueError(
            f"K64 bootstrap window [{start_frame}, {stop_frame}) exceeds "
            f"the {motion.frames}-frame source motion"
        )
    device = execution_state.device
    dtype = runtime.motion_mean.dtype
    raw_motion = encode_g1_yaw_delta_motion(
        torch.as_tensor(
            motion.root_pos[start_frame:stop_frame], device=device, dtype=dtype
        ).unsqueeze(0),
        torch.as_tensor(
            motion.root_rot_xyzw[start_frame:stop_frame], device=device, dtype=dtype
        ).unsqueeze(0),
        torch.as_tensor(
            motion.dof_pos_mujoco[start_frame:stop_frame], device=device, dtype=dtype
        ).unsqueeze(0),
    )
    history_q0, history_residual_raw, history_valid, source_boundary_state = (
        encode_k64_history_from_motion(
            codec=runtime.codec,
            statistics=runtime.statistics,
            raw_motion=raw_motion,
            motion_mean=runtime.motion_mean,
            motion_std=runtime.motion_std,
        )
    )
    history_residual = (
        history_residual_raw - runtime.residual_mean
    ) / runtime.residual_std
    difference = source_boundary_state - execution_state
    metadata = {
        "mode": "k64_pkl",
        "motion_path": str(Path(path).resolve()),
        "source_frames": int(motion.frames),
        "source_fps": float(motion.fps),
        "source_start_frame": start_frame,
        "source_stop_frame": stop_frame,
        "encoded_history_tokens": int(history_valid.sum().item()),
        "source_execution_s66_l2": float(torch.linalg.vector_norm(difference).item()),
        "source_execution_height_delta": float(difference[0, 0].item()),
        "source_execution_joint_rmse": float(
            torch.sqrt(torch.mean(difference[:, 5:34].square())).item()
        ),
    }
    return (
        history_q0,
        history_residual,
        history_valid,
        raw_motion,
        source_boundary_state,
        metadata,
    )


def _receive_execution_state(
    *,
    sonic_subscriber,
    sim_state_subscriber,
    synchronizer: SonicS66Synchronizer,
    timeout_seconds: float,
    minimum_feedback_index: int | None = None,
    not_before: float | None = None,
    sonic_records: list[dict],
    sim_state_records: list[dict],
) -> tuple[dict, dict] | None:
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    candidate = None
    while time.monotonic() < deadline:
        while True:
            sim_state = sim_state_subscriber.poll(timeout_ms=0)
            if sim_state is None:
                break
            sim_state_records.append(jsonable_feedback(sim_state))
            synchronizer.update_sim_state(sim_state)
        feedback = sonic_subscriber.poll(timeout_ms=20)
        if feedback is None:
            continue
        sonic_records.append(jsonable_feedback(feedback))
        synchronized = synchronizer.update_sonic_feedback(feedback)
        if synchronized is None:
            continue
        index = synchronized.get("sonic_feedback_index")
        if index is None:
            continue
        if minimum_feedback_index is not None and int(index) <= int(minimum_feedback_index):
            continue
        sim_state = synchronizer.latest_sim_state
        if sim_state is None:
            continue
        candidate = synchronized, sim_state
        if not_before is None or time.monotonic() >= not_before:
            return candidate
    return candidate if not_before is None else None


def _execution_dof_from_s66(execution: dict) -> np.ndarray:
    """Extract MuJoCo-order joint positions from the declared S66 layout."""
    s66 = np.asarray(execution.get("s66"), dtype=np.float32)
    if s66.shape != (66,) or not np.isfinite(s66).all():
        raise ValueError("synchronized SONIC execution state must contain finite S66")
    return s66[5:34].copy()


def _committed_reference_frame_count(
    *, frame_index: int, source_frames: int, source_fps: float, target_fps: float
) -> int:
    """Advance a chunked resampling clock without resetting its phase."""
    consumed_source_frames = int(round(int(frame_index) * float(source_fps) / float(target_fps)))
    next_frame_index = int(
        np.floor(
            (consumed_source_frames + int(source_frames))
            * float(target_fps)
            / float(source_fps)
            + 1e-8
        )
    )
    return max(1, next_frame_index - int(frame_index))


def _jit_inference_start_deadline(
    *, stream_started: float, commit_index: int, commit_seconds: float, budget_ms: float
) -> float:
    """Return the absolute time at which JIT inference must start."""
    return (
        float(stream_started)
        + int(commit_index) * float(commit_seconds)
        - float(budget_ms) / 1000.0
    )


def _apply_delayed_residual(
    synthetic_current: torch.Tensor,
    *,
    measured_previous: torch.Tensor | None,
    synthetic_previous: torch.Tensor | None,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Correct the current predicted end state with the previous C4 residual."""
    if measured_previous is None or synthetic_previous is None:
        return synthetic_current, None
    residual = measured_previous - synthetic_previous
    return synthetic_current + float(alpha) * residual, residual


def _collect_execution_until(
    *,
    deadline: float,
    sonic_subscriber,
    sim_state_subscriber,
    synchronizer: SonicS66Synchronizer,
    sonic_records: list[dict],
    sim_state_records: list[dict],
):
    """Drain telemetry until an absolute deadline without shifting that deadline."""
    latest = None
    while True:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0.0:
            return latest
        sim_state = sim_state_subscriber.poll(timeout_ms=0)
        while sim_state is not None:
            sim_state_records.append(jsonable_feedback(sim_state))
            synchronizer.update_sim_state(sim_state)
            sim_state = sim_state_subscriber.poll(timeout_ms=0)
        feedback = sonic_subscriber.poll(timeout_ms=min(5, max(0, int(remaining * 1000))))
        if feedback is not None:
            sonic_records.append(jsonable_feedback(feedback))
            synchronized = synchronizer.update_sonic_feedback(feedback)
            if synchronized is not None:
                latest = (synchronized, synchronizer.latest_sim_state)
        if remaining > 0.005:
            time.sleep(min(0.001, remaining))


def _execute_yaw_delta_commit(
    *,
    raw_commit: np.ndarray,
    raw_reference: np.ndarray | None = None,
    anchor_state: dict,
    execution_dof_mujoco: np.ndarray,
    previous_reference_dof: np.ndarray | None,
    frame_index: int,
    last_feedback_index: int,
    publisher,
    topic: str,
    sonic_subscriber,
    sim_state_subscriber,
    synchronizer: SonicS66Synchronizer,
    reference_adapter: SonicReferenceAdapter,
    execution_timeout: float,
    commit_seconds: float,
    sonic_records: list[dict],
    sim_state_records: list[dict],
    wait_for_execution: bool = True,
) -> tuple[dict, dict, np.ndarray, int, int, int, dict]:
    """Send a C4 or overlapping H8 reference, then return post-C4 S66."""
    if raw_reference is None:
        raw_reference = raw_commit
    committed_motion = g1_motion_from_yaw_delta_commit(
        raw_commit,
        base_position=anchor_state["base_position"],
        base_quat_wxyz=anchor_state["base_quat"],
        fps=G1_FPS,
    )
    reference_motion = g1_motion_from_yaw_delta_commit(
        raw_reference,
        base_position=anchor_state["base_position"],
        base_quat_wxyz=anchor_state["base_quat"],
        fps=G1_FPS,
    )
    reference_motion = reference_adapter.adapt(
        reference_motion,
        execution_dof_mujoco=execution_dof_mujoco,
        execution_quat_wxyz=anchor_state["base_quat"],
    )
    fields = reference_fields_from_commit(
        reference_motion,
        start_frame=0,
        stop_frame=reference_motion.frames,
        previous_dof_pos=previous_reference_dof,
        frame_index_start=frame_index,
    )
    publisher.send(pack_zmq_message(fields, topic=topic, version=1))
    sent_at = time.monotonic()
    committed_reference_frames = _committed_reference_frame_count(
        frame_index=frame_index,
        source_frames=raw_commit.shape[0],
        source_fps=G1_FPS,
        target_fps=reference_motion.fps,
    )
    if committed_reference_frames > reference_motion.frames:
        raise ValueError("reference preview is shorter than the committed interval")
    next_previous_dof = reference_motion.dof_pos_mujoco[committed_reference_frames - 1].copy()
    reference_frames = reference_motion.frames
    next_frame_index = frame_index + committed_reference_frames
    synthetic_anchor_state = {
        "base_position": committed_motion.root_pos[-1].astype(np.float32).copy(),
        "base_quat": committed_motion.root_rot_xyzw[-1][[3, 0, 1, 2]].astype(np.float32).copy(),
    }
    diagnostics = dict(reference_adapter.last_diagnostics)
    diagnostics["reference_sent_monotonic"] = sent_at
    diagnostics["committed_reference_frames"] = committed_reference_frames
    if not wait_for_execution:
        return (
            None,
            synthetic_anchor_state,
            next_previous_dof,
            next_frame_index,
            last_feedback_index,
            int(reference_frames),
            diagnostics,
        )
    received = _receive_execution_state(
        sonic_subscriber=sonic_subscriber,
        sim_state_subscriber=sim_state_subscriber,
        synchronizer=synchronizer,
        timeout_seconds=execution_timeout,
        minimum_feedback_index=last_feedback_index,
        not_before=sent_at + commit_seconds,
        sonic_records=sonic_records,
        sim_state_records=sim_state_records,
    )
    if received is None:
        raise TimeoutError("SONIC did not provide a new synchronized S66 after C4")
    execution, next_anchor_state = received
    return (
        execution,
        next_anchor_state,
        next_previous_dof,
        next_frame_index,
        int(execution["sonic_feedback_index"]),
        int(reference_frames),
        diagnostics,
    )


def _summarize_yaw_delta_motion(raw_motion: np.ndarray) -> dict:
    raw_motion = np.asarray(raw_motion, dtype=np.float32)
    if raw_motion.ndim != 2 or raw_motion.shape[-1] != 34:
        raise ValueError("raw_motion must have shape [T,34]")
    joint_span = np.ptp(raw_motion[:, 5:], axis=0)
    yaw_delta = np.arctan2(raw_motion[:, 3], raw_motion[:, 4])
    return {
        "frames": int(raw_motion.shape[0]),
        "max_joint_span_rad": float(joint_span.max()),
        "median_joint_span_rad": float(np.median(joint_span)),
        "root_height_span_m": float(np.ptp(raw_motion[:, 2])),
        "local_xy_path_m": float(np.linalg.norm(raw_motion[:, :2], axis=1).sum()),
        "absolute_yaw_change_rad": float(np.abs(yaw_delta).sum()),
    }


def run(args) -> None:
    if args.max_commits <= 0:
        raise ValueError("max_commits must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= args.feedback_alpha <= 1.0:
        raise ValueError("feedback_alpha must be in [0,1]")
    if args.jit_inference_budget_ms <= 0.0:
        raise ValueError("jit_inference_budget_ms must be positive")
    if args.feedback_source == "jit_measured" and args.plan_cache_dir:
        raise ValueError("jit_measured requires live generation, not --plan_cache_dir")
    if args.seed_mode == "k64_pkl" and not args.seed_motion_pkl:
        raise ValueError("--seed_motion_pkl is required with --seed_mode k64_pkl")
    if args.seed_mode == "cold" and args.seed_motion_pkl:
        raise ValueError("--seed_motion_pkl requires --seed_mode k64_pkl")
    if args.seed_mode == "cold" and args.seed_execution != "latent_only":
        raise ValueError("--seed_execution replay requires --seed_mode k64_pkl")
    commit_seconds = COMMIT_FRAMES / G1_FPS
    if args.jit_inference_budget_ms >= commit_seconds * 1000.0:
        raise ValueError("jit_inference_budget_ms must be shorter than one C4 commit")
    if args.execution_timeout < commit_seconds:
        raise ValueError(
            "execution_timeout must cover one C4 commit "
            f"({commit_seconds:.3f} seconds)"
        )
    try:
        import zmq
    except ImportError as error:
        raise RuntimeError("online SONIC runtime requires pyzmq") from error

    device = _device_from_args(args)
    runtime, bundle = _build_runtime(args, device)
    output_dir = Path(args.output_dir)
    sonic_records: list[dict] = []
    sim_state_records: list[dict] = []
    s66_records: list[dict] = []
    seed_s66_records: list[dict] = []
    generated_raw_commits: list[np.ndarray] = []
    generated_raw_plans: list[np.ndarray] = []
    synthetic_s66_records: list[dict] = []
    commit_records: list[dict] = []
    seed_commit_records: list[dict] = []
    synchronizer = SonicS66Synchronizer(fps=G1_FPS)
    reference_adapter = SonicReferenceAdapter(
        target_fps=args.sonic_reference_fps,
        safety_enabled=args.reference_safety == "conservative",
        max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
        max_yaw_velocity_rad_s=args.max_yaw_velocity_rad_s,
        ramp_seconds=args.ramp_seconds,
        joint_limit_margin_rad=args.joint_limit_margin_rad,
    )
    sonic_subscriber = SonicFeedbackSubscriber(
        host=args.feedback_host,
        port=args.feedback_port,
        topic=args.feedback_topic,
    )
    sim_state_subscriber = SonicFeedbackSubscriber(
        host=args.sim_state_host,
        port=args.sim_state_port,
        topic=args.sim_state_topic,
    )
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://*:{int(args.port)}")
    try:
        print(f"SONIC reference PUB bound to tcp://*:{args.port}", flush=True)
        time.sleep(max(float(args.startup_wait), 0.0))
        initial = _receive_execution_state(
            sonic_subscriber=sonic_subscriber,
            sim_state_subscriber=sim_state_subscriber,
            synchronizer=synchronizer,
            timeout_seconds=args.initial_state_timeout,
            sonic_records=sonic_records,
            sim_state_records=sim_state_records,
        )
        if initial is None:
            raise TimeoutError(
                "did not receive synchronized SONIC g1_debug and sonic_state; "
                "start MuJoCo telemetry, enable ZMQ mode, and press ] in SONIC"
            )
        execution, anchor_state = initial
        execution_tensor = torch.as_tensor(
            execution["s66"], device=device, dtype=runtime.motion_mean.dtype
        ).view(1, -1)
        previous_reference_dof = _execution_dof_from_s66(execution)
        frame_index = 0
        last_feedback_index = int(execution["sonic_feedback_index"])
        if args.seed_mode == "k64_pkl":
            (
                history_q0,
                history_residual,
                history_valid,
                seed_raw_motion,
                source_boundary_state,
                seed_metadata,
            ) = (
                _encode_k64_pkl_seed(
                    path=args.seed_motion_pkl,
                    start_frame=args.seed_start_frame,
                    runtime=runtime,
                    execution_state=execution_tensor,
                )
            )
            seed_metadata["execution"] = args.seed_execution
            if args.seed_execution == "replay":
                print("Replaying K64 seed to establish the physical history", flush=True)
                seed_raw_motion_np = seed_raw_motion.squeeze(0).detach().cpu().numpy()
                for seed_start in range(0, seed_raw_motion_np.shape[0], COMMIT_FRAMES):
                    seed_frame_index_start = frame_index
                    (
                        execution,
                        anchor_state,
                        previous_reference_dof,
                        frame_index,
                        last_feedback_index,
                        reference_frames,
                        reference_diagnostics,
                    ) = (
                        _execute_yaw_delta_commit(
                            raw_commit=seed_raw_motion_np[
                                seed_start : seed_start + COMMIT_FRAMES
                            ],
                            anchor_state=anchor_state,
                            execution_dof_mujoco=_execution_dof_from_s66(execution),
                            previous_reference_dof=previous_reference_dof,
                            frame_index=frame_index,
                            last_feedback_index=last_feedback_index,
                            publisher=publisher,
                            topic=args.topic,
                            sonic_subscriber=sonic_subscriber,
                            sim_state_subscriber=sim_state_subscriber,
                            synchronizer=synchronizer,
                            reference_adapter=reference_adapter,
                            execution_timeout=args.execution_timeout,
                            commit_seconds=commit_seconds,
                            sonic_records=sonic_records,
                            sim_state_records=sim_state_records,
                        )
                    )
                    seed_s66_records.append(execution)
                    seed_commit_records.append(
                        {
                            "frame_index_start": int(seed_frame_index_start),
                            "reference_frames": int(reference_frames),
                            "reference_diagnostics": reference_diagnostics,
                            "sonic_feedback_index": last_feedback_index,
                            "sim_time": execution.get("sim_time"),
                        }
                    )
                    print(
                        "Replayed K64 seed C4 "
                        f"{seed_start // COMMIT_FRAMES + 1}/"
                        f"{seed_raw_motion_np.shape[0] // COMMIT_FRAMES}: "
                        f"SONIC feedback {last_feedback_index}, ref "
                        f"{reference_frames}f@{args.sonic_reference_fps:g}Hz, "
                        f"|dq|<={reference_diagnostics['safe_max_joint_velocity_rad_s']:.2f}",
                        flush=True,
                    )
                execution_tensor = torch.as_tensor(
                    execution["s66"], device=device, dtype=runtime.motion_mean.dtype
                ).view(1, -1)
                difference_after_replay = source_boundary_state - execution_tensor
                seed_metadata["replayed_frames"] = int(seed_raw_motion_np.shape[0])
                seed_metadata["source_execution_s66_l2_after_replay"] = float(
                    torch.linalg.vector_norm(difference_after_replay).item()
                )
                seed_metadata["source_execution_joint_rmse_after_replay"] = float(
                    torch.sqrt(
                        torch.mean(difference_after_replay[:, 5:34].square())
                    ).item()
                )
            state = runtime.warm_start(
                execution_tensor,
                history_q0=history_q0,
                history_residual=history_residual,
                history_valid=history_valid,
            )
            startup_description = "K64-bootstrap"
        else:
            state = runtime.cold_start(execution_tensor)
            seed_metadata = {"mode": "cold"}
            startup_description = "cold-start"
        print(
            "Received initial S66; running strict-causal "
            f"{startup_description} D+C loop "
            f"for {args.max_commits} C4 commits",
            flush=True,
        )

        cached_commits = None
        cached_plans = None
        if args.plan_cache_dir:
            cache_dir = Path(args.plan_cache_dir).expanduser().resolve()
            cached_motion = np.asarray(
                np.load(cache_dir / "generated_motion.npy"), dtype=np.float32
            )
            cached_plans = np.asarray(np.load(cache_dir / "raw_plans.npy"), dtype=np.float32)
            expected_frames = int(args.max_commits) * COMMIT_FRAMES
            if cached_motion.shape != (expected_frames, 34):
                raise ValueError(
                    f"cached motion must have shape [{expected_frames},34], got {cached_motion.shape}"
                )
            if cached_plans.shape != (int(args.max_commits), 2 * COMMIT_FRAMES, 34):
                raise ValueError(
                    "cached plans must have shape "
                    f"[{args.max_commits},{2 * COMMIT_FRAMES},34], got {cached_plans.shape}"
                )
            cached_commits = cached_motion.reshape(
                int(args.max_commits), COMMIT_FRAMES, 34
            )
        generator_initial_s66 = state.execution_state.squeeze(0).detach().cpu().tolist()
        open_loop_started = None
        open_loop_deadline_misses = 0
        previous_synthetic_s66 = None
        jit_pending_state = None
        jit_condition_feedback_index = None
        pipelined_feedback = args.feedback_source in (
            "synthetic",
            "delayed_residual",
            "jit_measured",
        )
        for loop_index in range(int(args.max_commits)):
            deadline = None
            if args.feedback_source == "jit_measured" and loop_index > 0:
                deadline = open_loop_started + loop_index * commit_seconds
                inference_start_deadline = _jit_inference_start_deadline(
                    stream_started=open_loop_started,
                    commit_index=loop_index,
                    commit_seconds=commit_seconds,
                    budget_ms=args.jit_inference_budget_ms,
                )
                latest = _collect_execution_until(
                    deadline=inference_start_deadline,
                    sonic_subscriber=sonic_subscriber,
                    sim_state_subscriber=sim_state_subscriber,
                    synchronizer=synchronizer,
                    sonic_records=sonic_records,
                    sim_state_records=sim_state_records,
                )
                if latest is not None:
                    execution, _measured_anchor = latest
                    last_feedback_index = int(execution["sonic_feedback_index"])
                measured_condition = torch.as_tensor(
                    execution["s66"], device=device, dtype=runtime.motion_mean.dtype
                ).view(1, -1)
                state = runtime.accept_execution_feedback(
                    jit_pending_state, measured_condition
                )
                jit_condition_feedback_index = last_feedback_index
            if cached_commits is None:
                pending_state, generated = runtime.plan_commit(state)
            else:
                pending_state = None
                generated = SimpleNamespace(
                    raw_commit=torch.as_tensor(
                        cached_commits[loop_index], device=device, dtype=runtime.motion_mean.dtype
                    ).unsqueeze(0),
                    raw_plan=torch.as_tensor(
                        cached_plans[loop_index], device=device, dtype=runtime.motion_mean.dtype
                    ).unsqueeze(0),
                    latency_ms={"q0": 0.0, "residual": 0.0, "decode": 0.0, "total": 0.0},
                )
            raw_commit = generated.raw_commit.squeeze(0).detach().cpu().numpy()
            raw_plan = generated.raw_plan.squeeze(0).detach().cpu().numpy()
            generated_raw_commits.append(raw_commit.copy())
            generated_raw_plans.append(raw_plan.copy())
            packet_lateness_ms = 0.0
            if pipelined_feedback:
                if open_loop_started is None:
                    open_loop_started = time.monotonic()
                deadline = open_loop_started + loop_index * commit_seconds
                latest = _collect_execution_until(
                    deadline=deadline,
                    sonic_subscriber=sonic_subscriber,
                    sim_state_subscriber=sim_state_subscriber,
                    synchronizer=synchronizer,
                    sonic_records=sonic_records,
                    sim_state_records=sim_state_records,
                )
                if latest is not None:
                    execution, _measured_anchor = latest
                    last_feedback_index = int(execution["sonic_feedback_index"])
                packet_lateness_ms = max(0.0, (time.monotonic() - deadline) * 1000.0)
                if packet_lateness_ms > 2.0:
                    open_loop_deadline_misses += 1
            frame_index_start = frame_index
            if args.preview_mode == "c4_hold":
                generated_reference_adapter = reference_adapter
            else:
                # H8 windows overlap, so their content is resampled
                # independently. The committed frame clock remains continuous
                # inside _execute_yaw_delta_commit.
                generated_reference_adapter = SonicReferenceAdapter(
                    target_fps=args.sonic_reference_fps,
                    safety_enabled=args.reference_safety == "conservative",
                    max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
                    max_yaw_velocity_rad_s=args.max_yaw_velocity_rad_s,
                    ramp_seconds=args.ramp_seconds,
                    joint_limit_margin_rad=args.joint_limit_margin_rad,
                )
            (
                received_execution,
                anchor_state,
                previous_reference_dof,
                frame_index,
                last_feedback_index,
                reference_frames,
                reference_diagnostics,
            ) = _execute_yaw_delta_commit(
                raw_commit=raw_commit,
                raw_reference=(raw_plan if args.preview_mode == "h8_preview" else raw_commit),
                anchor_state=anchor_state,
                execution_dof_mujoco=_execution_dof_from_s66(execution),
                previous_reference_dof=previous_reference_dof,
                frame_index=frame_index,
                last_feedback_index=last_feedback_index,
                publisher=publisher,
                topic=args.topic,
                sonic_subscriber=sonic_subscriber,
                sim_state_subscriber=sim_state_subscriber,
                synchronizer=synchronizer,
                reference_adapter=generated_reference_adapter,
                execution_timeout=args.execution_timeout,
                commit_seconds=commit_seconds,
                sonic_records=sonic_records,
                sim_state_records=sim_state_records,
                wait_for_execution=not pipelined_feedback,
            )
            if received_execution is not None:
                execution = received_execution
            synthetic_s66 = motion_to_streaming_state(
                generated.raw_commit,
                fps=runtime.statistics.fps,
                state_spec=runtime.statistics.state_spec,
            )[:, -1]
            measured_s66 = torch.as_tensor(
                execution["s66"], device=device, dtype=runtime.motion_mean.dtype
            ).view(1, -1)
            delayed_residual = None
            if args.feedback_source == "measured":
                next_s66 = measured_s66
            elif args.feedback_source == "delayed_residual":
                next_s66, delayed_residual = _apply_delayed_residual(
                    synthetic_s66,
                    measured_previous=measured_s66,
                    synthetic_previous=previous_synthetic_s66,
                    alpha=args.feedback_alpha,
                )
            else:
                next_s66 = synthetic_s66
            if args.feedback_source == "jit_measured":
                jit_pending_state = pending_state
                step_index = int(pending_state.step_index)
            elif pending_state is not None:
                state = runtime.accept_execution_feedback(pending_state, next_s66)
                step_index = int(state.step_index)
            else:
                step_index = loop_index + 1
            s66_records.append(execution)
            synthetic_s66_records.append(
                {
                    "step_index": step_index,
                    "s66": synthetic_s66.squeeze(0).detach().cpu().tolist(),
                }
            )
            s66_delta = measured_s66 - synthetic_s66
            previous_synthetic_s66 = synthetic_s66.detach().clone()
            commit_records.append(
                {
                    "step_index": step_index,
                    "frame_index_start": int(frame_index_start),
                    "reference_frames": int(reference_frames),
                    "reference_diagnostics": reference_diagnostics,
                    "reference_sent_monotonic": reference_diagnostics[
                        "reference_sent_monotonic"
                    ],
                    "sonic_feedback_index": last_feedback_index,
                    "sim_time": execution.get("sim_time"),
                    "feedback_source": args.feedback_source,
                    "feedback_alpha": float(args.feedback_alpha),
                    "jit_condition_feedback_index": jit_condition_feedback_index,
                    "jit_inference_budget_ms": float(args.jit_inference_budget_ms),
                    "delayed_residual_available": delayed_residual is not None,
                    "delayed_residual_s66_l2": None
                    if delayed_residual is None
                    else float(torch.linalg.vector_norm(delayed_residual).item()),
                    "delayed_residual_joint_rmse": None
                    if delayed_residual is None
                    else float(torch.sqrt(torch.mean(delayed_residual[:, 5:34].square())).item()),
                    "packet_deadline_monotonic": deadline,
                    "packet_lateness_ms": packet_lateness_ms,
                    "measured_synthetic_s66_l2": float(torch.linalg.vector_norm(s66_delta).item()),
                    "measured_synthetic_joint_rmse": float(
                        torch.sqrt(torch.mean(s66_delta[:, 5:34].square())).item()
                    ),
                    "latency_ms": generated.latency_ms,
                }
            )
            print(
                f"Committed C4 step {step_index}: SONIC feedback "
                f"{last_feedback_index}, inference {generated.latency_ms['total']:.1f} ms, "
                f"ref {reference_frames}f@{args.sonic_reference_fps:g}Hz, "
                f"condition {args.feedback_source}, "
                f"|dq|<={reference_diagnostics['safe_max_joint_velocity_rad_s']:.2f}",
                flush=True,
            )
        if pipelined_feedback and open_loop_started is not None:
            latest = _collect_execution_until(
                deadline=open_loop_started + int(args.max_commits) * commit_seconds,
                sonic_subscriber=sonic_subscriber,
                sim_state_subscriber=sim_state_subscriber,
                synchronizer=synchronizer,
                sonic_records=sonic_records,
                sim_state_records=sim_state_records,
            )
            if latest is not None:
                execution, _measured_anchor = latest
    finally:
        publisher.close()
        context.term()
        sonic_subscriber.close()
        sim_state_subscriber.close()

    _write_json(output_dir / "sonic_feedback.json", sonic_records)
    _write_json(output_dir / "sim_state.json", sim_state_records)
    _write_json(output_dir / "s66_exec.json", s66_records)
    _write_json(output_dir / "synthetic_s66.json", synthetic_s66_records)
    _write_json(output_dir / "seed_s66_exec.json", seed_s66_records)
    generated_motion = np.concatenate(generated_raw_commits, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "generated_motion.npy", generated_motion)
    np.save(output_dir / "raw_plans.npy", np.stack(generated_raw_plans))
    _write_json(
        output_dir / "runtime.json",
        {
            "schema_version": "g1_paper_faithful_dc_sonic_runtime_v4",
            "experiment_id": args.experiment_id,
            "stage": bundle["stage"],
            "generator_checkpoint": args.generator_checkpoint,
            "plan_cache_dir": args.plan_cache_dir,
            "codec_checkpoint": bundle["codec_path"],
            "max_commits": int(args.max_commits),
            "commit_frames": COMMIT_FRAMES,
            "plan_tokens": 8,
            "commit_tokens": 4,
            "feedback_source": args.feedback_source,
            "feedback_alpha": float(args.feedback_alpha),
            "jit_inference_budget_ms": float(args.jit_inference_budget_ms),
            "preview_mode": args.preview_mode,
            "online_started_monotonic": commit_records[0]["reference_sent_monotonic"],
            "generator_initial_s66": generator_initial_s66,
            "sonic_reference_fps": args.sonic_reference_fps,
            "reference_safety": args.reference_safety,
            "max_joint_velocity_rad_s": args.max_joint_velocity_rad_s,
            "max_yaw_velocity_rad_s": args.max_yaw_velocity_rad_s,
            "ramp_seconds": args.ramp_seconds,
            "joint_limit_margin_rad": args.joint_limit_margin_rad,
            "cold_start": args.seed_mode == "cold",
            "seed": seed_metadata,
            "root_xy_tracking": False,
            "scheduler": {
                "mode": (
                    "jit_measured_absolute_deadline"
                    if args.feedback_source == "jit_measured"
                    else "absolute_deadline_double_buffer"
                    if pipelined_feedback
                    else "blocking_measured_feedback"
                ),
                "deadline_misses_over_2ms": int(open_loop_deadline_misses),
            },
            "seed_commit_records": seed_commit_records,
            "generated_motion_file": "generated_motion.npy",
            "generated_motion_activity": _summarize_yaw_delta_motion(
                generated_motion
            ),
            "commit_records": commit_records,
        },
    )


if __name__ == "__main__":
    run(parse_args())
