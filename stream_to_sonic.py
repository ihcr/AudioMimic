"""Play an exported offline G1 trajectory to SONIC without closing the loop.

The M0/M2/M4 evaluation exports contain absolute G1 joint positions and base
orientation at 30 Hz.  This player is deliberately separate from the online
diffusion runtime: it never consumes SONIC S66 feedback and therefore measures
only SONIC's ability to track a fixed generated trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from sonic_bridge import (
    G1_COMMIT_FRAMES,
    G1_FPS,
    G1Motion,
    SonicFeedbackSubscriber,
    SonicReferenceAdapter,
    SonicS66Synchronizer,
    jsonable_feedback,
    load_g1_motion,
    pack_zmq_message,
    reference_fields_from_commit,
    SONIC_REFERENCE_FROM_MUJOCO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True, help="offline G1 motion pickle")
    parser.add_argument(
        "--root_quat_order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="quaternion storage order in the input pickle",
    )
    parser.add_argument("--output_dir", default="", help="optional playback metadata directory")
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="allow replacing files in an existing non-empty output directory",
    )
    parser.add_argument("--port", default=5556, type=int)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--startup_wait", default=2.0, type=float)
    parser.add_argument(
        "--align_from_feedback_seconds",
        default=0.0,
        type=float,
        help="smoothly approach input frame zero from measured SONIC state before playback",
    )
    parser.add_argument(
        "--align_hold_seconds",
        default=0.0,
        type=float,
        help="hold input frame zero after alignment and before the original motion",
    )
    parser.add_argument(
        "--record_feedback",
        action="store_true",
        help="record SONIC/S66 telemetry for evaluation without using it to alter references",
    )
    parser.add_argument("--feedback_host", default="localhost")
    parser.add_argument("--feedback_port", default=5557, type=int)
    parser.add_argument("--feedback_topic", default="g1_debug")
    parser.add_argument("--sim_state_host", default="localhost")
    parser.add_argument("--sim_state_port", default=5559, type=int)
    parser.add_argument("--sim_state_topic", default="sonic_state")
    parser.add_argument("--sonic_reference_fps", default=50.0, type=float)
    parser.add_argument(
        "--playback_rate",
        default=1.0,
        type=float,
        help="motion speed multiplier applied before 30-to-50 Hz conversion",
    )
    parser.add_argument(
        "--packet_mode",
        choices=("c4", "full"),
        default="c4",
        help="c4 streams online-sized updates; full sends the complete fixed trajectory once",
    )
    parser.add_argument(
        "--preview_seconds",
        default=0.0,
        type=float,
        help=(
            "future reference horizon sent every C4 update; 0 sends only the "
            "committed C4, while 1.0 provides an Oracle one-second preview"
        ),
    )
    parser.add_argument(
        "--reference_safety",
        choices=("none", "conservative"),
        default="none",
        help="none preserves the exported trajectory exactly apart from 30-to-50 Hz resampling",
    )
    parser.add_argument("--max_joint_velocity_rad_s", default=1.5, type=float)
    parser.add_argument("--max_yaw_velocity_rad_s", default=0.5, type=float)
    parser.add_argument("--ramp_seconds", default=1.0, type=float)
    parser.add_argument("--joint_limit_margin_rad", default=0.05, type=float)
    parser.add_argument(
        "--max_seconds",
        default=0.0,
        type=float,
        help="0 plays the complete file; otherwise play this source-duration prefix",
    )
    return parser.parse_args()


def _slice_motion(motion: G1Motion, start: int, stop: int) -> G1Motion:
    return G1Motion(
        root_rot_xyzw=motion.root_rot_xyzw[start:stop],
        dof_pos_mujoco=motion.dof_pos_mujoco[start:stop],
        fps=motion.fps,
        root_pos=None if motion.root_pos is None else motion.root_pos[start:stop],
    )


def _wxyz_from_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion_xyzw, dtype=np.float32)[[3, 0, 1, 2]]


def _mujoco_from_sonic_dof(dof_sonic: np.ndarray) -> np.ndarray:
    dof_sonic = np.asarray(dof_sonic, dtype=np.float32)
    if dof_sonic.shape != (29,):
        raise ValueError(f"measured SONIC joints must have shape (29,), got {dof_sonic.shape}")
    dof_mujoco = np.empty_like(dof_sonic)
    dof_mujoco[SONIC_REFERENCE_FROM_MUJOCO] = dof_sonic
    return dof_mujoco


def _nlerp_xyzw(left: np.ndarray, right: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if float(np.dot(left, right)) < 0.0:
        right = -right
    values = (1.0 - fractions[:, None]) * left[None] + fractions[:, None] * right[None]
    return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32)


def retime_g1_motion(motion: G1Motion, playback_rate: float) -> G1Motion:
    """Time-scale a fixed trajectory while preserving the 30 Hz wire source contract."""
    playback_rate = float(playback_rate)
    if playback_rate <= 0.0:
        raise ValueError("playback_rate must be positive")
    if np.isclose(playback_rate, 1.0):
        return motion
    output_frames = max(1, int(round(motion.frames / playback_rate)))
    source_position = np.minimum(
        np.arange(output_frames, dtype=np.float64) * playback_rate,
        motion.frames - 1,
    )
    left = np.floor(source_position).astype(np.int64)
    right = np.minimum(left + 1, motion.frames - 1)
    fraction = (source_position - left).astype(np.float32)
    dof = (
        (1.0 - fraction[:, None]) * motion.dof_pos_mujoco[left]
        + fraction[:, None] * motion.dof_pos_mujoco[right]
    ).astype(np.float32)
    quat = np.empty((output_frames, 4), dtype=np.float32)
    for index, (left_index, right_index, amount) in enumerate(
        zip(left, right, fraction)
    ):
        quat[index] = _nlerp_xyzw(
            motion.root_rot_xyzw[left_index],
            motion.root_rot_xyzw[right_index],
            np.asarray([amount], dtype=np.float32),
        )[0]
    root_pos = None
    if motion.root_pos is not None:
        root_pos = (
            (1.0 - fraction[:, None]) * motion.root_pos[left]
            + fraction[:, None] * motion.root_pos[right]
        ).astype(np.float32)
    return G1Motion(
        root_rot_xyzw=quat,
        dof_pos_mujoco=dof,
        fps=motion.fps,
        root_pos=root_pos,
    )


def prepend_feedback_alignment(
    motion: G1Motion,
    feedback: dict,
    *,
    align_seconds: float,
    hold_seconds: float,
) -> tuple[G1Motion, int]:
    """Prepend a measured-state transition; leave the original motion untouched."""
    align_frames = int(round(float(align_seconds) * motion.fps))
    hold_frames = int(round(float(hold_seconds) * motion.fps))
    if align_frames <= 0:
        return motion, 0
    measured_dof = _mujoco_from_sonic_dof(feedback["body_q_measured"])
    measured_quat_wxyz = np.asarray(feedback["base_quat_measured"], dtype=np.float32)
    if measured_quat_wxyz.shape != (4,):
        raise ValueError("base_quat_measured must have shape (4,)")
    measured_quat_xyzw = measured_quat_wxyz[[1, 2, 3, 0]]
    u = np.linspace(0.0, 1.0, align_frames, endpoint=True, dtype=np.float32)
    smooth = u * u * (3.0 - 2.0 * u)
    align_dof = (
        (1.0 - smooth[:, None]) * measured_dof[None]
        + smooth[:, None] * motion.dof_pos_mujoco[0][None]
    ).astype(np.float32)
    align_quat = _nlerp_xyzw(measured_quat_xyzw, motion.root_rot_xyzw[0], smooth)
    prefix_dof = align_dof
    prefix_quat = align_quat
    if hold_frames > 0:
        prefix_dof = np.concatenate(
            (prefix_dof, np.repeat(motion.dof_pos_mujoco[:1], hold_frames, axis=0)), axis=0
        )
        prefix_quat = np.concatenate(
            (prefix_quat, np.repeat(motion.root_rot_xyzw[:1], hold_frames, axis=0)), axis=0
        )
    root_pos = motion.root_pos
    if root_pos is not None:
        prefix_pos = np.repeat(root_pos[:1], prefix_dof.shape[0], axis=0)
        root_pos = np.concatenate((prefix_pos, root_pos), axis=0)
    return (
        G1Motion(
            root_rot_xyzw=np.concatenate((prefix_quat, motion.root_rot_xyzw), axis=0),
            dof_pos_mujoco=np.concatenate((prefix_dof, motion.dof_pos_mujoco), axis=0),
            fps=motion.fps,
            root_pos=root_pos,
        ),
        prefix_dof.shape[0],
    )


def build_offline_reference_packets(
    motion: G1Motion,
    *,
    target_fps: float,
    safety_enabled: bool,
    max_joint_velocity_rad_s: float,
    max_yaw_velocity_rad_s: float,
    ramp_seconds: float,
    joint_limit_margin_rad: float,
    preview_seconds: float = 0.0,
    packet_mode: str = "c4",
):
    """Yield C4-timed SONIC packets, optionally with a sliding future preview."""
    if not np.isclose(motion.fps, G1_FPS, rtol=0.0, atol=1e-4):
        raise ValueError(f"offline SONIC playback requires {G1_FPS:g} Hz, got {motion.fps:g} Hz")
    if motion.frames <= 0:
        raise ValueError("offline motion must have at least one frame")

    if preview_seconds < 0.0:
        raise ValueError("preview_seconds must be non-negative")
    if packet_mode not in ("c4", "full"):
        raise ValueError(f"unsupported packet_mode: {packet_mode}")
    if packet_mode == "full" and preview_seconds != 0.0:
        raise ValueError("full packet mode does not use preview_seconds")
    if packet_mode == "full":
        adapter = SonicReferenceAdapter(
            target_fps=target_fps,
            safety_enabled=safety_enabled,
            max_joint_velocity_rad_s=max_joint_velocity_rad_s,
            max_yaw_velocity_rad_s=max_yaw_velocity_rad_s,
            ramp_seconds=ramp_seconds,
            joint_limit_margin_rad=joint_limit_margin_rad,
        )
        reference_motion = adapter.adapt(
            motion,
            execution_dof_mujoco=motion.dof_pos_mujoco[0],
            execution_quat_wxyz=_wxyz_from_xyzw(motion.root_rot_xyzw[0]),
        )
        fields = reference_fields_from_commit(
            reference_motion,
            start_frame=0,
            stop_frame=reference_motion.frames,
            frame_index_start=0,
        )
        yield 0, motion.frames, reference_motion, fields, dict(adapter.last_diagnostics)
        return
    preview_frames = (
        G1_COMMIT_FRAMES
        if preview_seconds == 0.0
        else max(G1_COMMIT_FRAMES, int(round(preview_seconds * motion.fps)))
    )
    initial_dof = motion.dof_pos_mujoco[0]
    initial_quat = _wxyz_from_xyzw(motion.root_rot_xyzw[0])
    previous_reference_dof = None
    reference_frame_index = 0
    continuous_adapter = None
    if preview_seconds == 0.0:
        continuous_adapter = SonicReferenceAdapter(
            target_fps=target_fps,
            safety_enabled=safety_enabled,
            max_joint_velocity_rad_s=max_joint_velocity_rad_s,
            max_yaw_velocity_rad_s=max_yaw_velocity_rad_s,
            ramp_seconds=ramp_seconds,
            joint_limit_margin_rad=joint_limit_margin_rad,
        )
    for source_start in range(0, motion.frames, G1_COMMIT_FRAMES):
        source_stop = min(source_start + preview_frames, motion.frames)
        # Contiguous C4 packets must share one resampling cursor so 30-to-50 Hz
        # conversion preserves phase (13, 13, 14 frames). Overlapping future
        # previews instead need an independent adapter for each source window.
        adapter = continuous_adapter
        if adapter is None:
            adapter = SonicReferenceAdapter(
                target_fps=target_fps,
                safety_enabled=safety_enabled,
                max_joint_velocity_rad_s=max_joint_velocity_rad_s,
                max_yaw_velocity_rad_s=max_yaw_velocity_rad_s,
                ramp_seconds=ramp_seconds,
                joint_limit_margin_rad=joint_limit_margin_rad,
            )
        reference_motion = adapter.adapt(
            _slice_motion(motion, source_start, source_stop),
            # The offline route intentionally has no physical-state dependency.
            execution_dof_mujoco=initial_dof,
            execution_quat_wxyz=initial_quat,
        )
        fields = reference_fields_from_commit(
            reference_motion,
            start_frame=0,
            stop_frame=reference_motion.frames,
            previous_dof_pos=previous_reference_dof,
            frame_index_start=reference_frame_index,
        )
        committed_stop = min(source_start + G1_COMMIT_FRAMES, motion.frames)
        if preview_seconds == 0.0:
            previous_reference_dof = reference_motion.dof_pos_mujoco[-1].copy()
            reference_frame_index += reference_motion.frames
        else:
            previous_reference_dof = motion.dof_pos_mujoco[committed_stop - 1].copy()
            reference_frame_index = int(round(committed_stop * target_fps / motion.fps))
        yield source_start, source_stop, reference_motion, fields, dict(adapter.last_diagnostics)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _drain_feedback(
    *,
    sonic_subscriber: SonicFeedbackSubscriber,
    sim_state_subscriber: SonicFeedbackSubscriber,
    synchronizer: SonicS66Synchronizer,
    sonic_records: list[dict],
    sim_state_records: list[dict],
    s66_records: list[dict],
) -> None:
    """Record controller telemetry only; never feed it into the reference path."""
    while True:
        sim_state = sim_state_subscriber.poll(timeout_ms=0)
        if sim_state is None:
            break
        sim_state_records.append(jsonable_feedback(sim_state))
        synchronizer.update_sim_state(sim_state)
    while True:
        feedback = sonic_subscriber.poll(timeout_ms=0)
        if feedback is None:
            break
        sonic_records.append(jsonable_feedback(feedback))
        synchronized = synchronizer.update_sonic_feedback(feedback)
        if synchronized is not None:
            s66_records.append(jsonable_feedback(synchronized))


def run(args: argparse.Namespace) -> None:
    if args.max_seconds < 0.0:
        raise ValueError("max_seconds must be non-negative")
    if args.preview_seconds < 0.0:
        raise ValueError("preview_seconds must be non-negative")
    if args.align_from_feedback_seconds < 0.0 or args.align_hold_seconds < 0.0:
        raise ValueError("alignment durations must be non-negative")
    if args.playback_rate <= 0.0:
        raise ValueError("playback_rate must be positive")
    if args.align_from_feedback_seconds > 0.0 and not args.record_feedback:
        raise ValueError("feedback alignment requires --record_feedback")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; choose a new run ID or use "
                "--overwrite_output"
            )
    motion_path = Path(args.pkl).expanduser().resolve()
    motion = load_g1_motion(motion_path)
    if args.root_quat_order == "wxyz":
        motion = G1Motion(
            root_rot_xyzw=motion.root_rot_xyzw[:, [1, 2, 3, 0]],
            dof_pos_mujoco=motion.dof_pos_mujoco,
            fps=motion.fps,
            root_pos=motion.root_pos,
        )
    input_motion_frames = motion.frames
    max_frames = motion.frames
    if args.max_seconds > 0.0:
        max_frames = min(max_frames, int(np.floor(args.max_seconds * motion.fps)))
        if max_frames <= 0:
            raise ValueError("max_seconds is shorter than one source frame")
        motion = _slice_motion(motion, 0, max_frames)
    selected_source_frames = motion.frames
    selected_source_duration_seconds = motion.frames / motion.fps
    motion = retime_g1_motion(motion, args.playback_rate)

    try:
        import zmq
    except ImportError as error:
        raise RuntimeError("offline SONIC playback requires pyzmq") from error

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://*:{int(args.port)}")
    packets: list[dict] = []
    reference_records: list[dict] = []
    emitted_reference_frames = 0
    sonic_records: list[dict] = []
    sim_state_records: list[dict] = []
    s66_records: list[dict] = []
    sonic_subscriber = None
    sim_state_subscriber = None
    synchronizer = None
    alignment_frames = 0
    if args.record_feedback:
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
        synchronizer = SonicS66Synchronizer(fps=G1_FPS)
    try:
        print(f"SONIC offline reference PUB bound to tcp://*:{args.port}", flush=True)
        startup_deadline = time.monotonic() + max(float(args.startup_wait), 0.0)
        while time.monotonic() < startup_deadline:
            if synchronizer is not None:
                _drain_feedback(
                    sonic_subscriber=sonic_subscriber,
                    sim_state_subscriber=sim_state_subscriber,
                    synchronizer=synchronizer,
                    sonic_records=sonic_records,
                    sim_state_records=sim_state_records,
                    s66_records=s66_records,
                )
            time.sleep(0.01)
        if args.align_from_feedback_seconds > 0.0:
            if not sonic_records:
                raise RuntimeError("no SONIC feedback received for initial-pose alignment")
            motion, alignment_frames = prepend_feedback_alignment(
                motion,
                sonic_records[-1],
                align_seconds=args.align_from_feedback_seconds,
                hold_seconds=args.align_hold_seconds,
            )
            print(
                f"Prepended measured-state alignment: {alignment_frames} frames "
                f"({alignment_frames / motion.fps:.3f}s)",
                flush=True,
            )
        started_at = time.monotonic()
        for packet_index, (source_start, source_stop, reference_motion, fields, diagnostics) in enumerate(
            build_offline_reference_packets(
                motion,
                target_fps=args.sonic_reference_fps,
                safety_enabled=args.reference_safety == "conservative",
                max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
                max_yaw_velocity_rad_s=args.max_yaw_velocity_rad_s,
                ramp_seconds=args.ramp_seconds,
                joint_limit_margin_rad=args.joint_limit_margin_rad,
                preview_seconds=args.preview_seconds,
                packet_mode=args.packet_mode,
            ),
            start=1,
        ):
            deadline = started_at + source_start / motion.fps
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                if synchronizer is not None:
                    _drain_feedback(
                        sonic_subscriber=sonic_subscriber,
                        sim_state_subscriber=sim_state_subscriber,
                        synchronizer=synchronizer,
                        sonic_records=sonic_records,
                        sim_state_records=sim_state_records,
                        s66_records=s66_records,
                    )
                time.sleep(min(remaining, 0.01))
            publisher.send(pack_zmq_message(fields, topic=args.topic, version=1))
            sent_at = time.monotonic()
            emitted_reference_frames += reference_motion.frames
            for frame_offset in range(reference_motion.frames):
                reference_records.append(
                    {
                        "frame_index": int(fields["frame_index"][frame_offset]),
                        "intended_time_seconds": float(fields["frame_index"][frame_offset])
                        / float(args.sonic_reference_fps),
                        "packet_index": packet_index,
                        "packet_sent_monotonic_seconds": sent_at,
                        "joint_pos": fields["joint_pos"][frame_offset].tolist(),
                        "joint_vel": fields["joint_vel"][frame_offset].tolist(),
                        "body_quat_wxyz": fields["body_quat_w"][frame_offset].tolist(),
                    }
                )
            packets.append(
                {
                    "packet_index": packet_index,
                    "source_start_frame": source_start,
                    "source_stop_frame": source_stop,
                    "reference_start_frame": int(fields["frame_index"][0]),
                    "reference_frames": reference_motion.frames,
                    "diagnostics": diagnostics,
                }
            )
            if packet_index == 1 or packet_index % 25 == 0:
                print(
                    f"Sent offline {args.packet_mode} packet {packet_index}: "
                    f"source [{source_start},{source_stop}), "
                    f"ref {reference_motion.frames}f@{reference_motion.fps:g}Hz",
                    flush=True,
                )
            if synchronizer is not None:
                _drain_feedback(
                    sonic_subscriber=sonic_subscriber,
                    sim_state_subscriber=sim_state_subscriber,
                    synchronizer=synchronizer,
                    sonic_records=sonic_records,
                    sim_state_records=sim_state_records,
                    s66_records=s66_records,
                )
        # The final packet describes the last C4 interval.  Keep the publisher
        # alive through that interval instead of reporting completion at its
        # send time (which is one C4 early).
        final_deadline = started_at + motion.frames / motion.fps
        while True:
            remaining = final_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if synchronizer is not None:
                _drain_feedback(
                    sonic_subscriber=sonic_subscriber,
                    sim_state_subscriber=sim_state_subscriber,
                    synchronizer=synchronizer,
                    sonic_records=sonic_records,
                    sim_state_records=sim_state_records,
                    s66_records=s66_records,
                )
            time.sleep(min(remaining, 0.01))
        elapsed = time.monotonic() - started_at
        finished_at = time.monotonic()
        print(
            f"Offline stream complete: {motion.frames} source frames "
            f"({motion.frames / motion.fps:.3f}s), {emitted_reference_frames} SONIC frames "
            f"in {elapsed:.3f}s",
            flush=True,
        )
    finally:
        publisher.close()
        context.term()
        if sonic_subscriber is not None:
            sonic_subscriber.close()
        if sim_state_subscriber is not None:
            sim_state_subscriber.close()

    if args.output_dir:
        digest = hashlib.sha256(motion_path.read_bytes()).hexdigest()
        _write_json(
            Path(args.output_dir) / "offline_sonic_playback.json",
            {
                "schema_version": "g1_offline_sonic_playback_v1",
                "motion_path": str(motion_path),
                "motion_sha256": digest,
                "input_root_quat_order": args.root_quat_order,
                "source_frames": motion.frames,
                "source_fps": motion.fps,
                "source_duration_seconds": motion.frames / motion.fps,
                "input_motion_frames": input_motion_frames,
                "selected_source_frames": selected_source_frames,
                "selected_source_duration_seconds": selected_source_duration_seconds,
                "playback_frames": motion.frames,
                "playback_duration_seconds": motion.frames / motion.fps,
                "playback_started_monotonic_seconds": started_at,
                "playback_finished_monotonic_seconds": finished_at,
                "playback_rate": float(args.playback_rate),
                "sonic_reference_fps": args.sonic_reference_fps,
                "preview_seconds": args.preview_seconds,
                "packet_mode": args.packet_mode,
                "alignment_frames": alignment_frames,
                "alignment_seconds": alignment_frames / motion.fps,
                "emitted_reference_frames": emitted_reference_frames,
                "reference_safety": args.reference_safety,
                "feedback_consumed": False,
                "feedback_recorded": bool(args.record_feedback),
                "sonic_feedback_records": len(sonic_records),
                "sim_state_records": len(sim_state_records),
                "s66_records": len(s66_records),
                "root_xy_tracking": False,
                "packets": packets,
            },
        )
        # Future-preview packets overlap. Keep the earliest published value for
        # each execution frame so this file always describes one 50 Hz timeline.
        unique_reference_records = {
            record["frame_index"]: record for record in reversed(reference_records)
        }
        ordered_reference_records = [
            unique_reference_records[index] for index in sorted(unique_reference_records)
        ]
        _write_json(
            Path(args.output_dir) / "reference.json",
            {
                "schema_version": "sonic_reference_timeline_v1",
                "fps": float(args.sonic_reference_fps),
                "records": ordered_reference_records,
            },
        )
        if args.record_feedback:
            output_dir = Path(args.output_dir)
            _write_json(output_dir / "sonic_feedback.json", {"records": sonic_records})
            _write_json(output_dir / "sim_state.json", {"records": sim_state_records})
            _write_json(output_dir / "s66_exec.json", {"records": s66_records})


if __name__ == "__main__":
    run(parse_args())
