#!/usr/bin/env python3
"""Diffusion vs SONIC side-by-side comparison pipeline.

Generates a diffusion motion for a single audio clip, then renders both the
raw diffusion output and the SONIC-executed output as a side-by-side video.

This uses run_full_song_eval's generation logic (which correctly handles
beat conditioning alignment) rather than test.py.

Usage:
  # Step 1: Generate diffusion motion
  conda run -n audiomimic python eval/diffusion_vs_sonic.py \\
    --wav data/edge_aistpp/wavs/gBR_sFM_cAll_d04_mBR5_ch06.wav \\
    --step 1

  # Step 2: Run SONIC and record (3 terminals — see printed instructions)

  # Step 3: Render side-by-side video
  conda run -n audiomimic python eval/diffusion_vs_sonic.py \\
    --wav data/edge_aistpp/wavs/gBR_sFM_cAll_d04_mBR5_ch06.wav \\
    --sonic-pkl eval/diffusion_vs_sonic_results/gBR_sFM_cAll_d04_mBR5_ch06_sonic.pkl \\
    --step 3

  # Step 4: Quantitative metrics comparison
  conda run -n audiomimic python eval/diffusion_vs_sonic.py \\
    --wav data/edge_aistpp/wavs/gBR_sFM_cAll_d04_mBR5_ch06.wav \\
    --sonic-pkl eval/diffusion_vs_sonic_results/gBR_sFM_cAll_d04_mBR5_ch06_sonic.pkl \\
    --step 4
"""

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SONIC_ROOT = Path("/home/tianhup/GR00T-WholeBodyControl")
G1_MODEL_PATH = "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"


# ──────────────────────────────────────────────────────────────────
# Step 1: Generate diffusion motion using run_full_song_eval logic
# ──────────────────────────────────────────────────────────────────

def step1_generate_diffusion(wav_path: str, checkpoint: str, out_dir: str,
                              use_beats: bool = True, beat_rep: str = "distance"):
    """Generate G1 motion from a single wav using the full-song pipeline."""
    import torch
    from EDGE import EDGE
    from eval.run_full_song_eval import (
        prepare_song_features,
        build_condition,
        slice_audio_for_long_generation,
        long_window_count,
        song_frame_count,
        HORIZON_FRAMES,
        STRIDE_FRAMES,
        FPS,
        slice_sort_key,
        clear_dir,
    )
    from model.diffusion import move_cond_to_device

    wav_path = str(Path(wav_path).resolve())
    out_dir = str(Path(out_dir).resolve())
    stem = Path(wav_path).stem

    print(f"\n{'='*60}")
    print(f"  STEP 1: Generating Diffusion motion")
    print(f"  Audio : {wav_path}")
    print(f"  Ckpt  : {checkpoint}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    os.makedirs(out_dir, exist_ok=True)

    # Prepare slices and features
    slice_dir = Path(out_dir) / "slices" / stem
    slice_dir.mkdir(parents=True, exist_ok=True)

    wav_slices, music_cond = prepare_song_features(
        Path(wav_path), slice_dir, "jukebox", output_stem=stem,
    )
    num_slices = len(wav_slices)
    print(f"  Slices: {num_slices}")
    print(f"  Music cond shape: {music_cond.shape}")

    # Build beat conditioning (matching the correct slice count)
    if use_beats:
        from test import build_beat_condition_slices
        beat_cond = build_beat_condition_slices(
            beat_source="audio",
            beat_rep=beat_rep,
            wav_path=wav_path,
            beat_file="",
            total_slices=num_slices,
            start_idx=0,
            num_slices=num_slices,
            fps=FPS,
            horizon=HORIZON_FRAMES,
            stride_frames=STRIDE_FRAMES,
        )
        cond = {"music": music_cond, "beat": beat_cond}
        print(f"  Beat cond shape: {beat_cond.shape}")
    else:
        cond = music_cond

    # Load model
    model = EDGE(
        "jukebox",
        checkpoint,
        use_beats=use_beats,
        beat_rep=beat_rep,
        lambda_beat=0.0,
        motion_format="g1",
    )
    model.eval()

    # Generate
    shape = (num_slices, model.horizon, model.repr_dim)
    cond = move_cond_to_device(cond, model.accelerator.device)
    model.diffusion.render_sample(
        shape,
        cond,
        model.normalizer,
        "fullsong",
        out_dir,
        fk_out=out_dir,
        name=[str(p) for p in wav_slices],
        sound=True,
        mode="long",
        render=False,
        metadata_audio_path=str(wav_path),
        metadata_stride_frames=STRIDE_FRAMES,
        metadata_total_frames=song_frame_count(wav_path),
        g1_fk_model_path=G1_MODEL_PATH,
        g1_root_quat_order="xyzw",
        g1_render_backend="mujoco",
        g1_render_width=960,
        g1_render_height=720,
        g1_mujoco_gl="egl",
    )

    torch.cuda.empty_cache()

    # Find the generated pkl
    candidates = sorted(glob.glob(os.path.join(out_dir, f"*{stem}*g1.pkl")))
    if not candidates:
        raise FileNotFoundError(f"No diffusion pkl found for {stem} in {out_dir}")
    diffusion_pkl = candidates[0]
    print(f"\n✓ Diffusion pkl: {diffusion_pkl}")
    return diffusion_pkl


# ──────────────────────────────────────────────────────────────────
# Step 2: Print SONIC recording instructions
# ──────────────────────────────────────────────────────────────────

def step2_print_sonic_instructions(diffusion_pkl: str, sonic_pkl: str):
    """Print instructions for manually running SONIC recording."""
    print(f"\n{'='*60}")
    print(f"  STEP 2: Record SONIC execution")
    print(f"{'='*60}")
    print(f"""
  This step requires 3 terminals. Run them in order:

  ── Terminal 1: MuJoCo simulator ──
  cd {SONIC_ROOT}
  source .venv_sim/bin/activate  # or conda activate sonic_sim
  python gear_sonic/scripts/run_sim_loop.py

  ── Terminal 2: Deploy node ──
  cd {SONIC_ROOT}/gear_sonic_deploy
  unset ROS_DISTRO RMW_IMPLEMENTATION AMENT_PREFIX_PATH COLCON_PREFIX_PATH
  bash deploy.sh --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose sim

  ── Terminal 3: Stream + Record ──
  cd {SONIC_ROOT}
  source .venv_sim/bin/activate
  python stream_audiomimic.py --pkl {diffusion_pkl}

  After streaming finishes:
  1. The sim loop captures executed qpos in its viewer
  2. Use the recording to extract executed motion as:
     {sonic_pkl}

  ⚠  Or use the automated recorder (if run_sim_record.py is available):
  python gear_sonic/scripts/run_sim_record.py --record_path {sonic_pkl}
  (instead of run_sim_loop.py in Terminal 1)

  Then re-run this script with --sonic-pkl {sonic_pkl} --step 3
""")


# ──────────────────────────────────────────────────────────────────
# Step 3: Render side-by-side video
# ──────────────────────────────────────────────────────────────────

def step3_render_side_by_side(diffusion_pkl: str, sonic_pkl: str,
                               wav_path: str, out_dir: str):
    """Render both motions with g1_visualization and merge with ffmpeg."""
    import subprocess

    print(f"\n{'='*60}")
    print(f"  STEP 3: Rendering side-by-side video")
    print(f"{'='*60}\n")

    os.environ["MUJOCO_GL"] = "egl"
    from eval.g1_visualization import render_g1_motion

    out_dir = str(Path(out_dir).resolve())
    wav_path = str(Path(wav_path).resolve())

    # Render diffusion
    print("  Rendering Diffusion motion...")
    diff_video = render_g1_motion(
        diffusion_pkl, out=out_dir, name=wav_path, sound=True,
        model_path=G1_MODEL_PATH, root_quat_order="xyzw",
        render_backend="mujoco", width=960, height=720,
        output_name="diffusion_raw",
    )
    print(f"  ✓ {diff_video}")

    # Render SONIC
    print("  Rendering SONIC executed motion...")
    sonic_video = render_g1_motion(
        sonic_pkl, out=out_dir, name=wav_path, sound=True,
        model_path=G1_MODEL_PATH, root_quat_order="xyzw",
        render_backend="mujoco", width=960, height=720,
        output_name="sonic_executed",
    )
    print(f"  ✓ {sonic_video}")

    # Merge side-by-side
    stem = Path(wav_path).stem
    merged = os.path.join(out_dir, f"{stem}_side_by_side.mp4")

    filter_str = (
        "[0:v]drawtext=text='Diffusion (Reference)'"
        ":fontcolor=white:fontsize=36:x=10:y=10"
        ":box=1:boxcolor=black@0.5:boxborderw=5[left];"
        "[1:v]drawtext=text='SONIC (Executed)'"
        ":fontcolor=cyan:fontsize=36:x=10:y=10"
        ":box=1:boxcolor=black@0.5:boxborderw=5[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    print(f"  Merging → {merged}")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", diff_video, "-i", sonic_video,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "0:a?",
        "-shortest",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        merged,
    ], check=True, capture_output=True)

    print(f"\n✓ Side-by-side video: {merged}")
    return merged


# ──────────────────────────────────────────────────────────────────
# Step 4: Quantitative comparison
# ──────────────────────────────────────────────────────────────────

def step4_compare_metrics(diffusion_pkl: str, sonic_pkl: str,
                          wav_path: str, out_dir: str):
    """Compute FK metrics for both and print a comparison table."""
    print(f"\n{'='*60}")
    print(f"  STEP 4: Quantitative Comparison")
    print(f"{'='*60}\n")

    from eval.g1_metrics import (
        load_g1_motion,
        evaluate_g1_fk_metrics,
        summarize_g1_motion,
    )

    ref = load_g1_motion(diffusion_pkl)
    exe = load_g1_motion(sonic_pkl)

    wav_path = str(Path(wav_path).resolve())
    ref["audio_path"] = wav_path
    exe["audio_path"] = wav_path

    ref_summary = summarize_g1_motion(ref)
    exe_summary = summarize_g1_motion(exe)
    ref_fk = evaluate_g1_fk_metrics(ref, G1_MODEL_PATH, root_quat_order="xyzw")
    exe_fk = evaluate_g1_fk_metrics(exe, G1_MODEL_PATH, root_quat_order="xyzw")

    print(f"  {'Metric':<30} | {'Diffusion':>12} | {'SONIC':>12} | {'Δ':>12}")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    rows = [
        ("Frames", ref_summary["frames"], exe_summary["frames"]),
        ("Root Height Mean", ref_summary["root_height_mean"], exe_summary["root_height_mean"]),
        ("Root Drift", ref_summary["root_drift"], exe_summary["root_drift"]),
        ("Root Vel Mean", ref_summary["root_velocity_mean"], exe_summary["root_velocity_mean"]),
        ("Joint Vel Mean", ref_summary["joint_velocity_mean"], exe_summary["joint_velocity_mean"]),
        ("Root Jerk Mean", ref_summary["root_smoothness_jerk_mean"], exe_summary["root_smoothness_jerk_mean"]),
        ("Joint Jerk Mean", ref_summary["joint_smoothness_jerk_mean"], exe_summary["joint_smoothness_jerk_mean"]),
    ]
    if ref_fk and exe_fk:
        rows.extend([
            ("G1FKBAS", ref_fk["G1FKBAS"], exe_fk["G1FKBAS"]),
            ("G1FKRoboPerformBAS", ref_fk["G1FKRoboPerformBAS"], exe_fk["G1FKRoboPerformBAS"]),
            ("G1BeatF1", ref_fk["G1BeatF1"], exe_fk["G1BeatF1"]),
            ("G1FootSliding", ref_fk["G1FootSliding"], exe_fk["G1FootSliding"]),
            ("G1GroundPenetration", ref_fk["G1GroundPenetration"], exe_fk["G1GroundPenetration"]),
            ("G1FootClearanceMean", ref_fk["G1FootClearanceMean"], exe_fk["G1FootClearanceMean"]),
        ])

    for name, rv, ev in rows:
        delta = ev - rv
        print(f"  {name:<30} | {rv:>12.4f} | {ev:>12.4f} | {delta:>+12.4f}")

    print()

    def _convert(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_convert(v) for v in obj]
        return obj

    report_path = os.path.join(out_dir, "comparison_metrics.json")
    report = {
        "diffusion": {"summary": ref_summary, "fk": ref_fk},
        "sonic":     {"summary": exe_summary, "fk": exe_fk},
    }
    with open(report_path, "w") as f:
        json.dump(_convert(report), f, indent=2)
    print(f"  ✓ Metrics saved: {report_path}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diffusion vs SONIC comparison")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--checkpoint",
                        default="runs/train/g1_lbeat_relative_finetune/weights/train-500.pt")
    parser.add_argument("--out", default="eval/diffusion_vs_sonic_results")
    parser.add_argument("--diffusion-pkl", default=None,
                        help="Pre-generated diffusion pkl (skip step 1)")
    parser.add_argument("--sonic-pkl", default=None,
                        help="Pre-recorded SONIC pkl (skip step 2)")
    parser.add_argument("--step", type=int, default=0,
                        help="Run specific step: 1=generate, 2=print SONIC instructions, "
                             "3=render, 4=metrics, 0=all")
    args = parser.parse_args()

    out_dir = str(Path(args.out).resolve())
    os.makedirs(out_dir, exist_ok=True)
    wav_path = str(Path(args.wav).resolve())
    stem = Path(wav_path).stem

    # Resolve diffusion pkl
    diffusion_pkl = args.diffusion_pkl
    if args.step in (0, 1) and diffusion_pkl is None:
        diffusion_pkl = step1_generate_diffusion(
            wav_path, args.checkpoint, out_dir)
    elif diffusion_pkl is None:
        candidates = sorted(glob.glob(os.path.join(out_dir, f"*{stem}*g1.pkl")))
        if candidates:
            diffusion_pkl = candidates[0]
            print(f"  Using existing diffusion pkl: {diffusion_pkl}")

    # Resolve sonic pkl
    sonic_pkl = args.sonic_pkl
    sonic_pkl_default = os.path.join(out_dir, f"{stem}_sonic.pkl")

    if args.step in (0, 2):
        if diffusion_pkl:
            step2_print_sonic_instructions(diffusion_pkl, sonic_pkl_default)
        if sonic_pkl is None and not os.path.exists(sonic_pkl_default):
            if args.step == 0:
                print("  ⚠  Skipping steps 3-4: no SONIC pkl available yet.")
                print(f"     After recording, re-run with --sonic-pkl <path> --step 3")
                return

    if sonic_pkl is None and os.path.exists(sonic_pkl_default):
        sonic_pkl = sonic_pkl_default

    if args.step in (0, 3) and diffusion_pkl and sonic_pkl:
        step3_render_side_by_side(diffusion_pkl, sonic_pkl, wav_path, out_dir)

    if args.step in (0, 4) and diffusion_pkl and sonic_pkl:
        step4_compare_metrics(diffusion_pkl, sonic_pkl, wav_path, out_dir)

    print(f"\n{'='*60}")
    print(f"  Results in: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
