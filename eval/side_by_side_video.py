import os
import sys
import time
import subprocess
import signal
from pathlib import Path

def main(wav_path, checkpoint, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    wav_path = os.path.abspath(wav_path)
    checkpoint = os.path.abspath(checkpoint)
    
    # 1. Generate Diffusion Motion
    print("=== 1. Generating Diffusion Motion ===")
    subprocess.run([
        sys.executable, "test.py",
        "--music_dir", os.path.dirname(wav_path),
        "--checkpoint", checkpoint,
        "--feature_type", "jukebox",
        "--use_beats",
        "--beat_rep", "distance",
        "--beat_source", "audio",
        "--motion_format", "g1",
        "--save_motions",
        "--motion_save_dir", output_dir,
        "--no_render"
    ], check=True)
    
    # Find the generated pkl
    wav_name = os.path.basename(wav_path).replace(".wav", "")
    diffusion_pkl = os.path.join(output_dir, f"0_0_{wav_name}_g1.pkl")
    if not os.path.exists(diffusion_pkl):
        raise RuntimeError(f"Diffusion failed to generate {diffusion_pkl}")

    sonic_pkl = os.path.join(output_dir, f"{wav_name}_sonic_executed.pkl")

    # 2. Start SONIC sim and Deploy node
    print("=== 2. Starting SONIC Simulation and Deploy Node ===")
    # disable onscreen to make headless rendering faster
    sim_proc = subprocess.Popen(
        ["../GR00T-WholeBodyControl/.venv_sim/bin/python", "gear_sonic/scripts/run_sim_record.py", "--record_path", sonic_pkl],
        cwd="/home/tianhup/GR00T-WholeBodyControl"
    )
    deploy_env = os.environ.copy()
    for k in ["ROS_DISTRO", "RMW_IMPLEMENTATION", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"]:
        deploy_env.pop(k, None)
    
    deploy_proc = subprocess.Popen(
        ["bash", "deploy.sh", "--input-type", "zmq", "--zmq-host", "localhost", "--zmq-port", "5556", "--zmq-topic", "pose", "sim"],
        cwd="/home/tianhup/GR00T-WholeBodyControl/gear_sonic_deploy",
        env=deploy_env
    )
    
    print("Waiting 10s for simulation to initialize and drop to ground...")
    time.sleep(10)
    
    # 3. Stream motion
    print("=== 3. Streaming Motion to SONIC ===")
    subprocess.run([
        "../GR00T-WholeBodyControl/.venv_sim/bin/python", "stream_audiomimic.py",
        "--pkl", diffusion_pkl
    ], cwd="/home/tianhup/GR00T-WholeBodyControl")
    
    print("=== 4. Stopping Simulation ===")
    sim_proc.send_signal(signal.SIGINT)
    deploy_proc.terminate()
    sim_proc.wait()
    deploy_proc.wait()
    
    if not os.path.exists(sonic_pkl):
        raise RuntimeError(f"SONIC failed to generate {sonic_pkl}")

    # 5. Render both motions side by side
    print("=== 5. Rendering Motions ===")
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    
    diff_mp4 = os.path.join(output_dir, "diff.mp4")
    sonic_mp4 = os.path.join(output_dir, "sonic.mp4")
    
    # Render Diffusion
    subprocess.run([
        ".venv311/bin/python", "-c",
        "from eval.g1_visualization import render_g1_motion; render_g1_motion("
        f"'{diffusion_pkl}', out='{output_dir}', name='{wav_path}', sound=True, "
        "model_path='third_party/unitree_g1_description/g1_29dof_rev_1_0.xml', "
        "root_quat_order='xyzw', render_backend='mujoco', width=960, height=720)"
    ], env=env, check=True)
    os.replace(os.path.join(output_dir, f"{wav_name}.mp4"), diff_mp4)
    
    # Render SONIC
    # Note: SONIC output might be slightly longer or shorter, visualization handles it.
    subprocess.run([
        sys.executable, "-c",
        "from eval.g1_visualization import render_g1_motion; render_g1_motion("
        f"'{sonic_pkl}', out='{output_dir}', name='{wav_path}', sound=True, "
        "model_path='third_party/unitree_g1_description/g1_29dof_rev_1_0.xml', "
        "root_quat_order='xyzw', render_backend='mujoco', width=960, height=720)"
    ], env=env, check=True)
    os.replace(os.path.join(output_dir, f"{wav_name}.mp4"), sonic_mp4)
    
    # Merge side by side
    merged_mp4 = os.path.join(output_dir, f"{wav_name}_side_by_side.mp4")
    
    # Draw text: Left Diffusion, Right SONIC
    # using complex filter
    filter_str = (
        "[0:v]drawtext=text='Diffusion (Reference)':fontcolor=white:fontsize=48:x=10:y=10:box=1:boxcolor=black@0.5[left];"
        "[1:v]drawtext=text='SONIC (Executed)':fontcolor=white:fontsize=48:x=10:y=10:box=1:boxcolor=black@0.5[right];"
        "[left][right]hstack=inputs=2[v];"
        "[0:a]anull[a]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", diff_mp4, "-i", sonic_mp4,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-shortest", # stop when the shortest stream ends
        merged_mp4
    ], check=True)
    
    print(f"=== DONE! Output saved to {merged_mp4} ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--checkpoint", default="runs/train/g1_lbeat_relative_finetune/weights/train-500.pt")
    parser.add_argument("--out", default="eval/side_by_side_results")
    args = parser.parse_args()
    main(args.wav, args.checkpoint, args.out)
