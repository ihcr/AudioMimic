"""
Asynchronous Streaming Inference for AudioMimic G1 motions.
Generates overlapping chunks of motion via DDIM Inpainting and streams them
over ZMQ to SONIC via Protocol v1 at a strict 30 FPS.
"""

import argparse
import glob
import json
import os
import queue
import sys
import threading
import time
from functools import cmp_to_key
from tempfile import TemporaryDirectory

import numpy as np
import torch
import zmq

from args import parse_test_opt
from data.slice import slice_audio
from dataset.motion_representation import decode_g1_motion

# IsaacLab mapping for ZMQ:
mujoco_to_isaaclab = [
    0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4,  10,
    16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
]

def pack_zmq_message(fields_dict: dict, topic: str = "pose", version: int = 1) -> bytes:
    fields_meta = []
    binary_parts = []
    HEADER_SIZE = 1280

    for name, arr in fields_dict.items():
        arr = np.ascontiguousarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            dtype_str = "f32" if arr.dtype == np.float32 else "f64"
        elif np.issubdtype(arr.dtype, np.integer):
            dtype_str = "i32" if arr.dtype == np.int32 else "i64"
        else:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        fields_meta.append({
            "name": name,
            "dtype": dtype_str,
            "shape": list(arr.shape),
        })

        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        binary_parts.append(arr.tobytes())

    header = {
        "v": version,
        "endian": "le",
        "count": 1,
        "fields": fields_meta,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes = header_json.ljust(HEADER_SIZE, b"\x00")

    return topic.encode("utf-8") + header_bytes + b"".join(binary_parts)


def consumer_thread(pose_queue, port, topic, fps=30.0, wav_file=None):
    """ZMQ streaming thread. Consumes poses and sends them to SONIC."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    print(f"[Consumer] ZMQ PUB bound on tcp://*:{port}, topic='{topic}'")

    print("[Consumer] Waiting for first frame...")
    frame_data = pose_queue.get()
    if frame_data is None:
        return
        
    if wav_file:
        import subprocess
        # Play audio in background
        subprocess.Popen(["ffplay", "-nodisp", "-autoexit", wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    dt = 1.0 / fps
    global_frame = 0
    last_dof_pos = None
    start_time = time.monotonic()

    while frame_data is not None:

        dof_pos = frame_data["dof_pos"]    # [29]
        root_rot = frame_data["root_rot"]  # [4]

        # Velocity finite difference
        if last_dof_pos is None:
            joint_vel = np.zeros_like(dof_pos)
        else:
            joint_vel = (dof_pos - last_dof_pos) / dt
        last_dof_pos = dof_pos

        # Mapping joints: MuJoCo -> IsaacLab order for SONIC encoder
        dof_pos_isaac = dof_pos[mujoco_to_isaaclab]
        joint_vel_isaac = joint_vel[mujoco_to_isaaclab]

        # Convert xyzw to wxyz if needed
        if abs(root_rot[3]) > abs(root_rot[0]):
            root_rot = root_rot[[3, 0, 1, 2]]

        fields = {
            "joint_pos":   dof_pos_isaac[np.newaxis, :],             # [1, 29]
            "joint_vel":   joint_vel_isaac[np.newaxis, :],           # [1, 29]
            "body_quat":   root_rot[np.newaxis, :],                  # [1, 4]
            "frame_index": np.array([global_frame], dtype=np.int32), # [1]
        }

        msg = pack_zmq_message(fields, topic=topic, version=1)
        sock.send(msg)
        global_frame += 1

        if global_frame % 30 == 0:
            print(f"[Consumer] Sent frame {global_frame} t={global_frame/fps:.1f}s")

        # Maintain strict FPS using global time to prevent drift
        target_time = start_time + global_frame * dt
        current_time = time.monotonic()
        remaining = target_time - current_time
        if remaining > 0:
            time.sleep(remaining)

        # Fetch next frame (None = EOF)
        frame_data = pose_queue.get()

    print("[Consumer] Playback finished.")
    sock.close()
    ctx.term()


@torch.no_grad()
def ddim_inpaint(model, shape, cond, constraint, sampling_timesteps=50):
    """DDIM loop with constraint injection to speed up inpainting (50 steps vs 1000 steps)."""
    device = model.accelerator.device
    diffusion = model.diffusion
    total_timesteps = diffusion.n_timestep
    eta = 1

    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    x = torch.randn(shape, device=device)
    
    # We must properly move cond to device
    if isinstance(cond, dict):
        cond = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond.items()}
    else:
        cond = cond.to(device)

    mask = constraint["mask"]
    value = constraint["value"]
    
    # Generate fixed noise for the constraint region to maintain coherence across steps
    noise = torch.randn_like(value)

    from tqdm import tqdm
    for time_idx, time_next_idx in tqdm(time_pairs, desc="DDIM Inpainting", leave=False):
        # 1. Enforce constraint: blend noisy value and x
        if time_idx >= 0:
            time_tensor = torch.full((shape[0],), time_idx, device=device, dtype=torch.long)
            noisy_value = diffusion.q_sample(value, time_tensor, noise=noise)
            x = noisy_value * mask + (1.0 - mask) * x

        # 2. Predict x_start
        time_cond = torch.full((shape[0],), time_idx, device=device, dtype=torch.long)
        pred_noise, x_start, *_ = diffusion.model_predictions(x, cond, time_cond, clip_x_start=diffusion.clip_denoised)

        # 3. Step backward in time
        if time_next_idx < 0:
            x = x_start
            continue

        alpha = diffusion.alphas_cumprod[time_idx]
        alpha_next = diffusion.alphas_cumprod[time_next_idx]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        step_noise = torch.randn_like(x)

        x = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * step_noise

    # Final enforcement at t=0
    x = value * mask + (1.0 - mask) * x
    return x


def producer_thread(opt, pose_queue, precache_features=False):
    """Diffusion generation thread."""
    from EDGE import EDGE as edge_cls
    from data.audio_extraction.baseline_features import extract as baseline_extract
    from data.audio_extraction.jukebox_features import extract as juke_extract

    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract

    print("[Producer] Loading model...")
    model = edge_cls(
        opt.feature_type,
        opt.checkpoint,
        use_beats=opt.use_beats,
        beat_rep=opt.beat_rep,
        lambda_beat=0.0,
        motion_format=opt.motion_format,
    )
    model.eval()

    device = model.accelerator.device
    horizon = model.horizon          # typically 150 frames (5 seconds at 30fps)
    stride_frames = horizon // 2     # typically 75 frames (2.5 seconds at 30fps)

    wav_files = glob.glob(os.path.join(opt.music_dir, "*.wav"))
    if not wav_files:
        print(f"[Producer] ERROR: No .wav files found in {opt.music_dir}")
        pose_queue.put(None)
        return
    wav_file = wav_files[0]

    print(f"[Producer] Slicing audio {wav_file}...")
    temp_dir = TemporaryDirectory()
    # 2.5s stride, 5.0s window
    slice_audio(wav_file, 2.5, 5.0, temp_dir.name)

    def stringintcmp_(a, b):
        ka = int(os.path.splitext(a)[0].split("_")[-1].split("slice")[-1])
        kb = int(os.path.splitext(b)[0].split("_")[-1].split("slice")[-1])
        if ka < kb: return -1
        if ka > kb: return 1
        return 0
    stringintkey = cmp_to_key(stringintcmp_)

    file_list = sorted(glob.glob(f"{temp_dir.name}/*.wav"), key=stringintkey)
    previous_half = None
    prev_decoded_dof = None  # previous chunk's decoded second half [stride_frames, 29]
    prev_decoded_rot = None  # previous chunk's decoded second half [stride_frames, 4]

    print(f"[Producer] Found {len(file_list)} chunks to process.")

    # 1. Extract Beat conditions for the whole audio
    beat_cond_all = None
    if opt.use_beats:
        import test as test_script
        print(f"[Producer] Extracting beat features from {wav_file}...")
        beat_cond_all = test_script.build_beat_condition_slices(
            beat_source=opt.beat_source,
            beat_rep=opt.beat_rep,
            wav_path=wav_file,
            beat_file=opt.beat_file,
            total_slices=len(file_list),
            start_idx=0,
            num_slices=len(file_list),
        )

    # Pre-cache Jukebox features if requested (removes feature extraction from the real-time loop)
    cached_features = None
    if precache_features:
        print(f"[Producer] Pre-caching {opt.feature_type} features for all {len(file_list)} slices...")
        cached_features = []
        for j, sf in enumerate(file_list):
            t0 = time.time()
            reps, _ = feature_func(sf)
            cached_features.append(reps)
            print(f"[Producer]   Feature {j+1}/{len(file_list)} extracted ({time.time()-t0:.1f}s)")
        print(f"[Producer] All features cached. Starting real-time DDIM generation.")

    for i, slice_file in enumerate(file_list):
        t_start = time.time()
        print(f"\n[Producer] ---> Processing chunk {i+1}/{len(file_list)} <---")
        
        # 2. Extract (or load cached) music features for this chunk
        if cached_features is not None:
            reps = cached_features[i]
        else:
            reps, _ = feature_func(slice_file)
        music_cond = torch.from_numpy(np.array([reps])).float().to(device)
        
        if opt.use_beats:
            beat_cond = beat_cond_all[i:i+1].to(device)
            cond = {"music": music_cond, "beat": beat_cond}
        else:
            cond = music_cond
        
        shape = (1, horizon, model.repr_dim)
        
        # 2. Denoise
        if previous_half is None:
            # First chunk: unconstrained DDIM
            print("[Producer] Running standard DDIM for first chunk...")
            samples = model.diffusion.ddim_sample(shape, cond)
        else:
            # Subsequent chunks: Inpaint to enforce overlap constraint
            print("[Producer] Running Inpaint DDIM to match overlap...")
            mask = torch.zeros(shape, device=device)
            mask[:, :stride_frames, :] = 1.0  # constrain first half
            
            value = torch.zeros(shape, device=device)
            value[:, :stride_frames, :] = previous_half
            
            constraint = {"mask": mask, "value": value}
            samples = ddim_inpaint(model, shape, cond, constraint=constraint)

        # 3. Unnormalize and decode
        samples_cpu = samples.detach().cpu()
        samples_unnorm = model.normalizer.unnormalize(samples_cpu)
        decoded = decode_g1_motion(samples_unnorm)
        
        root_rot = decoded["root_rot"][0] # [150, 4]
        dof_pos = decoded["dof_pos"][0]   # [150, 29]
        
        # 3.5 Crossfade: blend overlap region with previous chunk's decoded second half
        if prev_decoded_dof is not None:
            alpha = torch.linspace(1.0, 0.0, stride_frames)[:, None]  # [75, 1] fade from prev→curr
            # Blend DOF positions
            dof_pos[:stride_frames] = (
                alpha * prev_decoded_dof + (1.0 - alpha) * dof_pos[:stride_frames]
            )
            # Blend root rotation (lerp + normalize, adequate for small differences)
            root_rot[:stride_frames] = (
                alpha * prev_decoded_rot + (1.0 - alpha) * root_rot[:stride_frames]
            )
            root_rot[:stride_frames] = torch.nn.functional.normalize(
                root_rot[:stride_frames], dim=-1
            )
        
        # Save decoded second half for next chunk's crossfade
        prev_decoded_dof = dof_pos[stride_frames:].clone()
        prev_decoded_rot = root_rot[stride_frames:].clone()

        # 4. Save second half as constraint for the next chunk
        previous_half = samples[:, stride_frames:, :].to(device)
        
        # 5. Determine which frames to stream
        if i == len(file_list) - 1:
            # Last chunk: stream the entire chunk (we don't drop the second half)
            frames_to_push = range(horizon)
        else:
            # Normal chunk: stream only the first half (since the second half will be inpainted & pushed next)
            frames_to_push = range(stride_frames)

        # Push to Queue
        for f in frames_to_push:
            pose_queue.put({
                "dof_pos": dof_pos[f].numpy().astype(np.float32),
                "root_rot": root_rot[f].numpy().astype(np.float32)
            })

        elapsed = time.time() - t_start
        print(f"[Producer] Chunk {i+1} completed in {elapsed:.2f}s. Pushed {len(frames_to_push)} frames to stream.")

    # Signal completion
    pose_queue.put(None)
    temp_dir.cleanup()
    print("[Producer] Finished generating all chunks.")


if __name__ == "__main__":
    # 1. Parse custom args for ZMQ
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument("--port", type=int, default=5556)
    custom_parser.add_argument("--topic", default="pose")
    custom_parser.add_argument("--precache_features", action="store_true",
                               help="Pre-extract all Jukebox features before starting real-time DDIM generation")
    custom_args, remaining_argv = custom_parser.parse_known_args()

    # 2. Patch sys.argv so parse_test_opt doesn't fail on --port and --topic
    sys.argv = [sys.argv[0]] + remaining_argv
    opt = parse_test_opt()
    
    # 3. Thread-safe queue for frames
    pose_queue = queue.Queue()

    # Find wav file
    wav_files = glob.glob(os.path.join(opt.music_dir, "*.wav"))
    wav_file = wav_files[0] if wav_files else None

    # 4. Start Consumer Thread (ZMQ Streaming)
    consumer = threading.Thread(
        target=consumer_thread, 
        args=(pose_queue, custom_args.port, custom_args.topic, 30.0, wav_file),
        daemon=True
    )
    consumer.start()

    # 5. Start Producer Thread (Diffusion Generation) in main thread
    producer_thread(opt, pose_queue, precache_features=custom_args.precache_features)

    # Wait for consumer to finish playing
    consumer.join()
