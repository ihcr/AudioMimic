import time
import zmq
import numpy as np
import pickle
import torch

# 从 SONIC 移植过来的 pack 函数
def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 4) -> bytes:
    import json
    import struct
    
    HEADER_SIZE = 1280
    fields = []
    binary_data = []

    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            dtype_str = "f32" if value.dtype == np.float32 else "f64"
            fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
            
            if not value.flags["C_CONTIGUOUS"]:
                value = np.ascontiguousarray(value)
            if value.dtype.byteorder == ">":
                value = value.astype(value.dtype.newbyteorder("<"))
            binary_data.append(value.tobytes())

    header = {
        "v": version,
        "endian": "le",
        "count": 1,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes = header_json.ljust(HEADER_SIZE, b"\x00")
    
    return topic.encode("utf-8") + header_bytes + b"".join(binary_data)

def stream_motion(pkl_path, port=5555, fps=30):
    # 1. 启动 ZMQ Publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    print(f"ZMQ Publisher bound to port {port}")
    time.sleep(1) # 等待连接建立

    # 2. 读取生成的动作
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {pkl_path}")
        return
        
    root_pos = data["root_pos"]  # [T, 3]
    root_rot = data["root_rot"]  # [T, 4] quaternion
    dof_pos = data["dof_pos"]    # [T, 29]
    
    num_frames = root_pos.shape[0]
    print(f"Loaded motion with {num_frames} frames. Streaming at {fps} FPS...")

    # 3. 按帧率流式发送
    for i in range(num_frames):
        start_time = time.time()
        
        # 准备数据包，确保是 float32
        pose_data = {
            "root_pos": root_pos[i].numpy().astype(np.float32),
            "root_rot": root_rot[i].numpy().astype(np.float32),
            "dof_pos": dof_pos[i].numpy().astype(np.float32)
        }
        
        # 打包并发送
        msg = pack_pose_message(pose_data, topic="pose", version=4)
        socket.send(msg)
        
        # 控制帧率
        elapsed = time.time() - start_time
        sleep_time = max(0, (1.0 / fps) - elapsed)
        time.sleep(sleep_time)
        
        if i % 30 == 0:
            print(f"Sent frame {i}/{num_frames}")

    print("Stream finished.")
    socket.close()
    context.term()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stream motion via ZMQ to SONIC")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the generated .pkl file")
    parser.add_argument("--fps", type=int, default=30, help="Streaming FPS")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ Publisher port")
    args = parser.parse_args()

    stream_motion(args.pkl, port=args.port, fps=args.fps)
