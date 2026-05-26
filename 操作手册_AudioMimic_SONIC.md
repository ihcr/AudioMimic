# AudioMimic → SONIC Sim2Sim 操作手册

> 将 AudioMimic 生成的 G1 舞蹈动作通过 ZMQ 流式传输到 SONIC WBC 策略，在 MuJoCo 仿真中执行。

---

## 0. 前置条件

- AudioMimic 已生成 G1 格式的 `.pkl` 动作文件（位于 `~/AudioMimic/eval/g1_motions/`）
- SONIC 模型权重已下载（运行过 `python download_from_hf.py`）
- C++ deploy 已编译成功

---

## 1. 首次编译（只需一次）

> ⚠️ 关键：编译时必须 **unset ROS2 环境变量**，否则 CycloneDDS 会冲突。

```bash
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT=/usr
unset ROS_DISTRO RMW_IMPLEMENTATION AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
export CMAKE_PREFIX_PATH="/opt/onnxruntime:/usr/lib/x86_64-linux-gnu/cmake"
rm -rf build
just build
```

编译成功后 `target/release/g1_deploy_onnx_ref` 可执行文件就生成了。

---

## 2. 下载模型权重（只需一次）

```bash
cd ~/GR00T-WholeBodyControl
pip install huggingface_hub   # 如果没装过
python download_from_hf.py
```

验证文件：
```bash
ls -la ~/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/
# 应该有: model_decoder.onnx, model_encoder.onnx, observation_config.yaml

ls -la ~/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/
# 应该有: planner_sonic.onnx
```

---

## 2.5 可用 AudioMimic 模型一览

| 模型 | Checkpoint 路径 | 说明 |
|------|-----------------|------|
| **lbeat fine-tune (推荐)** | `runs/train/g1_lbeat_relative_finetune/weights/train-500.pt` | 在 BeatDistance 基线上 fine-tune，**增加了 beat 对齐 loss**，节拍卡点更精准。节奏向 checkpoint。 |
| BeatDistance 基线 | `runs/train/g1_aist_beatdistance_featurecache/weights/train-2000.pt` | 原始 BeatDistance 训练的稳定基线模型。 |

> 两个模型共享相同的推理接口（输入/输出 tensor shape `[T, 38]`），运行参数完全一致（`--feature_type jukebox --beat_rep distance`），只需替换 `--checkpoint` 路径即可切换。

如需下载新模型：
```bash
mkdir -p ~/AudioMimic/runs/train/g1_lbeat_relative_finetune/weights
wget -O ~/AudioMimic/runs/train/g1_lbeat_relative_finetune/weights/train-500.pt \
  "https://huggingface.co/wyksdsg/edge-g1-beatdistance/resolve/main/lbeat_relative_finetune/train-500.pt"
```

---

## 3. 启动流程（需要 3 个终端）

### Terminal 1 — MuJoCo 仿真器

```bash
cd ~/GR00T-WholeBodyControl && source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py
```

等待 MuJoCo 窗口打开，看到 G1 机器人模型。

### Terminal 2 — SONIC WBC 策略（C++ deploy）

```bash
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && \
export TensorRT_ROOT=/usr && \
unset ROS_DISTRO RMW_IMPLEMENTATION AMENT_PREFIX_PATH COLCON_PREFIX_PATH && \
./target/release/g1_deploy_onnx_ref lo \
  policy/release/model_decoder.onnx \
  reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq \
  --zmq-host localhost \
  --zmq-port 5556 \
  --zmq-topic pose \
  --output-type all \
  --disable-crc-check
```

> ⚠️ 首次运行会将 planner ONNX 转换为 TensorRT 引擎，需要 **3-5 分钟**，不要中断！
> 转换完成后会生成 `.trt` 缓存文件，下次启动会很快。

### Terminal 3 — AudioMimic 动作流（离线回放模式）

```bash
cd ~/GR00T-WholeBodyControl && source .venv_sim/bin/activate && \
python stream_audiomimic.py \
  --pkl ~/AudioMimic/eval/g1_motions/test_0_test_beat_g1.pkl \
  --loop
```

> 替换 `--pkl` 路径为你要播放的动作文件。

### Terminal 3 (替代方案) — AudioMimic 实时生成与流传输（在线推理模式）

如果你不想提前生成 `.pkl`，可以直接运行实时流式推理。它会一边用 GPU 生成动作一边实时发送给 SONIC（注意必须使用 `audiomimic` conda 环境）：

```bash
cd ~/AudioMimic && conda activate audiomimic && \
python stream_inference.py \
  --music_dir custom_music/ \
  --checkpoint runs/train/g1_lbeat_relative_finetune/weights/train-500.pt \
  --feature_type jukebox \
  --motion_format g1 \
  --use_beats --beat_rep distance --beat_source audio \
  --port 5556 \
  --precache_features
```

> 💡 也可以替换为基线模型 `runs/train/g1_aist_beatdistance_featurecache/weights/train-2000.pt`，两者接口完全兼容。

> ⚠️ 注意：两个 checkpoint 都**仅兼容** Jukebox 特征和 BeatDistance 的节拍表示，运行参数必须保持 `--feature_type jukebox` 与 `--beat_rep distance`，否则会导致模型不兼容或效果极差。

> ⚠️ 注意：添加 `--precache_features` 参数会在开始前一次性提取所有的 Jukebox 特征（需要一定时间）。提取完成后，DDIM 生成与 30 FPS 播放将同步进行，避免动作卡顿。如果不加该参数，则为完全实时流式模式，但由于 Jukebox 提取耗时较长，动作播放会逐渐落后于音频。

> 💡 **关于流式生成的平滑性与对齐优化：**
> 在流式推理中，动作是按 **5 秒的滑动窗口（Window）** 生成的，每次步进 **2.5 秒（Stride）**。为了保证动作连续且与音乐完美对齐，系统内置了以下优化：
> 1. **Crossfade 线性过渡混合**：在两个相邻 Chunk 间存在 2.5 秒的重叠区域（Overlap）。系统会对该区域的关节位置（DOF）进行线性插值，对根节点旋转（Root Quaternion）进行平滑过渡，消除分段拼接处的硬切和动作跳变。
> 2. **`--precache_features` 预加载**：剥离耗时的 Jukebox 特征提取，使得 DDIM 扩散模型能以足够的速度实时产出动作，避免缓冲区排空导致的画面定格和音乐错位。
> 3. **首帧对齐修复**：修复了 ZMQ Consumer 吞掉第一帧导致的持续性 33ms 时序偏移。

> 💡 **关于帧率（FPS）与控制频率：**
> AudioMimic 生成的参考动作和流式传输频率固定为 **30 FPS**，而 SONIC 控制器底层的实际控制频率远高于此（通常为几百 Hz）。SONIC 会在内部自动对 30 FPS 的参考轨迹进行平滑插值，因此参考帧率较低不会影响机器人底层物理控制的平滑度。

---

## 4. 操作顺序（在 Terminal 2 中按键）

| 步骤 | 按键 | 说明 |
|------|------|------|
| ① | `]` | 启动控制系统 |
| ② | 切到 MuJoCo 窗口按 `9` | 放下机器人到地面 |
| ③ | 回 Terminal 2 按 `Enter` | 开启 ZMQ streaming 模式（显示 `ZMQ STREAMING MODE: ENABLED`）|

---

## 5. 运行时按键参考

| 按键 | 功能 |
|------|------|
| `]` | 启动控制系统 |
| `Enter` | 切换 ZMQ streaming 开/关 |
| `O` | **紧急停止** — 立即停止控制并退出 |
| `I` | 重置基础朝向四元数和航向 |
| `Q` / `E` | 调整航向 ±0.1 rad |
| `T` | 播放当前参考动作（非 ZMQ 模式） |
| `N` / `P` | 切换下一个/上一个参考动作 |
| `R` | 重新开始当前动作 |

---

## 6. 停止流程

1. Terminal 2 按 `O` 停止策略
2. Terminal 3 按 `Ctrl+C` 停止 streamer
3. Terminal 1 关闭 MuJoCo 窗口

---

## 7. 下载音乐与生成新动作

### 7.1 从 YouTube 下载音乐

可以使用 `yt-dlp` 工具将 YouTube 视频直接下载为 `.wav` 文件并存放到 `custom_music/`：

```bash
cd ~/AudioMimic
yt-dlp -x --audio-format wav --audio-quality 0 -o "custom_music/%(title)s.%(ext)s" "<YOUTUBE_URL>"
```

### 7.2 生成动作 (离线模式)

在 AudioMimic 环境中：

```bash
conda activate audiomimic
cd ~/AudioMimic

python test.py \
  --music_dir custom_music/ \
  --checkpoint runs/train/g1_lbeat_relative_finetune/weights/train-500.pt \
  --feature_type jukebox \
  --motion_format g1 \
  --use_beats --beat_rep distance --beat_source audio \
  --no_render --save_motions \
  --motion_save_dir eval/g1_motions_lbeat \
  --out_length 15
```

> 💡 也可替换为基线模型 `runs/train/g1_aist_beatdistance_featurecache/weights/train-2000.pt`。

> ⚠️ 注意：两个 checkpoint 都仅支持 Jukebox + BeatDistance，参数不可更改。

生成的 `.pkl` 文件在指定的 `--motion_save_dir` 目录下。

---

---

## 8. 直接回放模式（跳过 SONIC WBC）

> 不经过 SONIC WBC 策略，直接把关节角通过 DDS 发给 MuJoCo sim。
> 适合快速验证动作效果，不需要 C++ deploy。

**只需 2 个终端：**

### Terminal 1 — MuJoCo 仿真器（同上）

```bash
cd ~/GR00T-WholeBodyControl && source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py
```

### Terminal 2 — 直接回放

```bash
cd ~/GR00T-WholeBodyControl && source .venv_sim/bin/activate && \
python playback_audiomimic_direct.py --pkl ~/AudioMimic/eval/g1_motions/test_0_test_beat_g1.pkl
```

**可选参数：**
- `--kp-scale 0.5` — 降低关节刚度（更柔顺）
- `--kd-scale 2.0` — 增加阻尼（减少振荡）
- `--fps 20` — 覆盖播放帧率

**区别对比：**

| | 经过 SONIC WBC（3 终端） | 直接回放（2 终端） |
|---|---|---|
| 平衡控制 | ✅ WBC 策略保持平衡 | ❌ 纯 PD 跟踪 |
| 适合 sim-to-real | ✅ 是 | ❌ 否 |
| 需要 C++ 编译 | ✅ 是 | ❌ 不需要 |
| 启动速度 | 慢（TRT 转换） | 快 |
| 动作保真度 | WBC 可能修正动作 | 原始动作 1:1 |

---

## 9. 常见问题

### Q: C++ 编译报 CycloneDDS 错误
**A:** 编译前必须 `unset ROS_DISTRO RMW_IMPLEMENTATION AMENT_PREFIX_PATH COLCON_PREFIX_PATH`，ROS2 Humble 的 CycloneDDS 和 bundled 版本冲突。

### Q: deploy 启动后卡在 TRT 转换
**A:** 正常，首次需要 3-5 分钟。转换后会缓存 `.trt` 文件。

### Q: 机器人站着不动
**A:** 确认在 Terminal 2 按了 `]` 启动策略，然后按 `Enter` 开启 ZMQ 模式。

### Q: MuJoCo 窗口没有出现
**A:** 确认 `.venv_sim` 已正确安装：`bash install_scripts/install_mujoco_sim.sh`

### Q: TensorRT 版本警告
**A:** 当前系统安装的是 TensorRT 10.16（DEB 包），官方要求 10.13。编译和推理可以工作，但如果动作效果异常，需要下载 TAR 包版本 10.13。

---

## 10. 数据调试与检查 (pkl schema)

如果发现机器人运动姿态异常，可以使用以下检查脚本快速验证 `.pkl` 文件的 Schema、关节顺序（MuJoCo vs IsaacLab）以及四元数格式（XYzw vs wxyz）。

**检查脚本 (`check_pkl.py`)**：
```python
import pickle
import numpy as np
import sys

def check_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"=== 检查文件: {pkl_path} ===")
    
    # 1. 检查 Schema
    expected_keys = {"dof_pos", "root_rot"}
    missing = expected_keys - set(data.keys())
    if missing:
        print(f"❌ 缺少必须的键: {missing}")
    else:
        print("✅ Schema 包含 dof_pos 和 root_rot")
    
    # 2. 检查 Shape
    dof_pos = np.array(data.get("dof_pos", []))
    root_rot = np.array(data.get("root_rot", []))
    print(f"   dof_pos shape: {dof_pos.shape}")
    print(f"   root_rot shape: {root_rot.shape}")
    
    if root_rot.shape[-1] == 4:
        # 3. 检查四元数格式 (通常根节点倾角不大时 w 接近 1 或 -1)
        # MuJoCo/AudioMimic 默认 [x, y, z, w]
        # IsaacLab/SONIC 预期 [w, x, y, z]
        first_quat = root_rot[0]
        if abs(first_quat[3]) > abs(first_quat[0]):
            print(f"⚠️ 四元数大概率是 [x, y, z, w] 格式。在发送给 SONIC 时需要转换为 [w, x, y, z] (流式脚本会自动处理)。")
        else:
            print(f"✅ 四元数大概率已经是 [w, x, y, z] 格式。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_pkl(sys.argv[1])
    else:
        print("请提供 .pkl 文件路径")
```

---

## 11. 运行后评估指标（Evaluation Metrics）

在完成实机或仿真运行后，我们关注以下几个核心指标来评估动作效果和系统性能：

1. **Tracking (位姿追踪误差)**：对比 AudioMimic 发送的参考关节角（Reference DOF）与 SONIC 执行的实际关节角（Executed DOF）。较小的 RMSE 代表底盘和 WBC 策略能够完美还原生成的动作。
2. **Stability (稳定性指标)**：统计仿真/实机中的摔倒次数、根节点的高度方差或异常姿态。用以评估生成的唯美舞蹈动作在面临物理重力、碰撞约束时的**可行性**。
3. **Latency (端到端延迟)**：评估从 AudioMimic 网络生成完成、ZMQ 流式传输网络开销，到 SONIC 接收命令并产生底层电机控制力矩的总延迟时间。
4. **BAS_executed (执行后节拍对齐分数)**：基于机器人**实际执行**的轨迹序列（而非网络直接输出的理论轨迹），重新计算 Forward Kinematics (FK) 并提取动作节拍，再与音乐节拍对比计算 Beat Alignment Score。这能真实反映舞蹈动作经过物理世界摩擦、惯性平滑后的**卡点效果**。

---

## 文件位置参考

| 文件 | 路径 |
|------|------|
| AudioMimic 动作 | `~/AudioMimic/eval/g1_motions/*.pkl` |
| ZMQ Streamer (离线) | `~/GR00T-WholeBodyControl/stream_audiomimic.py` |
| ZMQ Streamer (实时) | `~/AudioMimic/stream_inference.py` |
| MuJoCo Sim | `~/GR00T-WholeBodyControl/gear_sonic/scripts/run_sim_loop.py` |
| C++ Deploy 二进制 | `~/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref` |
| SONIC 策略模型 | `~/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/` |
| Planner 模型 | `~/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/` |
