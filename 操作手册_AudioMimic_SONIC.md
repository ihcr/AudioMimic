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
  --checkpoint runs/train/g1_aist_beatdistance_featurecache/weights/train-2000.pt \
  --feature_type jukebox \
  --motion_format g1 \
  --use_beats --beat_rep distance --beat_source audio \
  --port 5556 \
  --precache_features
```

> ⚠️ 注意：添加 `--precache_features` 参数会在开始前一次性提取所有的 Jukebox 特征（需要一定时间）。提取完成后，DDIM 生成与 30 FPS 播放将同步进行，避免动作卡顿。如果不加该参数，则为完全实时流式模式，但由于 Jukebox 提取耗时较长，动作播放会逐渐落后于音频。

> 💡 **关于流式生成的平滑性与对齐优化：**
> 在流式推理中，动作是按 2.5 秒的 Chunk 分段生成的。为了保证动作连续且与音乐完美对齐，系统内置了以下优化：
> 1. **Crossfade 线性过渡混合**：在两个相邻 Chunk 的重叠区域，对关节位置（DOF）进行线性插值，对根节点旋转（Root Quaternion）进行平滑过渡，消除分段拼接处的硬切和动作跳变。
> 2. **`--precache_features` 预加载**：剥离耗时的 Jukebox 特征提取，使得 DDIM 扩散模型能以足够的速度实时产出动作，避免缓冲区排空导致的画面定格和音乐错位。
> 3. **首帧对齐修复**：修复了 ZMQ Consumer 吞掉第一帧导致的持续性 33ms 时序偏移。

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
  --checkpoint runs/train/g1_aist_beatdistance_featurecache/weights/train-2000.pt \
  --feature_type jukebox \
  --motion_format g1 \
  --use_beats --beat_rep distance --beat_source audio \
  --no_render --save_motions \
  --motion_save_dir eval/g1_motions \
  --out_length 15
```

生成的 `.pkl` 文件在 `~/AudioMimic/eval/g1_motions/` 目录下。

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
e
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
