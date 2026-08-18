# AudioMimic / Musics2Dance → SONIC Sim2Sim 操作手册

> 当前推荐流程：先用固定 G1 轨迹验证 SONIC tracking，再使用 H8 双缓冲 online runtime 以 30 FPS 实时生成和执行。定量实验应关闭图像发布；录像作为单独复现实验运行。

---

## 0. 前置条件

- 已拉取 `~/Musics2Dance-prior-dev` 的 `prior-dev` 分支及 LFS 文件
- 已有 G1 格式的 `.pkl` 动作文件；推荐使用 `onlinegeneratedmotion/m2_predicted_fms/` 或 `m4_oracle_fms/` 中的 60 秒轨迹
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

实时性能和定量实验使用以下配置，只发布 SONIC state：

```bash
cd ~/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
  --enable-sonic-state-publish \
  --sonic-state-port 5559
```

该配置实测 MuJoCo state 约 194.7 Hz，realtime factor 约 0.965。需要录像时才使用下面的 offscreen 配置；图像发布会将 realtime factor 降至约 0.63，因此录像组不能用于实时性能结论。

```bash
cd ~/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
  --enable-sonic-state-publish \
  --enable-offscreen \
  --enable-image-publish \
  --camera-port 5555
```

等待 MuJoCo 窗口打开，看到 G1 机器人模型。`--enable-sonic-state-publish` 会额外发布真实 base state，供离线跟踪记录或在线 S66 闭环使用；`--enable-offscreen --enable-image-publish` 会发布 MuJoCo renderer 的相机画面，供 Terminal 4 直接录制。

### Terminal 2 — SONIC WBC 策略（C++ deploy）

```bash
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT=/usr
unset ROS_DISTRO RMW_IMPLEMENTATION AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
export CMAKE_PREFIX_PATH="/opt/onnxruntime:/usr/lib/x86_64-linux-gnu/cmake"

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
  --zmq-verbose \
  --output-type all \
  --disable-crc-check
```

> ⚠️ 首次运行会将 planner ONNX 转换为 TensorRT 引擎，需要 **3-5 分钟**，不要中断！
> 转换完成后会生成 `.trt` 缓存文件，下次启动会很快。

### Terminal 3 — M2/M4 离线完整轨迹流

以下命令是当前 SONIC 测试入口。它以连续 C4 packet 将 30 Hz 轨迹重采样到 SONIC 需要的 50 Hz，并转换至 SONIC policy joint order。

先测试 M2：这是用预测 future-music sidecar 生成的 60 秒轨迹。

```bash
cd ~/Musics2Dance-prior-dev
conda activate audiomimic

export MOTION_PKL=onlinegeneratedmotion/m2_predicted_fms/m2_train1234_sample1234_u100000_best_song098.pkl

python stream_to_sonic.py \
  --pkl "$MOTION_PKL" \
  --port 5556 \
  --topic pose \
  --sonic_reference_fps 50 \
  --reference_safety none \
  --record_feedback \
  --output_dir /tmp/sonic_openloop_m2_60s
```

该模式是 **open-loop tracking**：SONIC 反馈只保存到 `output_dir` 中的 `s66_exec.json`、`sim_state.json` 和 `sonic_feedback.json`，不会改变已经固定的后续 reference。

M4 仅替换轨迹路径；它使用 GT/oracle future-music condition，是离线效果上限，不是可实时部署的路线：

```bash
export MOTION_PKL=onlinegeneratedmotion/m4_oracle_fms/m4_train1234_sample1234_u100000_best_song098.pkl
```

> `stream_inference.py` 是旧 AudioMimic/EDGE 路线，不应用于本页的 V6f-X M2/M4 SONIC 对照。M2 的实时 generator、predicted-FMS sidecar 及其推理权重尚未发布；发布后接入 `run_g1_paper_faithful_dc_sonic.py` 完成 S66 闭环测试。

### Terminal 4（可选）— 录制 MuJoCo renderer 视频

在 Terminal 1 启动后、Terminal 3 前运行。该终端订阅 MuJoCo offscreen renderer 发布的 `third_person` 相机并直接编码 mp4，不是桌面/X11 截图，也不影响 SONIC reference 或 S66 记录。

```bash
cd ~/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host localhost \
  --camera-port 5555 \
  --fps 30 \
  --output-path /tmp/sonic_openloop_m2_60s/mujoco_renderer
```

出现 `SONIC Camera Viewer` 窗口后，先聚焦该窗口并按 `R` 开始录制，再启动 Terminal 3。Terminal 3 显示 `Offline stream complete` 后，回到 viewer 按 `R` 停止录制。所需的第三人称视频将写到：

```text
/tmp/sonic_openloop_m2_60s/mujoco_renderer/rec_<timestamp>/third_person.mp4
```

`third_person` 是 MuJoCo 的正面第三人称自由相机：每帧将镜头中心更新到 `pelvis`，但不使用机器人头部相机，也不随头部/躯干转动。默认距离 3.0 m、正面方位 180°、俯仰 -12°。当前 G1 MJCF 的 offscreen framebuffer 为 640×480，因此录像以该分辨率输出。

---

## 4. 操作顺序（在 Terminal 2 中按键）

| 步骤 | 按键 | 说明 |
|------|------|------|
| ① | `]` | 启动控制系统 |
| ② | 切到 MuJoCo 窗口按 `9` | 放下机器人到地面 |
| ③ | 回 Terminal 2 按 `Enter` | 开启 ZMQ streaming 模式（显示 `ZMQ STREAMING MODE: ENABLED`）|

完成第 ③ 步后不要再次按 `Enter`，否则会关闭 ZMQ streaming。然后再启动 Terminal 3。

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
| M2/M4 离线评估包 | `~/Musics2Dance-prior-dev/onlinegeneratedmotion/**/*.pkl` |
| AudioMimic K64 启动 seed | `~/AudioMimic/models/seeds/m2_train1234_sample1234_u100000_best_song098.pkl` |
| AudioMimic 在线模型 release | `~/AudioMimic/models/releases/pure-cf-d16-zero-200k-v1/` |
| ZMQ Streamer (当前离线) | `~/Musics2Dance-prior-dev/stream_to_sonic.py` |
| SONIC bridge / 在线闭环入口 | `~/Musics2Dance-prior-dev/sonic_bridge.py`, `run_g1_paper_faithful_dc_sonic.py` |
| MuJoCo 第三人称录像 | `/tmp/sonic_openloop_m2_60s/mujoco_renderer/rec_<timestamp>/third_person.mp4` |
| AudioMimic 旧在线 streamer | `~/AudioMimic/stream_inference.py` |
| MuJoCo Sim | `~/GR00T-WholeBodyControl/gear_sonic/scripts/run_sim_loop.py` |
| C++ Deploy 二进制 | `~/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref` |
| SONIC 策略模型 | `~/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/` |
| Planner 模型 | `~/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/` |

---

## 10. Online Generator 接入状态（2026-08-18）

### 10.0 模型安装与仓库边界

AudioMimic 是正式集成与部署仓库；Musics2Dance 保留训练、模型开发与 M0/M2/M4 离线评估。在本机从 Musics2Dance checkout 安装已验证的模型、codec、q0 和 K64 seed：

```bash
cd ~/AudioMimic
bash scripts/install_pure_cf_release.sh ~/Musics2Dance-prior-dev
```

脚本会执行 `SHA256SUMS` 校验。权重文件被 `.gitignore` 排除，Git 只保存 release README、manifest 和 SHA256 身份。当新模型保持 `K64/H8/C4 + S66 + d16 codec` 契约时，只需安装新 release 并替换命令中的 `MODEL_DIR`；如果改为 H8/C2 或修改 S66 语义，runtime 也必须做版本升级。

### 10.1 系统逻辑

当前在线模型一次规划 H8，并只提交前 C4：

```text
H8 = 16 motion frames @ 30 FPS = 0.533 s future
C4 =  8 motion frames @ 30 FPS = 0.267 s commit
```

SONIC 接口使用 protocol v1：

```text
joint_pos [T,29]    SONIC/IsaacLab joint order
joint_vel [T,29]
body_quat_w [T,4]   wxyz on wire
frame_index [T]     global 50 Hz reference index
catch_up [1]
```

模型和数据内部保持 MuJoCo joint order，只有在 `sonic_bridge.py` 的 wire boundary 才转换为 SONIC order。模型输出的 root XY 不发送给 SONIC；SONIC 只接收关节 reference 和 root orientation。

30 Hz reference 必须连续重采样到 50 Hz。每个 C4 对应的 SONIC 帧数不是固定 13，而是：

```text
13, 13, 14, 13, 13, 14, ...
```

### 10.2 已确认的问题和修复

| 项目 | 原问题 | 当前状态 |
|---|---|---|
| Quaternion | AIST raw GT 被 metadata 错标为 `wxyz` | raw GT 实际为 `xyzw`；wire 端转换为 `wxyz` |
| Resampler | 每个 C4 重建 adapter，30→50 Hz 相位反复归零 | C4 使用持续 adapter，保持 `13/13/14` |
| Frame index | online runtime 每次固定推进 13 帧 | 改为连续 committed clock |
| 起始状态 | 从 `macarena` 直接热切到任意 GT，最大跳变约 1.57 rad | GT 测试使用 3 s measured-pose alignment + 1 s hold |
| Online 调度 | inference 约 50 ms 与 C4 执行 267 ms 串行相加 | synthetic open-loop 改为绝对 deadline 双缓冲 |
| MuJoCo 性能 | 图像发布使仿真明显慢于真实时间 | 定量组关闭 image publish，RTF 从 0.63 提升到 0.965 |

### 10.3 SONIC 能力上限与接口消融

固定 reference 的结果表明 SONIC tracker 和 C4 分包均正常：

| Reference | 模式 | 结果 | 正式段全身 RMSE median | 最低 base height |
|---|---|---|---:|---:|
| M2 32 s | C4，连续 resampler | 稳定 | 0.156 rad | 0.660 m |
| M2 32 s | full sequence | 稳定 | 0.161 rad | 0.660 m |
| 平缓 AIST GT | aligned + full | 稳定 | 0.142 rad | 0.740 m |
| 平缓 AIST GT | aligned + C4 | 稳定 | 0.140 rad | 0.742 m |
| 激烈 AIST GT | aligned + full | GT 开始约 4.21 s 后摔倒 | 0.302 rad | 0.200 m |

平缓 GT 的 full 与 C4 结果基本一致，因此 C4 packet scheduling 不是 tracking 退化来源。激烈 GT 的腿部速度 P95 为 10.60 rad/s，而稳定 M2 约为 4.16 rad/s；该失败属于 reference 动态可执行性问题。

### 10.4 Open-loop 双缓冲

旧 runtime 是串行流程：

```text
inference 50 ms → publish → execute/wait 267 ms → next inference
```

实测发送周期 0.338 s，有效动作速度只有 23.65 FPS。修复后使用双缓冲和绝对 deadline：

```text
publish H8[k]
├─ SONIC executes C4[k]
└─ GPU generates H8[k+1]

deadline = t0 + k * 8 / 30
```

120 commits 的 32 s 验证结果：

| 指标 | 结果 |
|---|---:|
| Online wall-clock | 32.0003 s |
| Effective motion FPS | 29.9998 |
| Deadline misses > 2 ms | 0 |
| Packet lateness P95 | 1.08 ms |
| MuJoCo reset / fall | 0 / 0 |
| Minimum base height | 0.638 m |

关闭图像发布后的 30-commit 复验：generator 29.995 FPS，MuJoCo state 194.7 Hz，realtime factor 0.965，无摔倒。

当前实现已迁入正式 AudioMimic 仓库：

```text
~/AudioMimic/
```

关键文件：

```text
sonic_bridge.py
stream_to_sonic.py
run_g1_paper_faithful_dc_sonic.py
```

### 10.5 当前基线命令

定量运行时保持 Terminal 1 无图像发布，然后在 generator 终端执行：

```bash
cd ~/AudioMimic
conda activate audiomimic

MODEL_DIR=~/AudioMimic/models/releases/pure-cf-d16-zero-200k-v1
SEED_PKL=~/AudioMimic/models/seeds/m2_train1234_sample1234_u100000_best_song098.pkl

python run_g1_paper_faithful_dc_sonic.py \
  --generator_checkpoint "$MODEL_DIR/pure-cf-d16-zero-seed1234-update200000-inference.pt" \
  --q0_checkpoint "$MODEL_DIR/q0-d16-seed1234-update100000.pt" \
  --codec_checkpoint "$MODEL_DIR/codec-d16-seed1234-train300.pt" \
  --experiment_id EXP-20260804-v6f-x-pure-commit-forcing-final \
  --output_dir /tmp/online_open_h8_double_buffer_120 \
  --seed_mode k64_pkl \
  --seed_motion_pkl "$SEED_PKL" \
  --seed_start_frame 496 \
  --seed_execution replay \
  --feedback_source synthetic \
  --preview_mode h8_preview \
  --max_commits 120 \
  --nfe 10 \
  --q0_policy sample \
  --sampling_seed 1234 \
  --device cuda \
  --reference_safety none \
  --sonic_reference_fps 50 \
  --startup_wait 2 \
  --execution_timeout 2
```

### 10.6 Delayed Residual Closed-loop（已实现，待消融）

Open-loop 已作为实时基线固定。下一步不再使用旧的阻塞 measured-feedback 循环，而是在双缓冲基础上引入一拍延迟的状态残差：

```text
delta[k] = measured_end[k] - synthetic_end[k]
corrected_end[k+1] = synthetic_end[k+1] + alpha * delta[k]
```

实现参数为 `--feedback_source delayed_residual --feedback_alpha <value>`，与 synthetic Open-loop 共用绝对 deadline 双缓冲。首轮消融使用 `alpha = 0, 0.25, 0.5, 1.0`。其中 `alpha=0` 必须逐位复现当前 Open-loop；其余组比较 30 FPS deadline、C4 boundary jump、tracking RMSE、base height 和 survival。任何 feedback fusion 都不得阻塞下一次 H8 的生成或改变全局 deadline。

`alpha=0` 的首轮 Sim2Sim gate 已通过：有效 29.9998 FPS、deadline miss 0、无摔倒、最低 base height 0.670 m、MuJoCo RTF 0.960。两次独立物理运行不能直接要求 `generated_motion.npy` 逐位相同，因为 K64 replay 后的 measured initial S66 会有小幅变化；本次两组初始关节 S66 RMSE 为 0.0036 rad。`alpha=0` 的严格等价性由固定输入状态的单元测试保证，Sim2Sim gate 只要求调度和稳定性等价。

`alpha=0.25` 的 30-commit 结果：30.0004 FPS、deadline miss 0、无摔倒、最低 base height 0.688 m。相对 `alpha=0`，腿部 tracking RMSE median 从 0.115 降至 0.094 rad，P95 从 0.166 降至 0.140 rad；但全身 RMSE P95 从 0.265 升至 0.287 rad，手臂生成速度 P95 从 7.18 升至 10.58 rad/s。说明延迟反馈改善了腿部和基座稳定裕量，但统一 66D 增益会放大上肢动态，后续需检查 `alpha=0.5/1.0` 趋势并考虑分组增益。

`alpha=0.5` 仍保持 29.996 FPS、deadline miss 0，但全身 tracking RMSE median/P95 增至 0.241/0.371 rad，延迟残差 RMSE median/P95 增至 0.558/0.830，C4 边界速度诊断 P95/max 增至 113.23/120.99。手臂生成速度 P95/max 达到 25.40/49.37 rad/s。这说明统一 66D 的一拍延迟残差会形成正反馈放大，因此不继续执行 `alpha=1.0` 组。

### 10.7 JIT Measured-state Closed-loop

C4 表示 4 个 latent token，对应 8 个 30 FPS motion frame，因此物理执行窗口为 267 ms。推理通常约 50 ms，无需在 C4 开始时立即生成下一段。新增 `jit_measured` 模式：

```text
t = 0 ms        SONIC 开始执行 C4[k]
t ≈ 187 ms     取最新 measured S66，开始 H8[k+1] 推理
t ≈ 237 ms     推理完成
t = 267 ms      在绝对 deadline 发送 C4[k+1]
```

默认保留 80 ms 推理预算，使 condition 的测量延迟从约一个 C4 降低到约 80 ms，同时为 48--55 ms 的常规推理留出通信和抖动余量。该模式是 measured-state Closed-loop，不使用 `feedback_alpha`，也不使用 plan cache。

首轮 30-commit 验证命令：

```bash
cd ~/AudioMimic
conda activate audiomimic

MODEL_DIR=~/AudioMimic/models/releases/pure-cf-d16-zero-200k-v1
SEED_PKL=~/AudioMimic/models/seeds/m2_train1234_sample1234_u100000_best_song098.pkl

python run_g1_paper_faithful_dc_sonic.py \
  --generator_checkpoint "$MODEL_DIR/pure-cf-d16-zero-seed1234-update200000-inference.pt" \
  --q0_checkpoint "$MODEL_DIR/q0-d16-seed1234-update100000.pt" \
  --codec_checkpoint "$MODEL_DIR/codec-d16-seed1234-train300.pt" \
  --experiment_id EXP-20260804-v6f-x-pure-commit-forcing-final \
  --output_dir /tmp/online_jit_measured_h8_30 \
  --seed_mode k64_pkl \
  --seed_motion_pkl "$SEED_PKL" \
  --seed_start_frame 496 \
  --seed_execution replay \
  --feedback_source jit_measured \
  --jit_inference_budget_ms 80 \
  --preview_mode h8_preview \
  --max_commits 30 \
  --nfe 10 \
  --q0_policy sample \
  --sampling_seed 1234 \
  --device cuda \
  --reference_safety none \
  --sonic_reference_fps 50 \
  --startup_wait 2 \
  --execution_timeout 2
```

首轮 gate：有效动作速度接近 30 FPS，deadline miss 为 0，无 reset/fall。通过后再与 synthetic Open-loop 比较 tracking RMSE、base height、survival 和生成关节速度。

首轮 30-commit 结果：有效动作速度 30.0002 FPS，deadline miss 0，推理时间除首轮 warm-up 100.3 ms 外为 47--52 ms。最低 base height 0.664 m，无 fall/reset。全身 tracking RMSE median/P95 为 0.183/0.330 rad，腿部为 0.106/0.171 rad，手臂为 0.233/0.451 rad。相对 synthetic Open-loop，全身和腿部 median 略有改善，但全身与手臂 P95 变差；生成腿部速度 P95/max 为 4.33/22.29 rad/s，手臂为 9.25/46.93 rad/s，仍有少量极端尖峰。

该结果表明 JIT 调度可行，但 deadline 前 80 ms 的 measured S66 不能直接视为下一 C4 的边界状态。下一步需要对 root 和关节状态做约 80 ms 的时间对齐外推，再与当前 JIT 和 synthetic Open-loop 作同 seed 比较。

60 s 同 seed 复验进一步确认了该问题。Open-loop 与 JIT 均保持 30 FPS 且无 fall/reset，说明 SONIC 和调度层能长时运行；但 JIT 全身 RMSE P95 从 0.281 升至 0.345 rad，腿部从 0.192 升至 0.236 rad，手臂从 0.360 升至 0.465 rad。生成腿部速度 P95/max 从 2.56/7.95 升至 5.40/36.26 rad/s，手臂从 4.65/17.79 升至 10.60/71.01 rad/s。因此 `jit_measured` 直接反馈定义为失败基线：它不是 tracker 执行失败，而是未对齐且超出训练分布的 measured S66 改变了 generator rollout，导致更激进的 future plan。

### 10.8 SONIC 固定 Reference 动态能力曲线

使用同一条 M2 固定轨迹的前 32 s，不将 SONIC 反馈送回 generator，对动作做时间缩放。`stream_to_sonic.py` 新增 `--playback_rate`，在保持 30 Hz 源动作和 50 Hz SONIC 接口的前提下重采样固定轨迹。

| Playback | 全身 RMSE med/P95 | 腿部 RMSE med/P95 | 手臂 RMSE med/P95 | 最低 base height | Fall/reset | Ref velocity P95/max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75x | 0.134/0.207 | 0.107/0.174 | 0.152/0.266 | 0.659 m | 0/0 | 2.15/8.14 rad/s |
| 1.00x | 0.138/0.215 | 0.101/0.168 | 0.161/0.286 | 0.660 m | 0/0 | 2.91/11.13 rad/s |
| 1.25x | 0.145/0.241 | 0.104/0.173 | 0.171/0.320 | 0.657 m | 0/0 | 3.55/13.49 rad/s |
| 1.50x | 0.151/0.249 | 0.106/0.168 | 0.181/0.335 | 0.651 m | 0/0 | 4.24/15.85 rad/s |

四组均完整运行且腿部 tracking 没有随倍率显著恶化，性能下降主要集中在手臂。因此该实验给出的是 SONIC 能力的已验证下界，不是失败阈值：对该 M2 轨迹，SONIC 至少可稳定追踪到全身关节速度 P95 4.24 rad/s、最大 15.85 rad/s。JIT Closed-loop 生成的手臂速度 P95/max 为 10.60/71.01 rad/s，明显超出已验证固定轨迹动态范围，再次支持 generator rollout 是当前主要瓶颈。
