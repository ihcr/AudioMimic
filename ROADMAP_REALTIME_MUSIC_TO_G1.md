# AudioMimic：实时音乐驱动的 G1 机器人舞蹈路线图

更新日期：2026-08-18

## 1. 项目目标

AudioMimic 的目标是构建一个严格因果、可持续在线运行的机器人舞蹈系统：

```text
实时音频
  -> 因果音乐特征与未来音乐预测
  -> 在线 G1 motion generator
  -> 连续、可执行的短期 reference
  -> SONIC whole-body tracking
  -> MuJoCo 与真实 Unitree G1
```

最终系统应同时满足：

1. 只使用当前时刻已经到达的音乐，不读取真实未来音频。
2. 动作在节拍、节奏和高层音乐语义上与音乐匹配。
3. Generator、通信和 tracker 按统一时间戳实时运行，不降速播放。
4. 动作满足 G1 与 SONIC 的速度、加速度、jerk、关节限位和稳定性要求。
5. 系统能够连续运行 60 s 以上，并具备迟到、丢包和异常 reference 的 fallback。

## 2. 当前系统状态

### 2.1 已完成：Motion Generator 基础链路

- 已有 pure Commit Forcing d16 推理模型、q0 generator 和 matching codec。
- 在线契约为 `K64/H8/C4/S66`：
  - K64：64 个历史 latent token，对应约 4.27 s 动作历史；
  - H8：一次预测 8 个 latent token，即 16 个 30 FPS motion frame；
  - C4：每轮提交 4 个 latent token，即 8 个 motion frame、约 267 ms；
  - S66：root、关节位置、关节速度与加速度组成的物理边界状态。
- 常规 GPU 推理约 48--55 ms，能够在 C4 deadline 内完成。
- Open-loop 在线 rollout 已按约 30 FPS 连续运行 60 s，无 deadline miss、fall 或 reset。
- 模型、codec、q0 和 K64 seed 已安装到 AudioMimic，并通过 SHA256 校验。

当前模型仍是 **motion-only**。M2 PKL 仅用于提供 K64 启动历史，运行期间没有读取实时音乐。

### 2.2 已完成：SONIC 与中间通信

- Generator 内部使用 MuJoCo joint order，SONIC wire boundary 转换为 SONIC joint order。
- 已修正 quaternion 顺序、C4 frame index 和 30 Hz 到 50 Hz 连续重采样。
- C4 重采样保持 `13, 13, 14` 的连续相位，不再每包重置。
- 已支持 H8 preview、C4 commit、full sequence 和 C4 packet 两种离线模式。
- 固定 M2 和平缓 GT 的 full/C4 tracking 结果一致，排除 packet 切分是主要误差来源。
- 已支持 SONIC feedback、MuJoCo state 和 S66 telemetry 记录。
- 运行代码已经从临时实验树迁入正式 AudioMimic 仓库。

### 2.3 已完成：SONIC 能力与 Closed-loop 诊断

固定 M2 reference 的 `0.75x/1.0x/1.25x/1.5x` 动态能力实验均无 fall/reset。对该轨迹，SONIC 至少已验证到：

```text
全身关节速度 P95：4.24 rad/s
全身关节速度最大值：15.85 rad/s
```

性能下降主要集中在手臂，腿部 tracking 对倍率变化较稳定。这是已验证能力下界，不是 SONIC 的绝对失败阈值。

Closed-loop 诊断得到：

- `delayed_residual` 对完整 S66 使用统一增益会形成反馈放大；
- `jit_measured` 可维持 30 FPS，但 deadline 前 80 ms 的 measured S66 与真实 C4 boundary 时间错位；
- 60 s JIT rollout 的腿部速度 P95/max 为 5.40/36.26 rad/s，手臂为 10.60/71.01 rad/s；
- JIT tracking P95 明显差于 Open-loop，但两组均无 fall/reset；
- 阻塞等待真实 C4 boundary 后再推理只有约 24.3 FPS，不是有效在线方案。

因此当前主要瓶颈是 generator 对 delayed/noisy measured S66 的分布适应和动作可执行性，而不是 SONIC 协议或基础 tracking。

## 3. 尚未完成的核心链路

当前已经打通：

```text
在线无音乐动作生成 -> SONIC -> MuJoCo
```

尚未打通：

```text
实时音乐 -> 因果音乐条件 -> 在线动作生成 -> SONIC -> G1
```

缺口包括：

1. 实时音频采集、缓冲和时间戳。
2. 因果 beat、tempo、onset、phase 与高层音乐语义提取。
3. M2 predicted future-music sidecar 的在线 checkpoint 与推理入口。
4. 音乐条件真正进入 H8 generator，而不是仅使用 M2 动作 seed。
5. Audio、generator C4 和 SONIC reference 的统一绝对时钟。
6. Tracker-aware generator 训练与 boundary-state 时间对齐。
7. Reference feasibility 检测、H8 fallback 和真机安全状态机。

## 4. TODO List

### P0：形成实时音乐 Open-loop Demo

- [ ] 固定在线音乐特征接口，包括 feature schema、FPS、时间戳和因果可见范围。
- [ ] 确认 M2 sidecar 的 checkpoint、输入特征、future horizon 和训练配置。
- [ ] 将音频线程接入 AudioMimic：持续读取 wav 流或麦克风，不阻塞 generator。
- [ ] 在线提取因果 beat/tempo/onset/phase 特征并对齐到 30 FPS motion clock。
- [ ] 将 M2 predicted future-music condition 接入 H8/C4 generator。
- [ ] 保持当前稳定的 synthetic S66 Open-loop，暂不启用 measured-state Closed-loop。
- [ ] 同步播放音乐与 SONIC reference，记录端到端 audio-to-motion latency。
- [ ] 在 MuJoCo 完成至少 3 首未见音乐、每首 60 s 的实时生成演示。

验收标准：

```text
实时因果音频输入
30 FPS generator deadline miss <= 1%
无 reference 断流
60 s 无 fall/reset
视频中音乐与机器人动作时间同步
```

### P1：音乐-动作质量评估

- [ ] 实现 Beat Alignment、Beat F1、precision 和 recall 的统一评估。
- [ ] 增加动作对节拍响应强度，而不只检测动作 beat 是否存在。
- [ ] 增加 tempo consistency、beat phase error 和 onset-response latency。
- [ ] 增加运动多样性、motion energy、静止率和平均态检测。
- [ ] 增加速度、加速度、jerk、joint-limit、foot sliding 和 base stability 指标。
- [ ] 对 M0、M2 predicted 和 M4 oracle 使用相同歌曲、seed 与 duration 做消融。
- [ ] 对生成 reference 与 SONIC tracked motion 分别计算音乐一致性，量化 tracker 造成的节拍损失。

验收标准：M2 在严格因果条件下显著优于 M0，并报告与 M4 oracle 的差距。

### P2：Generator 可执行性改进

- [ ] 将 SONIC 已验证动态范围写入训练与评估配置，而不是运行时静默裁剪。
- [ ] 加入 joint velocity、acceleration、jerk 和 C4 boundary continuity loss。
- [ ] 对腿、腰、手臂使用分组损失与阈值，重点抑制手臂速度尖峰。
- [ ] 加入 joint limit、foot contact/sliding 和 base stability loss。
- [ ] 输出每个 H8 的 feasibility diagnostics 或 confidence。
- [ ] 对异常 H8 实现拒绝/回退策略，并保存触发原因。

验收标准：三 seed、每组 60 s 下，生成动态主要落在已验证 SONIC 范围内，tracking P95 不因在线 rollout 持续增长。

### P3：Tracker-aware Closed-loop Generator

- [ ] 明确定义 measured S66 的 measurement time、boundary time 和 age。
- [ ] 比较线性外推与 reference-aware boundary prediction。
- [ ] 采集 `(measured state, remaining reference, deadline boundary state)` 训练数据。
- [ ] 训练时加入 50--80 ms delay、tracking residual、measurement noise 和 feedback dropout。
- [ ] 进行多 C4 tracker-in-the-loop/self-forcing rollout，而不是只做单步 teacher forcing。
- [ ] 输入 current reference、boundary reference、tracking residual 和 remaining time。
- [ ] 比较 Open-loop、JIT-current、predicted-boundary 和非实时 boundary diagnostic。
- [ ] 若 H8/C4 反馈仍过慢，再训练 H8/C2；不要只在部署代码中直接改 commit 长度。

验收标准：Closed-loop 在相同音乐、seed 和初态下，tracking P95、survival 或抗扰性至少一项稳定优于 Open-loop，且生成动态不被反馈放大。

### P4：Runtime 安全与真机准备

- [ ] 为 audio、generator、reference 和 feedback 消息加入统一绝对时间戳。
- [ ] 实现 audio、generation、execution 三线程异步运行。
- [ ] 推理迟到时执行上一轮 H8 后半段，不让 SONIC reference 断流。
- [ ] 对 NaN/Inf、joint limit、速度、加速度、jerk 和 orientation jump 做发送前检查。
- [ ] 实现 feedback 丢失时的 Open-loop fallback 与平滑 hold。
- [ ] 在 MuJoCo 做 packet delay、dropout 和 inference jitter 压力测试。
- [ ] 确认真机 feedback 字段、joint order、quaternion 和仿真完全一致。
- [ ] 逐级执行低动态、标准动态和完整音乐真机实验。

## 5. 推荐执行顺序

```text
1. 冻结当前 SONIC protocol 与 Open-loop runtime
2. 接入实时因果音乐和 M2 sidecar
3. 完成 MuJoCo 实时音乐 Open-loop demo
4. 建立音乐质量与物理可执行性评估
5. 修改 generator 的动态约束与 tracker-aware 训练
6. 验证 predicted-boundary Closed-loop
7. 加入 fallback、安全检查和压力测试
8. 进入真实 G1
```

当前最优先的工作不是继续修改 SONIC，而是拿到并接入 **在线 M2 音乐条件模型与特征契约**。在此之前，系统只能证明实时动作生成和机器人 tracking，不能证明机器人在随实时音乐生成舞蹈。

## 6. 仓库责任边界

### AudioMimic

- 实时音频、music sidecar 推理与时钟同步；
- checkpoint inference runtime；
- SONIC bridge、MuJoCo/真机运行和安全 fallback；
- 端到端评估、视频和操作手册。

### Musics2Dance

- 数据处理与模型训练；
- M0/M2/M4、H8/C4 或 H8/C2 等研究消融；
- tracker-aware 与 feasibility loss；
- 导出带版本、manifest 和 SHA256 的兼容模型 release。

只要 motion/state/codec/music-condition 契约保持不变，AudioMimic 应能够仅替换模型 release；契约变化必须提升 runtime protocol 版本并提供 adapter。
