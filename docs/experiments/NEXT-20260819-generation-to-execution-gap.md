# Generation-to-Execution Gap 实验协议

更新日期：2026-08-19
状态：Phase A、Phase C已完成；Phase B、Phase D待执行
系统：AudioMimic motion generator + SONIC + Unitree G1/MuJoCo

本实验的指标定义、证据等级、统计规则和验收 gate 统一服从
[`AudioMimic Music-to-G1 Evaluation Map v1.0`](../evaluation/EVALUATION_MAP_MUSIC_TO_G1.md)。
本文件规定本次实验如何采集数据；Evaluation Map 规定这些数据必须如何评价。若两者
表述不一致，以 Evaluation Map 为准，并在修改前提升其版本号。

## 1. 研究问题

本实验量化 motion generator 输出的理想 reference 经 SONIC whole-body tracker 执行后损失了多少动作信息，重点回答：

1. SONIC 对不同身体部位的姿态、速度和动作幅度造成多大误差？
2. Tracking gap 中有多少来自时间延迟，有多少来自幅度和动作形状损失？
3. SONIC 对低频、中频和高频动作成分的保留能力分别如何？
4. Reference 中的 beat、onset、motion energy 和身体部位节奏在 tracked motion 中保留多少？
5. 哪些 reference 动态最容易造成 tracking 退化，并应转化为 generator 的训练约束？

这里定义：

```text
Generation-to-Execution Gap
= generator reference motion 与 SONIC tracked motion 之间的差异
```

该实验使用固定 reference，SONIC feedback 只用于记录，不返回 generator。因此实验测量的是 tracker 与机器人动力学对动作特征的影响，不混入 Closed-loop generator 的分布漂移。

完整评价对象分为两级：

```text
音乐 A
  -> motion generator
  -> 生成参考动作 M_ref
  -> SONIC whole-body tracker
  -> 实际执行动作 M_exec
```

评价必须分别回答：`M_ref`是否是合理且符合音乐的舞蹈，`M_exec`是否仍然合理且
符合音乐，以及从`M_ref`到`M_exec`损失了多少动作与音乐表达。任何只在reference
或只在execution上计算的单一指标，都不能独立支撑完整系统结论。

## 2. 核心假设

- H1：SONIC 对腿部低频动作的保留率高于手臂高频动作。
- H2：Raw tracking error 可分解为全局时间延迟与补偿延迟后的形状/幅度误差。
- H3：Reference 的高频功率、短促动作峰值和腕部 beat response 在 tracking 后衰减最明显。
- H4：Tracker会改变动作能量和音乐响应；这种变化应通过onset correlation、response lag和BAS retention实测，而不能预设所有音乐指标必然下降。
- H5：Reference velocity、acceleration、jerk 和 C4 boundary discontinuity 能预测 tracking gap。

## 3. 实验范围

### 3.1 第一阶段：最小可行实验

使用现有 M2 song098 固定 60 s reference，完成时间对齐与基础指标实现：

```text
1 route x 1 song x 1 seed x 3 repeats x 60 s
```

第一阶段输出：

- Raw 与 lag-compensated joint tracking RMSE；
- 各关节和身体分组 tracking lag；
- amplitude、energy 和频带功率保留率；
- base height、fall/reset 和 foot sliding；
- reference/tracked代表性关节曲线和关节热图。

### 3.2 第二阶段：模型路线对比

使用相同歌曲、动作起点、sampling seed 和时长比较：

| Route | 音乐条件 | 作用 |
|---|---|---|
| M0 | 无音乐 | 无条件动作基线 |
| M2 | predicted future-music | 严格因果目标模型 |
| M4 | oracle future-music | 非因果音乐条件上限 |

推荐规模：

```text
3 routes x 3 songs x 3 seeds x 3 repeats x 60 s
```

M2/M4必须使用匹配音频，并从PKL中的`audio_start_seconds`开始对齐，不得默认从歌曲0秒开始。

### 3.3 第三阶段：动态压力曲线

对同一固定reference做时间缩放：

```text
0.75x / 1.00x / 1.25x / 1.50x
```

必要时在MuJoCo中继续增加倍率，直到出现明确tracking退化或稳定性边界。能力曲线给出的是特定reference分布下的经验范围，不应直接表述为SONIC绝对硬上限。

## 4. 固定实验条件

每次实验必须记录并固定：

- AudioMimic Git commit；
- SONIC/GR00T-WholeBodyControl Git commit；
- SONIC policy与planner文件身份；
- motion PKL路径、SHA256、route、song ID、training seed和sampling seed；
- MuJoCo模型、初始姿态和场景；
- root quaternion order；
- source motion FPS与SONIC reference FPS；
- packet mode、preview horizon与reference safety设置；
- playback rate；
- startup wait、对齐段和hold段；
- feedback、sim-state和reference端口；
- 是否开启offscreen rendering或image publish；
- 主机、GPU、Python环境和实验日期。

定量实验默认关闭MuJoCo image publish，避免rendering降低realtime factor。视频实验单独运行，不与正式tracking数据混用。

## 5. 必须采集的原始数据

### 5.1 Reference

至少保存：

```text
reference frame_index
reference timestamp / intended execution time
joint_pos_target [T,29]
joint_vel_target [T,29]
root orientation target
source motion frame index
C4 packet index
```

### 5.2 SONIC Feedback

至少保存：

```text
SONIC feedback index
feedback receive monotonic time
controller/ROS timestamp（若有效）
body_q_target [T,29]
body_q_measured [T,29]
body_dq [T,29]
base_quat_target / measured
base_trans_target / measured
```

### 5.3 MuJoCo/Robot State

至少保存：

```text
sim_time或robot monotonic timestamp
base_position
base_quat
base_linear_velocity
base_angular_velocity
fall/reset标志或可推导字段
```

### 5.4 Music

至少保存：

```text
audio path与SHA256
audio_start_seconds
audio playback monotonic start time
beat/onset timestamps
因果音乐特征的frame index与timestamp（第二阶段）
```

## 6. 时间对齐协议

不能直接按数组下标比较reference与tracked motion。统一时间轴优先级为：

```text
reference frame_index/intended time
-> SONIC feedback timestamp/index
-> MuJoCo sim_time
```

处理步骤：

1. 删除startup、measured-pose alignment和hold段，只保留正式reference执行区间。
2. 将target与measured joint state插值到统一、单调的50 Hz时间轴。
3. 检查重复帧、缺帧、timestamp回退和sim reset。
4. 保留未补偿时延的Raw结果。
5. 通过cross-correlation估计全局与分组lag。
6. 将tracked motion按估计lag平移，再计算lag-compensated结果。
7. 音乐指标使用同一个audio clock，不允许分别为reference和tracked motion独立选择最优音乐偏移。

同时报告：

```text
Raw gap = 实际端到端执行误差
Lag-compensated gap = 补偿时间延迟后的动作形状/幅度误差
```

## 7. 完整评价体系

本节是本实验的执行摘要。完整 metric registry 及每项指标的适用条件见
[`eval/evaluation_map_v1.json`](../../eval/evaluation_map_v1.json)。

### 7.1 评价原则

本项目不使用单一综合分数替代多维评价。特别是，BAS只描述动作beat与音乐beat
的局部时间接近程度，不能评价动作自然性、表现力、风格、乐句结构或机器人可执行
性。当前motion-only在线模型没有显式beat控制，BAS只能作为辅助诊断；当模型加入
明确beat target或beat assignment后，再将Beat Precision/Recall/F1与BAP作为
可控性指标。

所有适用的动作和音乐指标都应同时计算在`M_ref`与`M_exec`上，并报告：

```text
Reference quality       = metric(M_ref)
Execution quality       = metric(M_exec)
Execution retention     = metric(M_exec) / metric(M_ref)
Execution degradation   = metric(M_exec) - metric(M_ref)
```

比值只用于具有合理零点且分母不接近零的非负指标；RMSE、lag和错误率主要报告差值，
避免使用难以解释的比值。`M_ref`与`M_exec`比较前先按第6节估计tracking lag；音乐
时钟保持固定，不得分别为reference和execution选择最优音乐偏移。

### 7.2 生成动作质量：M_ref

| 维度 | 核心指标 | 解释与限制 |
|---|---|---|
| 分布真实性 | FIDk、FIDg | 比较动力学与几何特征分布；只作辅助，不能替代人类判断 |
| 物理合理性 | PFC、FSR、ground penetration | 评价身体动力学、滑脚和地面穿透 |
| 平滑性 | velocity/acceleration/jerk P50、P95、P99、max | 检测异常尖峰和不可执行动态 |
| 在线连续性 | C4 boundary position/velocity jump | 评价相邻commit是否出现拼接跳变 |
| 长期活性 | motion energy、static ratio | 检测平均态、冻结和动作强度不足 |
| 长期非重复性 | pose repetition、self-similarity、autocorrelation | 检测短动作循环与模式坍缩 |
| 多样性 | Divk、Divg、同音乐跨seed距离 | 多样性应接近GT，不以越大越好 |
| Root与接触 | root/yaw drift、contact consistency、joint-limit margin | 评价长期漂移、接触和关节可行性 |

FIDk/FIDg在AIST++等有限数据集上可能与人类判断不一致，并可能被抖动等异常动态
影响。正式结果必须同时提供物理、长期连续性和人类评价，且FID使用固定特征提取器、
相同数据划分和相同序列长度。

### 7.3 音乐与舞蹈匹配：A-M_ref和A-M_exec

| 层次 | 指标 | 解释 |
|---|---|---|
| 局部拍点 | BAS | motion beat到最近music beat的接近程度，只作辅助 |
| 显式节拍控制 | Beat Precision/Recall/F1、BAP | 仅在有明确beat target/assignment时作为核心指标 |
| 动态响应 | onset-motion-energy correlation | 音乐onset与动作能量包络的相关程度 |
| 响应时间 | onset response lag、peak timing error | 音乐变化到动作响应的延迟 |
| 节奏速度 | tempo consistency、phase error | 动作周期与音乐tempo/phase是否一致 |
| 音乐语义 | Audio-motion R@1/R@3、MMDist | 音乐与动作在联合表示空间中的可检索性和距离 |
| 长期结构 | phrase-boundary response | 乐句、段落和强度变化是否引发动作组织变化 |
| 风格情绪 | genre/style/emotion consistency | 动作语义是否符合音乐类型与情绪 |

当前优先实现onset-energy correlation与response lag，因为它们不要求模型已经具备
显式beat assignment，也比BAS更直接地描述音乐变化是否引发动作响应。M2/M4必须
从PKL的`audio_start_seconds`播放匹配音频；没有可靠音频时钟的run不得进入音乐
指标统计。

### 7.4 SONIC执行质量：M_ref-M_exec

| 维度 | 指标 | 解释 |
|---|---|---|
| 稳定性 | Success Rate、time-to-fall、minimum base height | 是否完成动作及稳定裕量 |
| 姿态跟踪 | joint RMSE/EMPJPE | 实际关节与reference的误差 |
| 身体跟踪 | EMPKPE或key-body position error | 关键身体部位的空间误差 |
| 时序跟踪 | global/body-part lag、deadline miss | tracker延迟及系统实时性 |
| 动态保留 | amplitude、energy、band-power retention | 幅度、强度和频率细节保留程度 |
| 接触执行 | foot slip、contact timing error | reference接触是否被正确执行 |
| 音乐性保留 | BAS/onset/tempo/phrase retention | tracker执行后还保留多少音乐表达 |

### 7.5 人类审美与系统偏好

“舞蹈是否优美”没有可靠的单一自动指标。论文结果必须包含盲测或成对比较，问题
至少拆分为：自然性、舞蹈感、表现力、长期连贯性、节奏匹配、音乐风格/情绪匹配、
机器人执行后的表现力保留。分别比较generator与baseline、execution与baseline，
以及同一条`M_ref`与对应`M_exec`。

报告pairwise preference、Bradley-Terry或Elo分数及95% bootstrap置信区间。
视频应隐藏方法名称、随机化左右位置，并保证音乐、镜头、渲染和时长一致。人类评价
是审美与语义结论的主要证据，自动指标用于解释原因和保证可重复性。

### 7.6 实时系统指标

- audio-to-feature、feature-to-motion、motion-to-reference和reference-to-execution延迟；
- generator平均/P95/P99推理时间、30 FPS deadline miss rate；
- reference packet age、丢包、重复帧和fallback触发率；
- 连续60 s及更长rollout的survival和realtime factor。

### 7.7 与现有论文指标的对应关系

| 工作 | 主要评价 |
|---|---|
| [AIST++/FACT](../papers/aist-fact.pdf) | FIDk/FIDg、Distk/Distg、Beat Alignment、用户比较 |
| [EDGE](../papers/edge.pdf) | 大规模人类偏好/Elo、PFC、Beat Alignment、Distk/Distg，并质疑FID可靠性 |
| [Lodge](../papers/lodge.pdf) | 长序列FIDk/FIDg、FSR、Divk/Divg、BAS、用户评价、效率 |
| [Beat-It](../papers/beat-it.pdf) | PFC、BAS、Divk/Divg、KPD、BAP、用户评价；不使用FID |
| [RoboPerform](../papers/robopeform.pdf) | R@K、MMDist、BAS、Success、EMPJPE、EMPKPE、系统延迟 |
| [DiscoForcing](../papers/DiscoForcing.pdf) | FIDk/FIDg、FSR、Divk/Divg、BAS、ms/frame与FPS |

AudioMimic在这些工作基础上增加`M_ref -> M_exec`保留率，使motion generation质量、
音乐匹配和机器人执行损失能够被分别观察，而不是把generator与tracker混成一个结果。

## 8. Tracking Gap指标详细定义

### 8.1 Tracking Fidelity

关节位置RMSE：

```text
RMSE_q = sqrt(mean((q_tracked - q_reference)^2))
```

关节速度RMSE：

```text
RMSE_dq = sqrt(mean((dq_tracked - dq_reference)^2))
```

必须分别报告：

- Full body；
- Legs（前12个G1关节）；
- Waist（中间3个关节）；
- Arms（后14个关节）；
- 每个关节的median、P95和max。

Root报告orientation、yaw、base height和angular velocity误差。SONIC当前不直接追踪global root XY，不将global XY RMSE作为主要tracker失败指标。

### 8.2 Tracking Delay

对关节位置或低通后的关节速度做cross-correlation：

```text
lag_j = argmax crosscorr(q_reference_j, q_tracked_j)
```

报告每关节、身体分组和全身的median/P95 lag，单位同时使用frame和ms。

动作峰值延迟：

```text
peak timing error = t_tracked_peak - t_reference_peak
```

峰值匹配必须使用预先固定的时间容差，不能按每条轨迹事后调参。

### 8.3 Amplitude Retention

每关节稳健幅度：

```text
Amplitude = P95(q) - P5(q)
Amplitude retention = Amplitude_tracked / Amplitude_reference
```

同时报告每关节和身体分组的分布，reference幅度接近零的关节应排除或单独标记，避免比值发散。

### 8.4 Motion Energy Retention

```text
Energy = mean(||joint_velocity||^2)
Energy retention = Energy_tracked / Energy_reference
```

解释：

- `< 1`：动作强度被削弱；
- `约等于 1`：总体能量保留；
- `> 1`：可能存在补偿、振荡或tracking噪声。

### 8.5 Frequency Retention

使用Welch PSD，固定窗长、overlap和频段：

```text
Low:  0--1 Hz
Mid:  1--3 Hz
High: 3--8 Hz
```

```text
Band-power retention
= tracked band power / reference band power
```

分别报告legs、waist、arms。正式分析前检查50 Hz采样下的Nyquist限制与滤波设置。

### 8.6 Music-feature Retention

对reference motion和tracked motion使用完全相同的动作beat检测器与音乐beat：

```text
Beat F1 retention = Tracked Beat F1 / Reference Beat F1
BAS retention = Tracked BAS / Reference BAS
Beat recall drop = Reference Recall - Tracked Recall
Additional beat delay = Tracked timing error - Reference timing error
```

按身体部位报告：

```text
Wrist Beat F1 retention
Foot Beat F1 retention
Torso Beat F1 retention
Full-body Beat F1 retention
```

### 8.7 Contact与稳定性

- Foot-contact timing error；
- Contact-on-beat retention；
- Near-support-on-beat retention；
- Foot sliding变化；
- Ground penetration；
- Base height median/min；
- Fall/reset与首次失稳时间。

### 8.8 Reference Executability Predictors

每条reference计算：

- joint velocity P95/max；
- acceleration P95/max；
- jerk P95/max；
- C4 boundary position/velocity jump；
- root yaw velocity；
- joint-limit margin；
- legs/waist/arms分组动态。

分析这些变量与tracking RMSE、lag、energy retention和beat retention的相关性。

## 9. 结果图表

每组实验至少生成：

1. Reference与tracked代表关节曲线：腕、膝、踝、腰。
2. 每关节Raw与lag-compensated RMSE热图。
3. 每关节tracking lag热图。
4. Low/Mid/High频带功率保留率图。
5. Amplitude与energy retention身体分组图。
6. Reference与tracked Beat F1/BAS对比表。
7. Reference动态与tracking gap散点图。
8. Base height与fall/reset时间线。

论文结果至少拆成四张表，避免把generator质量、音乐匹配和tracker能力混成一个分数：

1. Reference Motion Quality：FIDk/g（辅助）、PFC、FSR、jerk、static、repetition、Div和C4 continuity。
2. Music-Motion Correspondence：BAS（辅助）、onset correlation、response lag、tempo/phase、R@K和MMDist。
3. Robot Execution：Success、RMSE/EMPJPE、EMPKPE、lag、foot slip、energy/amplitude/frequency retention。
4. Human Evaluation：naturalness、dance-likeness、expressiveness、long-term coherence和music appropriateness。

其中Robot Execution主表建议包含：

| Route | RMSE raw | RMSE aligned | Lag ms | Amp Ret. | Energy Ret. | High-freq Ret. | Beat F1 Ret. | Min height | Survival |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## 10. 数据目录与命名

建议统一写入：

```text
eval/generation_to_execution_gap/
  manifest.json
  <route>_<song>_<seed>_<repeat>/
    run_manifest.json
    offline_sonic_playback.json
    reference.npz
    sonic_feedback.json
    sim_state.json
    s66_exec.json
    metrics.json
    plots/
    video/                 # 可选，不作为正式定量输入
```

Run ID示例：

```text
m2_song098_seed1234_r01
```

任何重复运行不得覆盖旧目录。失败运行也必须保留manifest和失败原因。

## 11. 数据质量门槛

进入正式统计前，每次run必须满足：

- reference和feedback均非空且字段维度正确；
- 正式段时间戳严格单调；
- reference 50 Hz帧数与duration一致；
- feedback有效覆盖率不低于99%；
- 无无法解释的sim-time回退；
- 音频起点和reference起点可追溯；
- root quaternion与joint order在manifest中声明；
- 指标代码版本和Git commit已记录。

不满足门槛的run标记为invalid，不与正常run求均值。

## 12. 执行顺序

### Phase A：指标实现与单条轨迹

- [x] 建立统一reference/feedback时间轴。
- [x] 实现Raw与lag-compensated RMSE。
- [x] 实现lag、amplitude、energy和frequency retention。
- [x] 使用M2 song098跑3次60 s重复。
- [ ] 人工检查曲线和视频，验证正负lag方向与关节映射。

#### Phase A中期结果（2026-08-19）

同一条M2 song098/seed1234轨迹在`packet_mode=c4`、1.0x、无reference
safety下完成两次独立60 s运行：

| Run | Raw RMSE | Lag-comp. RMSE | Lag | Amp. retention | Energy retention | First fall |
|---|---:|---:|---:|---:|---:|---:|
| r01 | 0.2588 rad | 0.2330 rad | 200 ms | 0.924 | 0.571 | 35.2 s |
| r02 | 0.2730 rad | 0.2482 rad | 220 ms | 0.936 | 0.634 | 35.8 s |

Tracking指标只统计首次跌倒前的数据；survival使用完整60 s记录。两次首次失稳时间
接近，说明当前配置不能视为60 s稳定通过。该结论只适用于当前固定轨迹和C4播放
接口，不能据此声称所有M2轨迹均不可追踪。

r02在33--37 s区间的reference峰值包括右肘14.09 rad/s、左膝5.78 rad/s；
该区间包含较高动态，但尚不能单独证明它是跌倒原因。下一项使用完全相同的轨迹、
速度和SONIC policy，仅将`packet_mode`改为`full`，隔离C4 packet handoff影响。

`full_r01`同样失稳，首次跌倒为37.6 s；跌倒前Raw/lag-compensated RMSE为
0.2755/0.2539 rad，估计lag为180 ms。C4两次与full一次均在35--38 s首次
失稳，故C4 packet handoff不是主要失败来源。当前证据指向固定M2 reference的
动态与SONIC可执行范围不匹配。下一项对同一完整reference使用0.75x播放，检验
失稳是否随source动作片段移动，以及降低速度能否通过60 s/完整轨迹稳定性门槛。

0.75x full-packet组完整执行80.0 s且没有跌倒。相对1.0x full组，Raw RMSE从
0.2755降至0.2364 rad，lag-compensated RMSE从0.2539降至0.2142 rad，动作
能量保留率从0.505升至0.735；全身幅度保留率为0.943，估计lag仍为180 ms。
因此当前M2轨迹的主要部署瓶颈是原速reference的动态强度与SONIC能力不匹配，
而不是C4切包、60 s运行时长或单纯全局时延。频带保留率还需结合曲线检查，
因为大于1的高频功率可能来自控制振荡，不能直接解释为动作细节保留更好。

0.9x full-packet组在64.9/66.7 s首次跌倒，Raw/lag-compensated RMSE为
0.2709/0.2459 rad，能量保留率为0.691，未通过完整轨迹门槛，但稳定时间显著
长于1.0x。下一档测试0.85x。该组曾以相同run ID运行两次，第二次覆盖第一次，
因此只保留最后一次记录；streamer随后加入非空输出目录防覆盖检查。

0.85x组在47.8/70.6 s首次跌倒，对应原始轨迹约40.6 s；跌倒前Raw/
lag-compensated RMSE为0.2500/0.2275 rad，能量保留率为1.094。失稳时间没有
随playback rate严格单调变化，说明初态、policy扰动和控制振荡也会影响结果，
不能把单次运行解释为确定性的速度硬阈值。

Reference动态对比显示，稳定0.75x组全身velocity P95/max为2.29/8.59
rad/s，acceleration P95/max为24.3/132.1 rad/s^2，jerk P95为524.6
rad/s^3；失败0.85x组分别为2.57/9.56、30.0/169.2和663.8。该区间是
当前轨迹与当前SONIC policy下的经验范围，不是G1或SONIC的普适硬上限。后续
generator约束应同时覆盖velocity、acceleration、jerk和腿/臂分组，而不能只做
关节位置clip。

0.80x组在43.0/75.0 s首次跌倒，对应原始轨迹约34.4 s；Raw/
lag-compensated RMSE为0.2693/0.2355 rad，lag为280 ms，能量保留率为0.702。
该source-equivalent失稳位置与1.0x的35--38 s接近，进一步说明该动作区间是高风险
片段。由于目前只有0.75x单次通过，下一步不继续测试更多倍率，而是对0.75x做
独立重复，估计稳定率并排除偶然成功。

0.75x第二次运行在69.0/80.0 s首次跌倒，第一次完整通过，因此降速只能提高
成功概率，尚不能保证鲁棒执行。两次运行初始measured joint RMSE仅0.027 rad，
但失败组的分窗lag从前10 s的320 ms逐步增长到60--70 s的400 ms，分窗RMSE
从0.147增长到0.407 rad；成功组lag大多维持在140--220 ms。现有命令未显式
将measured pose平滑对齐到M2首帧，下一项应在1.0x加入3 s measured-state
alignment和1 s hold，先排除启动边界误差长期放大的影响。分析器默认排除该
alignment前缀，不将其计入正式tracking指标。

1.0x measured-state alignment首轮通过：3 s平滑对齐、1 s首帧hold后，完整
60 s M2动作无fall/reset，最低base height为0.651 m。排除4 s启动前缀后，
Raw RMSE为0.2445 rad，lag-compensated RMSE为0.2230 rad，全局lag为140 ms，
幅度/能量保留率为0.895/0.576。相对未对齐1.0x full组，Raw RMSE从0.2755
下降11.3%，lag-compensated RMSE从0.2539下降12.2%，且survival从37.6 s
提高到完整60 s。

因此此前无对齐实验不能作为“M2轨迹不可执行”的证据；它们主要证明从任意
measured pose直接跳到M2首帧会造成长期tracking divergence。当前新的工作假设
是：正确的启动边界能够使该M2轨迹稳定执行，但仍需至少两次独立alignment重复
验证成功率。速度、加速度和jerk仍需作为tracking fidelity与真机裕量指标，不能
再将其单独视为本次跌倒的已证实根因。

1.0x alignment第二次独立运行也完整通过：60 s无fall/reset，最低base height
0.625 m，Raw/lag-compensated RMSE为0.1681/0.1664 rad，全局lag为20 ms，
幅度/能量保留率为0.910/0.493。前两次alignment运行成功率为2/2；r01和r02的
tracking误差与lag存在运行间方差，但稳定性结论一致。完成r03后固定该启动协议，
再进入M0/M2/M4同条件对比。

第三次alignment运行同样完整通过，最终M2 aligned成功率为3/3。三次正式60 s
动作段的Raw RMSE为0.1933 +/- 0.0443 rad，lag-compensated RMSE为
0.1850 +/- 0.0329 rad，最低base height为0.633 +/- 0.016 m；幅度和能量
保留率分别为0.906 +/- 0.010和0.522 +/- 0.047。全局lag三次为140/20/20 ms，
存在运行间方差，但没有导致稳定性失败。至此冻结`3 s measured alignment + 1 s
hold + full packet + 1.0x`为固定轨迹SONIC评估协议。

### Phase B：音乐特征保留

- [x] 对reference/tracked使用同一动作beat检测器。
- [x] 实现onset-speed/impact correlation和response lag；tempo consistency待实现。
- [x] 实现BAS及其reference-to-execution retention，明确其仅是辅助指标。
- [ ] 在存在显式beat target时再实现Beat Precision/Recall/F1与BAP。
- [ ] 实现wrist/foot/torso/full-body分组音乐响应结果。
- [x] 验证预切片音乐与`audio_start_seconds`。

#### Phase B第一轮结果（2026-08-19）

完整报告见：
[`eval/motion_music_execution/first_round_20260819/FIRST_ROUND_ANALYSIS.md`](../../eval/motion_music_execution/first_round_20260819/FIRST_ROUND_ANALYSIS.md)。

在song098匹配子集中，M0/M2/M4的music-to-motion BAS分别为
`0.257/0.247/0.242`，best onset-impact correlation分别为
`0.032/0.025/0.031`。M2/M4没有在当前自动音乐指标上优于无音乐M0；所有impact
correlation均低于0.1，其最优lag不具有可靠解释。该结果进一步说明BAS不能单独验证
音乐条件是否有效，也不能支持“舞蹈优美”的结论。

M2 reference的motion energy为`1.934 +/- 0.015 rad^2/s^2`，jerk P95为
`718.1 +/- 3.3 rad/s^3`，static/repeated-pose ratio接近零，C4 boundary位置/
速度jump与普通帧之比为`0.989/0.960`。因此当前M2证据支持动作连续、活跃且没有
明显commit拼接异常，但审美质量仍需盲测。

M2经SONIC执行后的aggregate energy retention为`46.8 +/- 2.7%`；旧定义的
median-per-joint energy/amplitude retention为`52.2 +/- 4.7%`和
`90.6 +/- 1.0%`。三次执行全部稳定，但BAS平均下降0.035且方差较大。当前主要
问题是动态表达衰减，而不是60 s稳定性。

### Phase C：M0/M2/M4矩阵

- [x] 固定歌曲、seed、K64起点和duration。
- [x] 每条轨迹至少3次SONIC重复。
- [x] 汇总route均值和标准差（置信区间待扩展seed后报告）。
- [ ] 分析M2/M4音乐优势经过tracking后保留多少。

#### Phase C当前结果与结论（2026-08-19）

M0 song098/seed1234 aligned r01完整60 s通过，无fall/reset，最低base height
0.632 m；Raw/lag-compensated RMSE为0.1776/0.1759 rad，lag为20 ms，幅度/
能量保留率为0.913/0.475。该结果与M2 aligned r02/r03接近，当前单次M0证据
不支持“音乐条件M2更难追踪”。完成M0和M4各三次重复后再比较route组均值与方差。

M0三次aligned运行全部通过，Raw RMSE为0.1781 +/- 0.0007 rad，
lag-compensated RMSE为0.1763 +/- 0.0005 rad，lag为20.0 +/- 0.0 ms，最低
base height为0.6369 +/- 0.0055 m，幅度/能量保留率为0.9116 +/- 0.0012和
0.4682 +/- 0.0075。M2同样3/3稳定，但tracking运行间方差更大；主要由M2 r01
的140 ms lag造成。当前每route三次的样本量不足以声称M2显著更差，下一步按完全
相同协议采集M4三次。

M4三次完成后，M0/M2/M4 song098、seed1234、60 s矩阵结果为：

| Route | Success | Raw RMSE | Lag-comp. RMSE | Lag | Min height | Amp. ret. | Energy ret. | 3--8 Hz ret. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 3/3 | 0.1781 +/- 0.0007 | 0.1763 +/- 0.0005 | 20 +/- 0 ms | 0.6369 +/- 0.0055 m | 0.9116 +/- 0.0012 | 0.4682 +/- 0.0075 | 0.2269 +/- 0.0106 |
| M2 | 3/3 | 0.1933 +/- 0.0443 | 0.1850 +/- 0.0329 | 60 +/- 69 ms | 0.6327 +/- 0.0159 m | 0.9062 +/- 0.0095 | 0.5220 +/- 0.0472 | 0.3946 +/- 0.3081 |
| M4 | 3/3 | 0.1694 +/- 0.0006 | 0.1678 +/- 0.0005 | 20 +/- 0 ms | 0.6583 +/- 0.0049 m | 0.8715 +/- 0.0042 | 0.5030 +/- 0.0009 | 0.2443 +/- 0.0074 |

所有route在正确alignment下都稳定，音乐条件没有导致SONIC稳定性失败。M4的
tracking误差最低、base height裕量最高，但幅度保留率最低；M2方差主要由r01
的140 ms lag造成，r02/r03与M4接近。三条路线的动作能量仅保留约47--52%，
3--8 Hz功率保留约23--39%，说明主要generation-to-execution gap是快速动作细节
衰减，而不是fall/reset。下一阶段必须加入Beat F1、BAS、onset response和身体
部位节奏保留，才能判断音乐优势经过tracker后剩余多少。

三条route在本阶段的身份必须明确：M0是无音乐条件的frozen unconditional parent；
M2是使用predicted future-music sidecar生成的严格因果候选；M4是使用真实未来音乐
特征的oracle上限。这里使用的三条都是提前生成并保存的60 s PKL，由SONIC离线回放，
不是运行时读取实时音乐并在线生成的端到端实验。

分身体部位的Raw RMSE进一步显示，手臂误差持续高于腿部：M0腿/臂为
0.1581/0.2019 rad，M2为0.1759/0.2163 rad，M4为0.1534/0.1899 rad。全身
速度RMSE分别为1.1030、1.1519和0.9466 rad/s。该结果支持“快速上肢细节更难被
tracker保留”的方向，但还需要更多song和seed验证。

当前可以下的结论：

1. 在冻结的`3 s measured-state alignment + 1 s hold + full packet + 1.0x`
   协议下，当前M0/M2/M4固定轨迹均可被SONIC稳定执行60 s；此前无alignment摔倒
   主要是错误启动边界证据，不能用于否定M2可执行性。
2. 当前主要gap是fidelity而不是survival：幅度约保留87--91%，动作能量只保留
   47--52%，多数run的3--8 Hz功率只保留约21--25%。
3. M2的高频均值0.395不能解释为明显优于M0/M4，因为r01为0.750，而r02/r03仅为
   0.224/0.209；当前均值被单次tracking lag异常放大，方差0.308也明显过高。
4. M4在当前单轨迹上tracking RMSE最低，但样本只有song098、seed1234和三个tracker
   repeat；不能据此宣称M4生成质量或音乐匹配显著最好。
5. 现有数据没有计算reference/execution的音乐指标，也没有人类审美评价，因此尚不能
   回答M2/M4是否比M0更像舞蹈、更优美或更匹配音乐。

下一轮最小扩展为：至少3首歌x3个generation seed，每条reference做3次SONIC重复；
同时计算`M_ref`和`M_exec`上的onset-energy correlation、response lag、BAS及动作
质量指标，并加入成对人类评价。只有这样才能区分route差异、tracker随机性和单条动作
内容差异。

### Phase D：动态能力与模型约束

- [ ] 汇总0.75x--1.50x能力曲线。
- [ ] 建立动态指标与tracking gap的回归/相关分析。
- [ ] 给出generator的经验训练约束和质量门槛。
- [ ] 将结果反馈给tracker-aware generator训练。

## 13. 标准回放入口

固定轨迹正式评估使用已经冻结的离线Open-loop streamer协议。正式数据直接写入统一
目录，`r01`完成后重置MuJoCo和SONIC，再依次将`RUN_ID`改为`r02`和`r03`：

```bash
cd ~/AudioMimic
conda activate audiomimic

M2_PKL=~/Musics2Dance-prior-dev/onlinegeneratedmotion/m2_predicted_fms/m2_train1234_sample1234_u100000_best_song098.pkl
RUN_ID=m2_song098_seed1234_full_rate100_aligned_r01

python stream_to_sonic.py \
  --pkl "$M2_PKL" \
  --root_quat_order xyzw \
  --packet_mode full \
  --playback_rate 1.0 \
  --max_seconds 60 \
  --align_from_feedback_seconds 3 \
  --align_hold_seconds 1 \
  --output_dir "eval/generation_to_execution_gap/$RUN_ID" \
  --record_feedback \
  --feedback_port 5557 \
  --sim_state_port 5559 \
  --reference_safety none \
  --sonic_reference_fps 50 \
  --startup_wait 2
```

每个run应生成`offline_sonic_playback.json`、`reference.json`、
`sonic_feedback.json`、`sim_state.json`和`s66_exec.json`。三次采集完成后运行：

```bash
cd ~/AudioMimic
conda activate audiomimic

python eval/analyze_generation_execution_gap.py \
  --runs \
    eval/generation_to_execution_gap/m2_song098_seed1234_full_rate100_aligned_r01 \
    eval/generation_to_execution_gap/m2_song098_seed1234_full_rate100_aligned_r02 \
    eval/generation_to_execution_gap/m2_song098_seed1234_full_rate100_aligned_r03 \
  --output_dir eval/generation_to_execution_gap/phase_a_summary
```

每次repeat前重置MuJoCo与SONIC到相同初态。不要用旧版 streamer
产生的目录作为正式数据，因为旧记录没有`playback_started_monotonic_seconds`
和逐帧`reference.json`。

## 14. 完成标准

本实验路线完成时，应能够用数字回答：

1. SONIC平均和P95 tracking delay是多少？
2. 哪些关节和身体部位的Raw/lag-compensated误差最大？
3. 动作幅度、能量和高频成分分别保留多少？
4. `M_ref`本身是否自然、丰富、连续且物理合理？
5. `M_ref`与`M_exec`分别在BAS、onset response、tempo、语义和乐句层面多匹配音乐？
6. Reference的音乐性、动作幅度、能量和频率细节经过tracking后下降多少？
7. 人类是否认为动作自然、优美、有舞蹈感并与音乐匹配？
8. 哪些reference动态最能预测tracking失败或动作表达衰减？
9. Generator应采用哪些动态损失、阈值或tracker-aware训练策略？

该结果将作为后续Closed-loop必要性与generator优化方向的直接证据，而不是仅依赖视频主观判断。
