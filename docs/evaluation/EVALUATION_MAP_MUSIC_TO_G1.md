# AudioMimic Music-to-G1 Evaluation Map v1.2

更新日期：2026-08-23

## 1. 目的与适用范围

本文档规定 AudioMimic 从音乐条件生成到 SONIC 执行的统一评价协议。它不是一个可选
指标列表，而是后续模型比较、消融实验和论文表格必须遵守的评价地图。

所有指标的层级分类以
[`METRIC_TAXONOMY_MUSIC_DANCE_G1.md`](METRIC_TAXONOMY_MUSIC_DANCE_G1.md) 为准；
机器可读映射为 `eval/metric_taxonomy_v1.json`。任何新增指标必须先指定唯一的
`大模块 / 子模块`，corruption 操作只能作为校准方法，不能作为模型评分指标。

论文级 claim、三级 GT oracle、sealed-test 设计、实验矩阵和验收 gate 见
[`ICRA_GT_CALIBRATED_EVALUATION_PLAN.md`](ICRA_GT_CALIBRATED_EVALUATION_PLAN.md)。本文档定义
指标本身；论文计划定义每项结论需要哪些证据。

系统按以下因果链拆分：

```text
audio A -> motion generator -> reference motion M_ref
        -> SONIC tracker -> executed motion M_exec
```

评价必须回答四个彼此独立的问题：

1. `M_ref` 本身是否是自然、连续、丰富且物理合理的舞蹈；
2. `M_ref` 是否与输入音乐在节奏、动态、语义和长期结构上匹配；
3. SONIC 是否忠实、稳定地执行了同一条 `M_ref`；
4. `M_exec` 最终是否仍然是与音乐匹配、可观看且可部署的机器人舞蹈。

任何实验不得用一个总分替代上述四个问题。特别地，BAS 不能证明动作自然，FID 不能
证明动作好看，Success Rate 也不能证明 tracker 保留了舞蹈表现力。

## 2. 证据等级

| 等级 | 含义 | 使用规则 |
|---|---|---|
| Hard gate | 数据、稳定性或实时性的必要条件 | 任一失败时，该 run 不得被称为可部署结果，但失败样本仍必须计入成功率 |
| Core metric | 支撑论文主结论的核心指标 | 所有适用实验必须报告，不能选择性省略 |
| Supplementary | 有解释力但有已知局限的辅助指标 | 与 core metric 联合解释，不得单独作为验收依据 |
| Conditional | 仅在任务具有对应监督或控制目标时适用 | 不满足适用条件时标记 N/A，不得填 0 |

指标来源分为两类：

| 来源 | 含义 |
|---|---|
| Literature | 来自 FACT、EDGE、Lodge、Beat-It、RoboPerform、DiscoForcing、LRCM、MATHDance、PAMD、InfiniteDance、DanceBA 等本地论文 |
| AudioMimic | 为在线生成、SONIC 跟踪和表现力保留新增的系统指标 |

AudioMimic 新增指标不是冒充已有论文标准。论文中应明确写为 system-specific metrics，
并公开定义、实现和阈值。

本版的逐篇论文核对、指标取舍和 music feature policy 见
[`LITERATURE_METRIC_AUDIT_20260823.md`](LITERATURE_METRIC_AUDIT_20260823.md)。本版冻结后，
所有 generator、GMR/retargeting 和 SONIC execution 实验都必须使用同一套适用指标。
BAS 是 beat-alignment suite 的 **core** 指标，在所有 paired music-motion 实验中**强制报告**，不能省略。
它仍然不能单独代表完整音乐性或舞蹈质量；完整 beat-alignment suite 还必须联合
event Precision/Recall/F1（适用时）、onset/impact correlation、response lag、
tempo error 和 phase error。

指标有效性校准见
[`eval/benchmark_v1/gt/benchmark_validity_v1/REPORT.md`](../../eval/benchmark_v1/gt/benchmark_validity_v1/REPORT.md)。
当前已通过 GT corruption 验证的指标只对其声明的退化类型负责；未通过或尚未校准的
指标不能直接进入论文主结论。GT 本身被视为高质量经验分布，不强行设为每项指标的
`1.0`，也不要求每条 GT 在所有指标上都达到最大值。

## 2.1 冻结的主指标组

| 组 | 必须报告 | 说明 |
|---|---|---|
| Dance quality | FIDk/FIDg、PFC、FSR、Divk/Divg、velocity/acceleration/jerk、energy、freeze/repetition、penetration/jitter、human preference | 评价动作是否自然、连续、丰富、物理合理 |
| Music adaptation | BAS、event Precision/Recall/F1（适用时）、onset correlation、response lag、tempo/phase、R@K/MMDist（冻结 encoder 后） | 评价动作是否响应音乐节奏、动态、结构和风格 |
| SONIC execution | Success/TTF、EMPJPE/EMPKPE、raw/aligned RMSE、lag、amplitude/energy/band/contact retention | 评价 tracker 吃掉多少动作表现力 |
| Realtime system | deadline miss、latency P50/P95/P99、drop/fallback、realtime factor | 评价是否是真正在线系统 |

`BAS` 的解释必须和 beat count、event F1 或 onset response 一起出现。EDGE 已指出，
只按局部运动 beat 与音乐 beat 的距离会误罚 half-time/double-time 和合法的过渡动作，
所以 BAS 是重要证据，但不能作为唯一验收标准。

## 3. 总体 Evaluation Map

| 层级 | 评价对象 | 必答问题 | 核心证据 |
|---|---|---|---|
| P0 数据与协议 | A、M_ref、M_exec、时钟 | 输入格式和时间轴是否有效 | schema、coverage、audio offset、joint/quaternion order |
| G 生成质量 | M_ref | 是否是合格的长时 G1 舞蹈 | PFC、FSR、动态统计、C4 连续性、重复率、Div；FID 辅助 |
| M 音乐匹配 | A 与 M_ref | 动作是否确实由对应音乐解释 | BAS beat-alignment suite、R@K、MMDist、onset response、tempo/phase、phrase |
| X 可执行性 | M_ref | reference 是否位于 G1/SONIC 可执行域 | joint/root/contact limits、GT-calibrated dynamic envelope |
| T 跟踪质量 | M_ref 与 M_exec | SONIC 吃掉了多少动作信息 | Success、EMPJPE、EMPKPE、lag、amplitude/energy/band retention |
| E 执行后质量 | A 与 M_exec | 机器人实际跳出的动作是否仍匹配音乐 | 重算 G/M 指标并报告 retention/degradation |
| R 实时系统 | 全链路 | 是否严格因果且满足 deadline | 分阶段延迟、P95/P99、deadline miss、drop、realtime factor |
| H 人类评价 | M_ref、M_exec 视频 | 是否自然、像舞蹈、与音乐合适 | 盲测偏好、Bradley-Terry/Elo、95% CI |

## 4. P0：数据与协议硬门槛

| ID | 指标 | 判定规则 | 等级 | 来源 |
|---|---|---|---|---|
| P0-SCHEMA | Motion schema validity | `root_pos[T,3]`、`root_rot[T,4]`、`dof_pos[T,29]`，无 NaN/Inf，FPS 有效 | Hard gate | AudioMimic |
| P0-ORDER | Interface convention | G1 joint order、单位、坐标系、quaternion order 在 manifest 中固定并通过 round-trip test | Hard gate | AudioMimic |
| P0-TIME | Clock validity | audio、generator、reference 和 feedback 使用同一时间基准；记录 `audio_start_seconds` | Hard gate | DiscoForcing + AudioMimic |
| P0-COVER | Feedback coverage | 有效反馈覆盖率至少 99%，缺帧、重复帧和插值比例分别报告 | Hard gate | AudioMimic |
| P0-PAIR | Pairing validity | 比较 M0/M2/M4 时固定 song、clip start、length、generation seed；tracker 比较固定同一条 M_ref | Hard gate | AudioMimic |
| P0-LENGTH | Duration validity | 主长时实验每条至少 60 s；短时诊断不得混入主表 | Hard gate | Lodge/DiscoForcing + AudioMimic |

若 P0 失败，只能用于接口诊断，不得进入模型排名。

## 5. G：Generator 的动作质量

### 5.1 分布、物理和多样性

| ID | 指标 | 定义与解释 | 方向 | 等级 | 来源 |
|---|---|---|---|---|---|
| G-FID-K | FIDk | 生成与 GT 的 kinetic feature 分布距离 | ↓ | Supplementary | FACT、Lodge、DiscoForcing |
| G-FID-G | FIDg | 生成与 GT 的 geometric feature 分布距离 | ↓ | Supplementary | FACT、Lodge、DiscoForcing |
| G-PFC | Physical Foot Contact | 由身体加速度与足部静态接触关系评价运动学物理合理性 | ↓ | Core | EDGE、Beat-It |
| G-FSR | Foot Skating Ratio | 足部接近地面时仍发生明显水平滑动的帧比例 | ↓ | Core | Lodge、DiscoForcing |
| G-PEN | Ground penetration | 足部或关键 body 低于地面的深度和帧比例 | ↓ | Core | AudioMimic |
| G-DIV-K | Divk | kinetic feature 空间中的样本间距离 | → GT | Core | FACT、Lodge、Beat-It、DiscoForcing |
| G-DIV-G | Divg | geometric feature 空间中的样本间距离 | → GT | Core | FACT、Lodge、Beat-It、DiscoForcing |
| G-SEED-DIV | Same-audio seed diversity | 同一音乐跨 generation seed 的动作距离 | → GT | Core | AudioMimic |

FID 必须使用固定 extractor、数据 split、序列长度和采样数。EDGE 指出 AIST++ 小测试集上
FID 与人类判断可能不一致，Beat-It 因此不采用 FID；本项目只把 FID 作为辅助分布指标。
Div 也不是越高越好，抖动和不连续动作同样会提高距离，因此以 GT 区间为目标。

### 5.2 动态、连续性和长期行为

| ID | 指标 | 定义与解释 | 方向 | 等级 | 来源 |
|---|---|---|---|---|---|
| G-VEL | Joint velocity distribution | 全身及 legs/waist/arms 的 P50/P95/P99/max | → GT | Core | AudioMimic |
| G-ACC | Joint acceleration distribution | 检测突变和过强动态 | → GT | Core | AudioMimic |
| G-JERK | Joint jerk distribution | 检测高频抖动和不可执行尖峰 | ↓/→ GT | Core | AudioMimic |
| G-ENERGY | Motion energy | `mean(||dq||^2)`，衡量动作强度 | → GT | Core | AudioMimic |
| G-STATIC | Static ratio | 关节速度低于预注册阈值的帧比例 | → GT | Core | AudioMimic |
| G-FREEZE | Adaptive freeze proportion | 持续时间位于冻结区间且速度低于自适应阈值的帧比例 | → GT | Core | LRCM |
| G-FREEZE-LR | Freeze length regularity | 冻结片段长度标准差的倒数，检测异常长短冻结 | → GT | Supplementary | LRCM |
| G-REPEAT | Pose repetition | 间隔超过 2 s 仍近似相同的姿态比例及 self-similarity | ↓/→ GT | Core | AudioMimic |
| G-C4-POS | C4 position boundary jump | commit 边界位置跳变相对非边界的比值 | ≈1 | Core | AudioMimic |
| G-C4-VEL | C4 velocity boundary jump | commit 边界速度跳变相对非边界的比值 | ≈1 | Core | AudioMimic |
| G-ROOT | Root behavior | path length、net displacement、yaw drift、height distribution | → GT | Core | AudioMimic |
| G-JLIMIT | Joint-limit margin | 越界率、最小安全裕量、最大越界幅度 | 0 violations | Hard gate | AudioMimic |

当前 analyzer 已实现 velocity/acceleration/jerk、energy、static、repetition、C4 boundary
和 root 统计。PFC、FSR、FID/Div 的正式 G1 统一实现仍需接入固定 FK 和 extractor。

## 6. M：音乐与动作匹配

### 6.1 节奏与动态响应

| ID | 指标 | 定义与解释 | 方向 | 等级 | 来源 |
|---|---|---|---|---|---|
| M-BAS | Beat Alignment Score | 每个 kinematic beat 到最近 music beat 的指数加权距离 | ↑ | Core | FACT、EDGE、Lodge、Beat-It、DiscoForcing、RoboPerform |
| M-BEAT-F1 | Beat Precision/Recall/F1 | 在固定容差内评价预测 motion beat 对目标 beat 的命中和覆盖 | ↑ | Conditional | AudioMimic |
| M-BAP | Beat Assignment Precision | 生成动作是否服从指定的 beat assignment | ↑ | Conditional | Beat-It |
| M-KPD | Key Pose Distance | 指定 keyframe 上局部关节 Cartesian position 的 MSE | ↓ | Conditional | Beat-It |
| M-ONSET | Onset-motion response | onset strength 与 motion impact/energy envelope 的相关系数 | ↑ | Core | AudioMimic |
| M-RESP-LAG | Music response lag | onset 与 motion response 最大相关处的时差 | 0 | Core | AudioMimic |
| M-TEMPO | Tempo consistency | motion periodicity 与音乐 tempo/rhythm grid 的偏差 | ↓ | Core | AudioMimic |
| M-PHASE | Rhythm phase error | 匹配周期内动作 accent 相对音乐相位的圆周误差 | ↓ | Core | AudioMimic |
| M-RS | Rhythmic Score | 冻结/动作 accent 到期望 beat 的容差加权匹配分数 | ↑ | Supplementary | LRCM |

BAS 是 beat-alignment suite 的核心单向指标：它检查产生的 kinematic beat 是否靠近某个 music beat，并不要求每个
music beat 都有动作响应。高 BAS 可能来自很少的动作 beat，因此必须同时报告 motion
beat 数量、Precision/Recall/F1 或 onset response。当前模型没有显式 beat assignment 时，
BAP 和 Beat F1 只能标记 N/A，不能用来否定或证明整体音乐性。

`M-RESP-LAG` 只有在最佳相关系数达到预注册可靠性阈值时才解释；v1 使用 `r >= 0.10`。
音乐评价固定 audio clock，不得分别为 `M_ref` 和 `M_exec` 搜索能使分数最高的音频偏移。

### 6.2 音乐语义和长期结构

| ID | 指标 | 定义与解释 | 方向 | 等级 | 来源 |
|---|---|---|---|---|---|
| M-R1/R2/R3 | Audio-motion retrieval | 正确配对动作能否在 top-1/2/3 被对应音频检索 | ↑ | Core | RoboPerform |
| M-MMDIST | Multimodal Distance | 配对 audio-motion 在固定联合 embedding 中的平均距离 | ↓ | Core | RoboPerform |
| M-DS | Dance-music semantic retrieval score | 冻结 retrieval encoder 对配对 audio-motion 的语义匹配分数 | ↑ | Supplementary | MATHDance |
| M-PHRASE | Phrase-boundary response | 乐句/段落/强度边界附近动作结构变化的命中率和时差 | ↑/0 lag | Core | AudioMimic |
| M-STYLE | Genre/style/emotion consistency | 固定分类器或盲测对风格、情绪匹配的判断 | ↑ | Core | AudioMimic |

R@K 和 MMDist 只有在 audio-motion encoder 完成独立训练并冻结后才进入主表。该 encoder
不得在同一批 test clip 上调参。M0 是负控制，M4 oracle 是上界诊断；M2 必须在配对音乐
上显著优于 M0，才能声称模型使用了音乐条件。

## 7. X：Reference 可执行性

这一级在调用 SONIC 前执行，用于区分“generator 输出超出物理域”和“tracker 跟踪失败”。

| ID | 指标 | 定义与解释 | 方向 | 等级 | 来源 |
|---|---|---|---|---|---|
| X-JLIMIT | Reference joint-limit violation | G1 29 DoF 超出允许范围的帧率和最大幅度 | 0 | Hard gate | AudioMimic |
| X-DYN | Dynamic envelope violation | velocity/acceleration/jerk 超过 GT 与 SONIC capability envelope 的比例 | ↓ | Hard gate | AudioMimic |
| X-CONTACT | Contact feasibility | 支撑脚、COM、接触时序是否自洽 | ↑ | Core | EDGE/Lodge + AudioMimic |
| X-ROOT | Root feasibility | root height、roll/pitch、yaw rate、planar speed 是否处于可执行域 | ↑ | Hard gate | AudioMimic |
| X-INIT | Initial transition | measured-state alignment 后首段姿态/速度跳变 | ↓ | Hard gate | AudioMimic |

动态 envelope 必须先用 GT reference 在相同 SONIC 配置上标定。它是数据分析门槛，不是
运行时静默裁剪；如果 reference 违反 envelope，仍需保留原始轨迹并报告失败。

## 8. T：SONIC Tracker 质量

### 8.1 稳定性和误差

| ID | 指标 | 定义与解释 | 方向 | 等级 | 来源 |
|---|---|---|---|---|---|
| T-SUCC | Success Rate | 完成规定时长且未触发 fall/trajectory-deviation 条件的 run 比例 | ↑ | Hard gate | RoboPerform |
| T-TTF | Time to fall | 从正式动作开始到首次失败的时间，未摔倒为右删失 | ↑ | Core | AudioMimic |
| T-HMIN | Minimum base height | 最低 base height 及低于阈值的持续时间 | ↑ | Core | AudioMimic |
| T-EMPJPE | Mean per-joint position error | reference 与 execution 的 DoF rotation 平均误差，单位 rad | ↓ | Core | RoboPerform |
| T-EMPKPE | Mean per-keybody position error | FK key body 的平均位置误差，单位 m | ↓ | Core | RoboPerform |
| T-RMSE-RAW | Raw joint RMSE | 不做 lag 补偿的端到端误差 | ↓ | Core | AudioMimic |
| T-RMSE-ALIGN | Lag-compensated joint RMSE | 只补偿预定义 tracking lag 后的动作形状误差 | ↓ | Core | AudioMimic |
| T-LAG | Tracking lag | 全身、body group 和 per-joint cross-correlation lag | 0 | Core | AudioMimic |

Raw 与 lag-compensated 结果必须同时报告。正 lag 表示 execution 落后 reference。SONIC
当前不直接跟踪 global root XY，因此 global XY RMSE 不作为 tracker 主指标；root
orientation、angular velocity、base height 和 key-body positions 仍需报告。

失败 run 不得删除。T-SUCC/TTF 使用全部 run；误差指标只统计 first fall 前有效区间，
并明确有效时长。不能把摔倒后的仿真状态混入 RMSE 来夸大或掩盖 tracker 性能。

### 8.2 舞蹈表现力保留

| ID | 指标 | 定义 | 方向 | 等级 | 来源 |
|---|---|---|---|---|---|
| T-AMP | Amplitude retention | `(P95-P5)_exec / (P95-P5)_ref`，按关节和 body group 报告 | 1 | Core | AudioMimic |
| T-ENERGY | Energy retention | `mean(||dq_exec||^2) / mean(||dq_ref||^2)` | 1 | Core | AudioMimic |
| T-BAND-L/M/H | Band-power retention | Welch PSD 在 0-1、1-3、3-8 Hz 的功率保留率 | 1 | Core | AudioMimic |
| T-JERK | Jerk ratio | `jerk_P95_exec / jerk_P95_ref`，过低为细节损失，过高为振荡 | 1 | Core | AudioMimic |
| T-CONTACT | Contact retention | 支撑状态 F1、contact transition timing error、foot slip change | ↑/0 | Core | AudioMimic |

对 reference 幅度或 band power 接近零的关节，不计算不稳定比值，单独标记 inactive。
所有 retention 同时报告中位数、IQR、P5/P95，并按 legs/waist/arms 拆分。

## 9. E：执行后舞蹈质量

所有适用的 G 和 M 指标都应在 `M_ref` 与 `M_exec` 上使用同一实现重算：

```text
reference quality     = metric(M_ref)
execution quality     = metric(M_exec)
execution degradation = metric(M_exec) - metric(M_ref)
execution retention   = metric(M_exec) / metric(M_ref)
```

只有具有合理零点、分母不接近零且“比例”有物理意义的指标才报告 retention。FID、RMSE、
lag、error rate 和可能为负的 correlation 主要报告差值。音乐时钟固定，tracking lag 仅用于
`M_ref` 与 `M_exec` 的姿态误差对齐，不能用于事后移动音频以提高 `M_exec` 的音乐分数。

必须形成以下成对结果：

| 结果对 | 解释 |
|---|---|
| `G(M_ref)` 与 `G(M_exec)` | SONIC 对自然性、物理性、连续性和动态的影响 |
| `M(A,M_ref)` 与 `M(A,M_exec)` | SONIC 对节奏、动态响应和音乐语义的影响 |
| `H(M_ref)` 与 `H(M_exec)` | 人类感知到的表现力损失 |

## 10. R：实时系统评价

| ID | 指标 | 统计方式 | 等级 | 来源 |
|---|---|---|---|---|
| R-A2F | Audio-to-feature latency | mean/P50/P95/P99/max | Core | DiscoForcing + AudioMimic |
| R-F2M | Feature-to-motion latency | mean/P50/P95/P99/max | Core | DiscoForcing |
| R-M2REF | Motion-to-reference latency | queue、serialize、network 分项 | Core | AudioMimic |
| R-REF2EXEC | Reference-to-execution latency | reference timestamp 到 measured response | Core | RoboPerform + AudioMimic |
| R-DEADLINE | Deadline miss rate | 超过 C4 commit deadline 的比例及最长连续 miss | Hard gate | DiscoForcing + AudioMimic |
| R-DROP | Packet/drop/fallback rate | 丢包、重复帧、过期包、fallback 比例 | Hard gate | AudioMimic |
| R-RTF | Realtime factor | 有效动作时长 / wall-clock time | ≥1 | Hard gate | DiscoForcing |

只报告平均推理时间不够。严格因果实验还必须记录 audio lookahead、buffer 长度、H/C、
NFE、设备型号和 warm-up。离线生成轨迹可以评价 generator 和 tracker capability，但不能
作为“实时音乐生成”的系统证据。

## 11. H：人类评价

自动指标不能独立回答“动作是否优美”。正式论文至少包含以下独立问题：

| ID | 问题 | 比较对象 |
|---|---|---|
| H-NATURAL | 动作是否自然、无明显抖动或错误 | generator vs baselines；reference vs execution |
| H-DANCE | 动作是否像有组织的舞蹈 | M0/M2/M4/GT |
| H-EXPRESS | 动作是否有力度、层次和表现力 | reference vs execution |
| H-COHERENCE | 60 s 内是否连贯且不过度重复 | generator routes |
| H-RHYTHM | 动作重音和节奏是否匹配音乐 | M0/M2/M4；reference vs execution |
| H-STYLE | 动作风格/情绪是否适合音乐 | M0/M2/M4 |

采用随机化左右位置的盲式 pairwise study，音乐、镜头、渲染、时长和音量保持一致。
报告原始 preference、Bradley-Terry 或 Elo、参与者数量、有效比较数量和 95% bootstrap
置信区间。EDGE、FACT、Lodge 和 Beat-It 都使用人类评价；不同论文的题目和样本量不是
统一标准，因此 AudioMimic 在实验开始前冻结自己的 protocol。

本项目冻结的问卷措辞、素材门槛、随机化、排除规则和统计方法见
[`HUMAN_EVALUATION_PROTOCOL.md`](HUMAN_EVALUATION_PROTOCOL.md)。现有 debug composite
不得直接作为正式盲评素材。

## 12. 必做实验矩阵

### 12.1 Generator 比较

| 因素 | 最低要求 |
|---|---|
| Route | M0 unconditional、M2 predicted future-music、M4 oracle future-music、GT |
| Music | 至少 3 首 held-out 歌曲，覆盖不同 tempo/style |
| Generation | 每首每 route 至少 3 个 generation seeds |
| Duration | 每条 60 s；相同 clip start |
| 输出 | G、M、X 全部适用指标和盲测视频 |

M0 是音乐使用的负控制，M4 是 future-music 条件上界。M2 只与 M4 接近但不优于 M0 时，
不能声称模型学会了 music conditioning。

### 12.2 Tracker 比较

| 因素 | 最低要求 |
|---|---|
| Reference | 固定同一条 M_ref，不重新采样 generator |
| Repeats | 每条 reference 至少 3 次独立 SONIC runs |
| Initialization | `3 s measured-state alignment + 1 s hold` |
| Playback | 1.0x 主结果；0.75/0.8/0.9/1.25x 只作 capability curve |
| Packet | full 与 C4 分开报告，不混合 |
| 输出 | T、E、R 指标和 failure timestamp |

generation seed 和 tracker repeat 是两个不同的随机层，统计时不得混为独立 generation
样本。若同一条 reference 重复跟踪三次，说明的是 tracker 方差，不是模型多样性。

## 13. 统计与报告规则

1. 主要统计单位是完整 trajectory/clip，不把每一帧当作独立样本。
2. 报告 mean、median、SD/IQR 和 trajectory-level 95% bootstrap CI。
3. 模型比较使用同 song、start、seed 的 paired difference；多歌曲时把 song 作为分层或随机效应。
4. 多指标检验预先指定 primary endpoints，并对同一结论族进行多重比较校正。
5. 所有失败样本进入 Success Rate；其 pre-fall 区间可进入误差分析，但必须标记 truncation。
6. GT、M0、M2、M4 使用同一 metric implementation、阈值、FK、audio beat detector 和 feature extractor。
7. 主表同时给出绝对值和相对 GT/M0 的差值，不能只报告百分比提升。
8. 视频结论必须对应定量 run ID；不得从未进入统计的 cherry-picked 视频得出主结论。

## 14. AudioMimic v1 验收 Gate

以下是项目工程门槛，不是六篇论文共同认可的通用阈值。它们须先用 GT 和现有 SONIC
数据标定，在正式测试前冻结；标定后不得根据测试结果反向修改。

### 14.1 数据和实时 gate

| Gate | v1 目标 |
|---|---|
| P0 schema/order/time | 全部通过 |
| Feedback coverage | ≥99% |
| 60 s realtime factor | ≥1.0 |
| Generator deadline miss | ≤1% |
| NaN/Inf、joint-limit violation | 0 |

### 14.2 Tracker gate

| Gate | v1 provisional target |
|---|---|
| 60 s simulation Success Rate | ≥95% over full test matrix |
| Tracking lag P95 | ≤100 ms |
| Median amplitude retention | 0.85--1.15 |
| Median energy retention | ≥0.70 |
| Median 3--8 Hz power retention | ≥0.50 |

上述 retention 门槛必须由 GT capability test 复核。如果 GT 本身达不到，则应报告 SONIC
能力上限并重新设定训练域，不能把阈值降低到刚好让当前模型通过。

### 14.3 Music-conditioned model gate

M2 要进入“实时音乐驱动舞蹈”主结果，必须同时满足：

1. 相对 M0，在预注册的至少一个语义指标（R@K 或 MMDist）和一个时间结构指标
   （onset response、tempo/phase 或 phrase response）上取得 paired improvement；
2. 主要 improvement 的 trajectory-level 95% CI 不跨 0；
3. G-PFC、G-FSR、G-JERK、G-REPEAT 和 X gate 不显著劣于 M0；
4. 经过 SONIC 后，M2 的音乐优势在 `M_exec` 上仍存在；
5. 人类节奏/风格匹配 preference 支持自动指标结论。

FID、BAS 或单个成功视频均不能单独通过该 gate。

## 15. 论文来源映射

| 论文 | 本项目采用的评价思想 | 需要保留的限制 |
|---|---|---|
| [FACT/AIST++](../papers/aist-fact.pdf) | FIDk/FIDg、Distk/Distg、BeatAlign、用户比较 | BAS 为单向局部 beat 接近；原始评估是 human motion，不代表机器人可执行 |
| [EDGE](../papers/edge.pdf) | PFC、大规模 pairwise/Elo、BeatAlign、Dist | 明确质疑有限测试集上的 FID；PFC 不等于完整动力学稳定性 |
| [Lodge](../papers/lodge.pdf) | 长序列 FID/Div、FSR、BAS、效率和用户评价 | FSR 会把某些有意滑步也计为问题，需与 PFC/contact 联合解释 |
| [Beat-It](../papers/beat-it.pdf) | PFC、Div、BAS、KPD、BAP、用户评价 | BAP/KPD 只适用于显式 keypose/beat control；高 BAS 不代表服从指定 beat |
| [RoboPerform](../papers/robopeform.pdf) | R@K、MMDist、BAS、Success、EMPJPE、EMPKPE、部署延迟 | Success 不能替代动作质量；embedding 指标依赖固定且独立的 encoder |
| [DiscoForcing](../papers/DiscoForcing.pdf) | 严格因果/有界延迟、FID/FSR/Div/BAS、ms/frame 和 FPS | 实时吞吐不等于端到端 tracker 保真度，仍需 AudioMimic retention 指标 |

## 16. 当前实现状态

| 状态 | 指标 |
|---|---|
| 已实现并用于第一轮 | joint RMSE、lag-compensated RMSE、lag、root-relative EMPKPE、amplitude/energy/band-power retention、jerk、static、repetition、C4 jump、root statistics、BAS、onset correlation/lag、survival |
| 仓库已有原型，需统一或标定 | PFC、BAS/BAP、diversity、G1 FK、contact retention、foot sliding、ground penetration |
| 下一步实现 | tempo/phase、phrase response，以及经过 GT 标定的完整 executability envelope |
| 需要额外模型/研究 | 固定 G1 FID/Div extractor、audio-motion retrieval encoder、R@K/MMDist、style/emotion evaluator |
| 需要人工实验 | naturalness、dance quality、expressiveness、coherence、rhythm、style preference |

机器可读注册表见 [`eval/evaluation_map_v1.json`](../../eval/evaluation_map_v1.json)。第一轮
M0/M2/M4 与 SONIC 结果见
[`REPORT.md`](../../eval/motion_music_execution/m2_m3_gt_comparison_v2/REPORT.md)。
