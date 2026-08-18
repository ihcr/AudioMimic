# Rhythm-to-Motion Evaluation and Compound Rhythm Training Plan

更新日期：2026-06-17

## 1. 研究目标

当前阶段的首要任务不是继续训练新模型，而是建立一套更完整的 rhythm-to-motion evaluation protocol。现有结果主要依赖 BAS、Beat F1、G1Dist、G1Div 等指标，但这些指标仍不足以判断生成动作是否真正响应了输入节奏结构，也不能充分解释动作是否通过合理的 G1 身体运动完成节奏表达。

本阶段目标分为两步：

1. 建立 `rhythm_eval_suite`，用于评估节奏与动作之间的对应关系、身体部位响应以及机器人可执行性。
2. 在 evaluation protocol 稳定后，再训练 `compound_rhythm_v1` 条件模型，用更丰富的节奏特征替代当前过于简单的 1D/8D beat 表示。

该计划遵循 evaluation-first 原则：先明确如何评价“节奏是否被正确表达”，再训练新的 rhythm-conditioned generation model。

## 2. 现有 BAS 的不足

BAS 的基本思想是检测 motion beat 与 music beat 的时间接近程度。该指标在 AIST++、EDGE 等 music-to-dance 工作中较常见，适合衡量整体节拍对齐程度。但在当前 G1 机器人舞蹈生成任务中，仅使用 BAS 存在以下不足：

- 无法区分普通 beat、downbeat、accent、offbeat 和 subdivision。
- 无法判断动作是否响应了连续 rhythm phase，而不只是靠近某些离散 beat 点。
- 无法判断节奏响应来自手腕、脚部、躯干还是全身。
- 无法识别模型是否通过 wrist jitter 或 foot jitter 提高 beat 分数。
- 无法反映 beat 附近是否存在合理的 foot contact、near support 和机器人稳定性。
- 无法解释高 beat score 是否伴随 foot sliding、ground penetration、high foot lift 或 root instability。

因此，BAS 应保留为基础指标，但不能作为唯一结论依据。

## 3. Evaluation 指标体系

新的 evaluation protocol 分为四类：rhythm timing、compound rhythm structure、body response 和 robot feasibility。

### 3.1 Rhythm Timing Metrics

这组指标评估 motion beat 与目标 rhythm event 的时间关系。

| 指标 | 方向 | 定义 | 作用 | 来源 |
| --- | --- | --- | --- | --- |
| `BeatPrecision` | ↑ | 匹配到目标 beat 的 motion beat 数量 / 全部 motion beat 数量 | 衡量生成动作是否产生过多 false beat | 通用 event detection；项目内用于 motion beat matching |
| `BeatRecall` | ↑ | 被 motion beat 匹配到的目标 beat 数量 / 全部目标 beat 数量 | 衡量目标节拍是否被动作表达 | 通用 event detection；项目内用于 motion beat matching |
| `BeatF1` | ↑ | Precision 和 Recall 的调和平均 | 同时惩罚乱卡点和漏拍 | 通用 event detection |
| `BeatTimingMeanFrames` | → 0 | 匹配成功的 motion beat 相对 target beat 的平均帧偏移 | 判断动作整体提前或滞后 | 项目内扩展 |
| `BeatTimingStdFrames` | ↓ | 匹配偏移的标准差 | 判断 timing 是否稳定 | 项目内扩展 |
| `MotionBeatDensity` | 合理 | 单位时间内检测到的 motion beat 数量 | 判断动作事件是否过少或过多 | 项目内扩展 |
| `BeatDensityRatio` | → 1 | motion beat density / target beat density | 判断生成动作节奏密度是否匹配输入 | 项目内扩展 |
| `OffbeatFalsePositiveRate` | ↓ | 未匹配到目标 event 的 motion beat 比例 | 检测 off-beat jitter 或多余动作事件 | 项目内扩展 |

解释方式：

- Precision 高、Recall 低：动作卡点较准，但漏掉大量目标 beat。
- Precision 低、Recall 高：动作事件较多，但存在大量 false beat。
- Timing mean 偏正：动作整体滞后。
- Timing mean 偏负：动作整体提前。
- Density ratio 明显大于 1：可能存在 jitter 或过度动作事件。
- Density ratio 明显小于 1：动作过于平滑或缺少节奏事件。

### 3.2 Compound Rhythm Structure Metrics

这组指标用于复合节奏任务，不只考察主拍，还考察节奏层级和周期结构。

| 指标 | 方向 | 定义 | 作用 | 来源 |
| --- | --- | --- | --- | --- |
| `DownbeatHitRate` | ↑ | downbeat 附近是否存在显著 motion beat 或 energy response | 衡量小节强拍是否被动作强调 | music/dance rhythm analysis；项目内实现 |
| `AccentHitRate` | ↑ | accent event 附近是否存在动作强调 | 衡量模型是否区分强弱节奏 | rhythm/accent control 思路；项目内实现 |
| `WeakStrongContrast` | ↑ | 强拍响应强度与弱拍响应强度的差异 | 防止所有 beat 被同等处理 | 项目内实现 |
| `OffbeatHitRate` | ↑ | offbeat event 附近是否存在动作响应 | 评估副拍和切分节奏表达 | music rhythm analysis；项目内实现 |
| `SubdivisionHitRate` | ↑ | half/quarter/eighth/triplet 等 subdivision 是否被动作响应 | 评估复杂节奏型控制能力 | DiscoForcing rhythm pattern 思路启发 |
| `PhaseAlignmentScore` | ↑ | motion energy 或 motion beat 是否与 beat/bar phase 对齐 | 衡量动作是否连续跟随 rhythm phase | DiscoForcing periodic alignment、DeePhase 思路启发 |

这些指标不是 EDGE 原始标准指标，而是为复合节奏生成任务引入的结构化评估。DiscoForcing 的启发在于：节奏条件不应仅包含离散 beat pulse，还应包含 rhythm pattern 与 continuous periodic/phase alignment signal。

### 3.3 Body Response Metrics

这组指标评估节奏响应发生在哪些身体部位。

| 指标 | 方向 | 定义 | 作用 | 来源 |
| --- | --- | --- | --- | --- |
| `WristBeatF1` | 诊断 | 仅使用 wrist keypoints 检测 motion beat 后计算 F1 | 判断节奏响应是否集中在手腕 | 项目内 G1 诊断指标 |
| `FootBeatF1` | ↑ | 使用 foot/ankle keypoints 检测 motion beat 后计算 F1 | 判断脚步是否参与节奏 | 项目内 G1 诊断指标 |
| `TorsoBeatF1` | ↑ | 使用 pelvis/torso keypoints 检测 motion beat 后计算 F1 | 判断躯干和重心是否参与节奏 | 项目内 G1 诊断指标 |
| `FullBodyBeatF1` | ↑ | 使用全身 keypoints 检测 motion beat 后计算 F1 | 衡量整体身体节奏响应 | 项目内 G1 诊断指标 |
| `WristDominanceRatio` | ↓/适中 | wrist response 相对 full-body response 的比例 | 检测模型是否依赖手腕抖动换分 | 项目内 G1 诊断指标 |
| `BodyEnergyCorrelation` | ↑ | body-group motion energy 与 rhythm/accent strength 的相关性 | 判断动作强弱是否跟随节奏强弱 | 项目内扩展 |

分析原则：

- `WristBeatF1` 高而 `FootBeatF1`、`TorsoBeatF1` 低，说明模型可能主要通过手腕制造 beat event。
- `FootBeatF1` 高但 `FootSliding` 高，说明脚部参与节奏但物理质量不足。
- `TorsoBeatF1` 和 `FullBodyBeatF1` 较高通常更接近视觉上的全身舞蹈响应。

### 3.4 Robot Feasibility Metrics

这组指标面向 Unitree G1 和后续 SONIC/WBC tracking，主要衡量生成 reference motion 是否具有机器人可执行性。

| 指标 | 方向 | 定义 | 作用 | 来源 |
| --- | --- | --- | --- | --- |
| `FootContactOnBeatRate` | ↑ | target beat 附近是否至少一只脚接触地面 | 判断 beat 落点是否具备支撑条件 | 项目内 G1/SONIC 指标 |
| `NearSupportOnBeatRate` | ↑ | target beat 附近是否至少一只脚接近地面 | reference motion 中更稳健的支撑近似 | 项目内 G1/SONIC 指标 |
| `NoNearSupportRate` | ↓ | 没有 near-support 的帧比例 | 评估潜在 tracking 不稳定风险 | 项目内 G1/SONIC 指标 |
| `FootHighLiftRate` | ↓ | 脚高于阈值的帧比例 | 检测脚部悬空、高抬脚问题 | 项目内 G1 诊断指标 |
| `WristJerkMean` | ↓ | wrist keypoint jerk 均值 | 检测手腕高频抖动 | 动作平滑性指标，项目内适配 |
| `FootJerkMean` | ↓ | foot keypoint jerk 均值 | 检测脚部高频抖动 | 动作平滑性指标，项目内适配 |
| `FootSliding` | ↓ | 接触脚的水平滑动速度 | 检测物理不一致和 tracking 风险 | robot motion quality 常用指标 |
| `GroundPenetration` | ↓ | 脚或身体关键点低于地面的程度 | 检测穿地问题 | robot motion quality 常用指标 |
| `RootUpStability` | ↑ | root/torso up direction 的稳定性 | 检测侧翻、roll/pitch drift | 项目内 G1 诊断指标 |
| `RootAngularP95/P99` | ↓ | root angular velocity 的高分位数 | 检测极端 root rotation tail | 项目内 G1 诊断指标 |
| `JointRangeViolationRate` | ↓ | 关节超过 reference 合理范围的比例 | 检测异常姿态和不可执行动作 | 项目内 G1 质量指标 |

这类指标不是传统 music-to-dance 论文中的核心指标，而是 G1 robot-native generation 必需的质量门。其目的不是替代 rhythm metric，而是防止模型通过不合理动作提高 rhythm score。

## 4. 指标来源说明

### 4.1 文献中已有或强相关指标

| 指标/概念 | 相关文献或传统 | 本项目处理 |
| --- | --- | --- |
| BAS | AIST++、EDGE 及 music-to-dance 文献中的 beat alignment | 适配为 G1 raw 和 G1 FK 版本 |
| Diversity | EDGE/AIST++ 中的 diversity 类评价 | 适配为 `G1Div` |
| Distribution distance/FID 思路 | motion generation 常用分布质量评价 | 适配为 `G1Dist`，使用 G1 motion statistics |
| Periodic/phase representation | DiscoForcing、DeePhase | 用于后续 phase alignment metric 和 compound rhythm feature |
| Learned rhythmic code | DiscoForcing 的 VQ-PAE | 作为长期方向，短期不实现 |

### 4.2 从文献思想扩展而来

| 指标/概念 | 来源关系 | 本项目用途 |
| --- | --- | --- |
| Precision/Recall/F1 | 通用 event detection，与 beat alignment/BAP 兼容 | 评估 motion beat 与 target beat 的匹配质量 |
| Timing mean/std | event alignment 的自然扩展 | 分析动作提前、滞后和 timing 稳定性 |
| Downbeat/accent/subdivision hit rate | music rhythm analysis 与 DiscoForcing rhythm pattern 启发 | 评估复合节奏表达 |
| Condition sensitivity | control ablation 常用方法 | 判断模型是否真正使用 rhythm condition |

### 4.3 本项目自定义指标

| 指标/概念 | 设计原因 |
| --- | --- |
| Wrist/Foot/Torso Beat F1 | 判断节奏响应是否集中在某个 body group，尤其防止 wrist jitter |
| FootContactOnBeatRate | 检查 beat 落点是否具备支撑条件 |
| NearSupportOnBeatRate | 对 reference motion 提供比硬 contact 更稳健的支撑判断 |
| FootHighLiftRate | 捕捉当前 V5 暴露的脚悬空和异常高抬脚 |
| WristJerkMean / FootJerkMean | 捕捉端点高频抖动 |
| RootUpStability / RootAngularP99 | 捕捉长视频 root drift、侧翻和极端旋转 |

总结：

```text
论文指标提供 rhythm alignment 和 motion distribution 的基础；
DiscoForcing 提供 phase/pattern rhythm representation 的启发；
G1/SONIC 部署要求额外加入 body response、support/contact 和 endpoint smoothness 指标。
```

## 5. Evaluation 实施阶段

### 5.1 阶段一：扩展 G1 FK evaluation

首先扩展 `eval/g1_metrics.py`，输出当前不依赖新 rhythm label 的指标：

```text
Rhythm timing:
- G1BeatPrecision
- G1BeatRecall
- G1BeatF1
- G1BeatTimingMeanFrames
- G1BeatTimingStdFrames
- G1MotionBeatDensity
- G1BeatDensityRatio
- G1BeatOffbeatRate

Body response:
- G1WristBeatF1
- G1FootBeatF1
- G1TorsoBeatF1
- G1FullBodyBeatF1
- G1WristDominanceRatio

Robot feasibility:
- G1FootContactOnBeatRate
- G1NearSupportOnBeatRate
- G1NoNearSupportRate
- G1FootHighLiftRate
- G1WristJerkMean
- G1FootJerkMean
```

这一阶段只依赖已有 motion pkl、audio beat、G1 FK keypoints 和 foot points。

### 5.2 阶段二：扩展 leaderboard

更新 `eval/write_g1_metric_comparison.py`，输出新的 rhythm evaluation table。建议最小表头如下：

```text
Model
G1FKRoboPerformBAS
BeatPrecision
BeatRecall
BeatF1
BeatTimingMean/Std
MotionBeatDensity
OffbeatRate
WristBeatF1
FootBeatF1
TorsoBeatF1
FootContactOnBeat
NearSupportOnBeat
NoNearSupport
FootHighLift
WristJerk
FootSliding
GroundPenetration
G1Dist
G1Div
```

该表用于重新比较：

```text
1D GaussianBeat
8D beat-only
Librosa35
Wav2CLIP/STFT r02
V3b
V5
```

### 5.3 阶段三：建立 rhythm label 格式

复合节奏指标需要额外 rhythm annotation。建议定义统一的 rhythm label 文件或 `.npz` 字段：

```text
beat_frames
downbeat_frames
accent_frames
offbeat_frames
subdivision_frames
bar_phase
beat_phase
accent_strength
rhythm_density
tempo
```

在该格式稳定后，加入：

```text
DownbeatHitRate
AccentHitRate
OffbeatHitRate
SubdivisionHitRate
WeakStrongContrast
PhaseAlignmentScore
```

### 5.4 阶段四：condition sensitivity protocol

每个 rhythm-conditioned 模型至少需要以下 ablation：

```text
real rhythm
shifted rhythm +5 frames
shifted rhythm +10 frames
random rhythm
constant rhythm
zero rhythm
```

预期现象：

- `real rhythm` 在 rhythm metrics 上最好。
- `shifted rhythm` 应导致 motion beat timing 发生对应偏移。
- `random/constant/zero rhythm` 应明显降低 rhythm-to-motion 对应关系。
- 如果各条件差异不明显，说明模型没有充分使用 rhythm condition。

### 5.5 阶段五：failure panel

除 aggregate metrics 外，应自动导出 failure cases：

```text
wrist jerk highest
foot high-lift highest
no-near-support highest
ground penetration highest
beat recall lowest
offbeat false-positive highest
```

每个 failure case 应保存：

- motion 文件路径；
- audio/rhythm label；
- diagnostic plot；
- 对应 metric row；
- 如条件允许，保存 short render。

## 6. Evaluation 完成后的训练计划

在 `rhythm_eval_suite` 能够稳定重跑现有模型之后，再进入下一阶段训练。训练目标是构建比当前 1D GaussianBeat 和 8D beat-only 更完整的 `compound_rhythm_v1` 条件。

### 6.1 训练实验名称

建议实验名称：

```text
V6a_compound_rhythm_only_yaw_delta
```

基本设定：

```text
motion representation: g1_yaw_delta
condition: compound_rhythm_v1
without Wav2CLIP
without STFT
without Jukebox
```

该实验用于验证：在不引入音乐语义的情况下，结构化复合节奏是否能够稳定控制 G1 动作。

### 6.2 `compound_rhythm_v1` 建议特征

第一版建议使用手工 rhythm feature，而不是直接训练 learned rhythm encoder。这样便于调试和做 condition sensitivity。

#### A. Basic Beat Features

| 特征 | 维度 | 含义 |
| --- | ---: | --- |
| `beat_pulse` | 1 | beat frame 上的离散脉冲 |
| `gaussian_beat` | 1 | beat 附近的平滑 Gaussian 响应 |
| `dist_to_prev_beat_norm` | 1 | 当前帧到上一 beat 的归一化距离 |
| `dist_to_next_beat_norm` | 1 | 当前帧到下一 beat 的归一化距离 |
| `beat_phase_sin` | 1 | beat interval 内 phase 的 sine 编码 |
| `beat_phase_cos` | 1 | beat interval 内 phase 的 cosine 编码 |
| `beat_interval_norm` | 1 | 当前 beat interval 的归一化长度 |
| `onset_strength_norm` | 1 | 当前局部 onset 强度 |

这部分相当于当前 8D beat feature，是 `compound_rhythm_v1` 的基础。

#### B. Meter and Bar Features

| 特征 | 维度 | 含义 |
| --- | ---: | --- |
| `downbeat_pulse` | 1 | 小节第一拍或强拍位置 |
| `bar_phase_sin` | 1 | 小节周期内 phase 的 sine 编码 |
| `bar_phase_cos` | 1 | 小节周期内 phase 的 cosine 编码 |
| `beat_index_in_bar_norm` | 1 | 当前 beat 在小节中的位置 |
| `strong_beat_mask` | 1 | 强拍 mask |
| `weak_beat_mask` | 1 | 弱拍 mask |

用途：让模型区分小节结构，避免每个 beat 都生成同等强度动作。

#### C. Subdivision and Offbeat Features

| 特征 | 维度 | 含义 |
| --- | ---: | --- |
| `half_beat_pulse` | 1 | 半拍位置 |
| `quarter_beat_pulse` | 1 | 四分细分位置 |
| `eighth_beat_pulse` | 1 | 八分细分位置 |
| `offbeat_pulse` | 1 | 副拍位置 |
| `triplet_pulse` | 1 | 三连音/三等分位置 |
| `syncopation_mask` | 1 | 切分节奏位置 |

用途：支持复合节奏控制，使模型不只响应主拍，也能响应副拍、切分和细分节奏。

#### D. Accent, Tempo, and Density Features

| 特征 | 维度 | 含义 |
| --- | ---: | --- |
| `accent_strength` | 1 | 当前 rhythm event 的重音强度 |
| `local_onset_density` | 1 | 局部 onset 密度 |
| `rhythm_density` | 1 | 局部 rhythm event 密度 |
| `tempo_norm` | 1 | 归一化 tempo |
| `tempo_delta` | 1 | tempo 局部变化 |
| `beat_confidence` | 1 | beat tracking 置信度 |

用途：让模型区分强弱、稀疏/密集、稳定/变化的节奏段落。

#### E. Energy Features

| 特征 | 维度 | 含义 |
| --- | ---: | --- |
| `rms_energy_norm` | 1 | 局部 RMS 能量 |
| `energy_delta` | 1 | 能量变化率 |
| `low_band_onset` | 1 | 低频 onset，通常对应鼓点/低频冲击 |
| `mid_band_onset` | 1 | 中频 onset |
| `high_band_onset` | 1 | 高频 onset |

用途：提供 rhythm event 的强度和频段信息，帮助区分鼓点、旋律型 onset 和高频装饰音。

### 6.3 建议维度

第一版 `compound_rhythm_v1` 约为 31D：

```text
8D basic beat
6D meter/bar
6D subdivision/offbeat
6D accent/tempo/density
5D energy
= 31D
```

如果初版实现成本需要降低，可以先做 20D 子集：

```text
8D basic beat
3D downbeat/bar phase
4D offbeat/subdivision
3D accent/density/tempo
2D energy
= 20D
```

### 6.4 训练和 checkpoint 计划

建议先跑小规模可验证训练，不直接追求最终效果：

```text
checkpoints: 500, 1000, 1500
eval: full rhythm_eval_suite
ablation: real / shifted / random / constant / zero rhythm
render: selected short clips + 90s long clips
```

通过标准：

- `real rhythm` 明显优于 `random/constant/zero rhythm`。
- `shifted rhythm` 导致 motion beat timing 出现相应偏移。
- `BeatRecall` 提升不能伴随 `OffbeatFalsePositiveRate` 显著恶化。
- `WristDominanceRatio` 不应显著高于 V5。
- `FootContactOnBeatRate`、`NearSupportOnBeatRate` 不应明显低于 V5。
- `FootHighLiftRate`、`WristJerkMean`、`GroundPenetration` 不应明显恶化。

### 6.5 后续训练路线

如果 `V6a_compound_rhythm_only_yaw_delta` 证明 rhythm controllability 成立，再进入：

```text
V6b_contact_beatness_yaw_delta
```

目标：在 rhythm control 基础上加入 contact-aware beatness，降低 wrist 权重，引入 support/contact 约束。

如果 V6b 通过 robot feasibility gates，再进入：

```text
V7_semantic_rhythm_fusion
```

目标：在 compound rhythm control 之外，重新加入 Wav2CLIP semantic branch，用于补充音乐风格、情绪和音色信息。

## 7. 当前阶段不做的工作

在 rhythm evaluation 完成前，暂不进行以下工作：

- 不训练新 V6 模型；
- 不加入 Wav2CLIP 新语义分支；
- 不实现 DiscoForcing 的 VQ-PAE；
- 不实现 streaming diffusion；
- 不进行大规模 SONIC/WBC rollout。

这些工作应在 evaluation protocol 能稳定解释现有模型差异后再开展。

## 8. 最终执行顺序

建议执行顺序如下：

```text
1. Implement rhythm_eval_suite
2. Re-evaluate existing baselines
3. Build failure panel
4. Define rhythm label / compound_rhythm_v1 feature format
5. Train V6a_compound_rhythm_only_yaw_delta
6. Run condition sensitivity
7. Add contact-aware beatness
8. Add Wav2CLIP semantics
9. Validate through SONIC/WBC subset
```
