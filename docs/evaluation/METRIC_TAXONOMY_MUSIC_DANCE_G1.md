# AudioMimic 音乐-舞蹈-G1 指标分类体系 v1

更新日期：2026-08-23

冻结状态：**v1 已冻结**。本研究后续不得重定义、移动或删除现有指标；只有发现经过论文
依据和 GT calibration 支持的新指标时，才能新增 `v2`，并保留 v1 结果的可复现性。

本文档规定后续所有 benchmark、模型消融和 SONIC 实验的指标层级。报告必须按照
`大模块 -> 子模块 -> 指标` 展开，不能把 BAS、jerk、FID 和 tracking error 混在同一列
并解释成一个“舞蹈质量总分”。机器可读映射见
[`metric_taxonomy_v1.json`](../../eval/metric_taxonomy_v1.json)。

## 1. 总体结构

```text
P. 数据与协议有效性
|
+-- D. 舞蹈动作质量
|   +-- D1 分布真实性
|   +-- D2 多样性
|   +-- D3 平滑度与连续性
|   +-- D4 活力、冻结与重复
|   +-- D5 物理合理性
|
+-- M. 音乐-舞蹈适配
|   +-- M1 节拍与节奏
|   +-- M2 动态、tempo 与 phase 响应
|   +-- M3 乐句与段落结构
|   +-- M4 风格、情绪与跨模态语义
|
+-- X. G1 可执行性与 SONIC 跟踪
|   +-- X1 reference 可执行性
|   +-- X2 tracking fidelity
|   +-- X3 表现力 retention
|   +-- X4 稳定性与安全
|
+-- R. 在线实时系统
|
+-- H. 人类感知评价
```

每个模块回答不同问题：

| 大模块 | 核心问题 | 不能由什么替代 |
|---|---|---|
| D 舞蹈动作质量 | 不听音乐时，动作是否自然、丰富、平滑、像舞蹈？ | BAS 不能回答 |
| M 音乐-舞蹈适配 | 这段动作是否适合这一首音乐？ | FID、jerk 不能回答 |
| X G1/SONIC | reference 是否可执行，tracker 保留了多少动作？ | generator FID 不能回答 |
| R 在线系统 | 是否因果、实时、满足 deadline？ | 离线 PKL 成功不能回答 |
| H 人类感知 | 是否优美、有表现力、风格和情绪是否合适？ | 自动指标不能完全替代 |

## 2. 先区分音乐概念

`快/慢`、`Jazz/Hip-hop` 和 `开心/悲伤`不是同一个概念。

| 音乐属性 | 音乐学含义 | 示例 | 对舞蹈的预期影响 | 评价方式 |
|---|---|---|---|---|
| Tempo | 单位时间内的节拍速度 | 80 BPM、140 BPM，快/慢 | 动作周期和重心切换速度 | tempo error、periodicity |
| Beat/meter | 稳定脉冲、拍号和强弱拍组织 | 四拍、downbeat | 动作重音是否踩拍 | BAS、Beat P/R/F1、phase |
| Rhythm | 时值、切分、重音和停顿构成的时间模式 | syncopation、顿点 | 动作的停顿、爆发与连续组合 | rhythmic score、event pattern |
| Dynamics | 音量、onset、能量包络随时间变化 | crescendo、drop | 动作幅度、速度、impact 随音乐变化 | onset-impact corr、lag、energy response |
| Structure | 乐句、段落和重复结构 | intro/verse/chorus/drop | 动作主题、转场和重复是否对应段落 | phrase/section response |
| Genre/style | 由节奏、音色、编曲、历史语汇共同构成 | Jazz、Hip-hop、Classical | 动作词汇和身体使用方式 | style classifier、retrieval、人评 |
| Emotion/mood | 感知情绪和唤醒程度 | valence/arousal、紧张/舒缓 | 力度、姿态开放度、运动范围和速度 | 情绪模型或人工一致性评分 |
| Timbre/instrumentation | 声音质地和乐器构成 | drum、strings、vocal | 可能改变动作质感和局部响应 | audio embedding + counterfactual/human study |

因此，`slow/medium/fast` 是 tempo 分层，不是舞蹈风格；`Jazz` 是 genre/style；
`happy/sad` 是 emotion。三者必须分别报告，不能放在一个 style 列里。

### 2.1 模型输入特征与评价指标不是一回事

| 条件流 | 典型输入 | 主要表达 | 对应评价子模块 |
|---|---|---|---|
| Beat/rhythm stream | beat pulse、Gaussian beat、前后拍距离、phase sin/cos、beat interval | beat 位置、周期和 tempo | M1、M2 |
| Low-level acoustic stream | onset strength、RMS、spectral flux、chroma、MFCC | 动态、重音、和声与音色变化 | M2、M3 |
| Semantic stream | MERT、Wav2CLIP、Jukebox embedding | genre、instrumentation、长期语义和可能的情绪信息 | M3、M4 |

这些是 generator 的条件输入，不是模型分数。例如模型输入 `beat_phase_sin/cos`，最终仍要
用独立的 motion event/phase evaluator 判断动作是否真正使用了相位。风格和情绪也应优先
使用独立冻结的 evaluator 或盲评，避免用训练条件 encoder 自己给输出打分形成循环论证。

## 3. D：舞蹈动作质量

### D1 动作分布真实性

| 指标 | 代表什么 | 方向 | 论文来源与限制 |
|---|---|---|---|
| FIDk | 生成动作与 GT 在 kinetic feature 分布上的距离 | 低 | FACT、Lodge、DiscoForcing；extractor、时长和数据集必须固定 |
| FIDg | geometric feature 分布距离 | 低 | FACT、Lodge、DiscoForcing；不能单独表示优美 |

### D2 多样性与覆盖

| 指标 | 代表什么 | 方向 | 论文来源与限制 |
|---|---|---|---|
| Divk/Divg | 生成样本在 kinetic/geometric 空间的差异 | 接近同条件 GT | FACT、Lodge、Beat-It、DiscoForcing；不是越大越好 |
| Same-song seed diversity | 同一音乐不同 seed 是否生成不同但合理的舞蹈 | 接近 GT | AudioMimic；至少多个 seed 才有意义 |

### D3 平滑度与连续性

| 指标 | 代表什么 | 方向 | 论文来源与限制 |
|---|---|---|---|
| Velocity/acceleration | 动作速度和加速度分布，反映强度和急剧变化 | 接近同风格 GT | 常规运动学诊断；AudioMimic 固定实现 |
| Jerk | 加速度的一阶时间导数，检测高频抖动和不自然尖峰 | 不应显著高于 GT | 精确定义为 AudioMimic；相关论文更常报告 jitter，InfiniteDance 使用 jitter 类指标 |
| C4 boundary continuity | commit 边界是否出现位置/速度跳变 | 边界与非边界接近 | AudioMimic 在线系统指标 |
| Jitter | 高频小幅无意义摆动 | 低/接近 GT | InfiniteDance 等；需要固定频带和幅值定义 |

`jerk` 很低也不一定好：过度低通或平均态会同时降低 jerk 和舞蹈表现力。因此 jerk 必须
与 energy、static/freeze、FID 和人评一起解释。

### D4 活力、冻结与长期重复

| 指标 | 代表什么 | 方向 | 论文来源与限制 |
|---|---|---|---|
| Motion energy | 关节速度平方的平均，反映动作活动强度 | 接近同风格/tempo GT | AudioMimic；不是越高越好 |
| Static ratio | 速度低于固定阈值的帧比例 | 接近 GT | AudioMimic 精确定义；简单冻结诊断 |
| Freezing proportion | 持续低运动区间的比例 | 低/接近 GT | LRCM / Listen to Rhythm |
| Freeze length regularity | freeze 的持续长度是否异常规律 | 接近 GT | LRCM |
| Long-range repetition | 长时间 rollout 是否复制早期片段 | 低/接近 GT | AudioMimic；当前 detector 仍需加强 |

### D5 物理合理性

| 指标 | 代表什么 | 方向 | 论文来源与限制 |
|---|---|---|---|
| PFC | 足部接触与足部加速度是否一致 | 低 | EDGE、Beat-It；EDGE 明确提醒常用自动指标与人评可能不一致 |
| FSR | 接触地面时的足部滑动比例 | 低 | Lodge、DiscoForcing；有意滑步需结合 contact 判断 |
| Penetration | 足或身体穿地比例/深度 | 低 | PAMD、InfiniteDance、AudioMimic |
| Root stability | root height、roll/pitch、速度和漂移 | 条件化 GT 范围 | AudioMimic；机器人可执行性尤其重要 |

## 4. M：音乐-舞蹈适配

### M1 节拍、重音与节奏网格

| 指标 | 代表什么 | 方向 | 来源与限制 |
|---|---|---|---|
| BAS | motion beat 到最近 music beat 的接近程度 | 高 | FACT、EDGE、Lodge、Beat-It、RoboPerform、DiscoForcing；单向且可能奖励稀疏 beat |
| Beat Precision | motion beat 中有多少命中音乐 beat/onset | 高 | AudioMimic 固定事件协议 |
| Beat Recall | 音乐 beat 中有多少得到动作响应 | 高 | AudioMimic；避免动作很少却 BAS 虚高 |
| Beat F1 | Precision 与 Recall 的调和平均 | 高 | AudioMimic；FineDance freeze 校准当前为 WARN，不能单独 gate |
| BAP | 是否执行指定 beat assignment | 高 | Beat-It；只有显式 beat assignment 时适用 |
| KPD | 指定 beat key pose 与目标的距离 | 低 | Beat-It；无 key-pose target 时 N/A |
| Rhythmic Score | freeze/动作结构与节奏组织的一致性 | 高 | LRCM；需复现其固定定义 |

### M2 Tempo、phase 与动态响应

| 指标 | 代表什么 | 方向 | 来源与限制 |
|---|---|---|---|
| Tempo error | motion periodicity 与音乐 BPM 的差 | 低 | AudioMimic；快慢匹配，不代表 genre 匹配 |
| Phase error | motion accent 位于节拍周期中的相位误差 | 低 | AudioMimic；补充 BAS 的周期位置解释 |
| Onset-impact correlation | 音乐 onset/能量与动作 impact 的同步变化 | 高 | AudioMimic；相关低时不能解释 lag |
| Response lag | 动作响应相对 onset 的时间偏移 | 接近 0 | AudioMimic；必须同时报告相关强度 |
| Dynamics retention | 强弱、爆发和能量包络是否被动作表达 | 高 | AudioMimic；reference/execution 都要计算 |

### M3 乐句与长期结构

| 指标 | 代表什么 | 方向 | 来源与限制 |
|---|---|---|---|
| Phrase/section response | intro、verse、chorus、drop 附近是否有动作变化或转场 | 高、lag 接近 0 | Lodge 的 choreographic rules 支持该维度；AudioMimic evaluator 待冻结 |
| Long-horizon coherence | 段落之间动作是否连贯且避免长期冻结/漂移 | 高 | Lodge、DiscoForcing、人评 |

### M4 风格、情绪与跨模态语义

| 指标 | 代表什么 | 方向 | 来源与限制 |
|---|---|---|---|
| Audio-motion R@K | 给动作检索正确音乐，或反向检索的命中率 | 高 | RoboPerform；必须使用独立冻结 encoder |
| MMDist | 正确 audio-motion pair 的 embedding 距离 | 低 | RoboPerform；依赖 encoder |
| DS/DQ/DD | 跨模态语义、质量和多样性评价 | 按论文定义 | MATHDance；需先复现并验证与人评一致性 |
| Genre/style consistency | Jazz/Hip-hop/Classical 等风格是否一致 | 高 | 冻结分类器或盲评；当前尚未形成可靠 core metric |
| Emotion consistency | valence/arousal 或离散情绪是否一致 | 高 | affective classifier 或人工评分；当前待实现 |

风格和情绪不能从 BAS 推断。证明模型使用了音乐语义，至少需要 paired、wrong-song、
time-shift、null-audio 和 tempo-preserved counterfactual；否则模型可能只利用动作历史。

## 5. X：G1 可执行性与 SONIC 跟踪

| 子模块 | 指标 | 代表什么 |
|---|---|---|
| X1 reference 可执行性 | dynamic-envelope violation、joint-limit violation | generator 输出是否超出 tracker/机器人能力 |
| X2 tracking fidelity | EMPJPE、EMPKPE、raw/aligned RMSE、tracking lag | SONIC 是否跟上 reference |
| X3 表现力保留 | amplitude、energy、frequency band、contact、music-expression retention | tracker 吃掉了多少幅度、速度、节奏和接触特征 |
| X4 稳定安全 | success rate、time to fall、minimum base height | 是否能完整执行而不摔倒 |

`M_ref` 和 `M_exec` 必须使用相同的 D/M evaluator。tracker 误差之外还要报告：

```text
retention = execution characteristic / reference characteristic
```

这样才能区分“generator 没生成出来”和“SONIC 执行时损失了”。

## 6. R 与 H：在线系统和人类感知

### R 在线实时系统

- audio-to-feature、feature-to-motion、motion-to-reference、reference-to-execution latency；
- deadline miss、packet drop/stale/fallback、realtime factor；
- 因果窗口、audio lookahead、H/C、NFE 和硬件必须一起公开。

这些指标主要来自 DiscoForcing 的 streaming 评价思路，并由 AudioMimic 扩展到 SONIC 链路。

### H 人类感知

优美度没有可信的单一自动公式。盲评至少拆成：

| 子项 | 问题 |
|---|---|
| Naturalness | 动作是否自然、无明显抖动和错误？ |
| Dance quality/aesthetics | 是否像组织良好、可观看的舞蹈？ |
| Smoothness | 动作连接是否流畅而非僵硬或过度平滑？ |
| Expressiveness | 是否有力度、层次、幅度和表现力？ |
| Rhythm | 动作重音和停顿是否匹配音乐？ |
| Style | 舞蹈风格是否适合音乐 genre？ |
| Emotion | 动作表达的情绪是否适合音乐 mood？ |
| Long-horizon coherence | 长时间是否连贯、不过度重复或漂移？ |

FACT、EDGE、Lodge、Beat-It 均包含用户评价；EDGE 还明确说明常见自动指标可能不能准确
反映人类评价。因此自动指标用于解释“哪里好/哪里坏”，人评用于支撑“优美、自然、风格
合适”等最终感知 claim。

## 7. 三个容易混淆的项目

| 名称 | 它是什么 | 是否来自论文标准指标 | 应放在哪里 |
|---|---|---|---|
| Jerk | 运动学三阶导数统计，用于发现抖动和尖峰 | 精确定义为 AudioMimic；与论文 jitter/smoothness 问题相关 | D3 平滑度 |
| Low-pass energy response | 对 GT 做低通后检查 jerk/energy 是否按预期下降 | 否，它是 corruption calibration method | 不进入最终模型指标表 |
| Static ratio | 低于速度阈值的帧比例 | 精确定义为 AudioMimic；LRCM 有相关 freezing proportion | D4 活力/冻结 |

这三个项目不能并列为同一级“核心指标”：jerk 和 static ratio 是指标，low-pass response
是验证指标是否可靠的实验操作。

## 8. 后续论文表格模板

每个模型必须按模块分表，不使用一个加权总分：

1. `Table A - Dance Quality`：FID、Div、smoothness、energy/freeze/repeat、PFC/FSR；
2. `Table B - Music Adaptation`：rhythm、dynamics/tempo/phase、structure、style/emotion；
3. `Table C - G1 Execution`：executability、tracking、retention、stability；
4. `Table D - Online Runtime`：causality、latency、deadline、RTF；
5. `Table E - Human Study`：naturalness、aesthetics、smoothness、rhythm、style、emotion。

每张表都必须同时给出同条件 GT reference range，并按照 dataset、tempo、genre/style 和
时长分层。自动指标和人评可以相互验证，但不能替代彼此。

## 9. 本地论文依据

- [`aist-fact.pdf`](../papers/aist-fact.pdf)：FIDk/FIDg、diversity、BeatAlign 和用户评价；
- [`edge.pdf`](../papers/edge.pdf)：PFC、BeatAlign、diversity 与大规模用户评价，并讨论指标局限；
- [`lodge.pdf`](../papers/lodge.pdf)：长序列 FID/Div、FSR、BAS、结构规则和用户评价；
- [`beat-it.pdf`](../papers/beat-it.pdf)：PFC、BAS、BAP、KPD、diversity 和用户评价；
- [`DiscoForcing.pdf`](../papers/DiscoForcing.pdf)：严格因果、长时连续性、FID/FSR/Div/BAS 和实时性；
- [`robopeform.pdf`](../papers/robopeform.pdf)：audio-motion retrieval、BAS、机器人 tracking 和执行指标；
- [`LITERATURE_METRIC_AUDIT_20260823.md`](LITERATURE_METRIC_AUDIT_20260823.md)：完整逐篇审计表。
