# Literature Metric Audit and Frozen Feature Policy

更新日期：2026-08-23

本文档核对本地论文中实际使用的评价指标和音乐条件表示，并冻结 AudioMimic
后续 generator、retargeting 和 SONIC execution 实验的公共协议。

## 1. 论文指标核对

| 论文 | 主要评价指标 | 在 AudioMimic 中的处理 |
|---|---|---|
| AIST++ / FACT | FIDk、FIDg、BeatAlign/BAS、kinetic/geometric diversity、user study | 保留 FIDk/FIDg、Divk/Divg、BAS；补充 event F1、lag、phase 和人评 |
| EDGE | PFC、Beat Alignment、Distk/Distg、Elo/win rate；明确指出 beat alignment 的局限 | PFC 为动作质量核心；BAS 为 beat-alignment 核心但不能单独代表整体音乐性；保留盲测 |
| Lodge | FIDk/FIDg、Divk/Divg、BAS、foot refinement、user study、长序列稳定性 | 保留分布、节奏、脚部和长期连续性指标 |
| DiscoForcing | FIDk/FIDg、FSR、Divk/Divg、BAS、latency；比较 streaming deadline | 加入 FSR、P95/P99 latency、deadline miss 和 realtime factor |
| Beat-It | PFC、BAS、Divk/Divm、Beat Assignment Precision、Key Pose Distance | BAP/KPD 仅在存在显式 beat/keypose target 时启用，否则 N/A |
| RoboPerform | Success Rate、EMPJPE、EMPKPE、R@1/2/3、MMDist、BAS、joint/torque/slippage safety | 作为 SONIC tracking、audio-motion retrieval 和执行安全指标 |
| DanceBA / MambaDance / DGFM | FIDk/FIDg、Divk/Divg、PFC、BAS；强调 rhythm feature 与 diversity 的权衡 | 作为主流生成 benchmark 的共同基线 |
| PAMD | BAS、PFC、FID、Div、skating、floating、penetration、user study | 加入 reference/execution 的 contact 与 ground quality |
| MATHDance | FID、DIV、BAS，以及 retrieval-based DS/DQ/DD、R@5/10、rank；验证 metric-human alignment | 加入 retrieval 组，但必须使用独立冻结的 audio-motion encoder |
| LRCM / Listen to Rhythm | FIDk/FIDg、BAS、DIV、freezing proportion、Rhythmic Score、Length Regularity | 加入 freeze、节奏化 freeze 和 freeze length regularity |
| InfiniteDance | FIDk/FIDg、BAS、FSR、Jitter、Penetration；关注数据和长时泛化 | 加入 Jitter、penetration 和长时质量报告 |
| Audio-conditioned whole-body control | BAS、FDD/FID、affective consistency、long-horizon stability、latency；执行后重算 BAS | 作为 AudioMimic 的 reference/execution 对照协议 |

## 2. 冻结的 benchmark 结构

所有后续结果必须按以下顺序报告，不能用单个总分替代：

```text
source human oracle -> GMR/G1 reference -> SONIC execution
```

### A. Dance quality

- `FIDk`, `FIDg`：动作分布和 kinetic/geometric realism；以 GT 区间为目标，不追求无限大 diversity。
- `PFC`, `FSR`：脚部接触和 foot skating。
- `Divk`, `Divg`：多 seed 和同音乐多样性，必须与 GT 对照。
- `velocity`, `acceleration`, `jerk`, `motion energy`：动作强度、平滑性和尖峰。
- `freeze ratio`, `freeze length regularity`, `repetition ratio`：检测平均态、冻结和重复。
- `penetration`, `jitter`, `root/base stability`：机器人和 G1 reference 的可执行性。
- `human naturalness/dance/expressiveness`：自动指标不能替代的最终感知指标。

### B. Music adaptation

- `BAS`：beat-alignment suite 的核心指标；必须报告，但不能单独证明完整音乐性。
  它应与 beat event Precision/Recall/F1（适用时）、onset/impact correlation、response
  lag、tempo error 和 phase error 联合解释。
- `Beat Precision/Recall/F1`：有显式 beat target 时启用；没有 target 时标记 N/A。
- `onset-motion correlation`、`response lag`：动作能量是否响应音乐 onset，以及是否延迟。
- `tempo error`、`beat phase error`：节奏速度和相位是否一致。
- `R@1/R@2/R@3`、`MMDist`：audio-motion retrieval；需要独立冻结的 retrieval encoder。
- `phrase/section response`：音乐段落、强度和结构变化是否反映到动作。
- `style/emotion consistency`：必须通过冻结分类器或盲评，不从 BAS 推断。

### C. SONIC execution and system

- `Success Rate`, `Time to Fall`, minimum base height。
- `EMPJPE`, `EMPKPE`, raw/lag-compensated RMSE、tracking lag。
- amplitude、energy、band-power、contact retention。
- `deadline miss`, stage latency P50/P95/P99、packet/drop rate、realtime factor。

## 3. Music feature policy

这些是 generator 的输入条件，不是评价指标。后续 feature 实验统一使用以下分层：

### F0：必选节奏条件

保留当前 8D beat representation：

```text
beat_pulse
gaussian_beat
dist_to_prev_beat_norm
dist_to_next_beat_norm
beat_phase_sin
beat_phase_cos
beat_interval_norm
onset_strength_norm
```

原因是它同时给出离散 beat、局部平滑响应、相对 beat 位置、周期相位、tempo/interval
和 onset 强度。`beat_pulse` 与 `gaussian_beat` 不重复：前者保留事件位置，后者提供
可学习的平滑邻域。

### F1：推荐主线语义/声学条件

- **MERT**：作为主线高层音乐表示候选，承担 timbre、instrumentation、style、tempo
  context 等 beat-only 无法解释的内容；需要先做 causal/windowed inference latency audit。
- **Librosa low-level features**：保留 onset strength、RMS/energy、spectral flux、
  spectral centroid/bandwidth、MFCC/chroma/tempo 等可解释特征，负责局部声学动态和
  段落变化。它们不是 MERT 的替代，而是互补的低层条件。

推荐主模型条件：

```text
MERT semantic stream + Librosa acoustic stream + 8D beat stream
```

三个 stream 分别 adapter、normalize，再 fusion；不要把原始维度直接拼接。

当前代码状态要和研究建议区分开：仓库已经有 Wav2CLIP、STFT、Librosa baseline、8D
beat 和 Jukebox 的抽取/缓存路径；MERT 目前是推荐的下一步实现，还不是现成可直接
运行的主线 checkpoint。因而下一轮不应把 MERT 的结果写成已有结果，而应先固定其
因果窗口、输出 FPS、缓存格式和推理耗时，再进入同一评估表。

### F2：保留用于消融的条件

- **Wav2CLIP**：保留为已有历史 anchor 和 semantic ablation。它已经在本项目中有
  checkpoint/cache 和可比较结果，但不作为新主线，除非在固定 benchmark 上优于 MERT。
- **STFT**：作为 Librosa acoustic 的低级对照，不和完整 Librosa stream 同时作为默认主线。
- **Jukebox**：作为 EDGE-compatible heavy baseline，不作为实时部署默认条件，除非
  latency 和显存预算满足硬门槛。
- **GaussianBeat-only / 8D beat-only**：作为 beat-only lower bound，必须保留，
  用来证明语义和声学条件是否真的带来额外收益。

### F3：当前不单独加入

- 不能把 `beat_pulse`、`gaussian_beat`、phase 和 onset 再复制成多个近似 stream。
- 不在没有独立监督时把 motion-derived `motion_energy`/`motion_beatness` 当成可部署
  音乐输入；它们只能作为训练 target、预测 control 或 oracle ablation。
- 不把 Wav2CLIP、MERT、Jukebox 三个高层 encoder 同时堆入主模型；否则无法解释收益、
  latency 和 feature redundancy。

## 4. 冻结规则

1. 所有模型至少报告 `8D beat-only`、`MERT + Librosa + 8D beat` 和一个历史
   `Wav2CLIP + beat` anchor。
2. 所有 route 在 source oracle、G1 reference 和 SONIC execution 上使用同一套 A/B
   evaluator；BAS 每次都报告，作为 beat-alignment 主指标，不能省略；完整 beat suite
   必须同时报告其互补指标。
3. 音乐条件实验必须加入 paired、wrong-song、time-shift、tempo-preserved 和 null
   controls；只看 paired BAS 不足以证明音乐条件被使用。
4. 每首歌固定 clip、音频时钟、生成 seed、时长和 tracker repeat；失败执行保留在统计中。
5. 新增或删除指标必须提升 evaluation map 版本号，并记录原因；本版本之后不得只挑选
   对某个模型有利的指标。

## 5. 本地来源

逐篇核对的论文保存在 [`docs/papers/`](../papers/)；机器可读注册表为
[`eval/evaluation_map_v1.json`](../../eval/evaluation_map_v1.json)，实验说明见
[`eval/README.md`](../../eval/README.md)。
