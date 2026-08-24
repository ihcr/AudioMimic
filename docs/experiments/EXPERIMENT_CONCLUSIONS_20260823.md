# AudioMimic Experiment Conclusions and ICRA Claim Status

更新日期：2026-08-24

这是当前实验结论的总账。旧的实验文件保留原始命令、日志和中间结果；本文只记录
经过整理后可以用于研究判断和论文写作的结论。新实验完成后，先更新本文，再更新
`docs/experiments/INDEX.md` 和 `ROADMAP_REALTIME_MUSIC_TO_G1.md`。

## 1. 论文主线

目标不是单纯生成一段好看的 G1 motion，而是验证：

> 在严格因果和实时约束下，机器人能否根据已经到达的音乐在线生成有节奏、自然、
> 可执行的舞蹈，并在 SONIC 跟踪后保留音乐响应和动作表达。

系统链路为：

```text
arrived audio prefix -> music condition -> H8/C4 motion generator
-> G1 reference -> SONIC tracker -> executed dance
```

这里的 generator 训练数据已经是经过 GMR/retargeting 的 G1 motion，因此 generator
直接生成 G1 reference，不再经过第二次 GMR。SMPL/SMPLH 与 GMR-G1 是 GT 侧的
source/target 对照，不是 generator 推理链路中的两个连续模块。

推荐论文叙事顺序：

1. 先建立 G1 dance prior 和连续、可执行的 motion representation。
2. 再证明音乐条件确实改变动作，并且 paired music 优于 wrong/shifted/null。
3. 再证明 generator reference 可以被 SONIC 稳定执行。
4. 最后报告 execution 对动作质量和音乐响应的保留率，以及实时延迟。

## 2. 当前系统状态

| 对象 | 当前状态 | 可以支持的判断 |
|---|---|---|
| GT O-Human/O-G1 | AIST++ 1,408 + FineDance 203 | 已有 1,611 条 paired reference distribution |
| GMR/retargeting | 1,611 条 audit | 可报告 retargeting 前后音乐/动作指标变化 |
| M0 | unconditional parent | 可作为无音乐动作先验/执行 baseline |
| M2 | predicted future-music sidecar rollout | 有音乐条件轨迹，但当前正式 music matching 证据不足 |
| M3 | causal music-sidecar generator | sidecar 会改变轨迹，但 paired 尚未稳定优于 counterfactual |
| M4 | oracle future-music sidecar | 诊断 future-music 上限，不是可部署 online 方法 |
| SONIC | reference tracking pipeline | 接口和基础 tracking 成立，但会损失动态和高频表达 |

M2/M3/M4 的固定 PKL 是 offline rollout；只有真正的 causal audio-prefix inference 才能
支持“online music generation” claim。离线 PKL 可以评价 reference 和 tracker，但不能
单独证明实时音乐生成。

## 3. Benchmark 状态

### 3.1 指标已经冻结

动作质量、beat alignment/music matching、SONIC execution 和 online system 四组指标
定义见 [`../evaluation/EVALUATION_MAP_MUSIC_TO_G1.md`](../evaluation/EVALUATION_MAP_MUSIC_TO_G1.md)。

核心 beat-alignment 组为：

```text
BAS + reverse BAS + Beat Precision/Recall/F1
+ onset/impact correlation + response lag
+ tempo error + phase error
```

BAS 是核心，但不能单独代表舞蹈优美或完整音乐性。FID/Div、R@K/MMDist 只有在
extractor、split、序列长度、beat detector 和统计协议一致时才能直接比较论文数值。

### 3.2 GT benchmark

- 38 条 sealed calibration test：AIST++ 20、FineDance 18。
- 494 个 clean/jitter/lowpass/freeze/repeat 样本。
- 8 个跨数据集 corruption direction check 中 7 个通过。
- FineDance freeze Beat F1 为 WARN，说明该 detector 不能独立作为 FineDance 退化 gate。
- `repeat_similarity` 尚未通过方向性验证，暂不作为核心指标。
- 全量 GT 已按 dataset、tempo 和 style/genre 生成 32 个 strata 的 reference distribution。
- 本轮已按固定协议重算全量 GT oracle：AIST++ `1,408` 条、FineDance `203` 条，共 `1,611` 条 paired G1 sequence；逐条结果、dataset/style/tempo 分层和 D/M 分布均已更新。
- 全量 FineDance-G1 的 root-height 已单独审计：203 条中 186 条 root z 全程非负，17 条出现短时负 root z；这 17 条同时表现为脚部接近地面且穿透率较低，不能直接解释为 Breaking 等风格的舞蹈质量下降。

报告：[`stratified_audit_v1`](../../eval/results/benchmark_v1/gt/stratified_audit_v1/)、
[`stratified_audit_all_v1`](../../eval/results/benchmark_v1/gt/stratified_audit_all_v1/)。

38 条 sealed GT 的完整音乐-动作校准（BAS、Event F1、tempo、phase、impact
correlation 和 lag）见中文报告
[`gt_oracle_suite_v2/REPORT_ZH.md`](../../eval/results/benchmark_v1/gt/gt_oracle_suite_v2/REPORT_ZH.md)。
AIST++ 与 FineDance 的指标分布明显不同，后续模型和 SONIC 结果必须先做数据集/速度/风格
分层比较，不能用 pooled BAS 或单一总分直接排名。
差异来源和区分实验单独记录在
[`DATASET_DIFFERENCE_ANALYSIS_20260823.md`](../evaluation/DATASET_DIFFERENCE_ANALYSIS_20260823.md)。

该已有 G1 root/ground 诊断见
[`finedance_root_height_audit_v1/REPORT_ZH.md`](../../eval/results/benchmark_v1/gt/finedance_root_height_audit_v1/REPORT_ZH.md)。在 root 坐标约定修正前，D5 的
`root_height_min` 只保留为诊断字段；generator 和 SONIC execution 的正式比较优先使用
foot penetration、FSR/PFC proxy，以及其余冻结的 D/M/X 指标。

GT 不是每项指标都必须达到 1.0 的“唯一正确答案”，而是不同数据集、速度和风格下的
经验分布。模型结果应和匹配的 GT stratum 比较。

### 3.3 分模块 GT 基线与已有模型评分

已用同一套 D（舞蹈动作质量）和 M（音乐-舞蹈适配）指标在 AIST++ 1,408 条与
FineDance 203 条 paired G1 reference 上建立 dataset/style/tempo 条件分布。当前
dataset-level 中位数为：AIST++ `D energy=5.0692, jerk P95=1272.85,
BAS=0.2635, event F1=0.7059`；FineDance `D energy=4.2198, jerk P95=1103.69,
BAS=0.2237, event F1=0.6938`。两套数据使用相同指标实现，但绝对分布不同，后续
必须按 matched stratum 比较。

对现有模型 artifacts 的 GT 分布接近度评分（100 为接近对应分层 GT 中心，不是绝对
优美度）为：M2/M_ref 的 D/M 为 `47.0/68.1`，M3/M_ref 为 `51.6/62.9`，
M4/M_ref 为 `43.3/67.4`；SONIC 执行后分别为 M2 `44.7/58.8`、M3 `47.5/54.5`、
M4 `38.8/61.9`。这些结果仅用于 pipeline diagnosis，因为当前歌曲、seed、时长和
执行协议未平衡，不能作为最终模型排名。

报告：[`module_benchmark_v1/REPORT_ZH.md`](../../eval/results/benchmark_v1/gt/module_benchmark_v1/REPORT_ZH.md)。

全量中文 oracle 报告：[`gt_oracle_suite_all_v1/REPORT_ZH.md`](../../eval/results/benchmark_v1/gt/gt_oracle_suite_all_v1/REPORT_ZH.md)。需要注意，GT 原始 paired 数据可以直接建立 P/D/M 的 reference distribution；X（SONIC 执行）、R（在线 deadline/latency）和 H（人类感知）没有对应观测，已在全模块 inventory 中标为 pending，不能从 GT 分数推断。

这次分数不能被理解成只看 D/M 两个总分。taxonomy v1 共冻结 49 个指标，覆盖 P/D/M/X/R/H：
当前 D 的 jerk、energy、static ratio 已有 GT corruption 方向证据，M 的 BAS/Beat F1
只有部分方向证据；FID/Div、phrase、retrieval/MMDist、完整 X retention、R latency/deadline
和 H blind study 仍需补齐。完整清单见
[`module_calibration_audit.json`](../../eval/results/benchmark_v1/gt/module_benchmark_v1/module_calibration_audit.json)。
因此后续任何 M2/M3/M4 结果都必须逐模块给出原始值、matched-GT reference、SONIC retention
和 pending 项，不能用单个 BAS、单个 D/M 分数或 success rate 代替整个 benchmark。

## 4. 已完成实验结论

### 4.1 早期音乐条件和 beat feature

- GaussianBeat/8D beat-only 能改变节奏相关行为，但不足以表达完整音乐风格和动态。
- Wav2CLIP 能提供高层语义/风格信息，但单独不能解决节拍、动作幅度和 G1 接触问题。
- motion energy 能缓解平均态，motion beatness 能表达动作落点和顿感，但仍需要更强的
  G1 motion prior 和执行约束。
- 当前推荐条件组织：低层 Librosa/MERT + beat suite；Wav2CLIP 作为语义消融，不把
  所有 encoder 直接堆叠为主线。

### 4.2 M3 音乐因果消融

实验为 song012/065 × 3 seeds × paired/wrong/+4 s shifted/null，共 24 条 30 s 轨迹。

- 改变 music sidecar 会改变生成轨迹，说明模型确实使用了该条件。
- paired 相对 wrong 的 BAS 差约 `-0.0193`，impact correlation 差约 `+0.0023`。
- paired 在 BAS 和 impact correlation 上分别只胜出 `2/6` 和 `4/6`。

结论：可以声称“music sidecar 对生成有因果影响”，不能声称“当前 M3 已经学会了正确
音乐匹配”。

已有 M2 song098 三个 sampling seed 和 M3 song012/065 reference 已按全量 1,611 条
GT distribution 重新标定，结果见
[`m2_m3_gt_comparison_v2`](../../eval/results/motion_music_execution/m2_m3_gt_comparison_v2/)。
这只是 reference-level calibration：当前 M2/M3 仍集中在少数歌曲，不能代表跨数据集、
跨 tempo/style 的最终 generator benchmark。

### 4.3 M3 `M_ref` 首轮完整指标

M3 MRT2-conditioned release 已生成 `song012/065 x seed{1234,2345,3456}` 六条 60 秒
G1 reference，并完成 G1-native 全模块中的 reference-side 指标。结果见
[`eval/RESULTS.md`](../../eval/RESULTS.md) 和
[`m3_reference_metrics/REPORT.md`](../../eval/results/benchmark_v1/formal/m3_reference_metrics/REPORT.md)。

| 模块 | 结果 | 判断 |
|---|---:|---|
| 有限轨迹率 | 1.000 | 生成格式和数值有效 |
| D overall | 51.6 | 相对对应 GT 分层为中等接近度，不是绝对质量百分比 |
| M overall | 62.9 | 存在音乐适配信号，但仍不稳定 |
| FK BAS | 0.2708 | 有一定 beat 对齐 |
| Beat precision / recall / F1 | 0.3037 / 0.1866 / 0.2275 | 命中精度尚可，音乐 beat 覆盖不足 |
| Beat timing mean / std | -0.049 / 1.448 帧 | 已命中事件的时间偏差较小 |
| Offbeat false-positive rate | 0.6963 | 非拍动作比例较高 |
| Root drift / foot sliding | 1.535 m / 0.505 m | 需要和同音乐 O-G1 GT 做严格分层比较 |

当前最稳妥的结论是：M3 能生成有效的 G1 reference，部分节奏事件能够准确落在音乐
beat 附近，但没有稳定覆盖音乐中的全部节奏事件，动作动态表现力也偏弱。不能把
`BAS` 或单个 timing 指标解释成整体舞蹈质量，也不能把这六条 reference 与其他论文的
headline 数值直接排名。当前结果不包含 SONIC execution、tracking retention 或实时性
claim；这些属于下一层 paired `M_exec` 评估。

### 4.4 SONIC execution

- SONIC 基础接口、reference stream 和 tracking loop 已跑通。
- 统一 reference/execution 对照显示，tracker 能保留主要姿态，但会衰减 motion energy、
  1--3 Hz 动态和部分手臂/高频表达。
- 初始化、时钟、reference FPS、packet mode 和仿真负载会显著影响结果；失败 run 不能
  直接归因于 generator。
- closed-loop 的 measured state 存在时间错位和 tracker-generator domain gap；目前没有
  稳定优于 open-loop 的证据。

结论：SONIC 是可用 tracker，但当前 paper 主线应优先使用严格时钟的 open-loop online
generator + SONIC tracking；closed-loop 作为后续 tracker-aware extension。

### 4.5 GT capability 和 retargeting

- GT/O-G1/O-Exec 已建立三级 oracle 关系，四层正式记录已固定为
  O-Human / O-G1 / M_ref / M_exec。
- GMR 损失和 SONIC tracking 损失必须分别报告，不能把两者混成一个“模型误差”。
- O-Human/O-G1/O-Exec 的同一套 music-motion 指标可以作为 generator 和 tracker 的共同
  参考标准。
- 当前 38 条 O-Human -> O-G1 正式汇总见
  [`eval/results/benchmark_v1/formal/GT_REPORT.md`](../../eval/results/benchmark_v1/formal/GT_REPORT.md)。
  AIST++ 的 activity/root correspondence 在 retarget 后总体保持较好；FineDance
  明显偏低，说明需要先排查 SMPL/G1 配对、帧率、坐标和 beat 标注兼容性。这个结果
  是 retargeting audit，不应直接归因于 diffusion，也不应把 O-G1 当成无条件完美
  oracle。
- AIST++ 20 条和 FineDance 18 条 paired/wrong-song 完整性审计均显示 paired
  相关性平均高于 wrong-song；但 Top-1 只有 0.30 和 0.22，说明 benchmark 中的
  自动音乐指标必须和 counterfactual、分层分布及后续盲评一起使用。

## 5. 当前能写和不能写的 claim

### 可以支持

- 建立了一个从音乐到 G1 reference 再到 SONIC execution 的在线系统框架。
- 提出了/实现了严格因果、H8/C4 streaming motion generation interface。
- music sidecar 会对 motion trajectory 产生因果影响。
- SONIC 能执行部分生成 reference，并可量化动作和音乐响应的保留率。
- 建立了跨 AIST++/FineDance 的 GT-calibrated evaluation protocol。

### 目前不能支持

- 不能声称 M2/M3 已稳定优于 wrong-song、shifted 或 null condition。
- 不能声称当前模型已经对任意新歌实时生成高质量舞蹈；现有 cache/PKL 仍有限。
- 不能用单个 BAS、FID 或成功视频证明“舞蹈优美”或“音乐匹配”。
- 不能把 offline PKL replay 写成 online music generation。
- 不能把 closed-loop 当前结果写成已经优于 open-loop。

## 6. ICRA 主线缺口

1. 在统一 benchmark 上完成 M2/M3/M4 的多歌曲、多 seed、跨 tempo/style reference 评估。
2. 增加 paired、wrong-song、time-shifted、tempo-preserved、silence 五组正式 music
   counterfactual，验证 paired 是否真的更好。
3. 运行同一批 reference 的 SONIC execution，计算 beat suite、动作质量和 retention。
4. 完成 R@K/MMDist 的独立 audio-motion encoder calibration。
5. 完成 PFC、FSR、contact、foot-slip、FID/Div 等未统一实现的指标。
6. 完成 3 songs × 3 seeds × 3 SONIC repeats 和盲评。
7. 证明 causal audio-prefix inference 在连续 60 s 内满足 deadline、无 future leakage。

四层正式 manifest 已建立在 [`eval/results/benchmark_v1/formal/`](../../eval/results/benchmark_v1/formal/)。
它把同一音乐下的 O-Human、O-G1、M-ref 和 M-exec 固定在一条记录中；当前已有 38 条
GT records 和 5 条 model records。缺失的 M-ref/M-exec 明确标为 `pending`，在补齐前不
进入正式结果表。
已有的 M0/M2/M4 song098 SONIC repeats 与 M3 012/065 pair metrics 已另外汇总到
[`eval/results/benchmark_v1/formal/MODEL_REPORT.md`](../../eval/results/benchmark_v1/formal/MODEL_REPORT.md)。
它用于检查评估链路和 tracker retention，不能替代平衡的多歌曲、多 seed 正式实验。
当前 route-level 描述性均值见
[`MODEL_SUMMARY.md`](../../eval/results/benchmark_v1/formal/MODEL_SUMMARY.md)，不作为跨 route
排名。
Beat event Precision/Recall/F1、tempo error 和 phase error 已对有完整 motion/audio
的 22 条 artifact 补算，见
[`model_music_extended/REPORT.md`](../../eval/results/benchmark_v1/formal/model_music_extended/REPORT.md)
和中文说明
[`REPORT_ZH.md`](../../eval/results/benchmark_v1/formal/model_music_extended/REPORT_ZH.md)。
此前的 9 条 M0/M2/M4 SONIC execution 已从 feedback log 恢复为完整 measured-motion
PKL，因此这些旧记录的 event-level execution 指标已经补齐；未采集的正式矩阵单元仍为
pending。
正式扩展的 72-cell 采集矩阵见
[`eval/results/benchmark_v1/formal/EXPANSION_MATRIX.csv`](../../eval/results/benchmark_v1/formal/EXPANSION_MATRIX.csv)。
当前只有 5 个 balanced reference cell 和 3 个 execution cell；M3 pilot 与 M2
不同 training seed 的结果暂不混入 balanced sampling-seed 统计。
最新 `prior-dev` release 只有 unconditional pure Commit Forcing 权重，不能继续
生成新的 M2/M3 music-conditioned sample；当前 M2/M3 PKL 只能用于 offline 评估和
SONIC replay。checkpoint 边界与采集顺序见
[`MODEL_EXPANSION_STATUS_20260823.md`](../evaluation/MODEL_EXPANSION_STATUS_20260823.md)。

## 7. 当前下一步

正式扩展从 GT 两层校准进入模型四层对照：先固定 38 条 O-Human/O-G1 结果和
FineDance 的数据审计状态，再把每个模型的 G1 `M_ref` 与同一条 reference 的
`M_exec` 接入。由于 generator 已直接输出 G1，后续模型结果不再经过 GMR；GMR
只在 O-Human -> O-G1 这一层报告。

下一轮不再新增零散指标，而是固定以下矩阵：

```text
dataset: AIST++ / FineDance
tempo: slow / medium / fast
style: available strata
condition: paired / wrong / shifted / silence
model: M0 / M2 / M3 / M4
stage: O-Human(SMPL/SMPLH) / O-G1(GMR) / M_ref(G1) / M_exec(G1)
seed: at least 3
```

主表统一报告：

```text
dance quality + beat-alignment suite + semantic matching
+ SONIC retention + realtime system metrics
```

任何新模型都必须进入这个矩阵后，才能更新 ICRA 主结论。
