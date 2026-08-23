# AudioMimic Evaluation Hub

更新日期：2026-08-23

这里是 AudioMimic 的评估总入口。所有 generator、GMR/retargeting、SONIC
execution 和 end-to-end 实验都应使用同一套定义，并保留失败 run，不用单一总分替代
多维结果。

## 1. 先读什么

| 文件 | 内容 |
|---|---|
| [`EVALUATION_MAP_MUSIC_TO_G1.md`](EVALUATION_MAP_MUSIC_TO_G1.md) | 冻结的指标定义、方向、等级和来源 |
| [`ICRA_GT_CALIBRATED_EVALUATION_PLAN.md`](ICRA_GT_CALIBRATED_EVALUATION_PLAN.md) | GT oracle、sealed test、论文 claim 和验收规则 |
| [`LITERATURE_METRIC_AUDIT_20260823.md`](LITERATURE_METRIC_AUDIT_20260823.md) | FACT、EDGE、Lodge、Beat-It、RoboPerform、DiscoForcing 等论文的指标审计 |
| [`DATASET_DIFFERENCE_ANALYSIS_20260823.md`](DATASET_DIFFERENCE_ANALYSIS_20260823.md) | AIST++/FineDance 指标差异、GMR 混杂因素和区分实验 |
| [`METRIC_TAXONOMY_MUSIC_DANCE_G1.md`](METRIC_TAXONOMY_MUSIC_DANCE_G1.md) | 所有指标的“大模块 / 子模块 / 指标”分类、定义、来源和报告模板 |
| [`HUMAN_EVALUATION_PROTOCOL.md`](HUMAN_EVALUATION_PROTOCOL.md) | 盲评和 pairwise preference 协议 |
| [`../experiments/EXPERIMENT_CONCLUSIONS_20260823.md`](../experiments/EXPERIMENT_CONCLUSIONS_20260823.md) | 当前所有实验结论和 ICRA claim 状态 |

## 2. 评价对象

```text
audio A -> generator -> reference M_ref -> SONIC -> execution M_exec
```

- `M_ref`：generator 生成的动作是否自然、连续、多样、可执行。
- `A-M_ref`：动作是否响应对应音乐的节奏、动态、tempo、phase 和语义。
- `M_ref-M_exec`：SONIC 是否保留 reference 的动作和音乐响应。
- `A-M_exec`：机器人最终动作是否仍然符合音乐。

## 3. 冻结指标组

### 3.1 Dance quality

`PFC`、`FSR/foot skating`、velocity/acceleration/jerk、motion energy、静止率、
重复率、root/base stability、penetration、FIDk/FIDg、Divk/Divg 和人类评价。

FID/Div 只在 feature extractor、split、序列长度和采样策略一致时比较；FID 只作
辅助分布指标，Div 以 GT 分布为目标，不是越大越好。

### 3.2 Beat alignment and music matching

`BAS`、reverse BAS、Beat Precision/Recall/F1、onset/impact correlation、response
lag、tempo error、phase error，以及冻结 audio-motion encoder 后的 `R@K/MMDist`。

`BAS` 是核心 beat-alignment 指标，不是辅助指标；但它必须和其他 beat 指标一起解释。
它检查动作 beat 与音乐 beat 的接近程度，不能单独证明动作优美、音乐语义匹配或整体
音乐性。`BAP/KPD` 只有在模型有显式 beat assignment/key-pose target 时才启用。

### 3.3 SONIC execution

`Success rate`、time-to-fall、EMPJPE、EMPKPE、lag-compensated RMSE、amplitude/
energy retention、频段 retention、contact/foot-slip retention、root stability。

同一 reference 必须在 `M_ref` 和 `M_exec` 上重算适用的 G/M 指标，报告绝对值和
retention/degradation；失败 run 不能删除。

### 3.4 Online system

严格因果性、audio-to-feature、feature-to-motion、motion-to-reference、
reference-to-execution 的 P50/P95/P99 latency、deadline miss、stale condition、
packet drop、fallback 和 uninterrupted 60 s realtime factor。

## 4. Benchmark 层级

| 层级 | 数据 | 作用 |
|---|---|---|
| O-Human | AIST++/FineDance 原始 paired motion/audio | 自然舞蹈和音乐-动作配对分布 |
| O-G1 | 同一动作经过 GMR/retargeting 的 G1 reference | G1 目标动作分布及 retargeting delta；也是 generator 的训练域 |
| O-Exec | O-G1 经过 SONIC 的执行动作 | tracker 能力上限和 execution loss |
| M-ref | M0/M2/M3/M4 generator reference | 模型生成质量和音乐条件作用 |
| M-exec | 对应 generator reference 的 SONIC execution | 端到端最终表现 |

当前 GT reference distribution：AIST++ 1,408 条、FineDance 203 条，共 1,611 条。
由于 diffusion 直接在 O-G1 数据上训练，`M_ref` 本身就是 G1 motion，不应再做 GMR。
正式表中 SMPL/SMPLH、O-G1、M_ref、M_exec 是同一音乐样本下的四个平行对象：
SMPL 到 O-G1 报 retargeting delta，O-G1 到 M_ref 报 generator delta，M_ref 到
M_exec 报 tracking retention。
分层结果见 [`benchmark_v1/gt/stratified_audit_all_v1`](../../eval/benchmark_v1/gt/stratified_audit_all_v1/)。
其中 38 条 sealed calibration test（AIST++ 20、FineDance 18）用于指标方向验证，
不能和全量 calibration 混用来调模型。

## 5. 运行入口

详细命令在 [`eval/README.md`](../../eval/README.md)。常用结果：

- GT 分层 corruption：`eval/benchmark_v1/gt/stratified_corruptions_v1/`
- GT 分层方向审计：`eval/benchmark_v1/gt/stratified_audit_v1/`
- 全量 GT 分布：`eval/benchmark_v1/gt/stratified_audit_all_v1/`
- GT oracle 中文校准报告：`eval/benchmark_v1/gt/gt_oracle_suite_v2/REPORT_ZH.md`
- 全量 GT oracle 中文校准报告：`eval/benchmark_v1/gt/gt_oracle_suite_all_v1/REPORT_ZH.md`
- FineDance 全量窗口化 source/G1 分析：`eval/benchmark_v1/gt/finedance_windowed_v1/REPORT.md`
- FineDance 舞蹈风格/tempo 分层指标：`eval/benchmark_v1/gt/finedance_stratified_v1/REPORT.md`
- FineDance 舞蹈风格/tempo 中文报告：`eval/benchmark_v1/gt/finedance_stratified_v1/REPORT_ZH.md`
- GT benchmark 数据角色协议：`eval/benchmark_v1/gt/benchmark_protocol_v1/REPORT_ZH.md`
- AIST++/FineDance 全量分风格与 tempo 报告：`eval/benchmark_v1/gt/grouped_gt_all_v1/REPORT_ZH.md`
- FineDance-G1 root z / existing asset ground 诊断：`eval/benchmark_v1/gt/finedance_root_height_audit_v1/REPORT_ZH.md`
- GT 分模块 benchmark 与 M2/M3/M4 当前评分：`eval/benchmark_v1/gt/module_benchmark_v1/REPORT_ZH.md`
- D/M 指标方向校准：`eval/benchmark_v1/gt/module_benchmark_v1/CALIBRATION_REPORT_ZH.md`
- taxonomy v1 全模块覆盖清单：`eval/benchmark_v1/gt/module_benchmark_v1/metric_inventory.csv`
- GMR/retargeting：`eval/benchmark_v1/gt/retargeting_loss_v1/` 和 `retargeting_loss_all_v1/`
- M2/M3/GT 对照：`eval/motion_music_execution/m2_m3_gt_comparison_v2/`
- 四层正式 manifest：`eval/benchmark_v1/formal/`（O-Human/O-G1/M-ref/M-exec）
- GT 两层正式结果：`eval/benchmark_v1/formal/GT_REPORT.md`、
  `gt_human_to_g1.csv`、`gt_human_to_g1.json`
- paired/wrong-song 数据完整性审计：
  `eval/benchmark_v1/gt/aist_music_pairing_v1/`、
  `eval/benchmark_v1/gt/finedance_music_pairing_v1/`
- 已有模型阶段汇总：`eval/benchmark_v1/formal/MODEL_REPORT.md`、
  `model_stage_results.csv`、`model_stage_results.json`
- 当前 route-level 描述性汇总：`eval/benchmark_v1/formal/MODEL_SUMMARY.md`
- 扩展 beat event/tempo/phase 指标：
  `eval/benchmark_v1/formal/model_music_extended/REPORT.md`、
  [`REPORT_ZH.md`](../../eval/benchmark_v1/formal/model_music_extended/REPORT_ZH.md)
- 正式扩展采集矩阵：`eval/benchmark_v1/formal/EXPANSION_MATRIX.csv`、
  `EXPANSION_MATRIX.json`
- 模型扩展状态与 checkpoint 边界：
  `docs/evaluation/MODEL_EXPANSION_STATUS_20260823.md`
- M3 音乐因果消融：`eval/m3_music_ablation/formal_30s/aggregate_v2/`
- SONIC 对照：`docs/experiments/RESULT-20260820-generator-vs-sonic-baselines.md`

## 6. 当前 benchmark 结论

38 条 sealed GT 的 494 个 clean/退化样本中，jitter-jerk、lowpass-jerk、freeze-static
和 AIST++ freeze Beat F1 的方向检查通过；FineDance freeze Beat F1 为 WARN。因此
Beat F1 保留，但不能作为 FineDance 上单独的 corruption gate。`repeat_similarity`
暂不列为核心指标。

完整结论、未支持的 claim 和下一步实验以
[`EXPERIMENT_CONCLUSIONS_20260823.md`](../experiments/EXPERIMENT_CONCLUSIONS_20260823.md)
为准。

### GT 两层基线当前解释

`O-Human -> O-G1` 是 GMR/retargeting 基线，不能写成 generator error；
`O-G1 -> M_ref` 才是 generator gap，`M_ref -> M_exec` 才是 SONIC tracking
retention。当前 38 条校准结果中，AIST++ 的 activity/root correspondence
整体明显高于 FineDance；FineDance 的低 correspondence 和部分 beat 统计需要
先做格式、帧率和 beat 标注审计。因此 O-Human 和 O-G1 都要报告，但不能假设
每个数据集的 O-G1 都是同样理想的 oracle。

AIST++ 20 条测试配对和 FineDance 18 条测试配对都已经完成 paired/wrong-song
审计。两者的 paired 音乐平均相关性都高于 wrong-song control，但 Top-1 不是
100%，因此这一步只能证明数据中存在配对信号，不能把自动指标当作绝对的人类
音乐性真值。
