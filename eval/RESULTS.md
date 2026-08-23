# AudioMimic Evaluation Results

更新日期：2026-08-23

这是论文实验结果的唯一入口。完整指标按同一套模块计算，不用 BAS 或任意单一
总分替代整套评估。

## 1. 指标定义

- 指标总表、方向、来源和是否允许论文 claim：
  [EVALUATION_MAP_MUSIC_TO_G1.md](../docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md)
- 指标分类和报告模板：
  [METRIC_TAXONOMY_MUSIC_DANCE_G1.md](../docs/evaluation/METRIC_TAXONOMY_MUSIC_DANCE_G1.md)
- 论文指标审计：
  [LITERATURE_METRIC_AUDIT_20260823.md](../docs/evaluation/LITERATURE_METRIC_AUDIT_20260823.md)

指标分为：

1. Dance quality：energy、velocity、acceleration、jerk、static ratio、repetition、
   PFC、foot sliding、penetration、root stability、FID/Div 和人工评价。
2. Beat/rhythm alignment：BAS、reverse BAS、event precision/recall/F1、onset/
   impact correlation、response lag、tempo error、phase error。
3. Music adaptation：dynamic response、audio-motion retrieval、R@K/MMDist、style/
   tempo 分层和人工音乐性评价。
4. SONIC execution：tracking RMSE/EMPKPE、energy/amplitude/band retention、foot/contact
   retention、root stability、time-to-fall 和 success rate。
5. Online system：P50/P95/P99 latency、deadline miss、stale condition、packet drop、
   fallback 和 realtime factor。

## 2. 正式 benchmark 与结果位置

| 对象 | 结果入口 | 当前状态 |
|---|---|---|
| GT oracle，AIST++ 1,408 + FineDance 203 | [module benchmark report](benchmark_v1/gt/module_benchmark_v1/REPORT_ZH.md) | 已完成全量统计 |
| GT 全量分布和分风格/tempo 统计 | [grouped GT report](benchmark_v1/gt/grouped_gt_all_v1/REPORT_ZH.md) | 已完成 |
| 38 条 sealed calibration test | [GT oracle report](benchmark_v1/gt/gt_oracle_suite_v2/REPORT_ZH.md) | 已完成指标方向校准 |
| SMPL/O-Human -> G1/O-G1 | [retargeting results](benchmark_v1/gt/retargeting_loss_v1/) | 已完成 sealed audit；全量结果另列 |
| 四层正式协议 O-Human/O-G1/M-ref/M-exec | [formal benchmark](benchmark_v1/formal/README.md) | manifest 已建立 |
| M0/M2/M3/M4 reference 与 execution | [model summary](benchmark_v1/formal/MODEL_SUMMARY.md) | 已有描述性结果，尚未平衡 |
| M2/M3/GT 对照 | [comparison report](motion_music_execution/m2_m3_gt_comparison_v2/REPORT.md) | 已完成当前样本 |
| M3 paired/wrong/shifted/null 音乐消融 | [ablation aggregate](m3_music_ablation/formal_30s/aggregate_v2/REPORT.md) | 已完成首轮 |
| SONIC corrected 对比视频 | [MRT2 videos](mrt2_comparison/) | 已保留 corrected 版本 |

机器可读结果：

- GT module distributions：`benchmark_v1/gt/module_benchmark_v1/module_distributions.csv`
- 当前 model module scores：`benchmark_v1/gt/module_benchmark_v1/model_module_scores.csv`
- 当前 formal stage table：`benchmark_v1/formal/model_stage_results.csv`
- 扩展 beat/tempo/phase table：`benchmark_v1/formal/model_music_extended/metrics.csv`
- 指标 inventory：`benchmark_v1/gt/module_benchmark_v1/metric_inventory.csv`

## 3. 当前数值应如何解释

GT module report 给出 AIST++ 和 FineDance 各自的 reference distribution。模型分数
是模型结果相对于对应 dataset/style/tempo GT 分布的接近程度，不是“优美度绝对分”，
也不能把 AIST++ 和 FineDance 的 raw value 直接排名。

当前 M_ref/M_exec 的正式摘要在
[MODEL_SUMMARY.md](benchmark_v1/formal/MODEL_SUMMARY.md)。该表显示：

- M2/M3/M4 都已经有完整的 Dance、Beat/Music 和 SONIC retention 字段；
- M_exec 相比 M_ref 普遍出现 energy、jerk、FSR 或 BAS retention 变化，说明 tracker
  会改变 generator 的动作特征；
- M2/M4 主要来自 song098，M3 主要来自 song012/song065，样本数量和音乐没有平衡，
  因此当前只能作为 pipeline diagnosis，不能写成 M0/M2/M3/M4 的最终排名。

## 4. 论文当前可以声称什么

可以声称：

- 已建立跨 AIST++/FineDance 的多模块 GT benchmark；
- 已将原始 paired motion、G1 reference、generator reference 和 SONIC execution 分开；
- 已量化 GMR/retargeting、generator-to-execution 和 SONIC retention；
- 已完成 M3 音乐条件的 paired/wrong/shifted/null 因果消融框架。

暂时不能声称：

- 当前模型已经优于已有 music-to-dance 方法；
- 当前 M3 已经证明正确音乐条件优于 wrong/shifted/null；
- 固定的 M2/M3/M4 PKL 已经构成任意实时音乐输入的在线系统；
- SONIC 后的指标仍然等于 generator reference 的指标。

## 5. 复现入口

```bash
cd ~/AudioMimic
conda activate audiomimic

# 查看正式 benchmark 报告
less eval/RESULTS.md
less eval/benchmark_v1/gt/module_benchmark_v1/REPORT_ZH.md
less eval/benchmark_v1/formal/MODEL_SUMMARY.md

# 运行 benchmark 单元测试
python -m unittest \
  tests/test_module_benchmark.py \
  tests/test_module_calibration_audit.py \
  tests/test_metric_taxonomy.py \
  tests/test_grouped_gt_benchmark.py \
  tests/test_evaluation_map.py
```

原始 telemetry 和大体积 rollout 保存在本机归档目录，不属于 GitHub 公共结果：
`/tmp/audiomimic_eval_archive_20260823/`。
