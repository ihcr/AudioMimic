# Evaluation Workspace Layout

更新日期：2026-08-23

## Formal Results

- benchmark_v1/gt/: GT oracle、AIST++/FineDance 全量分布、retargeting audit、模块化指标和 calibration。
- benchmark_v1/formal/: O-Human、O-G1、M-ref、M-exec 四层正式 benchmark。
- generator_vs_tracker_baselines/: generator/reference 与 SONIC execution 的基线对照。
- motion_music_execution/m2_m3_gt_comparison_v2/: 当前 M2/M3/GT 对照表。
- m3_music_ablation/formal_30s/: M3 paired/wrong/shifted/null 音乐条件消融。
- mrt2_comparison/: 当前保留的 corrected M3 generator/SONIC 对比视频。

## Supporting Results

- gt_sonic_capability/: SONIC capability、接口和 execution 分析摘要。
- generation_to_execution_gap/: 保留的 phase-level 分析，不保留原始长 rollout。
- mrt2_metrics/: 当前 M3 execution metric JSON。
- human_study/: 当前人工评价资产和设计文件。

## Archive

历史 raw rollout、重复版本、无效视频和旧 sim2sim 输出已移至：

/tmp/audiomimic_eval_archive_20260823/

归档是可恢复的，不参与当前 benchmark。正式实验只应写入上述 canonical 目录。
