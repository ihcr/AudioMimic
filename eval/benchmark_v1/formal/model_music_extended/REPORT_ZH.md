# 扩展音乐与节奏指标报告

本报告在动作质量和 SONIC retention 指标之外，补充音乐-动作对应关系的事件级、
速度级和相位级评估。BAS 是 beat alignment 的核心指标，但不能单独代表舞蹈质量；
因此必须和事件覆盖、tempo、phase、onset/impact correlation 以及 response lag 一起解释。

## 指标定义

- **Event Precision / Recall / F1**：音乐 beat/onset 与动作 impact 事件在容差窗口内的精确率、召回率和 F1。
- **Tempo error (BPM)**：动作 impact 的估计速度与音乐 tempo 的绝对差，越低越好。
- **Phase error (cycles)**：动作 impact 相对音乐 beat 的周期相位误差，越低越好。
- **Speed / impact correlation**：动作速度包络或 impact 强度与音乐对应曲线的最佳相关性，越高越好。
- **BAS / reverse BAS**：音乐 beat 到动作 beat、以及动作 beat 到音乐 beat 的双向对齐分数。

## 当前结果

以下为同一路线内的平均值；`M_ref` 是生成器参考轨迹，`M_exec` 是 SONIC 执行后的测量轨迹。
这些结果用于检查评估链路和 execution retention，不作为未平衡矩阵上的最终模型排名。

| 路线 | 阶段 | n | Event P | Event R | Event F1 | Tempo error (BPM) | Phase error (cycles) | Speed corr | Impact corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | M_ref | 3 | 0.6349 | 0.7348 | 0.6803 | 35.0361 | 0.2613 | -0.0107 | 0.0244 |
| M0 | M_exec | 3 | 0.6227 | 0.7609 | 0.6849 | 40.9237 | 0.2558 | -0.0091 | 0.0422 |
| M2 | M_ref | 3 | 0.6176 | 0.7508 | 0.6777 | 39.6270 | 0.2531 | -0.0169 | 0.0252 |
| M2 | M_exec | 3 | 0.6227 | 0.7340 | 0.6738 | 39.0775 | 0.2493 | -0.0168 | 0.0079 |
| M3 | M_ref | 2 | 0.6676 | 0.7630 | 0.7074 | 25.3708 | 0.2543 | -0.0012 | 0.0207 |
| M3 | M_exec | 2 | 0.6151 | 0.7143 | 0.6551 | 32.6968 | 0.2428 | -0.0043 | 0.0136 |
| M4 | M_ref | 3 | 0.6270 | 0.7677 | 0.6902 | 40.9237 | 0.2606 | -0.0135 | 0.0303 |
| M4 | M_exec | 3 | 0.6226 | 0.7441 | 0.6780 | 32.4841 | 0.2535 | -0.0103 | 0.0059 |

## 数据状态

旧的 M0/M2/M4 SONIC 记录已经从 feedback log 导出为完整的 measured-motion PKL，
因此本报告已经包含这些记录的 `M_exec` 事件指标，不需要为这 9 次旧实验重新录制。
尚未采集的 72-cell 正式矩阵单元仍然标记为 pending；后续只有这些新单元需要重新运行
SONIC 并按相同协议记录。

本报告不把 BAS 当作唯一判断标准，也不把自动指标直接等同于‘舞蹈优美’。最终结论应结合
GT 分层分布、paired/wrong-song counterfactual、动作质量、音乐匹配和人工盲评。
