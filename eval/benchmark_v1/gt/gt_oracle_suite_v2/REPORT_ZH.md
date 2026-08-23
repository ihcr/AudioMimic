# 多数据集 GT Oracle Benchmark（中文）

本报告使用 AIST++ 和 FineDance 的 paired 音乐-舞蹈数据，建立生成器和 SONIC
执行结果的参考分布。它不是把每条真实舞蹈压缩成一个必须达到 1.0 的分数，
而是按数据集、速度和风格提供可比较的 calibration range。

- 评估范围：`test`
- 总序列数：**38**
- AIST++：**20** 条
- FineDance：**18** 条
- beat 指标使用固定 audio clock、固定 beat detector 和固定事件容差；模型和 SONIC 必须复用同一协议。

## 数据集分层的核心音乐指标

表中为每个数据集的中位数。BAS、Event F1、相关性越高越好；tempo/phase/lag 越接近 0 越好。

| 数据集 | n | BAS | Event F1 | Tempo error (BPM) | Phase error | Impact corr | Lag (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| aistpp | 20 | 0.2865 | 0.7143 | 23.9373 | 0.2747 | 0.1305 | 0.6174 |
| finedance | 18 | 0.2014 | 0.6623 | 27.7773 | 0.2551 | 0.0436 | 0.4333 |

## 指标含义

- **BAS**：动作 kinematic beat 到最近音乐 beat 的时间接近程度，是 beat-alignment 的核心指标，但不是唯一指标。
- **Beat Event F1**：动作 impact 对音乐 beat/onset 的命中与覆盖，联合 Precision、Recall 解释，避免少量动作 beat 造成虚高 BAS。
- **Tempo error**：动作 impact 速度与音乐节奏速度的 BPM 差异。
- **Phase error**：动作 accent 相对于音乐周期的位置误差。
- **Impact correlation**：动作 impact 强度与音乐 onset/能量曲线的相关性。
- **Lag**：两者最佳相关对应的时间偏移；只有相关性达到可靠性阈值时才解释。

## 如何用于模型和 tracker

1. `O-Human` 用于观察原始 SMPL/SMPLH paired 数据的自然分布。
2. `O-G1` 用于报告 GMR/retargeting 后的变化；这不是 diffusion 的生成误差。
3. `M_ref` 与匹配的 O-G1/GT 分层比较，报告 generator gap。
4. `M_exec` 使用同一条 reference 经过 SONIC 的测量轨迹，报告 tracking retention。

不同数据集的数值不能直接混合排名。最终论文还需要 paired/wrong-song、time-shifted、
tempo-preserved counterfactual，以及 R@K/MMDist 和人工盲评；自动指标不能单独等同于
‘舞蹈优美’或‘音乐语义匹配’。

完整逐条数据见 `gt_oracle_suite_metrics.json` 和 `gt_oracle_suite_summary.csv`。
