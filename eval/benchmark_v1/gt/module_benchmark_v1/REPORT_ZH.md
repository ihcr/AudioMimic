# GT 分模块 Benchmark 与 M2/M3/M4 当前评分 v1

本报告把指标分为 D（舞蹈动作质量）和 M（音乐-舞蹈适配）两大模块。GT 使用 AIST++ 1408 条和 FineDance 203 条 paired G1 reference；模型使用当前已有的 M2/M3/M4 reference/execution artifacts。

模型分数不是‘优美度绝对分’，而是模型结果与对应 dataset/style/tempo GT 分布的接近程度：100 表示接近该分层 GT 中心，越低表示偏离越大。AIST++ 与 FineDance 不使用 pooled raw value 直接排名。

## 模块定义

| module | 含义 | 指标 |
|---|---|---|
| D | 不看音乐时的动作质量、平滑性、活力和物理合理性 | energy, velocity, acceleration, jerk, static/repetition, FSR/PFC/penetration |
| M | 动作对对应音乐的节拍、动态、tempo 和 phase 适配 | BAS/reverse BAS, event P/R/F1, timing error, speed/impact correlation, lag, tempo/phase error |

## GT 数据覆盖

- AIST++: `1408` 条；10 个 genre。
- FineDance: `203` 条；多标签 style。
- 所有原始分布保存在 `module_distributions.csv`；便于直接读表的 style/tempo 模块摘要在 `gt_style_tempo_module_summary.csv`。

## GT 数据集模块基线

以下是同一实现、同一单位下的 dataset-level 中位数；它们用于描述分布，不能把 AIST++ 的绝对值当作 FineDance 的排名标准。完整 style/tempo 表见 `grouped_gt_all_v1/REPORT_ZH.md`。

| dataset | n | D energy | D jerk P95 | D FSR | D PFC | D penetration | M BAS | M event F1 | M impact corr | M tempo error | M phase error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aistpp | 1408 | 5.0692 | 1272.85 | 0.6321 | 0.1668 | 0.0000 | 0.2635 | 0.7059 | 0.0915 | 29.81 | 0.2675 |
| finedance | 203 | 4.2198 | 1103.69 | 0.3007 | 0.0241 | 0.0026 | 0.2237 | 0.6938 | 0.0396 | 21.01 | 0.2515 |

## 当前 M2/M3/M4 模块评分

| route | stage | n | target style | target tempo | D dataset | D style | D tempo | D overall | M dataset | M style | M tempo | M overall |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 | M_ref | 3 | Hiphop;Street | medium_90-130 | 43.5 | 36.5 | 60.9 | 47.0 | 64.8 | 68.9 | 70.5 | 68.1 |
| M2 | M_exec | 3 | Hiphop;Street | medium_90-130 | 42.6 | 38.6 | 52.9 | 44.7 | 55.2 | 56.9 | 64.2 | 58.8 |
| M3 | M_ref | 2 | Jazz;Locking;Street | mixed | 43.2 | 43.1 | 68.6 | 51.6 | 59.4 | 60.4 | 69.0 | 62.9 |
| M3 | M_exec | 2 | Jazz;Locking;Street | mixed | 42.8 | 44.5 | 55.1 | 47.5 | 50.4 | 50.9 | 62.3 | 54.5 |
| M4 | M_ref | 3 | Hiphop;Street | medium_90-130 | 41.3 | 34.0 | 54.6 | 43.3 | 63.4 | 66.8 | 72.0 | 67.4 |
| M4 | M_exec | 3 | Hiphop;Street | medium_90-130 | 36.0 | 30.8 | 49.6 | 38.8 | 58.8 | 59.8 | 67.0 | 61.9 |

## 论文指标对照

| 工作 | 其主要指标 | 在本 benchmark 的对应模块 | 可否直接比较 |
|---|---|---|---|
| FACT / AIST++ | FIDk/FIDg, Distk/Distg, BeatAlign, user study | D.realism/diversity, M.rhythm, H | 仅同 extractor、split、长度和 beat detector 时可比较 |
| EDGE | PFC, Beat Align, Distk/Distg, Elo/win rate | D.physics, M.rhythm, H | PFC/BAS 可复现，论文数值不能跨数据集直接搬用 |
| Lodge | FIDk/FIDg, FSR, Div, BAS, runtime, user study | D.realism/physics/diversity, M.rhythm, R/H | 需要统一 FineDance/AIST++ 协议 |
| Beat-It | PFC, BAS, Div, BAP, KPD, user study | D, M.rhythm, H | BAP/KPD 只有显式 beat/keypose target 才启用 |
| RoboPerform | R@1/2/3, MMDist, success, EMPJPE/EMPKPE | M.semantic, X.tracking/safety | 需要独立 audio-motion encoder 和相同 simulator |

## 论文中的典型报告值

下表只记录本地论文原文中的代表性结果，用于检查量纲和评估习惯；由于数据集、序列长度、feature extractor、beat detector 和统计协议不同，不能直接当作本项目的排名阈值。

| 来源/数据集 | 方法或 GT | PFC/FSR | BAS/BeatAlign | FID/Div 或其他 |
|---|---|---:|---:|---|
| EDGE / AIST++ | GT | PFC 1.332 | 0.24 | Distk 10.61, Distg 7.48 |
| EDGE / AIST++ | EDGE (w=2) | PFC 1.5363 | 0.26 | Distk 9.48, Distg 5.72 |
| Lodge / FineDance | GT | FSR 6.22% | 0.2120 | Divk 9.73, Divg 7.44 |
| Lodge / FineDance | Lodge (DDPM) | FSR 5.01% | 0.2397 | FIDk 45.56, FIDg 34.29 |
| Beat-It / AIST++ | GT | PFC 1.338 | 0.384 | Divk 9.773, Divm 7.212 |
| Beat-It / AIST++ | Ours (beat + keyframes) | PFC 0.966 | 0.661 | BAP 0.793 |
| RoboPerform / FineDance | Music-Motion retrieval | n/a | n/a | R@1 66.7, R@2 78.8, R@3 83.5, MM-Dist 1.154 |

当前 AudioMimic M_ref 的 BAS 均值约为：M2 `0.247`、M3 `0.263`、M4 `0.242`；它们接近部分 FineDance/AIST++ 论文的 BAS 量级，但不能据此宣称优于论文方法。

## 当前解释与限制

- GT 不是每项指标都应为 1；不同风格有不同动作强度和节奏策略。
- BAS 仍是 M.rhythm 核心指标，但必须和 event F1、tempo/phase、correlation 和 lag 一起报告。
- 当前 M2/M3/M4 的歌曲、seed、长度和 execution protocol 尚未完全平衡，因此评分是 pipeline 诊断，不是最终论文排名。
- M3 只有 012/065，M2/M4 主要是 098；下一轮正式实验要扩展到多歌曲、多 seed、跨 tempo/style。
- FID/Div、retrieval、人评和 X/R 模块仍需独立冻结 extractor/protocol，不能从当前 D/M 分数推断。
