# AudioMimic Evaluation Results

更新日期：2026-08-24

这是当前 `eval/` 的总报告。指标定义和解释见 [`README.md`](README.md)；本报告集中回答：

```text
GT benchmark -> M_ref -> SONIC -> M_exec
```

其中 D 是舞蹈动作质量，M 是音乐-舞蹈适配，X 是 G1/SONIC 执行，R 是在线系统，H 是人工感知。

## 1. 当前实验覆盖

| 对象 | 数据/模型 | 结果 |
|---|---|---|
| GT/O-G1 | AIST++ 1,408 条、FineDance 203 条 | D/M 全量分布、style/tempo 分组和 38 条校准集 |
| Retargeting | SMPL/O-Human -> G1/O-G1 | GMR 影响和质量审计 |
| Generator | M0/M2/M3/M4 | M_ref 的 D/M 描述性结果 |
| Tracker | SONIC | M_exec、tracking error、retention、稳定性 |
| Music ablation | M3 paired/wrong/shifted/null | 音乐条件因果对照，当前为首轮样本 |
| Online | generator、通信、SONIC | latency、deadline、状态新鲜度和 realtime factor |

## 2. GT D/M 基准

下面是同一实现、同一单位下的 dataset-level 中位数。AIST++ 和 FineDance 的 raw value
不能直接互相排名；模型应与对应 dataset/style/tempo 的 GT 分布比较。

| dataset | n | D energy | D jerk P95 | D FSR | D PFC | D penetration | M BAS | M event F1 | M impact corr | M tempo error | M phase error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIST++ | 1408 | 5.0692 | 1272.85 | 0.6321 | 0.1668 | 0.0000 | 0.2635 | 0.7059 | 0.0915 | 29.81 | 0.2675 |
| FineDance | 203 | 4.2198 | 1103.69 | 0.3007 | 0.0241 | 0.0026 | 0.2237 | 0.6938 | 0.0396 | 21.01 | 0.2515 |

完整 style/tempo 结果：
[`grouped_gt_all_v1/REPORT_ZH.md`](results/benchmark_v1/gt/grouped_gt_all_v1/REPORT_ZH.md)。

完整模块表：
[`module_benchmark_v1/REPORT_ZH.md`](results/benchmark_v1/gt/module_benchmark_v1/REPORT_ZH.md)。

## 3. M2/M3/M4 的 D/M 总评分

分数是相对于对应 GT 分层中心的接近度，100 表示接近该分层 GT 中心，不是“优美度绝对分”。

| route | stage | n | D dataset | D style | D tempo | D overall | M dataset | M style | M tempo | M overall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 | M_ref | 3 | 43.5 | 36.5 | 60.9 | 47.0 | 64.8 | 68.9 | 70.5 | 68.1 |
| M2 | M_exec | 3 | 42.6 | 38.6 | 52.9 | 44.7 | 55.2 | 56.9 | 64.2 | 58.8 |
| M3 | M_ref | 2 | 43.2 | 43.1 | 68.6 | 51.6 | 59.4 | 60.4 | 69.0 | 62.9 |
| M3 | M_exec | 2 | 42.8 | 44.5 | 55.1 | 47.5 | 50.4 | 50.9 | 62.3 | 54.5 |
| M4 | M_ref | 3 | 41.3 | 34.0 | 54.6 | 43.3 | 63.4 | 66.8 | 72.0 | 67.4 |
| M4 | M_exec | 3 | 36.0 | 30.8 | 49.6 | 38.8 | 58.8 | 59.8 | 67.0 | 61.9 |

这张表只能作为当前 pipeline diagnosis：M2/M4 主要是 song098，M3 是 song012/065，
歌曲、seed、时长和执行协议没有完全平衡，不能作为最终方法排名。

## 4. 当前 M_ref/M_exec 详细指标

详细动作质量 D：
[`formal/MODEL_SUMMARY.md#1-dance-quality`](results/benchmark_v1/formal/MODEL_SUMMARY.md)。

详细音乐适配 M：
[`formal/MODEL_SUMMARY.md#3-music-and-beat-alignment`](results/benchmark_v1/formal/MODEL_SUMMARY.md)。

其中包含：

- D：energy、amplitude、velocity、acceleration、jerk、static、repetition、C4 boundary jump；
- 物理：FSR、PFC、foot contact、penetration、root height、root displacement；
- M：speed/impact correlation、response lag、双向 BAS、audio/motion beat count；
- Beat event：precision、recall、F1、tempo error、phase error；
- X：energy/amplitude/jerk/FSR/BAS retention 和 execution lag。

## 5. 结果目录

| 模块 | 结果入口 |
|---|---|
| GT D/M benchmark | `results/benchmark_v1/gt/module_benchmark_v1/` |
| GT calibration 和 corruption | `results/benchmark_v1/gt/gt_oracle_suite_v2/`、`motion_corruptions_v1/` |
| GMR/retargeting | `results/benchmark_v1/gt/retargeting_loss_v1/`、`retargeting_loss_all_v1/` |
| 四层 protocol | `results/benchmark_v1/formal/` |
| M0/M2/M3/M4 reference/execution | `results/benchmark_v1/formal/`、`results/motion_music_execution/` |
| M3 音乐消融 | `results/m3_music_ablation/formal_30s/aggregate_v2/` |
| SONIC capability | `results/gt_sonic_capability/` |
| generation-execution gap | `results/generation_to_execution_gap/` |
| 视频和图 | `results/mrt2_comparison/`、`results/figures/` |

## 6. 当前结论和限制

- 已经建立 D/M 的 GT benchmark，并将 AIST++ 与 FineDance 分开按 style/tempo 校准。
- BAS 是 M1 beat 子模块的重要指标，但必须和双向 BAS、event P/R/F1、tempo、phase、impact
  correlation 和 response lag 一起解释。
- `M_exec` 必须重新计算 D/M，不能直接沿用 `M_ref` 分数；tracker retention 单独报告。
- 当前 M2/M3/M4 结果样本不平衡，尚不能证明模型优于已有方法，也不能证明任意实时音乐输入。
- FID/Div、retrieval、style/emotion 和人评需要固定 extractor、split、序列长度和协议后，
  才能进入最终论文排名。

## 7. 本轮执行状态（2026-08-23）

### 7.1 GT benchmark 校验

本轮重新执行了 GT benchmark 的有效性检查和模块汇总：

| 检查 | 状态 | 说明 |
|---|---|---|
| D 模块校验 | PASS | 6 项动作质量/物理扰动检查通过；jerk、低通能量、freeze/static 和 beat freeze 检查均可用 |
| M 模块校验 | PASS_WITH_CAVEAT | beat F1 校验通过；重复相似度仍需按数据集和风格分层解释 |
| 全量 GT 汇总 | 已完成 | AIST++ 1,408 条、FineDance 203 条已进入统一 module benchmark |
| 最终测试集冻结 | 未完成 | 当前协议仍处于 `final_test_pending`，不能据此做最终论文排名 |

校验明细见
[`benchmark_validity_v1/REPORT.md`](results/benchmark_v1/gt/benchmark_validity_v1/REPORT.md)，
模块汇总见
[`module_benchmark_v1/REPORT_ZH.md`](results/benchmark_v1/gt/module_benchmark_v1/REPORT_ZH.md)。

### 7.2 正式模型矩阵

正式矩阵固定为 `M0/M2/M3/M4 × 012/065/098 × 3 seeds × {M_ref,M_exec}`，共 72 个格子。
本轮扫描结果如下：

| 状态 | 数量 | 含义 |
|---|---:|---|
| M_ref available | 11 个矩阵格、15 个轨迹文件 | 有完整生成轨迹，可计算 generator 端 D/M；M2/M4 的 3 个文件是 training seed 变化，不是 3 个 sampling seed |
| M_exec available | 3 | 有对应 SONIC 记录，可计算 execution 端 D/M/X |
| pending | 58 | 缺生成轨迹、SONIC 配对记录，或 seed/song 尚未按正式协议补齐 |

矩阵明细见
[`formal/EXPANSION_MATRIX.json`](results/benchmark_v1/formal/EXPANSION_MATRIX.json)。
因此第 3 节的 M0/M2/M3/M4 表是当前 pipeline diagnosis，不是最终 ablation ranking。
矩阵现在显式区分 `sampling_seed` 和 `training_seeds`，避免把不同 checkpoint 与不同采样随机性混为一谈。

### 7.3 音乐条件的当前证据

M3 的 paired/wrong/shifted/null 消融已经完成首轮 24 条轨迹、6 个 paired blocks。
paired 条件相对 wrong 条件在 impact correlation 上赢得 4/6，在绝对 response lag 上赢得
4/6；相对 null 条件在 jerk 上赢得 6/6、impact correlation 上赢得 5/6。
但 BAS 只在 2/6 的 null 对比中获胜，说明“sidecar 改变了生成行为”已有证据，
“正确音乐配对已经稳定优于错误音乐”仍未被充分证明。

详细结果见
[`aggregate_v2/REPORT.md`](results/m3_music_ablation/formal_30s/aggregate_v2/REPORT.md)。

### 7.4 接下来唯一优先事项

1. 按同一 song、seed、时长、初始 K64 和 3 秒 alignment + 1 秒 hold 协议补齐正式矩阵。
2. 对每个 `M_ref` 生成严格配对的 `M_exec`，同时保留 reference/execution 两套 D/M/X 表。
3. 完成最终 test split 冻结后，再报告模型之间的排名；在此之前只报告均值、方差、
   分层分布和 retention，不写“优于”结论。

本轮扩展预检后，`mrt2-conditioned-v1` 已经落盘并完成校验，M3 的
`song012/065 × seed{1234,2345,3456}` 六个正式 `M_ref` 均已生成 60 秒轨迹，
每个 receipt 都记录了 checkpoint、sidecar、音频和运动文件的 SHA256。M2/M4 仍只有
既有 song098 轨迹，不能据此声称覆盖完整的音乐条件矩阵。

本轮已完成 M3-012/065 的 2 秒 CPU smoke，以及 GPU 环境下 6 条 60 秒正式生成：
checkpoint、test cache、K64 对齐、音频路径和 G1 motion pickle 均正常，每条输出
1800 帧且 `finite_motion=true`。下一步应在完全相同协议下先计算 6 条 `M_ref` 的完整
D/M 指标，再逐一接 SONIC 生成配对的 `M_exec`。

### 7.5 M3 六条 `M_ref` 首轮 G1 结果

六条轨迹已经完成 G1-native benchmark，结果文件为
[`m3_reference_metrics/metrics.json`](results/benchmark_v1/formal/m3_reference_metrics/metrics.json)，
报告为 [`REPORT.md`](results/benchmark_v1/formal/m3_reference_metrics/REPORT.md)。

| 模块 | 指标 | M3 `M_ref` 均值 | 说明 |
|---|---|---:|---|
| Rhythm | G1 FK BAS | 0.2708 | 音乐 beat 到动作 beat 的覆盖倾向 |
| Rhythm | Beat precision / recall / F1 | 0.3037 / 0.1866 / 0.2275 | 以音乐 beat detector 为 target，30 FPS |
| Rhythm | Beat timing mean / std | -0.049 / 1.448 frames | 匹配 beat 的相对时间误差 |
| Rhythm | Wrist / Foot / Torso beat F1 | 0.2292 / 0.2138 / 0.2409 | 身体部位节奏响应 |
| Motion quality | Root drift / path length | 1.535 m / 12.151 m | 生成 reference 的根部位移与路径长度 |
| Motion quality | Joint jerk mean | 446.758 | 连续性诊断，数值需和 GT 同协议校准 |
| Physical | Foot sliding / penetration | 0.505 / 0.0433 m | FK 后足部滑动和最低点穿地诊断 |
| Validity | Finite motion / range violation | 1.000 / 0.0103 | 数值有效率和关节范围越界率 |

这是一轮 generator-side diagnosis，不是最终论文排名。当前 pickle 没有
`designated_beat_frames`，因此旧式 BAP precision/recall 为 0，这个 0 只表示
“没有提供 designated beat 标注”，不能解释为动作没有 beat。`G1Dist/G1Div` 目前只用
两条 `012/065` reference cache，且尚未完成跨数据集尺度校准，也暂不用于模型优劣结论。
本轮只对 `M_ref` 做结论：六条轨迹均为有限值，说明生成链路和 G1 数据格式有效；
但 beat recall 只有 0.1866，且 offbeat false-positive rate 为 0.6963，说明当前
生成动作只覆盖了部分音乐 beat，不能据此声称已经形成稳定的音乐节奏控制。另一方面，
已匹配 beat 的 timing mean 为 -0.049 帧、std 为 1.448 帧，表示命中的事件时间偏差
较小；当前主要问题是 beat 事件覆盖率和非拍事件数量，而不是单个命中事件的时间偏移。
Root drift、foot sliding 和 jerk 仍需与同音乐的 O-G1 GT 分层分布比较后再判断是否
属于异常，不能用全局数值直接排名。`M_exec`、tracker retention 和实时性不属于本轮
reference-only 结论。
