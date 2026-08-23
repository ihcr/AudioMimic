# AIST++ 与 FineDance 指标差异分析

更新日期：2026-08-23

## 1. 当前观察

38 条 sealed GT 的 G1 benchmark 结果显示，两套数据的音乐-动作指标分布不同：

| 数据集 | n | 中位序列长度 | BAS | Beat Event F1 | Impact corr | Tempo error (BPM) | Phase error |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIST++ | 20 | 9.13 s | 0.2865 | 0.7143 | 0.1305 | 23.94 | 0.2747 |
| FineDance | 18 | 99.33 s | 0.2014 | 0.6623 | 0.0436 | 27.78 | 0.2551 |

这不能直接解释为 FineDance 的舞蹈质量更差。这里的指标是在 G1 reference 上计算的，
因此同时受到原始数据、音乐标注、序列长度、beat detector 和 GMR/retargeting 的影响。

## 1.1 原始 FineDance 是否损坏

对 18 条 FineDance sealed paired test 的原始 SMPL `.npy` 做了结构检查：

- 所有文件均存在，shape 为 `[T, 315]`，没有 NaN/Inf；
- 音频文件均存在，配对 manifest 中 `paired_valid=true`；
- motion 和 audio 的主要时长差异很小，但个别序列存在约 `0.37--0.50 s` 的尾部差异，评估时使用 common duration；
- 少数序列的零值比例明显偏高，可能反映未标注关节、补零或原始表示差异，需要在 GMR 前进一步审计。

因此，原始 FineDance **不是文件损坏**，但“文件可读”不等于“当前 beat evaluator 下的音乐性指标应当和 AIST++ 相同”。

### 全量 203 条 FineDance 检查

将检查扩展到 manifest 中全部 `203` 条 paired FineDance：

- `203/203` 条 source motion 均为有限值，shape 均为 `[T, 315]`；
- `12` 条存在音频/动作时长 warning，评估统一裁到 common duration；
- `6` 条零值比例超过 `0.20`，分别为 `010/042/137/139/143/145`，应单独检查是否存在补零或未标注关节；
- 全量 common duration 中位数约 `129.73 s`，范围约 `17.33--392.66 s`。

全量 source/target 的中位数如下：

| 数据集 | 层 | BAS | Event F1 | Impact corr | Tempo error (BPM) |
|---|---|---:|---:|---:|---:|
| AIST++ | source | 0.2445 | 0.6667 | 0.0859 | 28.50 |
| AIST++ | G1 target | 0.2591 | 0.7059 | 0.0853 | 25.76 |
| FineDance | source | 0.1201 | 0.4771 | 0.0282 | 20.91 |
| FineDance | G1 target | 0.2267 | 0.6829 | 0.0392 | 21.01 |

因此，FineDance 的 source 层在当前“音乐-动作对应”指标上的低分在全量数据上仍然成立，
不是 18 条 sealed test 的偶然；
但 GMR 后 source-to-G1 曲线对应关系仍然显著弱于 AIST++，两类问题必须分开报告。

## 2. 最强的当前证据：GMR 差异

同一批 source motion 经 GMR 后的 retargeting audit 给出：

| 数据集 | source->G1 activity corr | source->G1 root-speed corr | activity RMS ratio | root-speed RMS ratio | event count ratio | event timing error |
|---|---:|---:|---:|---:|---:|---:|
| AIST++ | 0.785 | 0.972 | 0.826 | 0.836 | 1.000 | 0.033 s |
| FineDance | 0.073 | 0.088 | 0.761 | 0.609 | 1.636 | 0.683 s |

FineDance 的 source-to-G1 曲线对应关系明显较弱，root 动态幅度也损失更多，且动作事件
数量增加、事件时间误差变大。这说明当前 FineDance 的低 BAS/impact correlation 至少
有一部分可能来自 GMR 的时序、坐标、root 或接触处理，而不是音乐-舞蹈配对本身。

同时，原始 source 层的中位数已经低于 AIST++：FineDance 的 source BAS/Event F1/Impact
correlation 约为 `0.1201/0.4653/0.0305`，AIST++ 约为 `0.2743/0.6583/0.1073`。
GMR 后 FineDance 的 BAS/Event F1 反而上升到约 `0.2316/0.6563`，但 source-to-G1
activity correlation 只有 `0.073`。这说明 GMR 可能改变了动作事件，使它在当前音乐
detector 下更容易命中，但没有忠实保留原始动作的连续动态。不能把所有差异都归因于 GMR，
也不能只看 G1 的 BAS 就认为 retargeting 成功。

## 3. 可能原因

### 3.1 数据组成和编舞风格不同

AIST++ 的测试片段较短，主要用于音乐-舞蹈生成 benchmark；FineDance 测试样本是更长的
跨风格序列，包含 Street、Jazz、Classic、Folk、Korean、Mix 等类别。不同风格的动作
accent 不一定都落在 beat/onset 上，可能表现为乐句、弱拍、持续姿态或上肢细节。因此
FineDance 的单一 motion-energy beat detector 可能漏掉有效舞蹈结构。

### 3.2 序列长度和统计稳定性不同

AIST++ sealed 序列中位数约 9 秒，FineDance 约 99 秒。短片段更容易保持局部节奏和单一
动作模式；长片段会包含 intro、transition、多个 phrase 和不同强度段落。全序列相关性
会被这些结构变化稀释，不能把两个数据集的 pooled mean 当成同一个任务。

### 3.3 beat/onset detector 与风格不完全匹配

当前 Event F1、BAS 和 impact correlation 依赖固定音频 beat/onset 提取和 G1 motion
energy。它对强拍清楚的片段更敏感，对切分复杂、弱拍、停顿、乐句型动作可能不稳定。
FineDance 的 freeze corruption 已经显示 Beat F1 在该数据集上只能作为 suite 的一部分，
不能单独作为质量 gate。

### 3.4 GMR/数据格式问题

FineDance 需要重点排查 source SMPL、audio、label、G1 retargeted motion 的时间起点、
root 坐标、朝向、四元数顺序、关节映射和接触脚定义。当前 retargeting 结果的 activity
和 root-speed correlation 过低，优先级高于继续调 diffusion。

### 3.5 指标定义本身的偏差

当前 motion beat 使用 G1 joint velocity/impact proxy，root drift 被刻意弱化，但 GMR
可能改变关节速度分布和局部极值。因而同一舞蹈在 SMPL 和 G1 上的事件数不一定一致，必须
分别报告 O-Human 与 O-G1，不能只看 G1 的绝对值。

## 4. 当前结论

目前可以确认：

1. AIST++ 与 FineDance 不能共享一个未经分层的“理想分数”。
2. FineDance 的差异不能直接归因于舞蹈风格；GMR 保真度问题是重要混杂因素。
3. O-Human、O-G1、M_ref、M_exec 必须分别评估，并报告相邻层的 delta/retention。
4. BAS 仍然是核心指标，但必须与 Event F1、tempo、phase、impact correlation、lag、
   motion event count 和人工判断联合使用。

## 5. 区分实验

后续固定以下顺序：

1. 已完成 source SMPL/SMPLH 的同套音乐指标检查：FineDance 的低 BAS/Event F1 在 GMR 之前已经存在，但需要区分数据分布和 detector 偏差。
2. 对 source 与 G1 做固定长度窗口评估，例如 8 秒、16 秒和完整序列，区分风格变化与长序列稀释。
3. 按 FineDance style、tempo 和动态强度分层，与 AIST++ 的对应层比较。
4. 对 FineDance 重新审计 audio/motion 起点、FPS、root、quat order、joint map 和接触定义。
5. 只有当 O-G1 的 retargeting delta 合理后，才把 FineDance 的 M_ref/M_exec 差异用于判断 generator 或 SONIC。

## 6. 全量窗口实验结果

已对全部 203 条 FineDance paired 序列使用固定、不重叠的 `5 s`、`16 s` 和 `full`
窗口评估；不足一个完整窗口的尾段不计入固定窗口统计。音频 beat detector、FPS、容差
和 music clock 与正式 retargeting audit 完全一致。

| 窗口 | 层 | n | BAS | Event F1 | Impact corr | Tempo error (BPM) | Phase error |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 s | source | 5405 | 0.0495 | 0.3529 | 0.0542 | 26.91 | 0.2786 |
| 5 s | G1 target | 5405 | 0.2048 | 0.7000 | 0.1305 | 26.95 | 0.2781 |
| 16 s | source | 1624 | 0.1286 | 0.5000 | 0.0365 | 22.27 | 0.2597 |
| 16 s | G1 target | 1624 | 0.2190 | 0.6866 | 0.0713 | 23.05 | 0.2594 |
| full | source | 203 | 0.1201 | 0.4771 | 0.0282 | 20.91 | 0.2523 |
| full | G1 target | 203 | 0.2267 | 0.6829 | 0.0392 | 21.01 | 0.2504 |

窗口结果排除了“只有长序列统计稀释”这一单一解释：FineDance source 在 5 s 窗口上
的 BAS/Event F1 更低，而不是更高。另一方面，G1 target 的 BAS 和 Event F1 在 5 s
窗口明显上升，说明当前 GMR 可能改变了局部 motion event，使动作更容易被 detector
匹配到音乐 beat；但这不能抵消 source-to-G1 activity correlation 仅约 `0.0495` 的问题。

原始结果文件见 `eval/benchmark_v1/gt/finedance_windowed_v1/`，其中：

- `windowed_metrics.csv/json`：逐序列、逐窗口、source/G1 两层结果；
- `windowed_summary.csv`：按窗口和层汇总；
- `REPORT.md`：可复现实验说明。

按舞蹈风格和 tempo 的全量汇总见 `eval/benchmark_v1/gt/finedance_stratified_v1/REPORT.md`，中文解读见
`eval/benchmark_v1/gt/finedance_stratified_v1/REPORT_ZH.md`。其中还包括“同风格是否更相似”的
探索性距离分析和置换检验。

数据角色已经冻结在 `eval/benchmark_v1/gt/benchmark_protocol_v1/REPORT_ZH.md`：38 条用于
指标校准，1611 条用于 GT reference 分布，最终论文 test split 尚未冻结。

当前判断因此更新为：FineDance 的差异首先包含原始 source 层的风格/表示/detector 问题，
其次包含 GMR 对局部事件和连续动态的重构问题；不能只用长序列长度解释，也不能只把问题
归因于 generator。这里的“低分”仅指当前音乐-动作对应指标，**不等于 FineDance 的总体
舞蹈质量下降**。

## 7. 不能用 BAS 推断舞蹈质量

全量 G1 oracle 的其他指标也必须同时报告。例如 AIST++/FineDance 的中位数为：

| 指标 | AIST++ G1 | FineDance G1 | 解释 |
|---|---:|---:|---|
| motion energy | 5.069 | 4.220 | 动作幅度/活动量，不是美观度 |
| jerk P95 | 1272.9 | 1103.7 | 平滑性与尖峰，低不一定更好，过低可能过平滑 |
| static ratio | 0.000 | 0.016 | 静止比例，需要结合舞蹈风格解释 |
| repeated pose ratio | 0.112 | 0.026 | 重复姿态代理，低不代表一定更自然 |
| root height min | 0.787 | 0.510 | G1 物理/retargeting 诊断，不是舞蹈审美指标 |
| Event F1 | 0.706 | 0.694 | 音乐事件覆盖，不是整体质量 |
| phase error | 0.267 | 0.252 | 节奏相位误差，越低越好，但只描述节奏层 |

这些数值不能组成一个总分。舞蹈质量至少要分成动作连续性、动作幅度、静止/重复、脚部
接触、root stability、音乐节奏适配和人工自然度几个维度。后续论文中应报告完整指标表，
而不是用 FineDance 的 BAS 或 Event F1 单独宣称“质量下降”。

相关结果：

- [GT Oracle 中文校准报告](../../eval/benchmark_v1/gt/gt_oracle_suite_v2/REPORT_ZH.md)
- [全量 1,611 条 GT 中文校准报告](../../eval/benchmark_v1/gt/gt_oracle_suite_all_v1/REPORT_ZH.md)
- [O-Human 到 O-G1 正式结果](../../eval/benchmark_v1/formal/GT_REPORT.md)
- [GMR retargeting audit](../../eval/benchmark_v1/gt/retargeting_loss_v1/REPORT.md)
