# SONIC Capability Calibration

更新日期：2026-08-20
状态：phase_a_complete_9_of_9，phase_b_complete_9_of_9

## 1. 目的

该实验先用 SONIC 自带的已知可追踪 reference 验证 AudioMimic 接口，再用真实数据集
retarget 后的 G1 reference 测量 SONIC 的能力范围。它回答：

1. 在低、中、高动态的 SONIC 已知分布和外部 GT 上，tracker 能稳定执行到什么强度；
2. 即使输入已知可追踪 reference，SONIC 会损失多少幅度、能量和高频细节；
3. M0/M2/M4 的 tracking gap 有多少是 tracker 固有损失，有多少来自 generator 分布偏移。

本实验遵循
[`AudioMimic Music-to-G1 Evaluation Map`](../evaluation/EVALUATION_MAP_MUSIC_TO_G1.md)。
它是 capability calibration，不替代 60 s 长时稳定性实验。

## 2. 两阶段 Reference

### 2.1 Phase A：接口能力基准

SONIC `reference/example` 中的 CSV 已经采用 SONIC/IsaacLab joint order 和 50 Hz。
[`prepare_sonic_capability_references.py`](../../eval/prepare_sonic_capability_references.py)
将其逆映射为 MuJoCo order、重采样为 30 Hz，再交给标准 wire adapter。这样可验证完整
`CSV -> AudioMimic contract -> ZMQ -> SONIC` 路径，且不会重复应用 joint permutation。

仅从时长至少 8 s、reference root height 全程不低于 0.6 m 的动作中分层：

| Level | SONIC reference | Duration | Joint velocity RMS | Root height min |
|---|---|---:|---:|---:|
| Low | `walking_quip_360_R_002__A428_M` | 9.10 s | 1.170 rad/s | 0.733 m |
| Medium | `macarena_001__A545_M` | 27.50 s | 1.334 rad/s | 0.651 m |
| High | `dance_in_da_party_001__A464_M` | 9.93 s | 2.423 rad/s | 0.638 m |

转换记录见
[`manifest.json`](../../eval/gt_sonic_capability/sonic_known_trackable_20260820/manifest.json)。

### 2.2 Phase B：retargeted GT 外部分布

从 568 条时长至少 10 s 的 GT 中，按 motion energy、velocity P95、acceleration P95 和
jerk P95 的平均百分位选择动态强度约 20%、50%、80% 的动作，并优先使用不同 genre。

| Level | Motion | Duration | Dynamic percentile | Energy | Velocity P95 | Jerk P95 |
|---|---|---:|---:|---:|---:|---:|
| Low | `gLH_sBM_cAll_d16_mLH0_ch09` | 11.97 s | 0.199 | 1.772 | 2.855 | 527.0 |
| Medium | `gKR_sBM_cAll_d28_mKR1_ch05` | 10.63 s | 0.501 | 5.960 | 5.617 | 1142.9 |
| High | `gMH_sBM_cAll_d22_mMH0_ch01` | 11.97 s | 0.801 | 9.571 | 6.551 | 1985.6 |

原始候选选择结果见
[`selection_manifest.json`](../../eval/gt_sonic_capability/selection_20260819/selection_manifest.json)。
原始 GT quaternion 为 `xyzw`。

Phase B 不再承担接口正确性证明。只有 Phase A 在相同初始化和 ZMQ 协议下通过后，Phase B
的失败才可解释为 retarget/reference 分布或 tracker capability 问题。

## 3. 固定协议

每条 reference 独立重复 3 次，共 9 runs：

```text
packet_mode = full
playback_rate = 1.0
alignment = 3 s measured state + 1 s hold
reference_safety = none
SONIC reference rate = 50 Hz
feedback = 5557
sim state = 5559
```

每个 repeat 必须完整重启 MuJoCo 和 SONIC，并遵循同一初始化状态机：

```text
MuJoCo starts with elastic band enabled
-> SONIC Init Done
-> enter CONTROL with bundled Macarena
-> wait 3 s
-> release elastic band
-> verify unassisted standing for 3 s
-> enable ZMQ streaming
-> start 3 s measured alignment + 1 s hold
-> execute reference
```

不能在 elastic band 仍启用时切入 ZMQ，再在 alignment 末端释放。该旧流程会产生与
reference 无关的瞬时跌倒，并污染 capability 结论。

不得使用运行时 velocity clipping。若中/高动态 reference 摔倒，保留失败数据并记录
time-to-fall；重启 SONIC 后继续下一 repeat，不能更换 reference。

## 4. 输出指标

- Success Rate、time-to-fall、minimum base height；
- raw/lag-compensated joint RMSE、tracking lag；
- root-relative EMPKPE；
- amplitude、energy、0--1/1--3/3--8 Hz retention；
- arms 3--8 Hz retention；
- height-derived contact F1，暂作诊断。

Phase-A baseline 采用成功 runs 的 error P95 和 retention P05，但只表示 SONIC
known-distribution 的接口/tracker 基线，不是独立 GT gate。Phase B 完成前保持
`provisional`，不得自动覆盖 Evaluation Map 中的正式门槛。

## 5. Phase-A 正式结果

2026-08-20 使用修正后的初始化顺序完成 9/9 runs。分析仅自动排除 metadata 中记录的
4 s alignment，不再额外 trim：

| Tier | Success | Min height mean | Raw RMSE | Aligned RMSE | Lag | EMPKPE | Amp. retention | Energy retention | 0--1 Hz | 1--3 Hz | 3--8 Hz | Arms 3--8 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | 3/3 | 0.731 m | 0.1162 | 0.1126 | 40 ms | 0.0630 m | 0.939 | 0.625 | 0.942 | 0.658 | 0.313 | 0.208 |
| Medium | 3/3 | 0.661 m | 0.1386 | 0.1375 | 20 ms | 0.0633 m | 0.945 | 0.666 | 0.860 | 0.631 | 0.620 | 0.509 |
| High | 3/3 | 0.660 m | 0.1751 | 0.1745 | 20 ms | 0.0891 m | 0.900 | 0.447 | 0.848 | 0.481 | 0.238 | 0.185 |

数值为每个 tier 三次 run 的均值。高动态组仍 3/3 稳定，但相对 low，aligned RMSE
增加 55%，EMPKPE 增加 41%，energy retention 从 0.625 降至 0.447。由此 Phase A
支持的结论是：接口和 tracker 基础稳定性成立，主要损失是动态表达、高频和手臂细节。

跨 9 条成功 runs 的 provisional SONIC-native baseline 为：aligned RMSE P95
`0.1747 rad`、EMPKPE P95 `0.0898 m`、amplitude retention P05 `0.896`、energy
retention P05 `0.444`、0--1/1--3/3--8 Hz retention P05 分别为
`0.847/0.472/0.230`。这些值不能称为外部 GT capability gate。

相同 reference 在 band 保持到 alignment 末端的旧流程出现瞬时跌倒，证明初始化顺序是
此前失败的混杂变量。正式 collection 仅接受修正后的状态机。

### 5.2 Phase-B retargeted GT

相同状态机、无 clipping 条件下，外部 retargeted GT 同样完成 9/9 runs：

| Tier | Success | Min height mean | Raw RMSE | Aligned RMSE | Lag | EMPKPE | Amp. retention | Energy retention | 0--1 Hz | 1--3 Hz | 3--8 Hz | Arms 3--8 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | 3/3 | 0.702 m | 0.1554 | 0.1540 | 20 ms | 0.1023 m | 0.921 | 0.670 | 0.868 | 0.596 | 0.890 | 0.599 |
| Medium | 3/3 | 0.697 m | 0.2025 | 0.1994 | 20 ms | 0.1133 m | 0.836 | 0.722 | 0.898 | 0.556 | 0.973 | 0.519 |
| High | 3/3 | 0.701 m | 0.3235 | 0.3231 | 20 ms | 0.1599 m | 0.810 | 0.656 | 0.623 | 0.719 | 0.721 | 0.279 |

因此外部分布 GT 的主要问题也不是 fall。相对 SONIC-native high，retargeted high 的
aligned RMSE 从 0.1745 增至 0.3231 rad（1.85 倍），EMPKPE 从 0.0891 增至
0.1599 m（1.80 倍），幅度保留从 0.900 降至 0.810。该差距是明确的
reference-domain tracking gap。

3--8 Hz retention 不能单独解释为动作质量；当 reference 高频功率很低或 execution
含跟踪振荡时，比值可接近或超过 1。必须与 RMSE、EMPKPE、绝对频带功率和视频共同报告。

外部 GT provisional gate：aligned RMSE P95 `0.3244 rad`、EMPKPE P95
`0.1608 m`、amplitude retention P05 `0.807`、energy retention P05 `0.627`。
当前每个 tier 只有一个 reference，正式论文 gate 仍需扩展 sequence/genre。

## 6. 采集与分析

采集命令由 selection manifest 中的 `planned_runs` 固定。SONIC 与 MuJoCo 已启动后，
每次只运行一个 repeat：

```bash
cd ~/AudioMimic
conda activate audiomimic

# Phase A: SONIC bundled known-trackable reference
bash scripts/run_sonic_known_trackable_capability.sh low 1

# Phase B: retargeted dataset GT
bash scripts/run_gt_sonic_capability.sh low 1
```

无窗口 Phase-A/Phase-B 全自动采集可执行：

```bash
python scripts/run_sonic_capability_suite.py \
  --levels low medium high \
  --repeats 1 2 3

python scripts/run_sonic_capability_suite.py \
  --source retargeted_gt \
  --levels low medium high \
  --repeats 1 2 3
```

将参数依次换成 `low 2`、`low 3`、`medium 1..3` 和 `high 1..3`。发生摔倒时先重启
SONIC/MuJoCo，再执行下一个 repeat；不要让 reset 后的状态混入同一 run。

Phase-A 完成 9 个 runs 后执行：

```bash
python eval/analyze_gt_sonic_capability.py \
  --manifest eval/gt_sonic_capability/sonic_known_trackable_20260820/manifest.json \
  --runs_root eval/gt_sonic_capability/known_trackable_runs \
  --output_dir eval/gt_sonic_capability/known_trackable_analysis/capability_9x_corrected
```

Phase-B 完成 9 个 retargeted GT runs 后执行：

```bash
python eval/analyze_gt_sonic_capability.py \
  --manifest eval/gt_sonic_capability/selection_20260819/selection_manifest.json \
  --runs_root eval/gt_sonic_capability/retargeted_gt_runs_v2 \
  --output_dir eval/gt_sonic_capability/retargeted_gt_analysis_v2
```

输出 `capability_analysis.json` 和 `capability_runs.csv`。若少于 3 个成功 GT runs，脚本
不会产生 empirical gate，只报告证据不足。
