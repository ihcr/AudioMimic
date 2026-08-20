# M0/M2/M4 与 SONIC 能力基线对照

更新日期：2026-08-20
状态：首轮完成

## 1. 目的

本实验把现有 M0、M2、M4 的生成动作和 SONIC 执行结果放到同一评价坐标中，回答两个问题：

1. 三类 generator reference 的动态强度接近低、中、高哪一档 retargeted GT；
2. SONIC 对这些 reference 的损失是否超过其在 SONIC-native 和 retargeted-GT 基线上的正常范围。

比较使用逐关节中位数的幅度、能量和频带保真率。该定义与 capability 实验一致；不使用容易被少数大幅度关节主导的全局总能量比替代。

## 2. 数据范围

- generator：song098 的现有导出，M0 两条，M2/M4 各三条；
- execution：每条 route 固定 seed1234 reference，SONIC 独立重复三次；
- SONIC-native baseline：low/medium/high 各三次；
- retargeted-GT baseline：low/medium/high 各三次。

该范围足以做工程诊断，但不足以支持跨歌曲、跨 seed 的统计结论，也不包含审美评价。

## 3. Generator Reference

| Route | 条件含义 | 最近 GT 动态档 | Energy | Velocity P95 | Acceleration P95 | Jerk P95 |
|---|---|---:|---:|---:|---:|---:|
| M0 | unconditional parent | low | 2.396 | 3.358 | 42.3 | 819.1 |
| M2 | predicted future-music sidecar | low | 1.934 | 3.049 | 37.3 | 718.1 |
| M4 | oracle future-music sidecar | low | 1.789 | 2.914 | 36.1 | 691.7 |

三类输出都落在所选 GT 的低动态区域。M0 动态最强，M4 最保守。因此当前主要问题不是 generator 超过了 SONIC 已验证的高动态稳定范围，而是低动态 reference 在执行中仍发生明显细节衰减。

## 4. SONIC 执行结果

| Route | 成功率 | Aligned RMSE | EMPKPE | 幅度保留 | 能量保留 | 0--1 Hz | 1--3 Hz | 3--8 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 3/3 | 0.1763 rad | 0.0790 m | 0.912 | 0.468 | 0.802 | 0.427 | 0.227 |
| M2 | 3/3 | 0.1850 rad | 0.0874 m | 0.906 | 0.522 | 0.808 | 0.413 | 0.395 |
| M4 | 3/3 | 0.1678 rad | 0.0755 m | 0.872 | 0.503 | 0.785 | 0.467 | 0.244 |

M4 的位姿与末端误差最低，其次为 M0、M2。三类动作幅度基本保留，但逐关节动态能量仅保留约一半，中频和高频损失尤为明显。M2 的 3--8 Hz 均值被一次振荡运行抬高，方差很大，不能解释为细节保真优于 M0/M4。

相对 SONIC-native low，M0/M2/M4 的 aligned RMSE 分别为 1.57/1.64/1.49 倍，EMPKPE 分别为 1.25/1.39/1.20 倍。相对 retargeted-GT low，RMSE 分别为 1.14/1.20/1.09 倍。这说明三类生成动作存在一定 reference-domain gap，但没有超出 retargeted-GT 低动态基线很多；更突出的差异是动态能量和中高频表达被 tracker 压缩。

## 5. 音乐匹配解释

现有 automatic metrics 不能证明 M2 或 M4 相对 unconditional M0 的音乐条件优势。song098 上的 BAS music-to-motion 为 M0/M2/M4 = 0.257/0.247/0.242，impact correlation 为 0.032/0.025/0.031。M0 不读取音乐，因此其分数只能作为偶然对齐基线。

当前结论仅是“尚无证据”，不是“音乐条件无效”。要验证音乐驱动，需要同歌曲、同 motion seed 的 condition shuffle / time shift / silence 消融，并扩展到多歌曲、多 seed。

## 6. 下一步

1. 固定当前 SONIC 初始化和 tracking 协议，不继续修改 tracker；
2. 扩展到至少 3 首歌曲 x 3 个 generation seeds x 3 次 SONIC repeats；
3. 对 reference 与 execution 同时计算 onset/beat-event transfer、motion-beat precision/recall/F1 与节奏相位误差；
4. 加入 condition shuffle、时间平移和 silence，检验 M2 是否真正使用音乐；
5. 完成盲评，分别评价自然度、音乐匹配、动作丰富度和机器人可执行性。

## 7. 复现

```bash
cd ~/AudioMimic
conda activate audiomimic
python eval/compare_generator_to_sonic_capability.py
```

机器可读结果见 [`comparison.json`](../../eval/generator_vs_tracker_baselines/20260820/comparison.json)，完整自动生成表见 [`COMPARISON.md`](../../eval/generator_vs_tracker_baselines/20260820/COMPARISON.md)。
