# AudioMimic Blind Human Evaluation Protocol v1.0

更新日期：2026-08-20
状态：协议冻结，song098 pilot 素材已生成，正式多歌曲素材待生成

## 1. 目标

自动指标不能独立判断动作是否自然、是否像舞蹈、是否与音乐合适，也不能完整描述 SONIC
损失的表现力。本协议把人类评价拆为三个相互独立的 pairwise studies：

| Study | 比较对象 | 音频 | 问题 |
|---|---|---|---|
| Generator quality | M0/M2/M4 reference | 静音 | H-NATURAL、H-DANCE、H-EXPRESS；长片另问 H-COHERENCE |
| Music match | M0/M2/M4 reference | 相同配对音乐 | H-RHYTHM、H-STYLE |
| Execution retention | 同一 reference 与其 SONIC execution | 静音主表；有声作为补充 | H-NATURAL、H-DANCE、H-EXPRESS |

三个 study 不合并成总分。Generator quality 不播放音乐，避免音乐偏好污染动作质量判断；
Music match 必须播放完全相同、响度归一化的音轨；Execution retention 只改变
reference/execution 身份。

## 2. 素材硬条件

每个 pair 必须固定：

- song、clip start、duration 和 generation seed；
- G1 模型、地面、背景、分辨率、FPS、相机轨迹和编码参数；
- 音频起点、响度和编码；
- 对 tracker study，固定同一条 `M_ref`，不得重新采样 generator。

像素内不得出现 M0/M2/M4、reference/execution、方法名、文件名、成功/失败等身份提示。
左右位置随机化。失败 run 不得从抽样框删除，应截取到首次失败并继续显示统一的失败标记，
防止只展示成功片段。

正式主表至少覆盖 3 首 held-out songs x 3 generation seeds。每条 reference 的 SONIC
执行使用 3 个独立 repeats。现有单 song098 调试视频不满足该条件。

## 3. Clip 设计

- 短时自然度/节奏：每段 12--20 s，三段固定窗口覆盖开头、中段和后段；
- 长期连贯性：使用 30--60 s 独立 trial，不由短片评分替代；
- 同一 participant 不重复看到完全相同的 pair；
- 左右视频同时开始，禁止循环速度不一致；
- Music match 使用固定音频时钟，不能为了提高某个方法得分单独平移音轨。

现有 60 s sequence 建议冻结三个 16 s 窗口：`[4,20)`、`[24,40)`、`[44,60)`。
启动 alignment 不进入正式窗口。

## 4. 问卷措辞

每题选择 `Left / Tie / Right`，不显示“更好模型”等诱导文本。

| ID | 显示问题 |
|---|---|
| H-NATURAL | 哪一侧动作更自然，抖动、突变或明显错误更少？ |
| H-DANCE | 哪一侧更像有组织、可观看的舞蹈？ |
| H-EXPRESS | 哪一侧的动作力度、层次和表现力更丰富？ |
| H-COHERENCE | 哪一侧在整段中更连贯，并且不过度停滞或重复？ |
| H-RHYTHM | 哪一侧的动作重音和节奏与音乐更匹配？ |
| H-STYLE | 哪一侧的动作风格和情绪更适合音乐？ |

每位 participant 最多完成每个 study 6 个正式 trials，共不超过 18 个，再加入 2 个明显
错误的 attention checks。预计时长控制在 20 分钟内。

## 5. Participant 与排除规则

- 预注册至少 24 名有效 participants；报告舞蹈/机器人经验和音频播放设备；
- 要求佩戴耳机完成 Music match；
- 开始前确认知情同意，收集最小化匿名信息；
- 仅按预注册规则排除：未完成、两个 attention checks 均失败、浏览器无法播放视频/音频；
- 不按评分方向、是否支持假设或“看起来异常”排除 participant；
- 保存全部原始匿名响应和排除日志。

真实受试者研究在启动前必须遵守所在机构的人体研究/伦理审批要求。

## 6. 随机化与统计

随机化单位是 participant。工具为每位 participant 独立打乱 trials，并让同一 canonical pair
在总体上尽量平衡左右位置。公开问卷只读取 blind clip ID；route 与 representation 映射只
保存在 private key 中，分析前不解盲。

每个问题分别报告：

1. raw left/right/tie counts；
2. route pair 的 win probability；
3. participant-aware、同时按 song 聚类的 bootstrap 95% CI；
4. 含 tie 的 Davidson/Bradley-Terry 模型，或在主文只报告预注册的 pairwise probability；
5. participant 数、有效 comparisons 和排除数量。

generation seed 与 tracker repeat 是嵌套随机因素，不能当成彼此独立的生成样本。多问题
检验使用 Holm correction，效应量与置信区间优先于只报告 p-value。

## 7. 素材登记与生成

当前 inventory：
[`assets_current_20260820.json`](../../eval/human_study/assets_current_20260820.json)。
现有视频均为调试 composite、缺少音频或 route/seed 覆盖不匹配，因此不进入正式问卷。

校验并生成 blinded study design：

```bash
cd ~/AudioMimic
conda activate audiomimic
python eval/build_human_pairwise_study.py
```

输出分为：

- `public_study.json`：问卷端使用的 blind IDs、问题和 participant assignments；
- `private_key.json`：blind ID 到 route/reference/execution 的映射，不提供给 participant；
- `asset_audit.json`：不合格素材及原因。

输出分别报告 `pilot_ready` 和 `paper_ready`。三类 study 均存在候选 pair 时
`pilot_ready=true`；只有满足 3 songs x 3 generation seeds x 3 tracker repeats 时，
`paper_ready=true`，并同步设置兼容字段 `ready=true`。

首套 song098/seed1234 pilot 已完成，6/6 素材通过审计并形成 9 个 canonical trials；结果见
[`RESULT-20260820-blind-video-pilot.md`](../experiments/RESULT-20260820-blind-video-pilot.md)。
