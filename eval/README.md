# AudioMimic Evaluation Map

本目录回答四个问题：

```text
audio -> generator -> M_ref -> SONIC -> M_exec
```

1. 生成的动作本身是否像自然、连续、可执行的舞蹈？
2. 动作是否真正匹配对应音乐的 beat、tempo、动态和风格？
3. SONIC 执行后保留了多少生成动作的表现力？
4. 整条链路是否满足实时性和稳定性要求？

`M_ref` 和 `M_exec` 必须使用同一套 D/M 指标。这样才能区分“generator 没生成出来”和
“tracker 执行时损失了”。BAS 是重要的节奏指标，但不能替代完整的 D/M 评估。

## 1. 评估对象

| 对象 | 含义 | 主要报告 |
|---|---|---|
| `GT/O-Human` | 原始数据集中的 SMPL 人类动作与音乐 | benchmark 上限和数据分布 |
| `GT/O-G1` | GT 经过 GMR/retargeting 后的 G1 reference | G1 表示和 retargeting 影响 |
| `M_ref` | generator 直接输出的 G1 reference | D + M |
| `M_exec` | SONIC 执行后的实际 motion/state | D + M + X |
| Online | 实时 feature、推理、通信和执行链路 | R + X |

## 2. 指标总表

### D：Dance quality，舞蹈动作质量

D 不看音乐，回答“这段动作本身是否自然、丰富、平滑、物理合理”。指标不是越大越好，
模型应接近同数据集、同 style、同 tempo 的 GT 分布。

| 子模块 | 指标 | 代表什么 | 理想方向 |
|---|---|---|---|
| D1 分布真实性 | `FIDk/FIDg`、`Distk/Distg` | kinetic/geometric feature 分布与 GT 的距离 | 低/接近 GT |
| D2 多样性 | `Divk/Divg`、same-song seed diversity | 不同样本是否有合理变化，而不是复制或塌缩 | 接近 GT |
| D3 连续性 | velocity、acceleration、`jerk P95` | 速度、加速度和高频抖动；jerk 过高通常表示不平滑 | 接近 GT |
| D3 边界连续性 | C4 boundary velocity jump | online commit 边界是否出现跳变 | 接近非边界 |
| D4 活力 | motion energy、amplitude | 动作幅度和活动强度 | 接近同组 GT |
| D4 冻结/重复 | static ratio、freeze length、long-range repetition | 是否陷入平均态、长时间不动或重复 | 接近 GT/较低 |
| D5 物理合理性 | `PFC`、`FSR`、penetration | 足部接触、足滑、穿地等物理质量 | PFC 高；其余低 |
| D5 稳定性 | root height、root displacement、roll/pitch | 根部漂移、身体高度和稳定性 | 接近 GT；不摔倒 |

### M：Music adaptation，音乐-舞蹈适配

M 回答“动作是否适合这一首音乐”。必须用 paired、wrong-song、time-shift 和 null/silence
对照，避免模型只利用动作历史。

| 子模块 | 指标 | 代表什么 | 理想方向 |
|---|---|---|---|
| M1 beat 对齐 | `BAS M->A` | motion beat 到最近 audio beat 的接近程度 | 高 |
| M1 beat 对齐 | `BAS A->M` / reverse BAS | audio beat 是否得到 motion 响应，防止动作太少造成 BAS 虚高 | 高 |
| M1 beat 事件 | beat precision、recall、`Beat F1` | 动作事件命中音乐 beat 的准确率、覆盖率和综合值 | 高 |
| M1 指定拍点 | `BAP`、`KPD` | 是否按指定 beat assignment 做出目标动作/key-pose | BAP 高、KPD 低；无 target 时 N/A |
| M2 时间节奏 | tempo error、phase error | 动作周期速度和重音位于拍子周期中的位置是否正确 | 低 |
| M2 动态响应 | speed correlation、impact correlation、response lag | 音乐 onset/能量变化是否引起同步的动作变化 | corr 高；lag 接近 0 |
| M2 强弱表达 | dynamic/energy response、amplitude response | crescendo、drop、重音等动态是否反映到动作幅度 | 高/接近 GT |
| M3 长期结构 | phrase/section response、long-horizon coherence | intro/verse/chorus/drop 的转场、段落和长期连贯性 | 接近 GT |
| M4 跨模态语义 | audio-motion `R@K`、MM-Dist | 正确音乐和动作能否互相检索，及 embedding 距离 | R@K 高、距离低 |
| M4 风格/情绪 | genre/style consistency、emotion consistency | Jazz/Hip-hop 等风格及 valence/arousal 是否匹配 | 高 |

`tempo error` 只表示快慢，不等于 genre；style 和 emotion 需要独立 evaluator 或盲评，不能
从 BAS 推断。

### X：G1/SONIC execution

| 子模块 | 指标 | 代表什么 |
|---|---|---|
| X1 可执行性 | joint-limit/dynamic-envelope violation | reference 是否超过机器人和 tracker 能力 |
| X2 跟踪误差 | RMSE、EMPJPE、EMPKPE、tracking lag | SONIC 实际动作与 reference 的差异 |
| X3 表现力保留 | energy/amplitude/jerk/band/contact/BAS retention | tracker 吃掉了多少动作幅度、速度、频段、接触和节奏 |
| X4 安全稳定 | success rate、time-to-fall、minimum base height | 是否能完整执行且不摔倒 |

### R：Online system

| 指标 | 代表什么 |
|---|---|
| P50/P95/P99 latency | 音频到 feature、推理、通信和 reference 提交的延迟 |
| deadline miss | 是否错过 SONIC 的执行窗口 |
| stale/measured-state age | generator 使用的状态是否过期 |
| packet drop/fallback | 通信丢包和异常时是否触发 fallback |
| realtime factor / publish rate | 仿真或系统是否以实时速度运行 |

### H：Human evaluation

自动指标不能完整表达“优美”。盲评至少报告 naturalness、dance quality、smoothness、
expressiveness、rhythm、style、emotion 和 long-horizon coherence。

### 当前正式矩阵状态

正式扩展计划为 `M0/M2/M3/M4 × song{012,065,098} × sampling seed{1234,2345,3456}`，
每个格子分别记录 `M_ref` 和配对的 `M_exec`。当前清单共 72 个计划格，其中 11 个
`M_ref` 格子有结果、3 个 `M_exec` 格子有结果；对应实际已有 15 个生成轨迹文件和
3 个 SONIC 执行记录。M2/M4 的现有三个文件是不同 `training_seed` 的 checkpoint，
而不是三个 `sampling_seed`，所以不能直接与 M0 的采样 seed 方差做等价比较。

矩阵清单：`results/benchmark_v1/formal/EXPANSION_MATRIX.json`。只有在同一音乐、同一
初始 K64、同一时长、同一 SONIC 对齐协议下同时拥有 `M_ref` 与 `M_exec`，该样本才进入
正式 generator/tracker 对比；其余只进入现状诊断。

MRT2 M3 正式生成使用 `Musics2Dance-prior-dev/scripts/infer_mrt2_system.py`。当前已完成
正式 `M3/song012/seed1234` 的 60 秒参考轨迹；从仓库根目录
运行时需要显式加入 `PYTHONPATH=.`；推荐先设置 `NUMBA_DISABLE_JIT=1`，避免旧 librosa
在服务器上写不可用的 numba cache。输出 pickle 才是后续 `M_ref` 评估和 SONIC 配对的输入。

G1 原生轨迹使用专门入口，不要调用只支持 SMPL/full_pose 的旧入口：

```bash
MUJOCO_GL=glfw NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
PYTHONPATH=. python -m eval.benchmark.run_g1_benchmark_eval \
  --motion_path <平铺后的G1-pkl目录> \
  --reference_motion_path <G1-GT-reference目录> \
  --output_dir <评估输出目录> \
  --method_name M3_M_ref \
  --feature_type mrt2_music_conditioned \
  --g1_root_quat_order xyzw
```

该入口包含 G1 BAS/RoboPerform BAS、beat precision/recall/F1、beat timing、
身体部位 beat F1、foot contact/sliding/penetration、root drift、速度/加速度/jerk、
关节范围和 G1 分布指标。没有 `designated_beat_frames` 时，旧式 BAP 不纳入结论，
应使用基于音乐 beat detector 的 precision/recall/F1。

### 当前 M3 `M_ref` 结果

M3 reference 已完成 `song012/065 x seed{1234,2345,3456}` 六条 60 秒 G1 轨迹的
reference-side 评估。这里的结果只回答“generator 输出了什么”，不包含 SONIC 执行、
控制器延迟或实时性判断。

| 模块 | 指标 | 当前均值 | 解释 |
|---|---|---:|---|
| Validity | Finite motion rate | 1.000 | 六条轨迹均为有限值，可进入后续评估 |
| Rhythm | G1 BAS / FK BAS | 0.2577 / 0.2708 | 音乐 beat 与动作 beat 的总体接近程度 |
| Rhythm | Beat precision / recall / F1 | 0.3037 / 0.1866 / 0.2275 | 命中准确率、音乐 beat 覆盖率和综合值 |
| Rhythm | Beat timing mean / std | -0.049 / 1.448 frames | 已匹配事件的时间偏差及稳定性 |
| Rhythm | Wrist / Foot / Torso F1 | 0.2292 / 0.2138 / 0.2409 | 不同身体部位的节奏响应 |
| Motion quality | Root drift / path length | 1.535 / 12.151 m | 生成 reference 的根部位移和路径长度 |
| Motion quality | Joint jerk mean | 446.758 | 关节高阶变化，作为连续性诊断 |
| Physical | Foot sliding / penetration | 0.505 / 0.0433 m | 足部滑动和穿地诊断 |
| Physical | Foot contact on beat | 0.6321 | 音乐 beat 附近的足部接触比例 |
| Validity | Joint range violation | 0.0103 | 参考动作超出统计关节范围的比例 |

这组结果的正确解释是：M3 输出格式有效，命中的 beat 事件时间偏差较小，但 beat
recall 偏低且 offbeat false-positive rate 为 0.6963，说明当前动作没有稳定覆盖
音乐中的全部节奏事件。因此不能只根据 BAS 或 timing mean 声称“随着音乐起舞”。
Root drift、foot sliding 和 jerk 必须和同音乐、同风格的 O-G1 GT 分布比较后解释；
`G1Dist/G1Div` 目前只有两条 reference cache，暂不作为模型排名依据。没有
`designated_beat_frames` 时 BAP 为 N/A，不能把 0 当作 beat 能力为零。

原始明细：
[`results/benchmark_v1/formal/m3_reference_metrics/REPORT.md`](results/benchmark_v1/formal/m3_reference_metrics/REPORT.md)。

## 3. 文件和结果位置

| 内容 | 入口 |
|---|---|
| 总结果报告 | [`RESULTS.md`](RESULTS.md) |
| 指标定义与论文映射 | [`../docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md`](../docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md) |
| 指标分类注册表 | [`../docs/evaluation/METRIC_TAXONOMY_MUSIC_DANCE_G1.md`](../docs/evaluation/METRIC_TAXONOMY_MUSIC_DANCE_G1.md) |
| GT 的 D/M 分布与模型模块分数 | `results/benchmark_v1/gt/module_benchmark_v1/REPORT_ZH.md` |
| GT 原始明细 | `results/benchmark_v1/gt/module_benchmark_v1/module_distributions.csv` |
| 模型 D/M/X 明细 | `results/benchmark_v1/formal/MODEL_SUMMARY.md` |
| M2/M3/M4 formal table | `results/benchmark_v1/formal/model_stage_results.csv` |
| 正式扩展矩阵 | `results/benchmark_v1/formal/EXPANSION_MATRIX.json`、`EXPANSION_MATRIX.csv` |
| M3 paired/wrong/shifted/null | `results/m3_music_ablation/formal_30s/aggregate_v2/REPORT.md` |
| SONIC gap 与 retention | `results/motion_music_execution/`、`results/generation_to_execution_gap/` |
| SONIC capability 上限 | `results/gt_sonic_capability/` |
| 视频 | `results/mrt2_comparison/`、`results/figures/` |

M3 六条正式参考轨迹的 SONIC 配对执行使用：
`eval/sonic/run_m3_reference_suite.sh`。SONIC 仿真必须先运行并监听 reference
端口 `5556`，执行脚本会按 `012/065 × 3 seeds` 逐条发送，使用 `3 s` measured-state
alignment、`1 s` hold、50 Hz SONIC reference 和 `full` packet。每条输出保存为
`results/benchmark_v1/formal/m3_execution/<run_id>/`。

## 4. 代码位置

```text
metrics/    g1_metrics.py, g1_kinematics.py, eval_bas_bap.py, eval_pfc.py, eval_diversity.py
benchmark/  GT 校准、模型评估、正式表格和报告生成
music/      音乐配对、beat/tempo/style 分析、M3 和 beat 消融
sonic/      tracking、execution gap、capability 和 retention
render/     G1/MuJoCo 视频
tools/      低频数据准备工具
```

所有新结果写入 `results/<experiment>/`，不要在 `eval/` 顶层新建散落的结果目录或重复指标脚本。
