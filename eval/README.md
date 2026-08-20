# AudioMimic Evaluation Guide

更新日期：2026-08-20

本文档是 AudioMimic 评估代码的操作入口。指标的正式定义、证据等级和论文映射以
[`docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md`](../docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md)
为准；本文说明数据怎样采集、命令怎样执行、输出怎样解释。

## 1. 评价对象

系统必须拆成三个对象评价，不能只报告一个 BAS 或一个成功率：

```text
audio A -> generator -> reference M_ref -> SONIC -> execution M_exec
```

1. `Generator`：`M_ref` 是否自然、连续、多样、物理合理。
2. `Music matching`：`M_ref` 是否与对应音乐的节奏、动态和语义匹配。
3. `Tracker`：SONIC 是否稳定、低延迟地保留 `M_ref` 的姿态和动态细节。
4. `End-to-end`：`M_exec` 最终是否仍然是一段与音乐匹配的舞蹈。

当前 M0/M2/M4 PKL 是固定的 60 s rollout。它们可以评价 motion quality 和
generation-to-execution gap，但不能单独证明实时音乐条件有效。真正的因果音乐消融需要
M2 checkpoint，并在固定 diffusion noise 下重新生成 paired、shifted、shuffled 和
silence 四组。

## 2. 环境与数据契约

```bash
cd ~/AudioMimic
conda activate audiomimic
```

每条 reference motion 必须满足：

```text
root_pos [T, 3]       metre
root_rot [T, 4]       manifest 中明确 xyzw 或 wxyz
dof_pos  [T, 29]      radian，固定 G1 joint order
fps                   正数
audio_start_seconds   音频相对动作起点
```

每个正式 SONIC run 应包含：

```text
offline_sonic_playback.json   运行配置与绝对起始时刻
reference.json                实际发送给 SONIC 的逐帧 reference
sonic_feedback.json           tracker feedback
sim_state.json                MuJoCo physical state
s66_exec.json                 重建的 execution boundary state
```

原始 telemetry 很大，只保存在本机；Git 提交分析脚本、manifest、`metrics.json`、
`summary.csv` 和实验报告。任何失败 run 都不能删除。

## 3. 标准评估流程

### Step 0：冻结实验因素

主实验至少使用 3 首 held-out 音乐、每首 3 个 generation seeds、每条 reference 3 次
独立 SONIC repeats。M0/M2/M4 必须固定 song、起点、长度和 generation seed。Tracker
比较必须使用同一条 `M_ref`，不能重新采样 generator。

主结果统一使用：

```text
duration: 60 s
playback: 1.0x
packet: full（C4 作为通信消融）
initialization: Macarena CONTROL -> release elastic band -> unassisted stand
transition: 3 s measured-state alignment + 1 s hold
SONIC reference rate: 50 Hz
```

### Step 1：标定 SONIC 能力上限

先跑 SONIC 自带 reference，再跑 retargeted GT。这样才能区分 tracker 能力、retarget
domain gap 和 generator 问题。

```bash
# 单次人工检查
bash scripts/run_sonic_known_trackable_capability.sh low 1
bash scripts/run_gt_sonic_capability.sh low 1

# low/medium/high x 3 repeats
python scripts/run_sonic_capability_suite.py \
  --levels low medium high \
  --repeats 1 2 3

python scripts/run_sonic_capability_suite.py \
  --source retargeted_gt \
  --levels low medium high \
  --repeats 1 2 3
```

分析：

```bash
python eval/analyze_gt_sonic_capability.py \
  --manifest eval/gt_sonic_capability/sonic_known_trackable_20260820/manifest.json \
  --runs_root eval/gt_sonic_capability/known_trackable_runs \
  --output_dir eval/gt_sonic_capability/known_trackable_analysis/capability_9x_corrected

python eval/analyze_gt_sonic_capability.py \
  --manifest eval/gt_sonic_capability/selection_20260819/selection_manifest.json \
  --runs_root eval/gt_sonic_capability/retargeted_gt_runs_v2 \
  --output_dir eval/gt_sonic_capability/retargeted_gt_analysis_v2
```

### Step 2：采集 M0/M2/M4 execution

下面以 M2 repeat 1 为例；M0、M4 和 `r02/r03` 只替换 PKL 与 `RUN_ID`。

```bash
MOTION_PKL=~/Musics2Dance-prior-dev/onlinegeneratedmotion/m2_predicted_fms/m2_train1234_sample1234_u100000_best_song098.pkl
RUN_ID=m2_song098_seed1234_full_rate100_aligned_r01

python stream_to_sonic.py \
  --pkl "$MOTION_PKL" \
  --root_quat_order xyzw \
  --packet_mode full \
  --playback_rate 1.0 \
  --max_seconds 60 \
  --align_from_feedback_seconds 3 \
  --align_hold_seconds 1 \
  --output_dir "eval/generation_to_execution_gap/$RUN_ID" \
  --record_feedback \
  --feedback_port 5557 \
  --sim_state_port 5559 \
  --reference_safety none \
  --sonic_reference_fps 50 \
  --startup_wait 2
```

每次 repeat 前重启 MuJoCo/SONIC 并恢复相同初态。不要覆盖旧 run；确需覆盖时显式使用
`--overwrite_output`。

### Step 3：计算 reference-to-execution gap

```bash
python eval/analyze_generation_execution_gap.py \
  --runs \
    eval/generation_to_execution_gap/m2_song098_seed1234_full_rate100_aligned_r01 \
    eval/generation_to_execution_gap/m2_song098_seed1234_full_rate100_aligned_r02 \
    eval/generation_to_execution_gap/m2_song098_seed1234_full_rate100_aligned_r03 \
  --output_dir eval/generation_to_execution_gap/phase_m2_final
```

输出 `metrics.json` 保存逐 run、逐 body group 结果，`summary.csv` 保存主表字段。默认在
first fall 处截断误差分析，但 Success/TTF 仍使用完整 run。只有诊断时才使用
`--include_post_fall_tracking`。

### Step 4：计算动作质量、音乐响应与 execution retention

```bash
python eval/analyze_motion_music_execution.py \
  --motion_root ~/Musics2Dance-prior-dev/onlinegeneratedmotion \
  --tracking_root eval/generation_to_execution_gap \
  --tracking_glob 'm[024]_song098_seed1234_full_rate100_aligned_r0*' \
  --output_dir eval/motion_music_execution/current
```

该步骤对 `M_ref` 和 `M_exec` 使用同一实现计算 velocity、acceleration、jerk、energy、
static ratio、repetition、C4 continuity、root、PFC/FSR proxy、BAS、onset correlation
和 response lag，并计算可定义的 retention/degradation。

### Step 5：与 SONIC-native 和 retargeted-GT 基线比较

```bash
python eval/compare_generator_to_sonic_capability.py \
  --route_summary eval/motion_music_execution/current/route_summary.json \
  --execution_metrics eval/motion_music_execution/current/execution_metrics.json \
  --native_capability eval/gt_sonic_capability/known_trackable_analysis/capability_9x_corrected/capability_analysis.json \
  --gt_capability eval/gt_sonic_capability/retargeted_gt_analysis_v2/capability_analysis.json \
  --gt_manifest eval/gt_sonic_capability/selection_20260819/selection_manifest.json \
  --output_dir eval/generator_vs_tracker_baselines/current
```

这里的 nearest dynamic tier 是解释性比较，不是新的质量总分。必须同时保留绝对误差、
稳定率和 retention。

### Step 6：检查固定轨迹的音乐配对敏感性

```bash
python eval/analyze_music_pairing_sensitivity.py \
  --motion_root ~/Musics2Dance-prior-dev/onlinegeneratedmotion \
  --tracking_root eval/generation_to_execution_gap \
  --tracking_glob 'm[024]_song098_seed1234_full_rate100_aligned_r0*' \
  --shift_step_seconds 2 \
  --output_dir eval/music_pairing_sensitivity/current
```

该步骤比较 correct clock、循环 time shift 和 wrong song，只能说明现有轨迹是否含有可检测
的 pairing evidence。它没有改变生成时 condition，因此不是 causal ablation。

### Step 7：生成盲评素材和问卷 manifest

统一渲染需要 SONIC simulation 环境中的 MuJoCo/EGL：

```bash
MUJOCO_GL=egl ~/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/render_song098_blind_pilot.py

conda activate audiomimic
python eval/build_human_pairwise_study.py \
  --assets eval/human_study/assets_song098_seed1234_pilot.json \
  --output_dir eval/human_study/design_song098_seed1234_pilot \
  --participants 24 \
  --trials_per_task 6 \
  --seed 20260820
```

`public_study.json` 和匿名媒体给参与者；`private_key.json` 不提交、不公开，直到评价结束。
正式实验必须满足 3 songs x 3 generation seeds，并为每条 execution 提供 3 repeats。

### Step 8：运行回归测试

```bash
conda run -n audiomimic python -m unittest \
  tests.test_offline_sonic_playback \
  tests.test_sonic_bridge \
  tests.test_evaluation_map \
  tests.test_generation_execution_gap \
  tests.test_generator_capability_comparison \
  tests.test_gt_capability_selection \
  tests.test_human_pairwise_study \
  tests.test_motion_music_execution \
  tests.test_music_pairing_sensitivity \
  tests.test_render_blind_g1_video
```

## 4. 指标速查与来源

### P0：数据和协议 gate

| ID | 含义 | 如何测 | 来源 |
|---|---|---|---|
| P0-SCHEMA | shape、FPS、NaN/Inf 有效 | load 时强校验 | AudioMimic |
| P0-ORDER | joint/quaternion/单位一致 | manifest + round-trip test | AudioMimic |
| P0-TIME | audio/reference/feedback 同一时钟 | 保存绝对 timestamp 和 audio offset | DiscoForcing + AudioMimic |
| P0-COVER | feedback 覆盖率 | 有效反馈帧/期望帧，另报缺失和插值 | AudioMimic |
| P0-PAIR | song、seed、时长配对有效 | manifest 字段完全匹配 | AudioMimic |
| P0-LENGTH | 长时结果至少 60 s | 按 trajectory 检查 | Lodge、DiscoForcing + AudioMimic |

### G：Generator 动作质量

| ID | 含义与计算 | 方向 | 来源/限制 |
|---|---|---|---|
| G-FID-K/G | kinetic/geometric feature 的生成-GT Fréchet distance | 低 | FACT、Lodge、DiscoForcing；固定 extractor 后才可报告 |
| G-PFC | 加速度与足部静态接触是否物理一致 | 低 | EDGE、Beat-It；不是完整动力学稳定性 |
| G-FSR | 足部接近地面时仍水平滑动的帧率 | 低 | Lodge、DiscoForcing；有意滑步需结合 contact 解释 |
| G-PEN | 足部/关键 body 穿地比例与深度 | 低 | AudioMimic |
| G-DIV-K/G | kinetic/geometric feature 的样本间距离 | 接近 GT | FACT、Lodge、Beat-It、DiscoForcing；不是越高越好 |
| G-SEED-DIV | 同一音乐不同 generation seed 的距离 | 接近 GT | AudioMimic |
| G-VEL/ACC/JERK | `dq/ddq/dddq` 的 P50/P95/P99/max | 接近 GT；jerk 不宜过高 | AudioMimic |
| G-ENERGY | `mean(||dq||^2)` | 接近 GT | AudioMimic |
| G-STATIC | 速度低于预注册阈值的帧率 | 接近 GT | AudioMimic |
| G-REPEAT | 相隔超过 2 s 的近重复姿态比例 | 低/接近 GT | AudioMimic |
| G-C4-POS/VEL | C4 边界 jump / 非边界 jump | 接近 1 | AudioMimic |
| G-ROOT | path、net displacement、yaw drift、height | 接近 GT | AudioMimic |
| G-JLIMIT | joint-limit 越界率、margin、最大越界 | 0 violation | AudioMimic hard gate |

### M：音乐与动作匹配

| ID | 含义与计算 | 方向 | 来源/限制 |
|---|---|---|---|
| M-BAS | kinematic beat 到最近 music beat 的指数距离 | 高 | FACT、EDGE、Lodge、Beat-It、DiscoForcing、RoboPerform；单向辅助指标 |
| M-BEAT-F1 | 固定容差下 motion beat 的 precision/recall/F1 | 高 | AudioMimic；仅有显式 beat target 时适用 |
| M-BAP | 动作是否服从指定 beat assignment | 高 | Beat-It；无 assignment 时为 N/A |
| M-KPD | 指定 key pose 的 Cartesian MSE | 低 | Beat-It；无 key pose 控制时为 N/A |
| M-ONSET | onset strength 与 motion impact/energy 的相关 | 高 | AudioMimic core metric |
| M-RESP-LAG | onset-response 最大相关的时差 | 接近 0 | AudioMimic；相关 `r < 0.10` 时不解释 lag |
| M-TEMPO | motion periodicity 与 tempo/rhythm grid 偏差 | 低 | AudioMimic，待实现 |
| M-PHASE | 动作 accent 对音乐相位的 circular error | 低 | AudioMimic，待实现 |
| M-R1/R2/R3 | 正确 audio-motion pair 的 retrieval top-K | 高 | RoboPerform；需要独立冻结 encoder |
| M-MMDIST | 配对 audio-motion joint embedding 距离 | 低 | RoboPerform；需要独立冻结 encoder |
| M-PHRASE | phrase boundary 附近动作结构变化命中与时差 | 高/零 lag | AudioMimic，待实现 |
| M-STYLE | genre/style/emotion 匹配 | 高 | AudioMimic；固定分类器或盲评 |

### X：发送 SONIC 前的可执行性

| ID | 含义 | 如何测 | 来源 |
|---|---|---|---|
| X-JLIMIT | reference 是否越过 G1 joint limit | 每关节 violation rate/max | AudioMimic |
| X-DYN | velocity/acceleration/jerk 是否超出 capability envelope | 对 SONIC-native 和 retargeted-GT 标定区间计超限率 | AudioMimic |
| X-CONTACT | 支撑、COM、接触时序是否自洽 | FK/contact proxy，后续接动力学 | EDGE/Lodge + AudioMimic |
| X-ROOT | height、roll/pitch、yaw rate、planar speed 是否可执行 | 与 GT capability envelope 比较 | AudioMimic |
| X-INIT | measured pose 到首帧的 transition jump | alignment 末端位置/速度差 | AudioMimic |

### T：SONIC tracking 与表现力保留

| ID | 含义与计算 | 方向 | 来源 |
|---|---|---|---|
| T-SUCC | 完整时长无 fall/deviation 的 run 比例 | 高 | RoboPerform |
| T-TTF/HMIN | 首次跌倒时间/最低 base height | 高 | AudioMimic |
| T-EMPJPE | reference-execution DoF rotation 平均误差，rad | 低 | RoboPerform |
| T-EMPKPE | FK key body 平均位置误差，m | 低 | RoboPerform |
| T-RMSE-RAW | 不补偿延迟的 joint RMSE | 低 | AudioMimic |
| T-RMSE-ALIGN | 仅补偿预定义 tracking lag 后的 joint RMSE | 低 | AudioMimic |
| T-LAG | execution 相对 reference 的 cross-correlation lag | 接近 0 | AudioMimic |
| T-AMP | `(P95-P5)_exec/(P95-P5)_ref` | 接近 1 | AudioMimic |
| T-ENERGY | `mean(||dq_exec||^2)/mean(||dq_ref||^2)` | 接近 1 | AudioMimic |
| T-BAND-L/M/H | 0-1/1-3/3-8 Hz Welch PSD retention | 接近 1 | AudioMimic |
| T-JERK | execution/reference jerk P95 | 接近 1 | AudioMimic |
| T-CONTACT | contact F1、transition timing、foot-slip change | 高/零误差 | AudioMimic |

`T-RMSE-RAW` 与 `T-RMSE-ALIGN` 必须同时报告。Global root XY 不是 SONIC 的直接 tracking
目标，不进入 tracker 主 RMSE；仍需报告 root orientation、height 和 key-body error。

### E：执行后质量

在相同 audio clock 下对 `M_exec` 重算所有适用 G/M 指标：

```text
degradation = metric(M_exec) - metric(M_ref)
retention   = metric(M_exec) / metric(M_ref)
```

只有具有合理零点且分母远离零的量才计算比例。FID、RMSE、lag、error rate 和可能为负的
correlation 主要报告差值，不能通过移动音频来事后提高 execution 分数。

### R：实时系统

| ID | 含义 | 报告方式 | 来源 |
|---|---|---|---|
| R-A2F | audio 到 causal feature 延迟 | mean/P50/P95/P99/max | DiscoForcing + AudioMimic |
| R-F2M | feature 到 motion 延迟 | mean/P50/P95/P99/max | DiscoForcing |
| R-M2REF | motion 到 SONIC reference 延迟 | queue/serialize/network 分项 | AudioMimic |
| R-REF2EXEC | reference timestamp 到 measured response | mean/P95/P99 | RoboPerform + AudioMimic |
| R-DEADLINE | 超过 C4 deadline 的比例 | rate + 最大连续 misses | DiscoForcing + AudioMimic |
| R-DROP | packet/drop/stale/fallback | 各自比例 | AudioMimic |
| R-RTF | motion duration / wall-clock duration | 至少 1.0 | DiscoForcing |

离线 PKL 回放不报告为实时音乐生成。在线实验还必须记录 audio lookahead、buffer、H/C、
NFE、GPU、warm-up 和是否读取未来音乐。

### H：人类评价

| ID | 问题 | 比较对象 |
|---|---|---|
| H-NATURAL | 动作是否自然、无抖动和错误 | generator baseline；reference vs execution |
| H-DANCE | 是否像有组织的舞蹈 | M0/M2/M4/GT |
| H-EXPRESS | 是否保留力度、层次和表现力 | reference vs execution |
| H-COHERENCE | 60 s 是否连贯且不过度重复 | generator routes |
| H-RHYTHM | 动作重音是否匹配音乐 | M0/M2/M4；reference vs execution |
| H-STYLE | 风格/情绪是否适合音乐 | M0/M2/M4 |

采用随机左右位置的 blinded pairwise study，报告 preference、Bradley-Terry/Elo、参与者
数量、有效比较数量和 participant-level bootstrap 95% CI。FACT、EDGE、Lodge 和 Beat-It
均使用人类评价；本项目的具体问题与排除规则见
[`HUMAN_EVALUATION_PROTOCOL.md`](../docs/evaluation/HUMAN_EVALUATION_PROTOCOL.md)。

## 5. 当前可报告与不可报告的结论

当前已实现：joint/lag-compensated RMSE、lag、EMPKPE、amplitude/energy/band retention、
velocity/acceleration/jerk、static、repetition、C4 jump、root、BAS、onset correlation/lag、
survival 和基础 FK/contact proxy。

当前仍需实现或冻结：正式 G1 FID/Div extractor、tempo/phase/phrase、joint-limit 与完整
executability envelope、独立 audio-motion retrieval encoder、R@K/MMDist、正式多歌曲盲评。

因此目前可以声称“固定 M0/M2/M4 reference 在正确初始化下可由 SONIC 稳定执行，但中高频
动态被压缩”；不能声称“实时音乐条件已经打通”或“M2 已经显著优于 M0”。

## 6. 论文来源

- FACT/AIST++：FIDk/FIDg、Dist/Div、BeatAlign 和用户比较。
- EDGE：PFC、BeatAlign、Dist、pairwise/Elo，并指出小测试集 FID 的局限。
- Lodge：长序列 FID/Div、FSR、BAS、效率和用户评价。
- Beat-It：PFC、Div、BAS、KPD、BAP 和用户评价。
- RoboPerform：R@K、MMDist、BAS、Success、EMPJPE、EMPKPE 和部署延迟。
- DiscoForcing：严格因果与有界延迟、FID/FSR/Div/BAS、ms/frame 和 FPS。
- AudioMimic：C4 continuity、动态 envelope、lag、retention、deadline/drop 和 end-to-end
  execution degradation。这些是项目系统指标，论文中必须明确标为 system-specific。
