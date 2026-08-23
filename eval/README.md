# AudioMimic Evaluation Guide

更新日期：2026-08-23

## Canonical workspace

当前正式评估只维护以下结果目录：

- `benchmark_v1/`: 正式 benchmark 总目录；其中 `gt/` 是 GT/GMR calibration，`formal/` 是 O-Human/O-G1/M-ref/M-exec 四层结果。
- `generator_vs_tracker_baselines/`: generator/reference 与 SONIC execution 的对照。
- `motion_music_execution/`: 当前 M0/M2/M3/M4 的音乐-动作与执行结果。
- `m3_music_ablation/`: M3 音乐条件因果消融。
- `mrt2_comparison/`: 当前保留的 M3 生成与 corrected SONIC 视频。
- `gt_sonic_capability/`: SONIC 能力上限和接口分析摘要。

历史 rollout、重复结果和旧视频已移至 `/tmp/audiomimic_eval_archive_20260823/`，
不参与当前 benchmark；需要复现旧实验时再从该目录恢复。

本文档是 AudioMimic 评估代码的操作入口。指标的正式定义、证据等级和论文映射以
[`docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md`](../docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md)
为准；论文 claim、GT 校准、sealed test、统计和实验矩阵以
[`ICRA_GT_CALIBRATED_EVALUATION_PLAN.md`](../docs/evaluation/ICRA_GT_CALIBRATED_EVALUATION_PLAN.md)
为准；本文说明数据怎样采集、命令怎样执行、输出怎样解释。

## 1. 评价对象

系统必须拆成三个对象评价，不能只报告一个 BAS 或一个成功率：

指标的统一层级分类见
[`METRIC_TAXONOMY_MUSIC_DANCE_G1.md`](../docs/evaluation/METRIC_TAXONOMY_MUSIC_DANCE_G1.md)，
机器可读注册表见 [`metric_taxonomy_v1.json`](metric_taxonomy_v1.json)。后续表格必须按
舞蹈动作质量、音乐-舞蹈适配、G1/SONIC、在线系统和人类感知分模块报告。

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

所有指标的论文来源、定义、证据等级和音乐条件输入策略见
[`docs/evaluation/LITERATURE_METRIC_AUDIT_20260823.md`](../docs/evaluation/LITERATURE_METRIC_AUDIT_20260823.md)。
当前固定三组输入对照：`8D beat-only` 下限、`MERT + Librosa + 8D beat` 主线，以及历史
`Wav2CLIP + beat` anchor。`BAS` 是 beat-alignment suite 的核心指标，每次都报告；
但不能单独作为完整音乐性或论文验收依据。它必须和 event F1/coverage、onset
correlation、response lag、tempo error 和 phase error 一起解释。

### Step 0：建立 GT benchmark manifest

先审计 FineDance/AIST++ 的 motion/audio 配对、G1 schema 和 split/cache 一致性：

```bash
cd ~/AudioMimic
python eval/build_gt_benchmark_manifest.py \
  --finedance-source-root data/finedance \
  --finedance-g1-root data/finedance-g1-retargeted \
  --finedance-root data/finedance_g1_fkbeats \
  --output-dir eval/benchmark_v1/gt/manifest_v2_finedance
```

本次 FineDance 审计产物：

- [`gt_benchmark_manifest.json`](benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json)
- [`AUDIT_REPORT.md`](benchmark_v1/gt/manifest_v2_finedance/AUDIT_REPORT.md)

当前 FineDance 原始 motion/G1 motion/audio 为 `203/203/207`；按同一数字 ID
配对后有 203 条有效配对，官方 cross-genre test 的 18 条全部可用。部分 WAV
比动作长，这是 FineDance 原始准备流程的正常情况；正式评估按 motion、G1 和
audio 的共同有效时长截断，不把它误判为错配。`audio-only` 的额外 ID 必须保留
在审计里，但不能进入 test。

旧的 `manifest_v1` 保留作历史记录；当前 FineDance 结果以 `manifest_v2_finedance`
为准。

### Step 0a：验证音乐-动作确实配对

文件存在和同 ID 还不能证明音乐与动作内容对应。对官方 18 条 test 运行同 ID
配对、逐条错配音乐和全量错配音乐对照：

```bash
cd ~/AudioMimic
MPLCONFIGDIR=/tmp/audiomimic-mpl-cache conda run -n audiomimic \
  python eval/audit_finedance_music_pairing.py \
  --g1-root data/finedance-g1-retargeted \
  --audio-root data/finedance/music_wav \
  --output-dir eval/benchmark_v1/gt/finedance_music_pairing_v1
```

产物：

- [`pairing_metrics.json`](benchmark_v1/gt/finedance_music_pairing_v1/pairing_metrics.json)
- [`pairing_summary.csv`](benchmark_v1/gt/finedance_music_pairing_v1/pairing_summary.csv)
- [`REPORT.md`](benchmark_v1/gt/finedance_music_pairing_v1/REPORT.md)

指标含义：

- `zero_lag_corr`：同一时刻 audio onset envelope 与 G1 关节运动能量的相关性；
- `best_lag_corr`：允许 `+-2 s` 时间偏移后最大相关性，用来诊断标注/起点延迟；
- `event_f1`：音频 onset 峰与运动能量峰在 `+-200 ms` 内的事件 F1；
- `dominant_rate_error_hz`：音频 onset 与运动能量在 `0.5--4 Hz` 主频差；
- `paired-vs-wrong margin`：同 ID 结果减去错配音乐结果，是音乐对应性的核心诊断；
- `top1_rate`：同一动作在所有候选 test 音乐中能否检索回自己的音乐。

这一阶段只验证 benchmark 的音乐-动作配对，不把相关性直接称作生成模型
music-to-dance score。最终论文还需要在生成 motion 上使用 beat phase、tempo、
lag-aware onset、音乐检索和人工盲评，并同时报告 reference 与 SONIC execution。

### Step 0b：生成 FineDance-G1 可训练 cache

当前本地已经生成 baseline 版本的 G1 cache：

```text
data/finedance_g1_fkbeats/
  train: 47,817 clips
  test:   3,265 clips
  motions_sliced: [150, 29] G1 DOF positions
  baseline_feats: [150, 35]
  beat_feats: motion/audio beat mask, distance and spacing
```

WAV 和 baseline feature 通过 symlink 复用 source tree；motion 是 G1 的
`root_pos/root_rot/dof_pos`，四个目录的 basename 已逐一校验一致。当前 beat
cache 的 audio beat 使用 STFT spectral-flux fast extractor，motion beat 使用
G1 MuJoCo FK；因此它适合大规模训练和诊断，但与旧 `librosa.beat_track`
缓存混用时必须在实验 manifest 中注明 extractor。

校验命令：

```bash
NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
  conda run -n audiomimic python data/validate_preprocessed_data.py \
  --data_path data/finedance_g1_fkbeats \
  --feature_type baseline --motion_format g1 \
  --use_beats --beat_rep distance --sample_count 1000
```

当前校验通过；观察到部分 FineDance retargeted clip 的 root height 为负值，
这是数据/retargeting 质量问题，不能通过提高验证阈值掩盖。训练前应增加
`root_height/contact/base-stability` 的过滤或修正，并把过滤规则固定在 manifest。
当前分布记录在
[`finedance_g1_quality_v1/REPORT.md`](benchmark_v1/gt/finedance_g1_quality_v1/REPORT.md)。

先生成逐 clip 质量 manifest。它只记录训练筛选候选，不删除文件；sealed test
始终完整保留：

```bash
python eval/build_finedance_g1_quality_manifest.py \
  --prepared-root data/finedance_g1_fkbeats \
  --output-dir eval/benchmark_v1/gt/finedance_g1_quality_v1
```

当前结果为：train `47,817` 个 clip，其中 `213` 个含负 root height，`330` 个
低于 `0.2 m`；test `3,265` 个 clip，其中 `20/66` 个分别触发这两个诊断条件。
这两个阈值不是最终删除规则，正式训练应至少比较 unfiltered 与固定过滤策略两组。

用真实 `AISTPPDataset` adapter 做一次小 batch smoke test，检查 motion/music/beat
三路 basename 和 tensor shape：

```bash
NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
  conda run --no-capture-output -n audiomimic \
  python eval/smoke_test_finedance_g1_dataset.py --batch-size 4
```

通过结果应为 train `47,817`、test `3,265`，motion `[4,150,38]`，music
`[4,150,35]`，beat `[4,150]`。这一步只验证数据接口，不代表模型已经训练完成。
正式训练前还要明确 beat estimator checkpoint；若暂时没有，可先用
`--use_beats --lambda_beat 0` 训练 beat-conditioned baseline，避免把 estimator
缺失误认为 FineDance 数据问题。

注意：训练和测试的 beat condition 必须都来自 audio。motion beat 只作为训练时的
`beat_target`，不能作为输入 condition，否则会把动作侧信息泄漏到 generator。
当前 `AISTPPDataset` 已按此规则修正；重新训练前必须重新记录实验配置。

原始审计命令（AIST++ 或没有 FineDance 原始资产的旧环境）：

```bash
python eval/build_gt_benchmark_manifest.py \
  --output-dir eval/benchmark_v1/gt/manifest_v1
```

本次审计产物：

- [`gt_benchmark_manifest.json`](benchmark_v1/gt/manifest_v1/gt_benchmark_manifest.json)
- [`AUDIT_REPORT.md`](benchmark_v1/gt/manifest_v1/AUDIT_REPORT.md)

当前 AIST++ 为 1408/1408 成对且 schema 有效，声明的 20 个 test sequence 与现有
processed test cache 完全一致。FineDance 的最新资产和配对审计见上面的
`manifest_v2_finedance`。

### Step 0c：校准动作质量指标

在正式比较模型前，先用 AIST++ test GT 生成可控退化样本。该实验只验证指标方向，
不产生模型排名，也不把退化样本当作论文结果：

```bash
cd ~/AudioMimic
python eval/build_gt_motion_corruptions.py \
  --manifest eval/benchmark_v1/gt/manifest_v1/gt_benchmark_manifest.json \
  --output-dir eval/benchmark_v1/gt/motion_corruptions_v1

NUMBA_DISABLE_JIT=1 conda run -n audiomimic \
  python eval/analyze_gt_motion_corruptions.py \
  --manifest eval/benchmark_v1/gt/motion_corruptions_v1/corruption_manifest.json \
  --output-dir eval/benchmark_v1/gt/motion_corruptions_v1
```

当前校准结果：20 条 test sequence 共生成 260 个样本；jitter 使 jerk P95 上升
`+1156.64 rad/s^3`，low-pass 使 jerk P95 下降 `-2266.46 rad/s^3`，freeze 使
静止比例上升 `+0.0986`，三个方向检查均通过。完整结果见
[`REPORT.md`](benchmark_v1/gt/motion_corruptions_v1/REPORT.md)。这说明 jerk、静止率和
运动能量可以作为动作质量维度；G1BAS/BeatF1 属于音乐-动作 beat-alignment 评价，
但不能单独代表舞蹈优美程度。beat 指标评价的是节奏、重音和顿感，不替代动作自然性、
物理合理性和人类感知评价。

Benchmark validity audit 已进一步检查指标是否真的对已知退化敏感：
[`benchmark_validity_v1`](benchmark_v1/gt/benchmark_validity_v1/)。jitter-jerk、
lowpass-jerk、lowpass-energy、freeze-static、freeze-beat-F1 和 freeze-BAS 通过了
方向检查；当前 `repeat_similarity` 没有稳定检测 repeat corruption，因此暂不列为
核心指标，必须先改进长时 self-similarity 定义和 repeat corruption 构造。这个 audit
验证的是“指标能否识别指定缺陷”，不是给 GT 一个绝对美学分数。

### Step 0d：跨数据集、节奏和风格的 GT 分层校准

为了避免指标只在总体平均值上有效，固定的 38 条 GT paired test（AIST++ 20 条、
FineDance 18 条）还要按数据集、BPM 和可用风格/genre 分层报告。慢速定义为
`BPM < 90`，中速为 `90 <= BPM < 120`，快速为 `BPM >= 120`。每一层使用同一套
动作质量和 beat-alignment 指标；BAS 是 beat-alignment suite 的核心成员，必须与
Beat F1、onset/impact correlation、response lag、tempo error 和 phase error 一起
解释，不能把某一层的 BAS 中位数当成“舞蹈优美”的唯一标准。

```bash
cd ~/AudioMimic
python -m eval.build_gt_stratified_corruptions

NUMBA_DISABLE_JIT=1 MUJOCO_GL=glfw \
  MPLCONFIGDIR=/tmp/audiomimic-mpl-cache XDG_CACHE_HOME=/tmp/audiomimic-cache \
  /home/tianhup/anaconda3/envs/audiomimic/bin/python \
  eval/analyze_gt_motion_corruptions.py \
  --manifest eval/benchmark_v1/gt/stratified_corruptions_v1/corruption_manifest.json \
  --output-dir eval/benchmark_v1/gt/stratified_corruptions_v1

python eval/audit_gt_benchmark_stratified.py
```

结果保存在 [`stratified_audit_v1`](benchmark_v1/gt/stratified_audit_v1/) 和
[`stratified_corruptions_v1`](benchmark_v1/gt/stratified_corruptions_v1/)。本轮 8 个跨
数据集方向检查中 7 个通过：jitter/low-pass/freeze 对 jerk、静止率以及 AIST++ 的
freeze Beat F1 均符合预期；FineDance 的 freeze Beat F1 为 `WARN`，说明该单一
event detector 在 FineDance 上不能单独作为退化判据，Beat F1 仍保留为核心节奏指标，
但必须和 BAS、coverage、lag、tempo/phase 共同报告。风格层中 `n < 3` 的结果只作
描述性统计，不能用于显著性结论；后续要扩展到全量 AIST++/FineDance 后再做风格比较。

FineDance-G1 root z / existing asset ground 诊断：
`eval/benchmark_v1/gt/finedance_root_height_audit_v1/REPORT_ZH.md`

全量 GT 的分层 reference distribution 已完成，使用 AIST++ 1,408 条和 FineDance
203 条 paired sequence。它不参与指标设计或 sealed test 选择，只用于给模型结果
提供数据集/tempo/style 条件化的参考范围：

```bash
python eval/audit_gt_benchmark_stratified.py \
  --gt-suite eval/benchmark_v1/gt/gt_oracle_suite_all_v1/gt_oracle_suite_metrics.json \
  --output-dir eval/benchmark_v1/gt/stratified_audit_all_v1 \
  --skip-corruption-checks
```

结果见 [`stratified_audit_all_v1`](benchmark_v1/gt/stratified_audit_all_v1/)。全量数据的
数据集层中位数为：AIST++ BAS `0.264`、Beat F1 `0.706`；FineDance BAS `0.224`、
Beat F1 `0.694`。慢/中/快 tempo 层的 Beat F1 中位数分别为 `0.606/0.682/0.750`，
因此不能把所有歌曲压成一个“理想分数”。FineDance 的小样本风格仍只作描述性统计，
正式风格显著性比较需要每层至少 3 条、最好更多歌曲。

按 AIST++ genre、FineDance style 和 tempo 输出全部 D/M 分组表：

```bash
python eval/build_grouped_gt_benchmark.py
```

FineDance-G1 root z 与 GMR ground 的诊断单独运行，不能用来删掉 sealed test：

```bash
python eval/audit_finedance_root_height.py
```

结果见 [`grouped_gt_all_v1`](benchmark_v1/gt/grouped_gt_all_v1/) 和
[`finedance_root_height_audit_v1`](benchmark_v1/gt/finedance_root_height_audit_v1/)。

按 D（舞蹈质量）和 M（音乐适配）模块，对 GT 以及当前 M2/M3/M4 的 `M_ref/M_exec`
做分层接近度评分：

```bash
python eval/build_module_benchmark.py
```

结果见 [`module_benchmark_v1`](benchmark_v1/gt/module_benchmark_v1/)。评分是相对于同
dataset/style/tempo GT 分布的接近度，不是绝对“优美度”分数。

注意：D/M 分模块表不是完整论文验收表。taxonomy v1 的全部模块必须同时检查：

```text
P 数据与协议：schema、joint order、clock/audio alignment、feedback coverage
D 舞蹈质量：FID/Div、dynamics、energy/static/freeze/repeat、PFC/FSR/penetration/root
M 音乐适配：BAS/Beat P-R-F1/BAP/KPD、onset/lag/tempo/phase、phrase、retrieval/MMDist/style/emotion
X G1/SONIC：reference envelope、joint/keybody error、amplitude/energy/band/contact retention、success/TTF
R 在线系统：deadline、分段 latency、realtime factor、stale/fallback/packet coverage
H 人类感知：naturalness、aesthetics、smoothness、expressiveness、rhythm、style/emotion
```

所有指标都要按上述模块分别报告，不能用 BAS 代表 M、用 jerk 代表 D，也不能用
SONIC success rate 代表 X 的全部 tracking fidelity。当前完整覆盖状态和缺失项见
[`metric_inventory.csv`](benchmark_v1/gt/module_benchmark_v1/metric_inventory.csv)。

在扩展模型前，先检查 D/M 指标对已知动作退化的方向是否合理：

```bash
python eval/build_module_calibration_audit.py
```

结果见 [`CALIBRATION_REPORT_ZH.md`](benchmark_v1/gt/module_benchmark_v1/CALIBRATION_REPORT_ZH.md)。
当前 D 模块方向检查通过；M 模块保留 BAS、Beat F1、event coverage、impact/速度相关性、
lag、tempo 和 phase 的联合 rhythm suite。FineDance 的 freeze Beat F1 为 WARN，
因此 Beat F1 不能单独作为跨数据集 gate，但不从 benchmark 中删除。
校准报告中的“全量冻结模块与指标覆盖”列出了 P/D/M/X/R/H 的每一项；未校准项目必须
在正式模型比较前补齐对应数据或明确标记为 pending。

### Step 0e：建立 FineDance-G1 GT oracle

在训练任何 generator 前，先对 sealed FineDance cross-genre test 的 18 条完整配对
序列计算动作质量和音乐对应性基准：

```bash
NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
  conda run --no-capture-output -n audiomimic \
  python eval/evaluate_finedance_gt_oracle.py \
  --manifest eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json \
  --output-dir eval/benchmark_v1/gt/finedance_gt_oracle_v1
```

产物为 `gt_oracle_metrics.json`、`gt_oracle_summary.csv` 和 `REPORT.md`。当前 18 条
GT 的均值为：motion energy `4.4891`，jerk P95 `1177.04 rad/s^3`，event F1
`0.6552`，event timing error `0.0965 s`，tempo error `27.64 BPM`，phase error
`0.2544 cycles`。这些是数据分布的参考范围，不是 generator 的目标分数，也不是
“优美程度”的绝对真值。SONIC execution retention 仍需对同一批 reference 单独采集。

### Step 0f：修正后 audio-beat baseline smoke

beat condition 泄漏修正后，先运行 1 epoch 的 baseline smoke，确认训练和评测链路
可以使用推理时可获得的 audio beat condition。该实验不是正式模型，也不用于论文
结论：

```text
checkpoint:
runs/train/finedance_g1_baseline_audio_beat_conditioned_smoke2/weights/train-1.pt
```

训练配置为 `feature_type=baseline`、`motion_format=g1`、`use_beats=True`、
`beat_rep=distance`、`lambda_beat=0`，并使用 `--no_render` 绕过无显示环境的
MuJoCo 视频渲染。训练完成于 93 个 batch，峰值 CUDA 显存约 25.1 GB。

用 sealed FineDance test 的 18 个片段做无渲染 smoke evaluation：

```bash
NUMBA_DISABLE_JIT=1 WANDB_MODE=offline MUJOCO_GL=glfw \
MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
conda run --no-capture-output -n audiomimic \
python eval/run_g1_dataset_eval.py \
  --checkpoint runs/train/finedance_g1_baseline_audio_beat_conditioned_smoke2/weights/train-1.pt \
  --feature_type baseline \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_baseline_beat_conditioned_backups \
  --motion_save_dir eval/finedance_baseline_audio_beat_smoke/motions \
  --metrics_path eval/finedance_baseline_audio_beat_smoke/metrics.json \
  --g1_table_path eval/finedance_baseline_audio_beat_smoke/g1_table.json \
  --motion_audit_path eval/finedance_baseline_audio_beat_smoke/motion_audit.json \
  --paper_report_path eval/finedance_baseline_audio_beat_smoke/paper_report.md \
  --use_beats --beat_rep distance --max_eval_clips 18 \
  --diagnostic_count 8 --enable_fk_metrics --g1_render_backend stick
```

本次 smoke 结果：`G1BAS=0.2843`、`G1FKBAS=0.2573`、`G1BeatF1=0.2611`、
`G1RoboPerformBAS=0.6121`、`G1Div=22.1094`、`G1Dist=42.7684`，
`RootDriftMean=0.0097`。结果文件为
[`metrics.json`](finedance_baseline_audio_beat_smoke/metrics.json)。其中
`RootHeightViolationRate=1.0` 反映当前 1 epoch smoke 输出的 root-height 统计仍未
满足正式质量要求；该 checkpoint 只能用于链路诊断。下一步是固定 quality manifest
后进行正式多 epoch 训练，再报告完整测试集和 reference/execution 两侧结果。

### Step 0f：多数据集 GT oracle calibration suite

为了判断 M2/M3 的动作质量和音乐对应性，不能只依赖单首歌曲或单个数据集的
GT。当前本地已审计并可统一对齐到 G1 的配对数据为：AIST++ crossmodal test
20 条，加上 FineDance cross-genre test 18 条，共 38 条完整音乐-动作序列。

运行统一 oracle suite：

```bash
NUMBA_DISABLE_JIT=1 MUJOCO_GL=glfw \
MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
conda run --no-capture-output -n audiomimic \
python eval/evaluate_gt_oracle_suite.py \
  --scope test \
  --output-dir eval/benchmark_v1/gt/gt_oracle_suite_v2
```

产物包括逐序列结果、AIST++/FineDance 分数据集统计和 pooled calibration：

- [`gt_oracle_suite_metrics.json`](benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_metrics.json)
- [`gt_oracle_suite_summary.csv`](benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_summary.csv)
- [`REPORT.md`](benchmark_v1/gt/gt_oracle_suite_v2/REPORT.md)

当前 38 条 held-out GT 的 pooled 参考分布为：event F1 `0.6784 +/- 0.1226`，
BAS `0.2904 +/- 0.1756`，absolute impact lag `0.5466 +/- 0.2551 s`，
phase error `0.2638 +/- 0.0416 cycles`，joint jerk P95 `1097.09 +/- 652.47`
rad/s^3。AIST++ 和 FineDance 必须同时报告，因为两者的风格、动作幅度、root
height 和节奏分布不同，不能把原始分数简单视为同一分布；报告中同时保留每个
数据集的 q10/median/q90 calibration range。

38 条是 sealed calibration，不是最终全量统计。校准通过后，已对全部已审计配对
数据运行扩展分布：AIST++ `1,408` 条、FineDance `203` 条，共 `1,611` 条。产物为
[`gt_oracle_suite_all_v1`](benchmark_v1/gt/gt_oracle_suite_all_v1/)，用于估计更稳定的
数据集内分布；论文主结果仍以 sealed 38 条和预注册测试协议为准。

对 M2/M3 的正式评估采用同样的核心指标：

```text
GT reference -> generator output -> SONIC execution
```

三者都计算动作质量和音乐匹配指标；SONIC 另外增加 tracking error、latency、
deadline miss、joint-limit、fall/stability 和 retention。GT oracle 只提供
参考分布，不是“优美程度”的绝对标签；自然性、舞蹈感和风格匹配仍需盲评。

当前本地没有另一套已完成 G1 retarget、音频配对、split 审计的第三方数据集，
因此暂不把未校验的数据混入主 oracle。后续新增数据必须先生成 manifest、验证
音乐-动作 ID 配对、统一 fps/root convention，再加入 suite。

### Step 0g：统一 GT/GMR/SONIC benchmark

本 benchmark 的目标不是比较 SMPL 和 G1 的关节位置，也不是建立一套复杂的
跨骨架几何误差。目标是用**同一套舞蹈质量指标和音乐适配指标**，分别评价同一
条配对数据在三个阶段的表现：

```text
原始 SMPL/SMPLH dance oracle
        -> GMR 后的 G1 reference
        -> SONIC executed motion
```

因此，原始 AIST++/FineDance motion 是 source oracle，GMR 生成的 G1 是 G1
oracle。两者都送入相同的 evaluator，前后差值就是 retargeting loss；G1 reference
到 SONIC execution 的差值就是 tracking/execution loss。SMPL、SMPLH 和 G1 的
骨架差异只在 evaluator 内部作为输入适配处理，不作为论文主结论。

运行 held-out 审计：

```bash
NUMBA_DISABLE_JIT=1 MUJOCO_GL=glfw \
MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
conda run --no-capture-output -n audiomimic \
python eval/evaluate_retargeting_loss.py \
  --scope test \
  --output-dir eval/benchmark_v1/gt/retargeting_loss_v1
```

扩展全量审计使用同一脚本和同一指标定义：

```bash
NUMBA_DISABLE_JIT=1 MUJOCO_GL=glfw \
MPLCONFIGDIR=/tmp/audiomimic-mpl-cache \
conda run --no-capture-output -n audiomimic \
python eval/evaluate_retargeting_loss.py \
  --scope all \
  --output-dir eval/benchmark_v1/gt/retargeting_loss_all_v1
```

当前审计覆盖 AIST++ 20 条和 FineDance 18 条，共 38 条 paired test 序列。输出：

- [`retargeting_loss_metrics.json`](benchmark_v1/gt/retargeting_loss_v1/retargeting_loss_metrics.json)：逐序列完整结果；
- [`retargeting_loss_summary.csv`](benchmark_v1/gt/retargeting_loss_v1/retargeting_loss_summary.csv)：便于统计和画图；
- [`REPORT.md`](benchmark_v1/gt/retargeting_loss_v1/REPORT.md)：方法、解释和 pooled summary。

全量扩展输出为 [`retargeting_loss_all_v1`](benchmark_v1/gt/retargeting_loss_all_v1/)。
其中 AIST++ 的 activity/root curve correlation 均值约为 `0.670/0.900`，FineDance
约为 `0.058/0.056`。这说明 GMR 的保真度具有明显数据集差异；FineDance 的 source
SMPLH 到 G1 转换必须单独分析，不能用 AIST++ 的结果替代。

benchmark 只保留两类主指标：

1. **舞蹈动作质量**：运动连续性、速度/加速度/jerk、动作能量、静止率、重复率、足部稳定性和整体可执行性。原始 motion、G1 reference、SONIC execution 都按同一指标定义报告；不同表示的转换只服务于计算指标，不报告逐关节位置差作为主结果。
2. **音乐适配程度**：onset/beat event Precision、Recall、F1，beat phase error，response lag，tempo error，onset-energy correlation，以及需要时的 BAS。它们直接回答动作是否踩拍、是否响应音乐结构、是否存在提前或延迟。

对 source 和 G1 使用完全相同的指标，`delta = G1 - source`。对于越大越好的指标，负值表示 retargeting loss；对于误差类指标，正值表示 retargeting loss。G1 reference 与 SONIC execution 再重复同样计算：

```text
source dance quality/music fit
        -> G1 dance quality/music fit       = GMR loss
        -> SONIC dance quality/music fit    = tracking loss
```

当前脚本中的 activity/root correlation、RMS ratio 和 event retention 仅作为动作结构诊断，帮助解释主 benchmark 的变化，不作为“骨架误差”或论文唯一结论。

   ```text
   source human motion
       -> G1 retargeted reference       = retargeting loss
       -> SONIC executed motion         = tracking/execution loss
   ```

本轮 38 条 pooled 结果的 activity/root 诊断值为 `0.4004/0.5323`，activity/root
RMS ratio 为 `0.7989/0.7145`；音乐 delta 为 `delta BAS=+0.0716`、
`delta event F1=+0.0956`、`delta impact correlation=+0.0091`、
`delta absolute impact lag=-0.1377 s`、`delta phase error=-0.0030 cycles`。
这些数值目前用于检查 pipeline 是否保留音乐响应，不直接宣称 GMR 改善了舞蹈。
正式论文的主表应放舞蹈质量和音乐适配两类 benchmark，source/G1/execution 三列
并列，另外报告 retargeting 和 tracking 的 delta。

分数据集看，AIST++ 的 activity/root curve 相关性均值为 `0.6990/0.9270`，
FineDance 为 `0.0687/0.0938`。这只是诊断信号，不能直接当作 FineDance 的
“骨架损失”；主 benchmark 仍应以统一的动作质量和音乐适配指标为准，并分别报告
两个数据集的分布。

其中 AIST++ 源 translation 按其 `smpl_scaling` 做归一化，FineDance 源 motion 使用
其 60 FPS、315 维 SMPLH 表示。后续不再把跨骨架位置差作为主 benchmark；如需定位
某个 GMR 映射问题，再单独使用这些结构诊断量。

### Step 0b：冻结实验因素

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

已完成的 `song098` pilot 输出在
[`gt_calibrated_m0_m2_m4_song098`](motion_music_execution/gt_calibrated_m0_m2_m4_song098)。
M0/M2/M4 各 3 次 SONIC execution 均完成 60 s，但总 motion-energy retention 分别为
`0.426 +/- 0.005`、`0.468 +/- 0.027`、`0.452 +/- 0.001`。这说明必须分别报告
`M_ref`、`M_exec` 和 retention；不能把 tracker 后的 BAS 或能量直接当作 generator
质量。该结果仍是单歌曲 pilot，不是最终论文结论。

该步骤对 `M_ref` 和 `M_exec` 使用同一实现计算 velocity、acceleration、jerk、energy、
static ratio、repetition、C4 continuity、root、PFC/FSR proxy、BAS、onset correlation
和 response lag，并计算可定义的 retention/degradation。

当前 M2/M3 reference 对 GT calibration 的统一对照已生成：
[`m2_m3_gt_comparison_v2`](motion_music_execution/m2_m3_gt_comparison_v2/)。目前包含
M2 song098 的 3 个训练 seed，以及 M3 song012/song065 各 1 条 60 s reference。该表是
描述性结果，不是单一总分排名；M3 仍需扩展到至少 3 首音乐和 3 个 generation seed。

### M3 音乐条件因果消融

M3 已完成第一轮正式 generator-level 因果实验：song012、song065，各使用
`1234/2345/3456` 三个 seed，并比较 `paired`、`wrong-song`、同一首歌 `+4 s`
错位和 `null sidecar`，共 24 条、每条约 30 s。每个 matched block 固定
checkpoint、unconditional parent、K64 history、起始帧和 sampling seed，只改变音乐条件。

逐条分析：

- `m3_music_ablation/formal_30s/song012/analysis_v2/`
- `m3_music_ablation/formal_30s/song065/analysis_v2/`

统一汇总：[`m3_music_ablation/formal_30s/aggregate_v2`](m3_music_ablation/formal_30s/aggregate_v2/)。
主要结果为：paired energy `1.829 +/- 0.213`、BAS `0.252 +/- 0.041`、impact
correlation `0.031 +/- 0.021`；wrong condition 的对应 BAS 为 `0.271 +/- 0.021`、
impact correlation 为 `0.029 +/- 0.029`。改变 sidecar 会造成约 `0.16--0.17 rad`
的 paired-joint RMSE，说明 M3 确实使用了音乐输入；但 paired 没有稳定优于
wrong/shifted/null，因此目前只能声称“音乐条件会改变生成”，不能声称“已经正确
随着目标音乐起舞”。

下一步先做 RMS-only、predicted-FMS-only、both、null 的拆分，并记录 sidecar 输出
幅度；只有 paired 在事件相关性、phase/lag 和 retrieval 指标上通过对照 gate，才进入
完整 SONIC matrix。`aggregate_m3_music_ablation.py` 可复用来汇总后续模型。

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
| M-BAS | kinematic beat 到最近 music beat 的指数距离 | 高 | FACT、EDGE、Lodge、Beat-It、DiscoForcing、RoboPerform；beat-alignment 核心指标 |
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
