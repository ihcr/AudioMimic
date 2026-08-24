# AudioMimic：实时音乐驱动的 G1 机器人舞蹈路线图

更新日期：2026-08-23

## 1. 项目目标

AudioMimic 的目标是构建一个严格因果、可持续在线运行的机器人舞蹈系统：

```text
实时音频
  -> 因果音乐特征与未来音乐预测
  -> 在线 G1 motion generator
  -> 连续、可执行的短期 reference
  -> SONIC whole-body tracking
  -> MuJoCo 与真实 Unitree G1
```

最终系统应同时满足：

1. 只使用当前时刻已经到达的音乐，不读取真实未来音频。
2. 动作在节拍、节奏和高层音乐语义上与音乐匹配。
3. Generator、通信和 tracker 按统一时间戳实时运行，不降速播放。
4. 动作满足 G1 与 SONIC 的速度、加速度、jerk、关节限位和稳定性要求。
5. 系统能够连续运行 60 s 以上，并具备迟到、丢包和异常 reference 的 fallback。

## 2. 当前系统状态

### 2.1 已完成：Motion Generator 基础链路

- 已有 pure Commit Forcing d16 推理模型、q0 generator 和 matching codec。
- 在线契约为 `K64/H8/C4/S66`：
  - K64：64 个历史 latent token，对应约 4.27 s 动作历史；
  - H8：一次预测 8 个 latent token，即 16 个 30 FPS motion frame；
  - C4：每轮提交 4 个 latent token，即 8 个 motion frame、约 267 ms；
  - S66：root、关节位置、关节速度与加速度组成的物理边界状态。
- 常规 GPU 推理约 48--55 ms，能够在 C4 deadline 内完成。
- Open-loop 在线 rollout 已按约 30 FPS 连续运行 60 s，无 deadline miss、fall 或 reset。
- 模型、codec、q0 和 K64 seed 已安装到 AudioMimic，并通过 SHA256 校验。

当前模型仍是 **motion-only**。M2 PKL 仅用于提供 K64 启动历史，运行期间没有读取实时音乐。

### 2.2 已完成：SONIC 与中间通信

- Generator 内部使用 MuJoCo joint order，SONIC wire boundary 转换为 SONIC joint order。
- 已修正 quaternion 顺序、C4 frame index 和 30 Hz 到 50 Hz 连续重采样。
- C4 重采样保持 `13, 13, 14` 的连续相位，不再每包重置。
- 已支持 H8 preview、C4 commit、full sequence 和 C4 packet 两种离线模式。
- 固定 M2 的完整60 s测试中，full与C4均在相近时间首次失稳，排除packet切分是
  主要失败来源；平缓GT的full/C4也表现一致。
- 已支持 SONIC feedback、MuJoCo state 和 S66 telemetry 记录。
- 运行代码已经从临时实验树迁入正式 AudioMimic 仓库。

### 2.3 已完成：SONIC 能力与 Closed-loop 诊断

2026-08-18进行的固定M2 `0.75x/1.0x/1.25x/1.5x`能力检查均无
fall/reset，但所有命令都先使用`--max_seconds 32`截取原轨迹前32 s，再进行
时间缩放。它们只证明前32 s source prefix在对应倍率下可执行，不能外推为完整
60 s轨迹通过。旧实验保存时长分别为42.67/32.00/25.60/21.33 s，但四组覆盖的
原始动作范围均止于source 32 s。

2026-08-19补做完整60 s source测试后得到：

| Playback | Packet | First fall / total |
|---|---|---:|
| 1.00x | C4 | 35.2/60.0 s、35.8/60.0 s |
| 1.00x | full | 37.6/60.0 s |
| 0.90x | full | 64.9/66.7 s |
| 0.85x | full | 47.8/70.6 s |
| 0.80x | full | 43.0/75.0 s |
| 0.75x | full | 80.0/80.0 s、69.0/80.0 s |

上述完整轨迹实验均未将当前measured pose平滑对齐到M2首帧，因此不能直接解释为
M2 generator不可执行。失败0.75x组的分窗tracking lag从约320 ms增长到400 ms，
表明错误启动边界会造成长时误差积累。

随后补做的`1.0x full + 3 s measured-state alignment + 1 s hold`首轮完整通过：
正式M2段60 s无fall/reset，最低base height 0.651 m；排除4 s启动前缀后，Raw/
lag-compensated RMSE为0.2445/0.2230 rad，全局lag为140 ms。未对齐1.0x full
组对应指标为0.2755/0.2539 rad、180 ms，并在37.6 s失稳。当前证据说明启动
边界处理是此前失败的主要因素之一，M2原速可执行性需要再做两次独立alignment
重复后才能定论。

第二次独立alignment运行同样完整通过：60 s无fall/reset，最低base height
0.625 m，Raw/lag-compensated RMSE为0.1681/0.1664 rad，全局lag为20 ms。
当前aligned成功率为2/2；完成第三次重复后，将该流程冻结为固定轨迹SONIC
tracking的标准启动协议。

第三次alignment运行也完整通过，最终成功率3/3。三次Raw RMSE均值/标准差为
0.1933/0.0443 rad，lag-compensated RMSE为0.1850/0.0329 rad，最低base
height为0.633/0.016 m。离线固定轨迹评估从此固定使用`3 s measured-state
alignment + 1 s hold + full packet + 1.0x`；无alignment结果仅作为启动边界
消融，不再用于判断generator可执行性。

同一标准协议下，M0/M2/M4 song098、seed1234均完成三次60 s回放并全部稳定，
成功率均为3/3。route级Raw RMSE分别为0.1781/0.1933/0.1694 rad，动作能量
保留率为0.468/0.522/0.503，3--8 Hz功率保留率为0.227/0.395/0.244。当前
主要gap是动作能量和高频细节衰减，而不是音乐条件导致稳定性失败。M2高频结果
方差较大，下一步需要扩展seed并计算音乐beat/onset特征在tracking前后的保留率。

2026-08-20 的 headless capability 复查又识别出一个独立初始化条件：elastic band
必须在内置 Macarena CONTROL 下释放，并在无约束站稳后才开启 ZMQ。若在 band
仍启用时进入 ZMQ、到 alignment 末端才释放，会产生与 reference 无关的瞬时跌倒。
按修正顺序完成 SONIC 原生 low/medium/high 各三次回放后，9/9 runs 全部无 fall。
三个 tier 的 aligned RMSE 为 0.1126/0.1375/0.1745 rad，root-relative EMPKPE
为 0.0630/0.0633/0.0891 m，能量保留率为 0.625/0.666/0.447。高动态组仍稳定，
但 pose error 增大且动态表达衰减明显。该实验给出 SONIC known-distribution baseline，
不是独立外部 GT gate；后续 GT capability 和 M0/M2/M4 复现实验均固定该初始化顺序。

同一协议下，retargeted GT low/medium/high 也完成 9/9 runs，全部无 fall。三个 tier
的 aligned RMSE 为 0.1540/0.1994/0.3231 rad，EMPKPE 为
0.1023/0.1133/0.1599 m，幅度保留率为 0.921/0.836/0.810。相对
SONIC-native high，外部 high GT 的 RMSE 和 EMPKPE 分别为 1.85 和 1.80 倍。
因此此前“激烈 GT 不可执行”的判断被撤销；真实问题是 retarget/reference domain
进入 SONIC 后的 fidelity gap，且强动态动作最明显。

将 M0/M2/M4 与上述双基线统一比较后，三类 generator reference 的动态统计均最接近
所选低动态 GT。三条路线的 aligned RMSE 为 0.1763/0.1850/0.1678 rad，分别是
SONIC-native low 的 1.57/1.64/1.49 倍，但只比 retargeted-GT low 高
1.14/1.20/1.09 倍。幅度保留率为 0.912/0.906/0.872，逐关节中位动态能量保留率
仅为 0.468/0.522/0.503。由此可将当前 generation-to-execution gap 更具体地表述为：
稳定性已经成立，reference-domain pose error 仍存在，tracker 对中高频动态表达有系统性
压缩。M4 的执行误差最低；现有单歌曲 automatic metrics 尚不能证明 M2/M4 的音乐条件
优势。完整结果见
[`RESULT-20260820-generator-vs-sonic-baselines.md`](docs/experiments/RESULT-20260820-generator-vs-sonic-baselines.md)。

进一步对现有固定轨迹做 song098 正确时钟、每 2 s 循环平移和 song065 错配诊断后，
M2/M4 reference 的 impact rank 仅为 26.4%/17.2%，BAS rank 为 25.3%/33.3%；
执行后也未显示正确配对优势。当前自动指标因此不能证明这些导出动作具有可检测的细粒度
音乐对齐。该实验不是生成时 condition intervention，不能据此断言音乐条件无效。真正的
因果验证仍需 M2 checkpoint，在固定 diffusion noise 下重新生成
`paired/shifted/shuffled/silence` 四组。结果见
[`RESULT-20260820-music-pairing-sensitivity.md`](docs/experiments/RESULT-20260820-music-pairing-sensitivity.md)。

旧32 s prefix实验曾观测到：

```text
全身关节速度 P95：4.24 rad/s
全身关节速度最大值：15.85 rad/s
```

性能下降主要集中在手臂，腿部tracking对倍率变化较稳定。这些数值仅是短prefix
上的观测，不是完整轨迹能力下界，也不是SONIC的绝对失败阈值。

Closed-loop 诊断得到：

- `delayed_residual` 对完整 S66 使用统一增益会形成反馈放大；
- `jit_measured` 可维持 30 FPS，但 deadline 前 80 ms 的 measured S66 与真实 C4 boundary 时间错位；
- 60 s JIT rollout 的腿部速度 P95/max 为 5.40/36.26 rad/s，手臂为 10.60/71.01 rad/s；
- JIT tracking P95 明显差于 Open-loop，但两组均无 fall/reset；
- 阻塞等待真实 C4 boundary 后再推理只有约 24.3 FPS，不是有效在线方案。

因此当前主要瓶颈是 generator 对 delayed/noisy measured S66 的分布适应和动作可执行性，而不是 SONIC 协议或基础 tracking。

## 3. 尚未完成的核心链路

当前已经打通：

```text
在线无音乐动作生成 -> SONIC -> MuJoCo
```

尚未打通：

```text
实时音乐 -> 因果音乐条件 -> 在线动作生成 -> SONIC -> G1
```

缺口包括：

1. 实时音频采集、缓冲和时间戳。
2. 因果 beat、tempo、onset、phase 与高层音乐语义提取。
3. M2 predicted future-music sidecar 的在线 checkpoint 与推理入口。
4. 音乐条件真正进入 H8 generator，而不是仅使用 M2 动作 seed。
5. Audio、generator C4 和 SONIC reference 的统一绝对时钟。
6. Tracker-aware generator 训练与 boundary-state 时间对齐。
7. Reference feasibility 检测、H8 fallback 和真机安全状态机。

## 4. TODO List

### 4.0 论文实验总路线（冻结入口）

后续所有模型、generator、SONIC 和端到端结果统一按照
[`ICRA_GT_CALIBRATED_EVALUATION_PLAN.md`](docs/evaluation/ICRA_GT_CALIBRATED_EVALUATION_PLAN.md)
执行。该计划定义三级 GT（O-Human、O-G1、O-Exec）、GT 指标验真、sealed test、
模型无关的 motion package、统计方法、主表以及论文 claim gate。

执行原则：先完成 GT 与指标校准，再进行模型排名；所有模型使用同一测试 manifest，
不按指标选择有利 seed 或歌曲子集；离线 PKL 回放只能作为 tracker 能力测试，不能
单独作为“实时音乐生成”证据。

### P0：形成实时音乐 Open-loop Demo

- [ ] 固定在线音乐特征接口，包括 feature schema、FPS、时间戳和因果可见范围。
- [ ] 确认 M2 sidecar 的 checkpoint、输入特征、future horizon 和训练配置。
- [ ] 将音频线程接入 AudioMimic：持续读取 wav 流或麦克风，不阻塞 generator。
- [ ] 在线提取因果 beat/tempo/onset/phase 特征并对齐到 30 FPS motion clock。
- [ ] 将 M2 predicted future-music condition 接入 H8/C4 generator。
- [ ] 保持当前稳定的 synthetic S66 Open-loop，暂不启用 measured-state Closed-loop。
- [ ] 同步播放音乐与 SONIC reference，记录端到端 audio-to-motion latency。
- [ ] 在 MuJoCo 完成至少 3 首未见音乐、每首 60 s 的实时生成演示。

验收标准：

```text
实时因果音频输入
30 FPS generator deadline miss <= 1%
无 reference 断流
60 s 无 fall/reset
视频中音乐与机器人动作时间同步
```

### P1：音乐-动作质量评估

Generation-to-Execution Gap的数据采集、时间对齐与指标协议见：
[`docs/experiments/NEXT-20260819-generation-to-execution-gap.md`](docs/experiments/NEXT-20260819-generation-to-execution-gap.md)。
所有指标定义、证据等级、统计方法和正式验收 gate 统一见：
[`docs/evaluation/ICRA_GT_CALIBRATED_EVALUATION_PLAN.md`](docs/evaluation/ICRA_GT_CALIBRATED_EVALUATION_PLAN.md)；
指标定义和证据等级见
[`docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md`](docs/evaluation/EVALUATION_MAP_MUSIC_TO_G1.md)；
机器可读注册表为 [`eval/evaluation_map_v1.json`](eval/evaluation_map_v1.json)。

- [ ] 分别评价生成reference `M_ref`、SONIC执行动作`M_exec`及两者之间的retention。
- [x] 完成 AIST++ 音频/G1 motion schema、成对文件和 split/cache 一致性审计，生成 GT benchmark manifest。
- [x] 用 AIST++ test GT 完成第一轮动作质量指标校准：jitter、low-pass、freeze 的方向检查通过。
- [x] 将统一动作质量、音乐响应和 execution-retention evaluator 跑通 M0/M2/M4 song098：9/9 SONIC runs 完成 60 s。
- [x] 补齐 FineDance-G1 raw motion、WAV、label 和 retargeted G1 资产；203 条同 ID 配对有效，官方 cross-genre test 18/18 可用，审计见 `eval/results/benchmark_v1/gt/manifest_v2_finedance/`。
- [x] 完成 FineDance test 的 paired-vs-wrong-song 音乐对应性审计：paired 最佳相关性 `0.1655`，全错配均值 `0.0884`，平均 margin `+0.0771`；这证明存在音乐-动作对应信号，但不等于最终 musicality 分数。
- [x] 生成 FineDance-G1 baseline+beat prepared tree：train `47,817`、test `3,265`；motion/audio/baseline/beat basename 全部一致，仓库校验通过。目录为 `data/finedance_g1_fkbeats/`，metadata 记录了 extractor。
- [x] 生成 FineDance-G1 逐 clip quality manifest，并将 root-height 规则改为可审计的训练候选过滤；当前不删除文件，sealed test 保持完整。train 有 `213` 个负 root clip、`330` 个低于 `0.2 m`，test 有 `20/66` 个对应诊断样本。
- [x] 用真实 `AISTPPDataset` 完成 FineDance-G1 baseline+beat smoke test：train/test `47,817/3,265`，motion/music/beat batch shape 为 `[4,150,38]`、`[4,150,35]`、`[4,150]`。
- [x] 建立 FineDance-G1 sealed GT oracle：18 条完整 paired G1/audio 序列，补齐 motion quality、event F1、tempo error、lag 和 phase error；结果见 `eval/results/benchmark_v1/gt/finedance_gt_oracle_v1/`。
- [x] 修正 beat condition 的 train/test 泄漏：train/test 都使用 audio beat，motion beat 只作为 beat loss target。
- [x] 完成修正后 audio-beat baseline 的 1 epoch smoke train/eval：18/18 FineDance test clips 生成和 FK 指标跑通；checkpoint 与结果见 `eval/finedance_baseline_audio_beat_smoke/`。该结果仅用于链路诊断，不作为论文结果。
- [x] 建立多数据集 GT oracle calibration suite：AIST++ held-out 20 条 + FineDance held-out 18 条，共 38 条 paired G1/audio 序列；输出 per-dataset 与 pooled motion/music reference distributions，见 `eval/results/benchmark_v1/gt/gt_oracle_suite_v2/`。校准后扩展到 AIST++ 1,408 条 + FineDance 203 条，见 `eval/results/benchmark_v1/gt/gt_oracle_suite_all_v1/`。
- [x] 建立统一 GT/GMR/SONIC benchmark：对同一音乐的原始 SMPL/SMPLH、G1 reference 和 SONIC execution 使用同一套舞蹈质量与音乐适配指标；38 条 source→G1 配对审计结果见 `eval/results/benchmark_v1/gt/retargeting_loss_v1/`。retargeting 与 tracking 只用前后 delta 区分，不把跨骨架位置差作为主结论。
- [x] 完成全量 source→G1 retargeting audit：1,611 条有效配对；扩展结果见 `eval/results/benchmark_v1/gt/retargeting_loss_all_v1/`。AIST++ 与 FineDance 分布分开报告，不用 pooled 数值掩盖 FineDance 的 GMR 保真度问题。
- [x] 将已有 M2/M3 generator reference 按 GT calibration 统一出表：M2 song098 三 seed、M3 song012/065 各一条 60 s，结果见 `eval/results/motion_music_execution/m2_m3_gt_comparison_v2/`。
- [x] 完成 M3 `M_ref` 首轮 G1-native 全模块评估：song012/065 × 3 sampling seeds，共 6 条 60 s reference；FK BAS `0.2708`、Beat F1 `0.2275`、Beat recall `0.1866`、timing mean/std `-0.049/1.448` 帧。该结果只支持 reference-side diagnosis，不包含 SONIC execution 或实时性 claim。
- [x] 完成 M3 正式 generator-level 音乐条件因果消融：song012/065 × seed1234/2345/3456 × paired/wrong/+4s shifted/null，共 24 条 30 s 轨迹；逐条结果见 `eval/results/m3_music_ablation/formal_30s/song012/analysis_v2/`、`song065/analysis_v2/`，汇总见 `aggregate_v2/`。
- [x] 确认 M3 sidecar 对生成有因果作用，但当前 paired condition 尚未证明优于 wrong/shifted/null：paired 相对 wrong 的 BAS 差为 `-0.0193`、impact correlation 差为 `+0.0023`，paired 胜出分别为 `2/6`、`4/6`；不能据此宣称已经学会正确音乐对齐。
- [x] 完成论文指标审计并冻结 evaluation map v1.2：BAS 升级为 beat-alignment suite 的核心指标，同时固定 event coverage/F1、onset response、lag、tempo/phase 等互补 beat 指标；动作质量、音乐适配、SONIC 执行和实时系统四组指标保持分开报告。来源与定义见 `docs/evaluation/LITERATURE_METRIC_AUDIT_20260823.md`。
- [x] 用 AIST++ GT corruption 验证 benchmark 指标方向：20 条 GT、260 个 clean/退化样本；jerk、low-pass energy、static ratio 和 freeze event F1 通过；`repeat_similarity` 未通过，暂不作为核心指标。结果见 `eval/results/benchmark_v1/gt/benchmark_validity_v1/`。
- [x] 将 GT benchmark 按数据集、tempo 和可用 style/genre 分层验证：AIST++ 20 条 + FineDance 18 条，共 494 个 clean/退化样本；8 个跨数据集方向检查中 7 个通过。FineDance 的 freeze Beat F1 为 WARN，因此 Beat F1 不能单独作为退化判据，但继续作为 beat-alignment suite 的核心指标，与 BAS、coverage、lag、tempo/phase 联合报告。结果见 `eval/results/benchmark_v1/gt/stratified_audit_v1/`。
- [x] 将 GT reference distribution 扩展到全量 paired 数据：AIST++ 1,408 条 + FineDance 203 条，共 1,611 条、32 个 dataset/tempo/style strata；全量结果只用于条件化 calibration，不回写 sealed test 或指标设计。结果见 `eval/results/benchmark_v1/gt/stratified_audit_all_v1/`。
- [x] 建立四层正式实验 manifest：同一音乐固定 O-Human(SMPL/SMPLH)、O-G1(GMR)、M-ref(G1 generator) 和 M-exec(SONIC G1)；当前接入 38 条 GT 和已有模型记录，缺失 execution 显式标为 pending。结果见 `eval/results/benchmark_v1/formal/`。
- [x] 汇总 38 条 O-Human -> O-G1 两层正式 GT 结果，统一报告 GMR 前后 activity、root、BAS、Beat F1、impact、tempo/phase 等变化；结果见 `eval/results/benchmark_v1/formal/GT_REPORT.md`。AIST++ retargeting correspondence 较稳定，FineDance correspondence 偏低，后者先作为数据/格式审计项，不作为 generator 退化结论。
- [x] 完成 AIST++ 20 条与 FineDance 18 条 paired/wrong-song 音乐配对完整性审计；两套数据的 paired 平均相关性高于 wrong-song，但 Top-1 分别为 `0.30/0.22`，不能把自动指标视为绝对音乐性 oracle。结果见 `eval/results/benchmark_v1/gt/aist_music_pairing_v1/` 和 `finedance_music_pairing_v1/`。
- [x] 将已有 M0/M2/M4 song098 的 9 次 SONIC repeat 与 M3 012/065 的 paired `M_ref/M_exec` 指标统一汇总；结果见 `eval/results/benchmark_v1/formal/MODEL_REPORT.md`。该表是现有进度记录，不是平衡的模型排名。
- [x] 生成现有 route-level 描述性汇总 `eval/results/benchmark_v1/formal/MODEL_SUMMARY.md`；仅用于检查动作质量与 tracker retention，不做跨歌曲/跨协议排名。
- [x] 对已有完整 motion/audio artifact 补算 Beat event Precision/Recall/F1、tempo error 和 phase error；结果见 `eval/results/benchmark_v1/formal/model_music_extended/REPORT.md` 和 `REPORT_ZH.md`。此前 9 条 M0/M2/M4 SONIC 记录已从 feedback log 恢复 measured-motion PKL，旧 execution event 指标已补齐；未采集矩阵单元仍显式 pending。
- [x] 固化正式扩展采集矩阵：4 routes × 3 songs（012/065/098）× 3 sampling seeds（1234/2345/3456）× `M_ref/M_exec`，共 72 个 target cells；当前 5 个 reference、3 个 execution 已有，其余显式 pending，见 `eval/results/benchmark_v1/formal/EXPANSION_MATRIX.csv`。
- [ ] 获取或发布可复现 M2/M3 music-conditioned inference checkpoint、feature-cache contract、training/sampling seed 和 audio alignment metadata；当前 prior-dev release 只有 unconditional pure Commit Forcing，不能继续生成新的 M2/M3 PKL。
- [x] 冻结音乐条件输入对照：`8D beat-only` 下限、`MERT + Librosa + 8D beat` 主线、`Wav2CLIP + beat` 历史 anchor；STFT/Jukebox 只作为消融，不同时堆叠多个高层 encoder。
- [ ] 在固定 quality manifest 下比较 unfiltered train、`root_z >= 0` 和 retarget correction 三种训练策略；不得清洗 sealed test。
- [ ] 正式重新训练 FineDance-G1 beat-conditioned baseline（使用修正后的 audio beat condition，`lambda_beat=0`），再补齐 G1 beat estimator checkpoint 后训练带可学习 beat loss 的版本。
- [ ] 将校准后的动作质量指标扩展为正式 evaluator：FIDk/FIDg（辅助）、PFC、FSR、motion energy、静止率、重复率和C4连续性。
- [x] 建立 GT 音乐匹配 calibration：BAS（beat-alignment 核心）、onset-energy correlation、response lag、tempo/phase 和 event Precision/Recall/F1 已在 38 条 paired GT 上统一统计；R@K/MMDist 仍需在生成结果和错配对照上补齐。
- [ ] 仅在加入显式beat target后，将Beat Precision/Recall/F1和BAP作为节拍可控性指标。
- [ ] 增加速度、加速度、jerk、joint-limit、foot sliding和base stability指标。
- [ ] 增加盲测：自然性、舞蹈感、表现力、长期连贯性、节奏和音乐风格匹配。
- [x] 冻结盲评问题、素材匹配、左右随机化、排除规则和 participant-aware 统计协议。
- [x] 完成 song098/seed1234 的统一无标签 pilot：M0/M2/M4 reference 与 execution 共 6 段，6/6 审计合格。
- [ ] 将相同 renderer 扩展到 3 songs x 3 generation seeds x 3 SONIC repeats。
- [ ] 对 M0、M2 predicted 和 M4 oracle 使用相同歌曲、seed 与 duration 做消融。
- [ ] 对生成 reference 与 SONIC tracked motion 分别计算音乐一致性，量化 tracker 造成的节拍损失。
- [ ] 将统一 benchmark 跑通 source oracle、M3/M2 generator reference 和 SONIC execution 四层，主表固定为舞蹈质量、音乐适配、retargeting delta、generator delta、tracking delta 五组结果；M_ref 已是 G1 输出，不再重复 GMR。
- [ ] 扩展 M2/M3 正式 generator benchmark 至至少 3 首音乐、3 个 generation seed，并在相同 GT calibration 下比较 reference 与 SONIC execution；当前 M3 已完成 2 首歌 × 3 seed 的因果 pilot，下一步补第 3 首和 RMS/FMS 拆分消融。
- [x] 对现有固定轨迹完成正确时钟、循环 time-shift 和 wrong-song 配对敏感性诊断。
- [ ] 获得 M2 checkpoint 后固定采样噪声，重新生成 paired/shifted/shuffled/silence 四组，完成因果条件消融。
- [ ] 将 FineDance 的音乐对应审计扩展为正式生成评估：paired、time-shifted、wrong-song、tempo-preserved 和 silence 五组，加入 beat phase、tempo、lag-aware onset、retrieval R@K/MMDist，并分别计算 reference/execution。
- [ ] 将分层 GT 校准扩展到全量 AIST++/FineDance，并保证每个主要 tempo/style stratum 至少 3 条序列；在此之前，小样本 style 结果只作描述性报告。

验收标准：在多歌曲、多generation seed和SONIC重复下，M2在严格因果条件下相对
M0的音乐匹配提升具有统计依据，同时动作质量与执行成功率不退化；报告与M4 oracle
的差距，以及SONIC对动作能量、高频细节和音乐表达的保留率。BAS不得单独作为验收依据。

### P2：Generator 可执行性改进

- [ ] 将 SONIC 已验证动态范围写入训练与评估配置，而不是运行时静默裁剪。
- [ ] 加入 joint velocity、acceleration、jerk 和 C4 boundary continuity loss。
- [ ] 对腿、腰、手臂使用分组损失与阈值，重点抑制手臂速度尖峰。
- [ ] 加入 joint limit、foot contact/sliding 和 base stability loss。
- [ ] 输出每个 H8 的 feasibility diagnostics 或 confidence。
- [ ] 对异常 H8 实现拒绝/回退策略，并保存触发原因。

验收标准：三 seed、每组 60 s 下，生成动态主要落在已验证 SONIC 范围内，tracking P95 不因在线 rollout 持续增长。

### P3：Tracker-aware Closed-loop Generator

- [ ] 明确定义 measured S66 的 measurement time、boundary time 和 age。
- [ ] 比较线性外推与 reference-aware boundary prediction。
- [ ] 采集 `(measured state, remaining reference, deadline boundary state)` 训练数据。
- [ ] 训练时加入 50--80 ms delay、tracking residual、measurement noise 和 feedback dropout。
- [ ] 进行多 C4 tracker-in-the-loop/self-forcing rollout，而不是只做单步 teacher forcing。
- [ ] 输入 current reference、boundary reference、tracking residual 和 remaining time。
- [ ] 比较 Open-loop、JIT-current、predicted-boundary 和非实时 boundary diagnostic。
- [ ] 若 H8/C4 反馈仍过慢，再训练 H8/C2；不要只在部署代码中直接改 commit 长度。

验收标准：Closed-loop 在相同音乐、seed 和初态下，tracking P95、survival 或抗扰性至少一项稳定优于 Open-loop，且生成动态不被反馈放大。

### P4：Runtime 安全与真机准备

- [ ] 为 audio、generator、reference 和 feedback 消息加入统一绝对时间戳。
- [ ] 实现 audio、generation、execution 三线程异步运行。
- [ ] 推理迟到时执行上一轮 H8 后半段，不让 SONIC reference 断流。
- [ ] 对 NaN/Inf、joint limit、速度、加速度、jerk 和 orientation jump 做发送前检查。
- [ ] 实现 feedback 丢失时的 Open-loop fallback 与平滑 hold。
- [ ] 在 MuJoCo 做 packet delay、dropout 和 inference jitter 压力测试。
- [ ] 确认真机 feedback 字段、joint order、quaternion 和仿真完全一致。
- [ ] 逐级执行低动态、标准动态和完整音乐真机实验。

## 5. 推荐执行顺序

```text
1. 冻结当前 SONIC protocol 与 Open-loop runtime
2. 接入实时因果音乐和 M2 sidecar
3. 完成 MuJoCo 实时音乐 Open-loop demo
4. 建立音乐质量与物理可执行性评估
5. 修改 generator 的动态约束与 tracker-aware 训练
6. 验证 predicted-boundary Closed-loop
7. 加入 fallback、安全检查和压力测试
8. 进入真实 G1
```

当前最优先的工作不是继续修改 SONIC，而是拿到并接入 **在线 M2 音乐条件模型与特征契约**。在此之前，系统只能证明实时动作生成和机器人 tracking，不能证明机器人在随实时音乐生成舞蹈。

人类评价协议和随机化工具已完成，song098/seed1234 pilot 已生成 6 个合格 clips 和 9 个
canonical trials，`pilot_ready=true`。但正式问卷尚未启动，`paper_ready=false`；必须在获得
多歌曲、多 seed generator 输出后，按
[`HUMAN_EVALUATION_PROTOCOL.md`](docs/evaluation/HUMAN_EVALUATION_PROTOCOL.md) 统一渲染，
不能直接使用旧对比视频形成论文用户研究。

## 6. 仓库责任边界

### AudioMimic

- 实时音频、music sidecar 推理与时钟同步；
- checkpoint inference runtime；
- SONIC bridge、MuJoCo/真机运行和安全 fallback；
- 端到端评估、视频和操作手册。

### Musics2Dance

- 数据处理与模型训练；
- M0/M2/M4、H8/C4 或 H8/C2 等研究消融；
- tracker-aware 与 feasibility loss；
- 导出带版本、manifest 和 SHA256 的兼容模型 release。

只要 motion/state/codec/music-condition 契约保持不变，AudioMimic 应能够仅替换模型 release；契约变化必须提升 runtime protocol 版本并提供 adapter。
