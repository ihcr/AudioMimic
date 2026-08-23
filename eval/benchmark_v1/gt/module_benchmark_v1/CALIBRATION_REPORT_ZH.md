# D/M Benchmark 指标方向校准报告 v1

本报告不是模型排名，而是检查冻结的 D（舞蹈动作质量）和 M（音乐-舞蹈适配）指标能否识别已知动作退化。输入为 38 条 sealed GT 的 clean、jitter、low-pass、freeze 和 repeat 变体。

## 校准结论

- D 模块：jerk P95 和 static ratio 的方向检查全部通过，可以继续作为动作质量子模块的核心指标。
- M 模块：Beat F1 在 AIST++ freeze 检查通过，但在 FineDance 中为 WARN；Beat F1 不能单独作为跨数据集退化 gate。
- BAS 不因该 WARN 被删除。BAS、Beat F1、event precision/recall、impact correlation、lag、tempo error 和 phase error 仍应作为完整 rhythm suite 联合报告。
- 这一步只验证‘指标对指定退化是否敏感’，不能把 GT 的分数解释为绝对优美度，也不能替代人类盲评。

## 逐项映射

| module | submodule | metric | dataset | check | expected | result | high-clean delta |
|---|---|---|---|---|---|---|---:|
| D | smoothness | jerk_p95 | aistpp | jitter_jerk_p95_rad_s3 | increase | **PASS** | +1135.3794 |
| D | smoothness | jerk_p95 | aistpp | lowpass_jerk_p95_rad_s3 | decrease | **PASS** | -2266.4634 |
| D | liveliness | static_ratio | aistpp | freeze_static_ratio_below_0p05_rad_s | increase | **PASS** | +0.0986 |
| M | rhythm | event_f1 | aistpp | freeze_G1BeatF1 | decrease | **PASS** | -0.0567 |
| D | smoothness | jerk_p95 | finedance | jitter_jerk_p95_rad_s3 | increase | **PASS** | +858.4909 |
| D | smoothness | jerk_p95 | finedance | lowpass_jerk_p95_rad_s3 | decrease | **PASS** | -2463.3446 |
| D | liveliness | static_ratio | finedance | freeze_static_ratio_below_0p05_rad_s | increase | **PASS** | +0.0119 |
| M | rhythm | event_f1 | finedance | freeze_G1BeatF1 | decrease | **WARN** | +0.0018 |

## 全部冻结模块与指标覆盖

下面列出 taxonomy v1 中的全部指标。‘方向校准’只表示当前已有 GT corruption 或 validity audit 对该指标做过敏感性检查；‘未校准’不表示指标无效，而表示正式模型实验前仍需完成相应的 GT/counterfactual/人评校准。

| module | submodule | metric | 当前实现状态 | 当前校准状态 | level |
|---|---|---|---|---|---|
| P 数据与评估协议有效性 | P.interface 数据与接口约定 | P0-SCHEMA Motion schema validity | prototype | not_directionally_calibrated | hard_gate |
| P 数据与评估协议有效性 | P.interface 数据与接口约定 | P0-ORDER Interface convention validity | prototype | not_directionally_calibrated | hard_gate |
| P 数据与评估协议有效性 | P.time 配对、时钟与覆盖 | P0-TIME Clock and audio alignment validity | prototype | not_directionally_calibrated | hard_gate |
| P 数据与评估协议有效性 | P.time 配对、时钟与覆盖 | P0-COVER Feedback coverage | implemented | not_directionally_calibrated | hard_gate |
| D 舞蹈动作质量 | D.realism 动作分布真实性 | G-FID-K Kinetic FID | missing | not_implemented | core |
| D 舞蹈动作质量 | D.realism 动作分布真实性 | G-FID-G Geometric FID | missing | not_implemented | supplementary |
| D 舞蹈动作质量 | D.diversity 多样性与覆盖 | G-DIV-K Kinetic diversity | prototype | not_directionally_calibrated | core |
| D 舞蹈动作质量 | D.diversity 多样性与覆盖 | G-DIV-G Geometric diversity | prototype | not_directionally_calibrated | core |
| D 舞蹈动作质量 | D.smoothness 平滑度与局部连续性 | G-DYNAMICS Joint dynamics distribution | implemented | core_validated | core |
| D 舞蹈动作质量 | D.smoothness 平滑度与局部连续性 | G-C4-CONT C4 boundary continuity | implemented | not_directionally_calibrated | core |
| D 舞蹈动作质量 | D.liveliness 活力、冻结与长期重复 | G-ENERGY Motion energy | implemented | core_validated_for_lowpass | core |
| D 舞蹈动作质量 | D.liveliness 活力、冻结与长期重复 | G-STATIC Static ratio | implemented | core_validated | core |
| D 舞蹈动作质量 | D.liveliness 活力、冻结与长期重复 | G-FREEZE Adaptive freeze proportion | missing | not_implemented | core |
| D 舞蹈动作质量 | D.liveliness 活力、冻结与长期重复 | G-FREEZE-LR Freeze length regularity | missing | not_implemented | supplementary |
| D 舞蹈动作质量 | D.liveliness 活力、冻结与长期重复 | G-REPEAT Long-term pose repetition | implemented | needs_revision | core |
| D 舞蹈动作质量 | D.physics 接触、滑步与身体物理合理性 | G-PFC Physical Foot Contact | prototype | not_directionally_calibrated | core |
| D 舞蹈动作质量 | D.physics 接触、滑步与身体物理合理性 | G-FSR Foot Skating Ratio | prototype | not_directionally_calibrated | core |
| D 舞蹈动作质量 | D.physics 接触、滑步与身体物理合理性 | G-PEN Ground penetration | prototype | not_directionally_calibrated | core |
| D 舞蹈动作质量 | D.physics 接触、滑步与身体物理合理性 | G-ROOT Root behavior | implemented | not_directionally_calibrated | core |
| M 音乐-舞蹈适配 | M.rhythm 节拍、重音与节奏网格 | M-BAS Beat Alignment Score | implemented | supplementary_only | supplementary |
| M 音乐-舞蹈适配 | M.rhythm 节拍、重音与节奏网格 | M-BEAT-F1 Beat precision, recall and F1 | prototype | music_diagnostic_validated | conditional |
| M 音乐-舞蹈适配 | M.rhythm 节拍、重音与节奏网格 | M-BAP Beat Assignment Precision | prototype | not_directionally_calibrated | conditional |
| M 音乐-舞蹈适配 | M.rhythm 节拍、重音与节奏网格 | M-KPD Key Pose Distance | missing | not_implemented | conditional |
| M 音乐-舞蹈适配 | M.rhythm 节拍、重音与节奏网格 | M-RS Rhythmic Score | missing | not_implemented | supplementary |
| M 音乐-舞蹈适配 | M.dynamics 速度、相位与动态响应 | M-ONSET Onset-motion response | implemented | not_directionally_calibrated | core |
| M 音乐-舞蹈适配 | M.dynamics 速度、相位与动态响应 | M-RESP-LAG Music response lag | implemented | not_directionally_calibrated | core |
| M 音乐-舞蹈适配 | M.dynamics 速度、相位与动态响应 | M-TEMPO-PHASE Tempo consistency and phase error | missing | not_implemented | core |
| M 音乐-舞蹈适配 | M.structure 乐句与段落结构响应 | M-PHRASE Phrase-boundary response | missing | not_implemented | core |
| M 音乐-舞蹈适配 | M.semantic 风格、情绪与跨模态语义 | M-RETRIEVAL Audio-motion R@1, R@2 and R@3 | missing | not_implemented | core |
| M 音乐-舞蹈适配 | M.semantic 风格、情绪与跨模态语义 | M-MMDIST Multimodal Distance | missing | not_implemented | core |
| M 音乐-舞蹈适配 | M.semantic 风格、情绪与跨模态语义 | M-DS Dance-music semantic retrieval score | missing | not_implemented | supplementary |
| X 机器人可执行性与跟踪保真 | X.reference 跟踪前 reference 可执行性 | X-DYN Reference dynamic-envelope violation | missing | not_implemented | hard_gate |
| X 机器人可执行性与跟踪保真 | X.reference 跟踪前 reference 可执行性 | X-JLIMIT Reference joint-limit violation | prototype | not_directionally_calibrated | hard_gate |
| X 机器人可执行性与跟踪保真 | X.tracking 关节与关键身体跟踪误差 | T-EMPJPE Mean per-joint position error | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.tracking 关节与关键身体跟踪误差 | T-EMPKPE Mean per-keybody position error | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.tracking 关节与关键身体跟踪误差 | T-RMSE-RAW Raw joint RMSE | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.tracking 关节与关键身体跟踪误差 | T-RMSE-ALIGN Lag-compensated joint RMSE | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.tracking 关节与关键身体跟踪误差 | T-LAG Tracking lag | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.retention 动作与音乐表现力保留 | T-AMP Amplitude retention | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.retention 动作与音乐表现力保留 | T-ENERGY Motion-energy retention | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.retention 动作与音乐表现力保留 | T-BAND Frequency-band power retention | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.retention 动作与音乐表现力保留 | T-CONTACT Contact retention | prototype | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.retention 动作与音乐表现力保留 | E-MUSIC-RET Executed music-expression retention | implemented | not_directionally_calibrated | core |
| X 机器人可执行性与跟踪保真 | X.safety 稳定性与安全 | T-SUCC Execution success rate | implemented | not_directionally_calibrated | hard_gate |
| X 机器人可执行性与跟踪保真 | X.safety 稳定性与安全 | T-TTF Time to fall | implemented | not_directionally_calibrated | core |
| R 在线实时系统 | R.runtime 延迟、deadline 与实时率 | R-DEADLINE Generator deadline miss rate | prototype | not_directionally_calibrated | hard_gate |
| R 在线实时系统 | R.runtime 延迟、deadline 与实时率 | R-LATENCY End-to-end latency decomposition | prototype | not_directionally_calibrated | core |
| R 在线实时系统 | R.runtime 延迟、deadline 与实时率 | R-RTF Realtime factor | prototype | not_directionally_calibrated | hard_gate |
| H 人类感知评价 | H.perception 自然性、舞蹈感、优美度、节奏、风格与情绪 | H-PREFERENCE Blind pairwise preference | manual | human_study_required | core |

## 对 benchmark 的决定

| 模块 | 当前状态 | 后续使用方式 |
|---|---|---|
| D / smoothness | PASS | 报告 jerk、velocity/acceleration、energy，并结合 static ratio、FSR/PFC 和物理稳定性解释 |
| D / liveliness | PASS | 报告 motion energy、static ratio 和重复率；不能把能量越大直接当作越优美 |
| M / rhythm | PASS_WITH_CAVEAT | BAS 与 Beat F1、事件覆盖、impact/速度相关性、lag、tempo 和 phase 一起报告 |
| 其它 M 子模块 | 待校准 | structure 和 semantic 必须分别做 phrase、retrieval、MMDist、style/emotion 评估，不能由 rhythm 代替 |
| X / SONIC | 待完整采集 | reference、tracking、retention、安全四组全部报告，不能只看 success rate 或 RMSE |
| R / online | 待完整采集 | latency、deadline、RTF、stale/fallback 等一起报告，离线 PKL 不算 online 证据 |
| H / perception | 待盲评 | naturalness、aesthetics、smoothness、expressiveness、rhythm、style/emotion 分开评分 |

## 下一步

1. 保持本报告中的 D/M 指标定义不变，扩展到全量 AIST++ 1,408 条和 FineDance 203 条 GT 的 style/tempo 分层。
2. 在同一歌曲、同一音频起点和同一统计窗口上扩展 M2/M3/M4，多 seed 分别计算 M_ref 和 M_exec。
3. 对正式模型结果同时报告原始指标、匹配 GT 分布的条件分数，以及 SONIC retention；不使用单一总分替代分模块结果。
4. 对 FineDance 继续保留 Beat F1，但先检查 beat detector、音频起点和动作事件定义，再决定是否做数据集专属校准。
