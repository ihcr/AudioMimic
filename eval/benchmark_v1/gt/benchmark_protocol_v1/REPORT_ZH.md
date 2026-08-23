# GT Benchmark Protocol v1（中文）

本协议把‘指标校准’、‘GT 分布估计’和‘论文最终测试’分开，避免把全量数据混作一个分数。

## 当前数据状态

| 数据角色 | AIST++ | FineDance | 总数 | 用途 |
|---|---:|---:|---:|---|
| calibration_38 | 20 | 18 | 38 | 指标方向和 corruption 校准 |
| full_reference_1611 | 1408 | 203 | 1611 | 条件化 GT 分布和风格/tempo 分析 |
| final_test | 待冻结 | 待冻结 | 待冻结 | M2/M3/M_exec 最终论文比较 |

## 使用规则

1. `calibration_38` 可以用于确定指标方向、corruption 严重程度和实现错误。
2. `full_reference_1611` 只用于估计 dataset/style/tempo 条件下的 GT reference range，不能用于挑选最佳 checkpoint。
3. `final_test` 必须在模型选择、阈值和特征方案冻结后单独划分；在它冻结前，不能声称论文最终结果。
4. 生成动作和 SONIC execution 都必须与相同条件的 GT reference 比较，不能把不同风格或不同 tempo 的 GT 合并成一个理想分数。

## 当前 benchmark 结论

当前 benchmark 已经可以做诊断性 motion-generation 评估，但仍处于 calibration/reference 阶段。
jerk、低通能量、static ratio 的方向校准较稳定；FineDance 的 Beat F1 对 freeze corruption 未表现出稳定下降，
因此 Beat F1 必须和 BAS、coverage、impact correlation、lag、tempo、phase 联合报告，不能单独作为质量 gate。
repeat similarity、FID/Div、retrieval 和人类审美评分仍需独立校准。

## 生成模型评估入口

模型结果应报告三层：`M_ref`、`M_exec`、`M_exec / M_ref retention`。每一层同时报告动作质量、
音乐适配和运行/跟踪指标；‘优美度’和风格真实性需要人工盲评，不从 BAS 反推。

机器可读协议：`protocol.json`。
