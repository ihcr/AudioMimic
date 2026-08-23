# FineDance-G1 Root Height Audit v1

这是一项已有 G1 资产/retargeting/坐标约定诊断，不是舞蹈质量评分，也不修改或删除 benchmark 数据。
`root_height_min_m` 只看 root_pos 的 z 轴；脚部穿透和接触使用 FK 后的地面相对高度单独统计。

## 判定规则

- `valid_root_z`: root z 全部不小于 0。
- `transient_root_drop_feet_near_ground`: root z 有负值，但中位数 >= 0.60 m、负值比例 <= 0.05、脚部穿透比例 <= 0.02 且最低脚高 >= -0.12 m。
- `root_drop_with_physical_warning`: 负 root z 同时伴随较低根部中位高度或明显脚部物理异常。
- 以上阈值仅用于定位问题，不能作为训练集删选或论文质量分数。

## 总体

| diagnostic label | clips |
|---|---:|
| transient_root_drop_feet_near_ground | 17 |
| valid_root_z | 186 |

## 风格分布

| style | clips | labels |
|---|---:|---|
| Breaking | 13 | transient_root_drop_feet_near_ground: 13 |
| Chinese | 4 | valid_root_z: 4 |
| Choreography | 6 | valid_root_z: 6 |
| Classic | 48 | transient_root_drop_feet_near_ground: 2, valid_root_z: 46 |
| Dai | 10 | valid_root_z: 10 |
| DunHuang | 2 | valid_root_z: 2 |
| Folk | 28 | valid_root_z: 28 |
| HanTang | 11 | valid_root_z: 11 |
| Hiphop | 18 | valid_root_z: 18 |
| Jazz | 21 | transient_root_drop_feet_near_ground: 1, valid_root_z: 20 |
| Korean | 35 | transient_root_drop_feet_near_ground: 1, valid_root_z: 34 |
| Kun | 1 | valid_root_z: 1 |
| Locking | 3 | valid_root_z: 3 |
| Miao | 10 | valid_root_z: 10 |
| Mix | 45 | transient_root_drop_feet_near_ground: 1, valid_root_z: 44 |
| Popping | 21 | valid_root_z: 21 |
| ShenYun | 34 | transient_root_drop_feet_near_ground: 2, valid_root_z: 32 |
| Street | 80 | transient_root_drop_feet_near_ground: 14, valid_root_z: 66 |
| Urban | 6 | valid_root_z: 6 |
| Wei | 8 | valid_root_z: 8 |
| jiewu | 2 | valid_root_z: 2 |

## 解释

当前应把负 root z 视为已有 G1 资产的 root/ground convention 待核查信号，不能直接解释为 Breaking 或其他风格的动作质量下降。
若 root z 出现负值但脚部穿透很低，优先检查 root 轨迹的轴定义、全局平移与地面参考；若两者同时异常，再检查上游 retargeting 的姿态/地面约束。
在问题修正前，D5 的 `root_height_min` 只作为诊断字段，generator 与 SONIC execution 的正式比较应优先使用 foot penetration、FSR/PFC proxy 及其他冻结指标。

## 文件

- `root_height_audit.csv`: 每条 FineDance 序列的诊断记录。
- `root_height_audit.json`: 阈值、原始路径、完整记录和汇总。
