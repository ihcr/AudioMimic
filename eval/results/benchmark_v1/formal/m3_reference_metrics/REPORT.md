# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 6
- Finite motion rate: 1.0
- Beat alignment: 0.25774732824236063
- RoboPerform BAS: 0.42077767285400053
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 1.5351567566394806
- Root flat range mean: 2.468096931775411
- Root height mean: 0.8436762789885203
- Joint position std mean: 0.28422042230765027
- Joint position range mean: 1.6984564065933228
- Joint range violation rate: 0.010309706257982118
- G1 feature distance: 1228.7269287109375
- G1 diversity: 1086.3083516438803

## FK Metrics

- FK beat alignment: 0.2708164230724462
- FK RoboPerform BAS: 0.4448632517847247
- Beat F1: 0.22747532233353385
- Beat precision: 0.3037157047726504
- Beat recall: 0.18656410256410258
- Beat offbeat false-positive rate: 0.6962842952273496
- Beat timing mean frames: -0.04927248677248675
- Beat timing std frames: 1.4480134050051372
- Motion beat density: 1.0166666666666666
- Wrist beat F1: 0.22924553095017344
- Foot beat F1: 0.21383141572805106
- Torso beat F1: 0.24086892798139126
- Foot contact on beat: 0.6321367521367521
- Near support on beat: 0.9965299145299146
- No-near-support rate: 0.01074074074074074
- Foot high-lift rate: 0.011481481481481481
- Wrist jerk mean: 293.8726012087603
- Foot sliding: 0.5053563217322031
- Ground penetration: 0.04330946132540703

## Table Row

```json
{
  "Files": 6,
  "G1 Beat Align.": 0.25774732824236063,
  "G1 Beat F1": 0.22747532233353385,
  "G1 Beat Match": 0.0,
  "G1 Beat Offbeat Rate": 0.6962842952273496,
  "G1 Beat Precision": 0.3037157047726504,
  "G1 Beat Recall": 0.18656410256410258,
  "G1 Beat Timing Mean": -0.04927248677248675,
  "G1 Beat Timing Std": 1.4480134050051372,
  "G1 FK Beat Align.": 0.2708164230724462,
  "G1 FK RoboPerform BAS": 0.4448632517847247,
  "G1 Foot Beat F1": 0.21383141572805106,
  "G1 Foot Contact On Beat": 0.6321367521367521,
  "G1 Foot High Lift": 0.011481481481481481,
  "G1 Foot Sliding": 0.5053563217322031,
  "G1 Motion Beat Density": 1.0166666666666666,
  "G1 Near Support On Beat": 0.9965299145299146,
  "G1 No Near Support": 0.01074074074074074,
  "G1 RoboPerform BAS": 0.42077767285400053,
  "G1 Torso Beat F1": 0.24086892798139126,
  "G1 Wrist Beat F1": 0.22924553095017344,
  "G1 Wrist Jerk": 293.8726012087603,
  "G1Dist": 1228.7269287109375,
  "G1Div": 1086.3083516438803,
  "Joint Pos. Range": 1.6984564065933228,
  "Joint Pos. Std": 0.28422042230765027,
  "Joint Range Viol.": 0.010309706257982118,
  "Method": "M3_M_ref",
  "Root Drift": 1.5351567566394806,
  "Root Flat Range": 2.468096931775411,
  "Root Height Max": 0.8944784800211588,
  "Root Height Min": 0.7332343757152557
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
