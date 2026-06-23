# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.22473221960169712
- RoboPerform BAS: 0.48111499962484705
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.7158810496045017
- Root flat range mean: 0.96837384079399
- Root angular velocity p99: 2.9523979961795432
- Root angular velocity max: 14.956616401672363
- Root up-z p01: 1.0
- Root tilt >60deg rate: 0.0
- Root height mean: 0.8180525238809688
- Joint position std mean: 0.3454828437266566
- Joint position range mean: 1.2878861544645217
- Joint range violation rate: 0.010760944183344775
- G1 feature distance: 3.990786552429199
- G1 diversity: 16.560869664641054

## FK Metrics

- FK beat alignment: 0.2604100884879021
- FK RoboPerform BAS: 0.46927837639360204
- Beat F1: 0.2339781314717912
- Beat recall: 0.19029774192285628
- Beat timing mean frames: 0.07856632392620141
- Beat timing std frames: 0.4176427335961854
- Beat density ratio: 0.5279997068093528
- Unmatched motion beat rate: 0.6440619143471923
- Wrist beat F1: 0.26183300262392895
- Foot beat F1: 0.21415925929930069
- Torso beat F1: 0.19049256938756534
- Wrist dominance ratio: 1.119288789371167
- Foot contact on beat rate: 0.4196492295409297
- Near support on beat rate: 0.8727467035777572
- No near support rate: 0.17900357325165903
- Foot high-lift rate: 0.11118121490556405
- Wrist jerk mean: 1974.1318133131567
- Foot jerk mean: 1010.9372719744946
- Foot sliding: 0.8260536285992969
- Ground penetration: 0.0756852775812149

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.22473221960169712,
  "G1 Beat Density Ratio": 0.5279997068093528,
  "G1 Beat F1": 0.2339781314717912,
  "G1 Beat Match": 0.0,
  "G1 Beat Recall": 0.19029774192285628,
  "G1 FK Beat Align.": 0.2604100884879021,
  "G1 FK RoboPerform BAS": 0.46927837639360204,
  "G1 Foot Beat F1": 0.21415925929930069,
  "G1 Foot Contact On Beat": 0.4196492295409297,
  "G1 Foot High Lift Rate": 0.11118121490556405,
  "G1 Foot Jerk Mean": 1010.9372719744946,
  "G1 Foot Sliding": 0.8260536285992969,
  "G1 Ground Penetration": 0.0756852775812149,
  "G1 Near Support On Beat": 0.8727467035777572,
  "G1 No Near Support Rate": 0.17900357325165903,
  "G1 RoboPerform BAS": 0.48111499962484705,
  "G1 Torso Beat F1": 0.19049256938756534,
  "G1 Unmatched Motion Beat Rate": 0.6440619143471923,
  "G1 Wrist Beat F1": 0.26183300262392895,
  "G1 Wrist Dominance": 1.119288789371167,
  "G1 Wrist Jerk Mean": 1974.1318133131567,
  "G1Dist": 3.990786552429199,
  "G1Div": 16.560869664641054,
  "Joint Pos. Range": 1.2878861544645217,
  "Joint Pos. Std": 0.3454828437266566,
  "Joint Range Viol.": 0.010760944183344775,
  "Method": "G1 train-1000",
  "Root Ang. Vel. Max": 14.956616401672363,
  "Root Ang. Vel. P99": 2.9523979961795432,
  "Root Drift": 0.7158810496045017,
  "Root Flat Range": 0.96837384079399,
  "Root Height Max": 0.8767815604874405,
  "Root Height Min": 0.730882758221619,
  "Root Tilt >60 Rate": 0.0,
  "Root Up Z P01": 1.0
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
