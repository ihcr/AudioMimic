# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.20717977311847136
- RoboPerform BAS: 0.4209757758373147
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.2709437547146042
- Root flat range mean: 0.4456017358630759
- Root angular velocity p99: 3.4537138793738675
- Root angular velocity max: 72.26371002197266
- Root up-z p01: 0.8747431028663291
- Root tilt >60deg rate: 0.017270035732516592
- Root height mean: 0.8148932359970045
- Joint position std mean: 0.2949353406990195
- Joint position range mean: 1.0516371532361868
- Joint range violation rate: 0.01168386404041471
- G1 feature distance: 9.19995403289795
- G1 diversity: 20.536898664086255

## FK Metrics

- FK beat alignment: 0.23110010256837643
- FK RoboPerform BAS: 0.4198832819174301
- Beat F1: 0.19131768856998568
- Beat recall: 0.1545532187221651
- Beat timing mean frames: 0.01850944359367024
- Beat timing std frames: 0.33458492643617743
- Beat density ratio: 0.5340834127391336
- Unmatched motion beat rate: 0.7116585466273245
- Wrist beat F1: 0.1862834563099666
- Foot beat F1: 0.19609466939825268
- Torso beat F1: 0.20535028719421058
- Wrist dominance ratio: 0.9736865299928735
- Foot contact on beat rate: 0.8517343808471682
- Near support on beat rate: 0.9776764706282318
- No near support rate: 0.023454823889739664
- Foot high-lift rate: 0.030365492598264425
- Wrist jerk mean: 1171.8644088188478
- Foot jerk mean: 1289.0614862736306
- Foot sliding: 0.5964300314663334
- Ground penetration: 0.0802772045135498

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.20717977311847136,
  "G1 Beat Density Ratio": 0.5340834127391336,
  "G1 Beat F1": 0.19131768856998568,
  "G1 Beat Match": 0.0,
  "G1 Beat Recall": 0.1545532187221651,
  "G1 FK Beat Align.": 0.23110010256837643,
  "G1 FK RoboPerform BAS": 0.4198832819174301,
  "G1 Foot Beat F1": 0.19609466939825268,
  "G1 Foot Contact On Beat": 0.8517343808471682,
  "G1 Foot High Lift Rate": 0.030365492598264425,
  "G1 Foot Jerk Mean": 1289.0614862736306,
  "G1 Foot Sliding": 0.5964300314663334,
  "G1 Ground Penetration": 0.0802772045135498,
  "G1 Near Support On Beat": 0.9776764706282318,
  "G1 No Near Support Rate": 0.023454823889739664,
  "G1 RoboPerform BAS": 0.4209757758373147,
  "G1 Torso Beat F1": 0.20535028719421058,
  "G1 Unmatched Motion Beat Rate": 0.7116585466273245,
  "G1 Wrist Beat F1": 0.1862834563099666,
  "G1 Wrist Dominance": 0.9736865299928735,
  "G1 Wrist Jerk Mean": 1171.8644088188478,
  "G1Dist": 9.19995403289795,
  "G1Div": 20.536898664086255,
  "Joint Pos. Range": 1.0516371532361868,
  "Joint Pos. Std": 0.2949353406990195,
  "Joint Range Viol.": 0.01168386404041471,
  "Method": "G1 train-1000",
  "Root Ang. Vel. Max": 72.26371002197266,
  "Root Ang. Vel. P99": 3.4537138793738675,
  "Root Drift": 0.2709437547146042,
  "Root Flat Range": 0.4456017358630759,
  "Root Height Max": 0.8665517081893056,
  "Root Height Min": 0.7471410331942403,
  "Root Tilt >60 Rate": 0.017270035732516592,
  "Root Up Z P01": 0.8747431028663291
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
