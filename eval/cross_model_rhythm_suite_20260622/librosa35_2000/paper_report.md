# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.24128070540031524
- RoboPerform BAS: 0.4730023221456952
- Designated beat precision: 0.33597014925373136
- Designated beat recall: 0.16499303672212856
- Root drift mean: 0.2021512396271365
- Root flat range mean: 0.3631996290698351
- Root angular velocity p99: 3.0653894927888725
- Root angular velocity max: 85.12373352050781
- Root up-z p01: 0.9589674761047042
- Root tilt >60deg rate: 0.0008024502297090352
- Root height mean: 0.8476101209157094
- Joint position std mean: 0.21503756182984224
- Joint position range mean: 0.8423036350818724
- Joint range violation rate: 0.0005869285877734946
- G1 feature distance: 9.254408836364746
- G1 diversity: 11.366099631642493

## FK Metrics

- FK beat alignment: 0.2544357456958737
- FK RoboPerform BAS: 0.4503674442887729
- Beat F1: 0.2138846395721516
- Beat recall: 0.17570365511005043
- Beat timing mean frames: 0.018586013272077588
- Beat timing std frames: 0.38277123239102445
- Beat density ratio: 0.534999633511691
- Unmatched motion beat rate: 0.683381285107549
- Wrist beat F1: 0.2156299614360263
- Foot beat F1: 0.20417288005708595
- Torso beat F1: 0.21793534326472316
- Wrist dominance ratio: 1.0081601084929053
- Foot contact on beat rate: 0.942613907701963
- Near support on beat rate: 0.9966385911179173
- No near support rate: 0.0009964267483409904
- Foot high-lift rate: 0.0032240939254721797
- Wrist jerk mean: 968.6031377858257
- Foot jerk mean: 1128.3351601009242
- Foot sliding: 0.5305995295642894
- Ground penetration: 0.03516579046845436

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.24128070540031524,
  "G1 Beat Density Ratio": 0.534999633511691,
  "G1 Beat F1": 0.2138846395721516,
  "G1 Beat Match": 0.33597014925373136,
  "G1 Beat Recall": 0.17570365511005043,
  "G1 FK Beat Align.": 0.2544357456958737,
  "G1 FK RoboPerform BAS": 0.4503674442887729,
  "G1 Foot Beat F1": 0.20417288005708595,
  "G1 Foot Contact On Beat": 0.942613907701963,
  "G1 Foot High Lift Rate": 0.0032240939254721797,
  "G1 Foot Jerk Mean": 1128.3351601009242,
  "G1 Foot Sliding": 0.5305995295642894,
  "G1 Ground Penetration": 0.03516579046845436,
  "G1 Near Support On Beat": 0.9966385911179173,
  "G1 No Near Support Rate": 0.0009964267483409904,
  "G1 RoboPerform BAS": 0.4730023221456952,
  "G1 Torso Beat F1": 0.21793534326472316,
  "G1 Unmatched Motion Beat Rate": 0.683381285107549,
  "G1 Wrist Beat F1": 0.2156299614360263,
  "G1 Wrist Dominance": 1.0081601084929053,
  "G1 Wrist Jerk Mean": 968.6031377858257,
  "G1Dist": 9.254408836364746,
  "G1Div": 11.366099631642493,
  "Joint Pos. Range": 0.8423036350818724,
  "Joint Pos. Std": 0.21503756182984224,
  "Joint Range Viol.": 0.0005869285877734946,
  "Method": "G1 train-2000",
  "Root Ang. Vel. Max": 85.12373352050781,
  "Root Ang. Vel. P99": 3.0653894927888725,
  "Root Drift": 0.2021512396271365,
  "Root Flat Range": 0.3631996290698351,
  "Root Height Max": 0.8808590369604266,
  "Root Height Min": 0.7962070924856031,
  "Root Tilt >60 Rate": 0.0008024502297090352,
  "Root Up Z P01": 0.9589674761047042
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
