# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.24348203285873515
- RoboPerform BAS: 0.4723610163185976
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.2810404571422175
- Root flat range mean: 0.6249156354922248
- Root angular velocity p99: 5.907132820766525
- Root angular velocity max: 93.90775299072266
- Root up-z p01: 0.9029918275821446
- Root tilt >60deg rate: 0.00044512506380806527
- Root height mean: 0.8177193117178235
- Joint position std mean: 0.3120745927506165
- Joint position range mean: 1.1875690442405835
- Joint range violation rate: 0.004095404059073067
- G1 feature distance: 5.7821807861328125
- G1 diversity: 14.092856849767085

## FK Metrics

- FK beat alignment: 0.24294892768253504
- FK RoboPerform BAS: 0.45169106934440323
- Beat F1: 0.21063100003935523
- Beat recall: 0.1687402186957585
- Beat timing mean frames: 0.06784620433165608
- Beat timing std frames: 0.3759175297129574
- Beat density ratio: 0.5148794253463315
- Unmatched motion beat rate: 0.6754217382020072
- Wrist beat F1: 0.2153191377428469
- Foot beat F1: 0.20227171879948289
- Torso beat F1: 0.2070218734307926
- Wrist dominance ratio: 1.0222575865025363
- Foot contact on beat rate: 0.5642938084396061
- Near support on beat rate: 0.9619514119987052
- No near support rate: 0.057715160796324654
- Foot high-lift rate: 0.0382052067381317
- Wrist jerk mean: 1132.1501097317641
- Foot jerk mean: 1049.5454876924755
- Foot sliding: 0.7548638079606373
- Ground penetration: 0.051726266741752625

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.24348203285873515,
  "G1 Beat Density Ratio": 0.5148794253463315,
  "G1 Beat F1": 0.21063100003935523,
  "G1 Beat Match": 0.0,
  "G1 Beat Recall": 0.1687402186957585,
  "G1 FK Beat Align.": 0.24294892768253504,
  "G1 FK RoboPerform BAS": 0.45169106934440323,
  "G1 Foot Beat F1": 0.20227171879948289,
  "G1 Foot Contact On Beat": 0.5642938084396061,
  "G1 Foot High Lift Rate": 0.0382052067381317,
  "G1 Foot Jerk Mean": 1049.5454876924755,
  "G1 Foot Sliding": 0.7548638079606373,
  "G1 Ground Penetration": 0.051726266741752625,
  "G1 Near Support On Beat": 0.9619514119987052,
  "G1 No Near Support Rate": 0.057715160796324654,
  "G1 RoboPerform BAS": 0.4723610163185976,
  "G1 Torso Beat F1": 0.2070218734307926,
  "G1 Unmatched Motion Beat Rate": 0.6754217382020072,
  "G1 Wrist Beat F1": 0.2153191377428469,
  "G1 Wrist Dominance": 1.0222575865025363,
  "G1 Wrist Jerk Mean": 1132.1501097317641,
  "G1Dist": 5.7821807861328125,
  "G1Div": 14.092856849767085,
  "Joint Pos. Range": 1.1875690442405835,
  "Joint Pos. Std": 0.3120745927506165,
  "Joint Range Viol.": 0.004095404059073067,
  "Method": "G1 train-1500",
  "Root Ang. Vel. Max": 93.90775299072266,
  "Root Ang. Vel. P99": 5.907132820766525,
  "Root Drift": 0.2810404571422175,
  "Root Flat Range": 0.6249156354922248,
  "Root Height Max": 0.8719244379361958,
  "Root Height Min": 0.7375523641226303,
  "Root Tilt >60 Rate": 0.00044512506380806527,
  "Root Up Z P01": 0.9029918275821446
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
