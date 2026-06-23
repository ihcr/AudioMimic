# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.22370973925600682
- RoboPerform BAS: 0.4321492173346528
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.27261551888759433
- Root flat range mean: 0.4480509028979684
- Root angular velocity p99: 3.1391978486150216
- Root angular velocity max: 93.32207489013672
- Root up-z p01: 0.9520320853170173
- Root tilt >60deg rate: 0.0008677896886166411
- Root height mean: 0.8359728917411057
- Joint position std mean: 0.2329195534334066
- Joint position range mean: 0.9085598788619224
- Joint range violation rate: 0.0010800725211666754
- G1 feature distance: 8.911333084106445
- G1 diversity: 12.844476913495592

## FK Metrics

- FK beat alignment: 0.23770586626231816
- FK RoboPerform BAS: 0.42452843904726517
- Beat F1: 0.19792435487917806
- Beat recall: 0.16067114227503998
- Beat timing mean frames: 0.013654925982644206
- Beat timing std frames: 0.3606565590383083
- Beat density ratio: 0.5413765300886902
- Unmatched motion beat rate: 0.7050500947738966
- Wrist beat F1: 0.19169848462113556
- Foot beat F1: 0.2028538391835724
- Torso beat F1: 0.21335772614862136
- Wrist dominance ratio: 0.9685441932507849
- Foot contact on beat rate: 0.9292470083590254
- Near support on beat rate: 0.9962670348198679
- No near support rate: 0.0014088820826952525
- Foot high-lift rate: 0.004845329249617151
- Wrist jerk mean: 1058.150017735172
- Foot jerk mean: 1173.285416120464
- Foot sliding: 0.5524556990878323
- Ground penetration: 0.04081939160823822

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.22370973925600682,
  "G1 Beat Density Ratio": 0.5413765300886902,
  "G1 Beat F1": 0.19792435487917806,
  "G1 Beat Match": 0.0,
  "G1 Beat Recall": 0.16067114227503998,
  "G1 FK Beat Align.": 0.23770586626231816,
  "G1 FK RoboPerform BAS": 0.42452843904726517,
  "G1 Foot Beat F1": 0.2028538391835724,
  "G1 Foot Contact On Beat": 0.9292470083590254,
  "G1 Foot High Lift Rate": 0.004845329249617151,
  "G1 Foot Jerk Mean": 1173.285416120464,
  "G1 Foot Sliding": 0.5524556990878323,
  "G1 Ground Penetration": 0.04081939160823822,
  "G1 Near Support On Beat": 0.9962670348198679,
  "G1 No Near Support Rate": 0.0014088820826952525,
  "G1 RoboPerform BAS": 0.4321492173346528,
  "G1 Torso Beat F1": 0.21335772614862136,
  "G1 Unmatched Motion Beat Rate": 0.7050500947738966,
  "G1 Wrist Beat F1": 0.19169848462113556,
  "G1 Wrist Dominance": 0.9685441932507849,
  "G1 Wrist Jerk Mean": 1058.150017735172,
  "G1Dist": 8.911333084106445,
  "G1Div": 12.844476913495592,
  "Joint Pos. Range": 0.9085598788619224,
  "Joint Pos. Std": 0.2329195534334066,
  "Joint Range Viol.": 0.0010800725211666754,
  "Method": "G1 train-2000",
  "Root Ang. Vel. Max": 93.32207489013672,
  "Root Ang. Vel. P99": 3.1391978486150216,
  "Root Drift": 0.27261551888759433,
  "Root Flat Range": 0.4480509028979684,
  "Root Height Max": 0.8714709506465684,
  "Root Height Min": 0.7862043963092027,
  "Root Tilt >60 Rate": 0.0008677896886166411,
  "Root Up Z P01": 0.9520320853170173
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
