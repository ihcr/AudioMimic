# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.23367977350899044
- RoboPerform BAS: 0.4643282207316514
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.06564324937673831
- Root flat range mean: 0.22818774061539934
- Root angular velocity p99: 5.55330718451584
- Root angular velocity max: 93.67179870605469
- Root up-z p01: 0.8633080905414833
- Root tilt >60deg rate: 0.002636038795303726
- Root height mean: 0.8255621249332544
- Joint position std mean: 0.3472447294410986
- Joint position range mean: 1.2944760343519142
- Joint range violation rate: 0.012899684920173913
- G1 feature distance: 6.055990219116211
- G1 diversity: 18.483817323426507

## FK Metrics

- FK beat alignment: 0.22856655753212665
- FK RoboPerform BAS: 0.4567924463783697
- Beat F1: 0.2049890106126372
- Beat recall: 0.16153603500990413
- Beat timing mean frames: -0.013482826514985785
- Beat timing std frames: 0.3453495480930276
- Beat density ratio: 0.4761416110826065
- Unmatched motion beat rate: 0.6654094827586207
- Wrist beat F1: 0.2100902210392328
- Foot beat F1: 0.20283466033768494
- Torso beat F1: 0.20079566034471144
- Wrist dominance ratio: 1.024885287320281
- Foot contact on beat rate: 0.39012975386391247
- Near support on beat rate: 0.8623213395168624
- No near support rate: 0.18426544155181215
- Foot high-lift rate: 0.07553854007146503
- Wrist jerk mean: 663.9920328761849
- Foot jerk mean: 889.6515329480956
- Foot sliding: 0.7373051720980845
- Ground penetration: 0.048324037343263626

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.23367977350899044,
  "G1 Beat Density Ratio": 0.4761416110826065,
  "G1 Beat F1": 0.2049890106126372,
  "G1 Beat Match": 0.0,
  "G1 Beat Recall": 0.16153603500990413,
  "G1 FK Beat Align.": 0.22856655753212665,
  "G1 FK RoboPerform BAS": 0.4567924463783697,
  "G1 Foot Beat F1": 0.20283466033768494,
  "G1 Foot Contact On Beat": 0.39012975386391247,
  "G1 Foot High Lift Rate": 0.07553854007146503,
  "G1 Foot Jerk Mean": 889.6515329480956,
  "G1 Foot Sliding": 0.7373051720980845,
  "G1 Ground Penetration": 0.048324037343263626,
  "G1 Near Support On Beat": 0.8623213395168624,
  "G1 No Near Support Rate": 0.18426544155181215,
  "G1 RoboPerform BAS": 0.4643282207316514,
  "G1 Torso Beat F1": 0.20079566034471144,
  "G1 Unmatched Motion Beat Rate": 0.6654094827586207,
  "G1 Wrist Beat F1": 0.2100902210392328,
  "G1 Wrist Dominance": 1.024885287320281,
  "G1 Wrist Jerk Mean": 663.9920328761849,
  "G1Dist": 6.055990219116211,
  "G1Div": 18.483817323426507,
  "Joint Pos. Range": 1.2944760343519142,
  "Joint Pos. Std": 0.3472447294410986,
  "Joint Range Viol.": 0.012899684920173913,
  "Method": "G1 train-1000",
  "Root Ang. Vel. Max": 93.67179870605469,
  "Root Ang. Vel. P99": 5.55330718451584,
  "Root Drift": 0.06564324937673831,
  "Root Flat Range": 0.22818774061539934,
  "Root Height Max": 0.8727148575403059,
  "Root Height Min": 0.7648992687418121,
  "Root Tilt >60 Rate": 0.002636038795303726,
  "Root Up Z P01": 0.8633080905414833
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
