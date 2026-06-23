# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.24449472737834008
- RoboPerform BAS: 0.4675838853838644
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.3045258902894707
- Root flat range mean: 0.495361150342509
- Root angular velocity p99: 3.6563686941455025
- Root angular velocity max: 41.988182067871094
- Root up-z p01: 0.9091541134611937
- Root tilt >60deg rate: 0.005031138335885657
- Root height mean: 0.8295077657535285
- Joint position std mean: 0.2952318708390956
- Joint position range mean: 1.1145819478341668
- Joint range violation rate: 0.0022797697628980304
- G1 feature distance: 10.232069969177246
- G1 diversity: 14.312384800589122

## FK Metrics

- FK beat alignment: 0.26286748988763886
- FK RoboPerform BAS: 0.45628849887922013
- Beat F1: 0.22687682146360966
- Beat recall: 0.1882285735400675
- Beat timing mean frames: 0.07219353897761248
- Beat timing std frames: 0.41689292035460657
- Beat density ratio: 0.5487795939309537
- Unmatched motion beat rate: 0.6633498063309737
- Wrist beat F1: 0.22628645507070172
- Foot beat F1: 0.21780398532017894
- Torso beat F1: 0.2227620669245515
- Wrist dominance ratio: 0.9973978549721412
- Foot contact on beat rate: 0.8564655199618266
- Near support on beat rate: 0.9789957966144797
- No near support rate: 0.021794793261868298
- Foot high-lift rate: 0.023045431342521697
- Wrist jerk mean: 1594.926628198737
- Foot jerk mean: 1480.0543749873234
- Foot sliding: 0.686467742317486
- Ground penetration: 0.2168484628200531

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.24449472737834008,
  "G1 Beat Density Ratio": 0.5487795939309537,
  "G1 Beat F1": 0.22687682146360966,
  "G1 Beat Match": 0.0,
  "G1 Beat Recall": 0.1882285735400675,
  "G1 FK Beat Align.": 0.26286748988763886,
  "G1 FK RoboPerform BAS": 0.45628849887922013,
  "G1 Foot Beat F1": 0.21780398532017894,
  "G1 Foot Contact On Beat": 0.8564655199618266,
  "G1 Foot High Lift Rate": 0.023045431342521697,
  "G1 Foot Jerk Mean": 1480.0543749873234,
  "G1 Foot Sliding": 0.686467742317486,
  "G1 Ground Penetration": 0.2168484628200531,
  "G1 Near Support On Beat": 0.9789957966144797,
  "G1 No Near Support Rate": 0.021794793261868298,
  "G1 RoboPerform BAS": 0.4675838853838644,
  "G1 Torso Beat F1": 0.2227620669245515,
  "G1 Unmatched Motion Beat Rate": 0.6633498063309737,
  "G1 Wrist Beat F1": 0.22628645507070172,
  "G1 Wrist Dominance": 0.9973978549721412,
  "G1 Wrist Jerk Mean": 1594.926628198737,
  "G1Dist": 10.232069969177246,
  "G1Div": 14.312384800589122,
  "Joint Pos. Range": 1.1145819478341668,
  "Joint Pos. Std": 0.2952318708390956,
  "Joint Range Viol.": 0.0022797697628980304,
  "Method": "G1 train-1000",
  "Root Ang. Vel. Max": 41.988182067871094,
  "Root Ang. Vel. P99": 3.6563686941455025,
  "Root Drift": 0.3045258902894707,
  "Root Flat Range": 0.495361150342509,
  "Root Height Max": 0.877431435889891,
  "Root Height Min": 0.7601126676794729,
  "Root Tilt >60 Rate": 0.005031138335885657,
  "Root Up Z P01": 0.9091541134611937
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
