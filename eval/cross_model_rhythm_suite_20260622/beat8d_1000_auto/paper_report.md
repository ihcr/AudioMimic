# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.21686918856503312
- RoboPerform BAS: 0.43830987880414085
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.2893826382037874
- Root flat range mean: 0.48925692845951724
- Root angular velocity p99: 3.8288319076772512
- Root angular velocity max: 72.14185333251953
- Root up-z p01: 0.9020038265035492
- Root tilt >60deg rate: 0.009153649821337418
- Root height mean: 0.823280376077246
- Joint position std mean: 0.2902083978627396
- Joint position range mean: 1.1003409973101812
- Joint range violation rate: 0.003217897942299907
- G1 feature distance: 9.117654800415039
- G1 diversity: 14.488784340523273

## FK Metrics

- FK beat alignment: 0.229761477149506
- FK RoboPerform BAS: 0.4295888447204291
- Beat F1: 0.1934135128789019
- Beat recall: 0.15505894345399085
- Beat timing mean frames: 0.028703420112302196
- Beat timing std frames: 0.3332928561050713
- Beat density ratio: 0.5178479806494173
- Unmatched motion beat rate: 0.7041047416843595
- Wrist beat F1: 0.1922859369593088
- Foot beat F1: 0.19515176425735434
- Torso beat F1: 0.2060750186498379
- Wrist dominance ratio: 0.9942178372386694
- Foot contact on beat rate: 0.8315835763441835
- Near support on beat rate: 0.9709840482811387
- No near support rate: 0.030764675855028074
- Foot high-lift rate: 0.027137314956610516
- Wrist jerk mean: 1180.0095568259728
- Foot jerk mean: 1333.563646990527
- Foot sliding: 0.6480405568346312
- Ground penetration: 0.1579919457435608

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.21686918856503312,
  "G1 Beat Density Ratio": 0.5178479806494173,
  "G1 Beat F1": 0.1934135128789019,
  "G1 Beat Match": 0.0,
  "G1 Beat Recall": 0.15505894345399085,
  "G1 FK Beat Align.": 0.229761477149506,
  "G1 FK RoboPerform BAS": 0.4295888447204291,
  "G1 Foot Beat F1": 0.19515176425735434,
  "G1 Foot Contact On Beat": 0.8315835763441835,
  "G1 Foot High Lift Rate": 0.027137314956610516,
  "G1 Foot Jerk Mean": 1333.563646990527,
  "G1 Foot Sliding": 0.6480405568346312,
  "G1 Ground Penetration": 0.1579919457435608,
  "G1 Near Support On Beat": 0.9709840482811387,
  "G1 No Near Support Rate": 0.030764675855028074,
  "G1 RoboPerform BAS": 0.43830987880414085,
  "G1 Torso Beat F1": 0.2060750186498379,
  "G1 Unmatched Motion Beat Rate": 0.7041047416843595,
  "G1 Wrist Beat F1": 0.1922859369593088,
  "G1 Wrist Dominance": 0.9942178372386694,
  "G1 Wrist Jerk Mean": 1180.0095568259728,
  "G1Dist": 9.117654800415039,
  "G1Div": 14.488784340523273,
  "Joint Pos. Range": 1.1003409973101812,
  "Joint Pos. Std": 0.2902083978627396,
  "Joint Range Viol.": 0.003217897942299907,
  "Method": "G1 train-1000",
  "Root Ang. Vel. Max": 72.14185333251953,
  "Root Ang. Vel. P99": 3.8288319076772512,
  "Root Drift": 0.2893826382037874,
  "Root Flat Range": 0.48925692845951724,
  "Root Height Max": 0.8743466376619353,
  "Root Height Min": 0.7517618255348874,
  "Root Tilt >60 Rate": 0.009153649821337418,
  "Root Up Z P01": 0.9020038265035492
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
