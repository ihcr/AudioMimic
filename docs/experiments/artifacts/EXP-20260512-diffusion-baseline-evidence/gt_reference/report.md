# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.2129773536208284
- RoboPerform BAS: 0.4090914675013971
- Designated beat precision: 0.0
- Designated beat recall: 0.0
- Root drift mean: 0.31531355906849096
- Root height mean: 0.8042436706663821
- Joint range violation rate: 0.001815071024977557
- G1 feature distance: 0.0
- G1 diversity: 15.488929744619554

## FK Metrics

- FK beat alignment: 0.21595194600273604
- FK RoboPerform BAS: 0.39999880713753083
- Beat F1: 0.17699683968564853
- Beat timing mean frames: 0.023466054109239416
- Beat timing std frames: 0.2837533329201329
- Foot sliding: 0.21758346407001075
- Ground penetration: 0.039205025881528854

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.2129773536208284,
  "G1 Beat F1": 0.17699683968564853,
  "G1 Beat Match": 0.0,
  "G1 FK Beat Align.": 0.21595194600273604,
  "G1 FK RoboPerform BAS": 0.39999880713753083,
  "G1 Foot Sliding": 0.21758346407001075,
  "G1 RoboPerform BAS": 0.4090914675013971,
  "G1Dist": 0.0,
  "G1Div": 15.488929744619554,
  "Joint Range Viol.": 0.001815071024977557,
  "Method": "G1 ground_truth_with_audio",
  "Root Drift": 0.31531355906849096,
  "Root Height Max": 0.8577975434329572,
  "Root Height Min": 0.7253741872666624
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
