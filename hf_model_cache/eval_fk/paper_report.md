# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 186
- Finite motion rate: 1.0
- Beat alignment: 0.5409093742377377
- Designated beat precision: 0.4469606674612634
- Designated beat recall: 0.25100401606425704
- Root drift mean: 0.13193057093178473
- Root height mean: 0.882642497298538
- Joint range violation rate: 0.00046224199728092945
- G1 feature distance: 6.617173194885254
- G1 diversity: 8.247352061481955

## FK Metrics

- FK beat alignment: 0.5208861469852919
- Beat F1: 0.2909873043886327
- Beat timing mean frames: 0.027777777777777776
- Beat timing std frames: 0.511774942278862
- Foot sliding: 0.524482373428601
- Ground penetration: 0.06008222699165344

## Table Row

```json
{
  "Files": 186,
  "G1 Beat Align.": 0.5409093742377377,
  "G1 Beat F1": 0.2909873043886327,
  "G1 Beat Match": 0.4469606674612634,
  "G1 FK Beat Align.": 0.5208861469852919,
  "G1 Foot Sliding": 0.524482373428601,
  "G1Dist": 6.617173194885254,
  "G1Div": 8.247352061481955,
  "Joint Range Viol.": 0.00046224199728092945,
  "Method": "G1 train-2000",
  "Root Drift": 0.13193057093178473,
  "Root Height Max": 0.898532207614632,
  "Root Height Min": 0.8593541524743521
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
