# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 3265
- Finite motion rate: 1.0
- Beat alignment: 0.3045893558201103
- RoboPerform BAS: 0.5416480441754296
- Designated beat precision: 0.41837553820331314
- Designated beat recall: 0.2213000849224118
- Root drift mean: 0.24899306027218698
- Root height mean: 0.8365356394561838
- Joint range violation rate: 0.000808082941683829
- G1 feature distance: 8.629511833190918
- G1 diversity: 12.234703898705753

## FK Metrics

- FK beat alignment: 0.31924628008581024
- FK RoboPerform BAS: 0.5141870079510646
- Beat F1: 0.2910016355141776
- Beat timing mean frames: 0.025601254284255824
- Beat timing std frames: 0.5038724830650808
- Foot sliding: 0.5476596917894314
- Ground penetration: 0.040557220578193665

## Table Row

```json
{
  "Files": 3265,
  "G1 Beat Align.": 0.3045893558201103,
  "G1 Beat F1": 0.2910016355141776,
  "G1 Beat Match": 0.41837553820331314,
  "G1 FK Beat Align.": 0.31924628008581024,
  "G1 FK RoboPerform BAS": 0.5141870079510646,
  "G1 Foot Sliding": 0.5476596917894314,
  "G1 RoboPerform BAS": 0.5416480441754296,
  "G1Dist": 8.629511833190918,
  "G1Div": 12.234703898705753,
  "Joint Range Viol.": 0.000808082941683829,
  "Method": "G1 train-1000",
  "Root Drift": 0.24899306027218698,
  "Root Height Max": 0.8704405633997954,
  "Root Height Min": 0.7810490296682939
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
