# G1 Robot-Native Evaluation Report

This report uses kinematic G1 metrics. SMPL-only physical and diversity metrics are not reported.

- Generated clips: 186
- Finite motion rate: 1.0
- Beat alignment: 0.5728312580651221
- Designated beat precision: 0.4586728754365541
- Designated beat recall: 0.2637215528781794
- Root drift mean: 0.13350983736135305
- Root height mean: 0.8831702445143013
- Joint range violation rate: 0.0001075268817204301
- G1 feature distance: 6.778853416442871
- G1 diversity: 7.543248104254058

## FK Metrics

- FK beat alignment: 0.5515417970600949
- Beat F1: 0.3298751591758183
- Beat timing mean frames: -0.06191756272401432
- Beat timing std frames: 0.5916017327577837
- Foot sliding: 0.587887765619383
- Ground penetration: 0.0808684229850769

## Table Row

```json
{
  "Files": 186,
  "G1 Beat Align.": 0.5728312580651221,
  "G1 Beat F1": 0.3298751591758183,
  "G1 Beat Match": 0.4586728754365541,
  "G1 FK Beat Align.": 0.5515417970600949,
  "G1 Foot Sliding": 0.587887765619383,
  "G1Dist": 6.778853416442871,
  "G1Div": 7.543248104254058,
  "Joint Range Viol.": 0.0001075268817204301,
  "Method": "G1 train-2000",
  "Root Drift": 0.13350983736135305,
  "Root Height Max": 0.8985322079350871,
  "Root Height Min": 0.8614486811622497
}
```

## Deferred Metrics

- Contact quality, link tracking error, and simulator success need a controller rollout.
- PFC, Distg, Distk, Divk, and Divm are SMPL-body metrics and are intentionally omitted here.
