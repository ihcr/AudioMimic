# EXP-20260522-gaussian-beat-condition-ablation

Status: finished
Owner: yukun
Created: 2026-05-22
Last Updated: 2026-05-22

## Research Question

Does the pure GaussianBeat checkpoint actually use the 1-D beat condition, or is its larger motion amplitude mostly coming from the unconditional dance prior?

## Hypothesis

If real GaussianBeat, shifted beat, random beat, constant beat, and unconditional/no-beat generation produce similar G1 benchmark scores and amplitude metrics, then the standalone GaussianBeat condition is too weak to control motion. In that case GaussianBeat should be treated as a weak lower-bound prior probe, and the long-term direction should move toward richer and explicitly structured conditioning rather than relying on a single beat scalar.

## Baseline Or Control

- Checkpoint: `runs/train/EXP-20260520-finedance-g1-gaussian-beat_r01_linear/weights/train-1000.pt`.
- Control condition: cached real `data/finedance_g1_fkbeats/test/gaussian_beat_feats`.
- Existing real-control motions may be reused from `eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/motions`, but metrics are re-written under this experiment so the comparison uses the current evaluator and amplitude fields.

## Intervention

Evaluate five inference-only condition variants while keeping the model, seed, split, generation mode, guidance defaults, and evaluator fixed:

1. `real`: original cached GaussianBeat feature.
2. `shift_p10`: GaussianBeat feature delayed by 10 frames with edge padding.
3. `random`: per-clip GaussianBeat feature replaced by a deterministic random permutation of other cached test features.
4. `constant`: all frames set to the train-set mean GaussianBeat value.
5. `no_beat_uncond`: unconditional CFG sample by setting diffusion `guidance_weight=0`; real condition tensor is still passed only for shape/device plumbing.

## Invariant Controls

- Branch/worktree: `/home/tianhup/Desktop/Musics2Dance`.
- Dataset: `data/finedance_g1_fkbeats`, test split, `3265` clips.
- Motion format: G1, 150-frame clips at 30 FPS.
- Feature type: `gaussian_beat`, `feature_fusion=linear`.
- Checkpoint guidance for conditioned variants: checkpoint/default `guidance_weight=2`.
- Random seed: `1234`.
- Batch size: `32` unless memory requires lowering.
- FK metrics: enabled with `third_party/unitree_g1_description/g1_29dof_rev_1_0.xml` and `xyzw` root quaternion order.

## Data And Cache Contract

- Only inference conditions and generated/eval artifacts are touched.
- No training cache, processed data, raw data, or checkpoint is modified.
- `constant` computes its scalar from `data/finedance_g1_fkbeats/train/gaussian_beat_feats` and records it in the manifest.
- `random` records the deterministic permutation seed in the manifest.

## Execution Plan

Command shape:

```bash
.venv311/bin/python -m eval.run_gaussian_beat_condition_ablation \
  --checkpoint runs/train/EXP-20260520-finedance-g1-gaussian-beat_r01_linear/weights/train-1000.pt \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_gaussian_beat_dataset_backups \
  --output_root eval/EXP-20260522-gaussian-beat-condition-ablation \
  --real_motion_dir eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/motions \
  --batch_size 32 \
  --seed 1234 \
  --enable_fk_metrics
```

## Evaluation Plan

For each variant, write:

- `{variant}/metrics.json`
- `{variant}/g1_table.json`
- `{variant}/motion_audit.json`
- `{variant}/paper_report.md`
- generated motions under `{variant}/motions` when generation is required

Then write:

- `comparison_g1_metrics.json`
- `comparison_g1_metrics.md`
- `manifest.json`

Primary decision metrics:

- Rhythm/control: `G1BAS`, `G1FKBAS`, `G1BeatF1`, `G1RoboPerformBAS`, `G1FKRoboPerformBAS`.
- Motion quality: `G1Dist`, `G1Div`, `G1FootSliding`, `G1GroundPenetration`, `RootDriftMean`, `RootSmoothnessJerkMean`, `JointSmoothnessJerkMean`.
- Anti-average signal: `JointPositionStdMean`, `JointPositionRangeMean`, `RootFlatRangeMean`.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-22 | spec | ready | this file | Preparing inference-only ablation script and full-test evaluation. |
| 2026-05-22 | implementation smoke | passed | `eval/run_gaussian_beat_condition_ablation.py`; `tests/test_gaussian_beat_condition_ablation.py`; `/tmp/gb_ablation_smoke/comparison_g1_metrics.md` | Added a reusable runner for real, shifted, random, constant, and unconditional GaussianBeat conditions. Focused tests passed, and a 4-clip smoke generated/evaluated `real` and `shift_p10` successfully. |
| 2026-05-22 | full ablation benchmark | passed | `eval/EXP-20260522-gaussian-beat-condition-ablation/comparison_g1_metrics.md`; per-variant `metrics.json` files | Ran all five `3265`-clip variants with FK metrics enabled. |
| 2026-05-22 | DOF sensitivity analysis | passed | `eval/g1_condition_sensitivity.py`; `eval/EXP-20260522-gaussian-beat-condition-ablation/condition_sensitivity_vs_no_beat_uncond.md`; `eval/EXP-20260522-gaussian-beat-condition-ablation/condition_sensitivity_vs_real.md` | Added paired clip-level DOF/root similarity metrics against `no_beat_uncond` and `real` baselines. |

## Results

Full comparison table: `eval/EXP-20260522-gaussian-beat-condition-ablation/comparison_g1_metrics.md`.

| Variant | G1BAS | G1FKBAS | G1BeatF1 | G1RoboPerformBAS | G1FKRoboPerformBAS | G1Dist | G1Div | JointPositionStdMean | JointPositionRangeMean | RootFlatRangeMean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `real` | 0.2072 | 0.2311 | 0.1913 | 0.4210 | 0.4199 | 9.2000 | 20.5369 | 0.2949 | 1.0516 | 0.4456 |
| `shift_p10` | 0.2079 | 0.2283 | 0.1881 | 0.4230 | 0.4149 | 9.0453 | 18.5562 | 0.2879 | 1.0416 | 0.4464 |
| `random` | 0.2079 | 0.2299 | 0.1896 | 0.4200 | 0.4152 | 9.2062 | 20.7178 | 0.2960 | 1.0558 | 0.4408 |
| `constant` | 0.1940 | 0.2097 | 0.1782 | 0.4316 | 0.4209 | 9.6833 | 11.3504 | 0.2580 | 0.9900 | 0.4481 |
| `no_beat_uncond` | 0.2063 | 0.2171 | 0.1844 | 0.4262 | 0.4196 | 4.4485 | 16.5251 | 0.2836 | 1.0599 | 0.4010 |

Key deltas against `real`:

- `shift_p10` and `random` are effectively tied with `real` on `G1BAS` and close on `G1FKBAS` / `G1BeatF1`. They also preserve nearly the same motion-amplitude statistics.
- `constant` lowers `G1BAS`, `G1FKBAS`, `G1BeatF1`, and especially `G1Div`, but it does not collapse motion-to-music RoboPerform BAS.
- `no_beat_uncond` keeps `G1BAS` near `real` and has similar RoboPerform BAS, while changing quality/diversity statistics. This means the trained generator and dataset prior can already produce music-beat-correlated motion without a faithful beat input.

Paired DOF sensitivity reports:

- Against `no_beat_uncond`: `eval/EXP-20260522-gaussian-beat-condition-ablation/condition_sensitivity_vs_no_beat_uncond.md`.
- Against `real`: `eval/EXP-20260522-gaussian-beat-condition-ablation/condition_sensitivity_vs_real.md`.

| Baseline | Variant | DofRMSE | DofPearson | DofDelta/BaselineStd | ActiveDOFs | DofStdRatio | RootRMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| `no_beat_uncond` | `real` | 0.7073 | 0.4524 | 2.7547 | 28.8735 | 1.1476 | 0.6227 |
| `no_beat_uncond` | `shift_p10` | 0.6648 | 0.4736 | 2.5836 | 28.8772 | 1.1164 | 0.6140 |
| `no_beat_uncond` | `random` | 0.7094 | 0.4527 | 2.7572 | 28.8855 | 1.1530 | 0.6224 |
| `no_beat_uncond` | `constant` | 0.5348 | 0.5477 | 2.0474 | 28.8637 | 0.9943 | 0.4340 |
| `real` | `shift_p10` | 0.7092 | 0.4616 | 2.8066 | 28.7308 | 1.1261 | 0.6256 |
| `real` | `random` | 0.8063 | 0.3778 | 3.2076 | 28.9127 | 1.1829 | 0.7101 |
| `real` | `constant` | 0.6875 | 0.3909 | 2.7016 | 28.9357 | 1.0268 | 0.5474 |
| `real` | `no_beat_uncond` | 0.7073 | 0.4524 | 2.7822 | 28.8735 | 1.1167 | 0.6227 |

Top changed DOFs across these paired comparisons are consistently upper-body joints, especially shoulder pitch/yaw and elbow joints. This says the conditioned branch is changing full-body samples, not just a tiny beat channel, but those changes do not track the exact real beat timing because shifted/random beat conditions produce benchmark and DOF-amplitude patterns close to real.

## Current Conclusion

The standalone 1-D GaussianBeat condition is too weak to be a reliable beat-timing control signal in the current pipeline. The condition/guidance branch is not inert: paired DOF metrics show conditioned samples can differ substantially from `no_beat_uncond`. The problem is more specific and more important: real beats, +10-frame shifted beats, and random per-clip beats produce nearly the same benchmark scores, similar DOF amplitudes, and similar paired-distance profiles. The model reacts to having a GaussianBeat-like condition, but it does not use the exact beat timing strongly enough.

This supports moving toward richer conditioning rather than spending more effort on pure GaussianBeat alone. GaussianBeat remains useful as a lower-bound diagnostic or auxiliary rhythm channel, but should be combined with denser audio features, explicit beat-phase/tempo/bar structure, and stronger condition-use diagnostics.

## Next Action

Design the next conditioning experiment around richer audio structure and include a condition-sensitivity check like this ablation in the evaluation protocol.
