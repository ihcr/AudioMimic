# EXP-20260622-cross-model-rhythm-suite

Status: finished
Owner: yukun
Created: 2026-06-22
Last Updated: 2026-06-22

## Research Question

Does the new rhythm eval suite explain cross-model quality differences that the previous G1 metrics could not explain?

## Scope

This is an evaluation-only comparison over existing saved full-eval motions. No model was retrained or resampled.

Representative rows:

- `librosa35_2000`
- `gaussian_beat_1d_1000`
- `wav2clip_stft_r02_2000`
- `v3_r03_1000_pred`
- `v3b_1500_pred`
- `v5_yaw_1000_pred`
- `beat8d_1000_auto`
- `beat8d_beatness_1000_pred`

All rows use `3265` test motions and reference motions from `data/finedance_g1_fkbeats/test/motions_sliced`.

## Evaluation Additions

The new suite keeps the previous G1/FK beat and distribution metrics, then adds:

- Beat-density diagnostics: `G1BeatDensityRatio`, `G1UnmatchedMotionBeatRate`
- Body response diagnostics: wrist, foot, torso, and full-body beat F1 plus `G1WristDominanceRatio`
- Support/contact diagnostics: `G1FootContactOnBeatRate`, `G1NearSupportOnBeatRate`, `G1NoNearSupportRate`, `G1FootHighLiftRate`
- Endpoint diagnostics: `G1WristJerkMean`, `G1FootJerkMean`
- Per-file `failure_panel.json` outputs

## Artifacts

- Summary JSON: `eval/cross_model_rhythm_suite_20260622/summary.json`
- Full comparison JSON: `eval/cross_model_rhythm_suite_20260622/comparison.json`
- Full comparison Markdown: `eval/cross_model_rhythm_suite_20260622/comparison.md`
- Analysis Markdown: `eval/cross_model_rhythm_suite_20260622/analysis.md`
- Acceptance report: `eval/cross_model_rhythm_suite_20260622/acceptance_report.md`
- Per-model eval outputs: `eval/cross_model_rhythm_suite_20260622/<model>/`

## Key Results

| Model | FKBAS | BeatF1 | ContactBeat | NoSupport | HighLift | WristJerk | Dist | Div | Pen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `librosa35_2000` | 0.254 | 0.214 | 0.943 | 0.001 | 0.003 | 969 | 9.254 | 11.366 | 0.035 |
| `v3_r03_1000_pred` | 0.229 | 0.205 | 0.390 | 0.184 | 0.076 | 664 | 6.056 | 18.484 | 0.048 |
| `v3b_1500_pred` | 0.243 | 0.211 | 0.564 | 0.058 | 0.038 | 1132 | 5.782 | 14.093 | 0.052 |
| `v5_yaw_1000_pred` | 0.260 | 0.234 | 0.420 | 0.179 | 0.111 | 1974 | 3.991 | 16.561 | 0.076 |
| `beat8d_1000_auto` | 0.230 | 0.193 | 0.832 | 0.031 | 0.027 | 1180 | 9.118 | 14.489 | 0.158 |
| `beat8d_beatness_1000_pred` | 0.263 | 0.227 | 0.856 | 0.022 | 0.023 | 1595 | 10.232 | 14.312 | 0.217 |

## Current Conclusion

The new suite explains substantially more than the previous metric set.

The old metrics can rank models by `G1FKBAS`, `G1BeatF1`, `G1Dist`, and `G1Div`, but they cannot explain whether a score came from wrist-heavy motion, missing foot support, high-lift/hovering feet, bad event density, or endpoint jerk.

The clearest example is `v5_yaw_1000_pred`: old metrics make it look like the best rhythm/quality checkpoint (`G1BeatF1=0.234`, `G1Dist=3.991`, `G1Div=16.561`), while the new suite exposes the failure mode (`G1WristBeatF1=0.262`, `G1FootBeatF1=0.214`, `G1FootContactOnBeatRate=0.420`, `G1NoNearSupportRate=0.179`, `G1FootHighLiftRate=0.111`, `G1WristJerkMean=1974`).

The `beat8d_beatness_1000_pred` row proves predicted beatness is rhythm-active relative to `beat8d_1000_auto`, but also shows why it should not be promoted: rhythm improves, while `G1Dist=10.232`, `G1GroundPenetration=0.217`, and endpoint jerk are poor.

The training-check-acceptance verdict is to reject `beat8d_beatness_1000_pred` as a mainline checkpoint and accept the new rhythm suite as the default checkpoint gate. Evidence and split metric-direction tables are recorded in `eval/cross_model_rhythm_suite_20260622/acceptance_report.md`.

2026-06-30 evaluation policy update: use `v3b_1500_pred` and the single 8D raw-diffusion row `beat8d_1000_auto` as stable raw-diffusion reference anchors for future model comparisons. This is a qualitative/render-informed role, not a claim that either row wins every metric or is the final accepted model. They are useful because render inspection shows them as comparatively natural and stable relative to later failure modes, while their metrics expose different strengths and weaknesses:

- `v3b_1500_pred` is the balanced raw-diffusion anchor for overall motion naturalness, rhythm, and distribution behavior.
- `beat8d_1000_auto` is the single-8D raw-diffusion anchor for beat/control behavior.

Do not treat other 8D variants, including `beat8d_beatness_1000_pred`, as stable baselines. They remain useful diagnostic rows, but not comparison anchors, because the added beatness variant improves some rhythm metrics while failing distribution/penetration and endpoint-quality checks.

Future acceptance reports should compare every new route against both anchors on the same fixed clips, seeds, render settings, and metric suite. Metrics are diagnostic evidence, not the final judge. A checkpoint should not be promoted only because `G1BeatF1`, `G1FKBAS`, or another single metric improves; qualitative render inspection and robot feasibility diagnostics must agree. If a model beats these anchors on rhythm but regresses in support/contact, ground behavior, endpoint jerk, root stability, or visual naturalness, report it as a failure mode or metric-hacking risk rather than a better model.

## Next Action

Use this suite as the default ckpt500/ckpt1000 acceptance gate for V6a/contact-support-aware experiments and future prior/latent routes. Include `v3b_1500_pred` and `beat8d_1000_auto` in matched render and metric comparisons. Do not accept a checkpoint on beat metrics alone.
