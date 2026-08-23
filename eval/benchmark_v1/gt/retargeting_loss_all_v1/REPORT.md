# GMR Retargeting Loss Audit

This audit compares paired source human motion with retargeted G1 motion using the same audio.
Because SMPL/SMPLH and G1 do not share joint semantics, activity/root/music proxies are used; this is not a direct joint position error.

Scope: **all**, sequences: **1611**.

## Interpretation

Positive `delta_*` means the target G1 result is larger/worse for that metric; negative means the target is lower/better. For BAS and event F1, negative is a loss. For absolute lag, positive is a loss.

## Pooled summary

| metric | mean | std | median | q10 | q90 |
|---|---:|---:|---:|---:|---:|
| `duration_seconds` | 28.675585 | 48.794404 | 9.566667 | 7.333333 | 68.333333 |
| `source_fps` | 60.000000 | 0.000000 | 60.000000 | 60.000000 | 60.000000 |
| `target_fps` | 29.990532 | 0.022280 | 30.000000 | 29.932280 | 30.000000 |
| `activity_curve_best_corr` | 0.592652 | 0.287758 | 0.654347 | 0.108283 | 0.908790 |
| `activity_curve_best_lag_seconds` | -0.008649 | 0.261587 | 0.033333 | -0.166667 | 0.033333 |
| `root_speed_curve_best_corr` | 0.793676 | 0.301752 | 0.929733 | 0.120442 | 0.984109 |
| `activity_rms_ratio_target_over_source` | 0.790485 | 0.276108 | 0.813043 | 0.437333 | 1.002025 |
| `root_speed_rms_ratio_target_over_source` | 0.806720 | 0.134150 | 0.822113 | 0.626690 | 0.864284 |
| `activity_event_count_ratio` | 1.177920 | 0.554183 | 1.029851 | 0.842105 | 1.700000 |
| `activity_event_median_error_seconds` | 0.672150 | 2.918966 | 0.066667 | 0.033333 | 0.666667 |
| `delta_bas` | 0.025595 | 0.090060 | 0.024879 | -0.081337 | 0.131653 |
| `delta_event_f1` | 0.055894 | 0.143686 | 0.040476 | -0.113852 | 0.240646 |
| `delta_impact_correlation` | 0.002200 | 0.035787 | 0.001294 | -0.036570 | 0.045427 |
| `delta_absolute_impact_lag` | -0.014235 | 0.365162 | 0.000000 | -0.500000 | 0.500000 |
| `delta_tempo_error_bpm` | -2.824570 | 23.409255 | -0.000000 | -31.790389 | 25.723107 |
| `delta_phase_error_cycles` | -0.001040 | 0.050093 | -0.000274 | -0.060084 | 0.056656 |

The most important outputs for the music-to-G1 story are `delta_bas`, `delta_event_f1`, `delta_absolute_impact_lag`, `delta_phase_error_cycles`, and the activity/root curve correlations. These quantify whether retargeting itself removes musical response before SONIC execution.

Raw source and target motion quality should still be reported separately. A retargeted G1 motion can have a different absolute jerk or energy scale while preserving the temporal structure of the source dance.
