# Unified GT/GMR/SONIC Benchmark: Retargeting Stage

This audit applies the same dance-quality and music-adaptation evaluator to paired source
SMPL/SMPLH motion and its retargeted G1 reference using the same audio. It measures the
change introduced before SONIC execution. It is not a direct cross-skeleton joint-position
error; activity/root diagnostics are secondary explanations only.

Scope: **test**, sequences: **38**.

## Interpretation

Positive `delta_*` means the target G1 result is larger/worse for that metric; negative means the target is lower/better. For BAS and event F1, negative is a loss. For absolute lag, positive is a loss.

## Pooled summary

| metric | mean | std | median | q10 | q90 |
|---|---:|---:|---:|---:|---:|
| `duration_seconds` | 50.113158 | 49.817107 | 11.966667 | 7.776667 | 112.126667 |
| `source_fps` | 60.000000 | 0.000000 | 60.000000 | 60.000000 | 60.000000 |
| `target_fps` | 29.996436 | 0.015122 | 30.000000 | 30.000000 | 30.000000 |
| `activity_curve_best_corr` | 0.400438 | 0.361102 | 0.270114 | -0.005436 | 0.890045 |
| `activity_curve_best_lag_seconds` | -0.105263 | 0.486095 | 0.033333 | -0.930000 | 0.363333 |
| `root_speed_curve_best_corr` | 0.532326 | 0.425446 | 0.768348 | 0.026185 | 0.980930 |
| `activity_rms_ratio_target_over_source` | 0.798925 | 0.192414 | 0.805511 | 0.651605 | 0.944738 |
| `root_speed_rms_ratio_target_over_source` | 0.714499 | 0.172633 | 0.816066 | 0.466211 | 0.866370 |
| `activity_event_count_ratio` | 1.284826 | 0.393383 | 1.180348 | 0.850000 | 1.737564 |
| `activity_event_median_error_seconds` | 0.939035 | 1.819297 | 0.183333 | 0.033333 | 3.221667 |
| `delta_bas` | 0.071560 | 0.098617 | 0.084184 | -0.036590 | 0.178758 |
| `delta_event_f1` | 0.095633 | 0.121166 | 0.117771 | -0.068750 | 0.221384 |
| `delta_impact_correlation` | 0.009135 | 0.057425 | 0.002251 | -0.048301 | 0.066469 |
| `delta_absolute_impact_lag` | -0.137719 | 0.377292 | -0.066667 | -0.666667 | 0.330000 |
| `delta_tempo_error_bpm` | 0.456359 | 28.305449 | -0.000000 | -36.394984 | 35.614973 |
| `delta_phase_error_cycles` | -0.002959 | 0.030155 | 0.001711 | -0.039144 | 0.037359 |

The benchmark headline should use two groups: (1) dance quality, including continuity,
energy, jerk, freeze/repetition and stability; and (2) music adaptation, including event
Precision/Recall/F1, phase error, response lag, tempo error, onset-energy correlation and,
when applicable, BAS. The same table must be produced for source oracle, G1 reference and
SONIC execution. `delta` from source to G1 is retargeting loss; `delta` from G1 to SONIC is
tracking loss.

The most useful current diagnostics for the music-to-G1 story are `delta_bas`,
`delta_event_f1`, `delta_absolute_impact_lag`, `delta_phase_error_cycles`, and the
activity/root curve correlations. These quantify whether retargeting changes musical
response before SONIC execution, but they are not a substitute for the full benchmark.

Raw source, G1 reference and SONIC execution quality should be reported in the same
benchmark table. Absolute energy or jerk may change with representation, so the paper should
report normalized values and paired deltas, not pretend that a raw SMPL joint coordinate and
a G1 joint coordinate are directly comparable.
