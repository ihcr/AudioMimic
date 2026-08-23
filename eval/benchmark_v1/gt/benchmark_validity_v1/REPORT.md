# GT Benchmark Validity Audit v1

This audit uses paired human-dance GT and controlled corruptions. It validates whether
a metric responds to a known defect; it does not assign an aesthetic score to GT.

- Input: `/home/tianhup/AudioMimic/eval/benchmark_v1/gt/motion_corruptions_v1/per_sequence_metrics.csv`
- Sequences: 20
- Corrupted/clean records: 260

## Directional Checks

| check | metric | expected | result | high-clean |
|---|---|---|---|---:|
| jitter_jerk | jerk_p95_rad_s3 | increasing | **PASS** | +1156.6358 |
| lowpass_jerk | jerk_p95_rad_s3 | decreasing | **PASS** | -2266.4634 |
| lowpass_energy | motion_energy_rad2_s2 | decreasing | **PASS** | -3.7644 |
| freeze_static | static_ratio_below_0p05_rad_s | increasing | **PASS** | +0.0986 |
| freeze_beat_f1 | G1BeatF1 | decreasing | **PASS** | -0.0567 |
| freeze_bas | G1BAS | decreasing | **PASS** | -0.0350 |
| repeat_similarity | repeat_similarity | increasing | **WARN** | -0.0195 |

## Decision

The validated core currently includes jerk, low-pass energy response, static ratio,
and event F1 for their stated use cases, plus BAS as a core beat-alignment metric.
BAS can be insensitive to defects or reward sparse motion beats, so it must be combined
with event coverage, onset response, lag, tempo and phase. The current repeat similarity
implementation is not accepted as a core metric until a stronger repetition corruption
and a long-range self-similarity measure are added.

FIDk/FIDg, Divk/Divg, retrieval R@K/MMDist, style/emotion and human preference are
not rejected; they remain unvalidated because their fixed extractors or human protocol
have not yet been run on this benchmark.

Clean GT is treated as a high-quality empirical reference distribution. It is not assigned
a score of 1.0, and not every GT clip is expected to maximize every metric.
