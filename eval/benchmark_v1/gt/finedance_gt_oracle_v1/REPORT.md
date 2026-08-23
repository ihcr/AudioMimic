# FineDance-G1 GT Oracle Evaluation

This report evaluates the frozen 18-sequence FineDance cross-genre test using paired retargeted G1 motion and audio. It is model-independent and does not use a generator checkpoint.

Test sequences: **18**.

The report separates motion quality from music correspondence. Event F1 matches audio beat events to motion impact events within the configured tolerance; tempo error is absolute BPM difference; phase error is nearest event offset normalized by the audio beat period after the measured lag.

## Aggregate metrics

| metric | mean | std | count |
|---|---:|---:|---:|
| `quality.motion_energy_rad2_s2` | 4.489076427292083 | 2.0831848817455434 | 18 |
| `quality.joint_jerk_abs_rad_s3.p95` | 1177.04236972124 | 476.1994378689399 | 18 |
| `quality.static_ratio_speed_below_008` | 0.02552029125275898 | 0.0303404211028291 | 18 |
| `quality.repeated_pose_ratio_rms008_after2s` | 0.04472004541833832 | 0.06663344847904415 | 18 |
| `quality.physical.fsr_ground_calibrated_proxy` | 0.3028883753084635 | 0.10027891808814908 | 18 |
| `quality.physical.pfc_proxy` | 0.03153912004529655 | 0.027146424682183434 | 18 |
| `quality.root_height_min_m` | 0.5181812894862019 | 0.26154912467418895 | 18 |
| `music.speed_best_correlation` | 0.05081984588687053 | 0.06338527909529101 | 18 |
| `music.impact_best_correlation` | 0.05342581442263582 | 0.03905349669340319 | 18 |
| `music.impact_best_lag_seconds` | 0.22592592592592597 | 0.5267599021499564 | 18 |
| `music.bas_music_to_motion` | 0.22291296147762887 | 0.06474614394169531 | 18 |
| `music.event_f1.f1` | 0.6551666219327252 | 0.06372586668263708 | 18 |
| `music.event_f1.median_abs_timing_error_seconds` | 0.09653565129755409 | 0.010749021016356065 | 18 |
| `music.audio_bpm` | 114.61296575731166 | 25.430227523169254 | 18 |
| `music.motion_impact_bpm` | 128.352595680182 | 11.900913705028637 | 18 |
| `music.tempo_abs_error_bpm` | 27.644705110839716 | 18.252548379092374 | 18 |
| `music.phase.mean_phase_error_cycles` | 0.2543972695365161 | 0.01213445194398281 | 18 |

## Interpretation

These values define the empirical GT reference range; they are not a claim that the dataset is perfectly musical or physically executable on the robot. Generated references must be compared against this same test set, and SONIC execution must be evaluated separately through reference-to-execution retention.

Existing paired-vs-wrong-song retrieval results remain in `eval/benchmark_v1/gt/finedance_music_pairing_v1/`; this report adds the full motion-quality and event/tempo/phase profile.
