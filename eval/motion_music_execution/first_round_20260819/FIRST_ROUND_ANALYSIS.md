# M0/M2/M4 Motion--Music--Execution First-Round Analysis

Date: 2026-08-19
Protocol: fixed 60 s PKLs, song-aligned audio, SONIC `3 s alignment + 1 s hold + full + 1.0x`

## 1. Scope

This is an automatic metric audit, not a final aesthetic evaluation.

- Reference generation: all nine exported PKLs. The matched-song098 comparison contains two M0, three M2, and three M4 trajectories; the remaining M0 trajectory uses song065.
- Execution: one fixed song098/seed1234 reference per route, each replayed through SONIC three times.
- Music: pre-sliced `098_t000128_60s.wav` or `065_t000128_60s.wav`, already aligned to the PKL `audio_start_seconds=4.267 s` source offset.
- M0 is unconditional, M2 uses predicted future-music, and M4 uses oracle future-music.

The execution repeats estimate tracker variability, not generator variability. The sample size is too small for significance claims.

## 2. Reference Motion Quality

Matched song098 results, mean +/- population standard deviation:

| Route | n | Energy | Velocity P95 | Accel. P95 | Jerk P95 | Static | Repeat | FSR proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 2 | 2.396 +/- 0.091 | 3.358 +/- 0.043 | 42.32 +/- 0.50 | 819.1 +/- 6.2 | 0.58% | 0.22% | 8.42% |
| M2 | 3 | 1.934 +/- 0.015 | 3.049 +/- 0.013 | 37.32 +/- 0.07 | 718.1 +/- 3.3 | 0.02% | 0.00% | 8.75% |
| M4 | 3 | 1.789 +/- 0.222 | 2.914 +/- 0.170 | 36.10 +/- 3.09 | 691.7 +/- 64.6 | 0.15% | 0.00% | 5.50% |

Units are rad/s, rad/s2, and rad/s3 for the three derivative columns. FSR is the fraction of frames where either foot is below 5 cm and moves faster than 0.20 m/s under the evaluation FK model; it is a transparent proxy, not a calibrated replacement for dataset-standard FSR/PFC.

For M2 specifically:

- Median joint amplitude is `0.651 +/- 0.026 rad`.
- Energy is 19.3% below the matched M0 pack; jerk P95 is 12.3% lower. M2 is less aggressive than M0 but is not static.
- Static and strict nonlocal repeated-pose rates are approximately zero. There is no evidence of freezing or exact short-loop collapse under the stated thresholds.
- C4 boundary/non-boundary position and velocity jump ratios are `0.989` and `0.960`. Commit boundaries are not more discontinuous than ordinary frames.
- Root path length is `10.32 +/- 0.39 m`, while net planar displacement is only `0.62 +/- 0.17 m`. The trajectory moves around substantially without a large one-way drift.
- Minimum generated root height is `0.711 +/- 0.023 m`, and no reference FK ground penetration was detected with the world-z check.

These results support continuity, activity, and moderate dynamics. They do not establish that M2 is visually beautiful: naturalness, choreography, expressiveness, and style still require blinded human ratings.

## 3. Music Correspondence

| Route | n | BAS music-to-motion | BAS motion-to-music | Best onset-impact corr. |
|---|---:|---:|---:|---:|
| M0 | 2 | 0.257 +/- 0.035 | 0.420 +/- 0.065 | 0.032 +/- 0.005 |
| M2 | 3 | 0.247 +/- 0.004 | 0.386 +/- 0.007 | 0.025 +/- 0.008 |
| M4 | 3 | 0.242 +/- 0.015 | 0.394 +/- 0.019 | 0.031 +/- 0.001 |

The onset-impact correlations are all below `0.1`, so their optimized lag estimates are explicitly marked unreliable. Speed correlation is slightly negative for all routes, which is compatible with dance beats occurring at pauses or direction changes rather than speed maxima.

The current automatic metrics do not show an M2 or M4 music advantage over unconditional M0. In fact, matched M0 has a slightly higher mean BAS, despite not consuming music. This is direct evidence that BAS alone cannot validate the music-conditioning route. Possible explanations remain confounded: weak use of music conditioning, insufficient metrics, selection of only one song, and very small sample size.

## 4. SONIC Execution Loss

| Route | Success | Raw RMSE | Aligned RMSE | Lag | Aggregate energy retention | BAS change | Impact corr. change |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 3/3 | 0.1781 +/- 0.0006 | 0.1763 +/- 0.0004 | 20 +/- 0 ms | 42.6 +/- 0.5% | -0.032 +/- 0.004 | -0.026 +/- 0.002 |
| M2 | 3/3 | 0.1933 +/- 0.0361 | 0.1850 +/- 0.0268 | 60 +/- 57 ms | 46.8 +/- 2.7% | -0.035 +/- 0.056 | -0.013 +/- 0.006 |
| M4 | 3/3 | 0.1694 +/- 0.0005 | 0.1678 +/- 0.0004 | 20 +/- 0 ms | 45.2 +/- 0.1% | -0.066 +/- 0.008 | -0.024 +/- 0.006 |

Root-relative FK and frequency-retention results:

| Route | EMPKPE mean | EMPKPE P95 | Low 0--1 Hz | Mid 1--3 Hz | High 3--8 Hz | Contact F1 L/R |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.0790 +/- 0.0004 m | 0.1815 +/- 0.0013 m | 80.2 +/- 0.2% | 42.7 +/- 0.1% | 22.7 +/- 0.9% | 0.700 / 0.530 |
| M2 | 0.0874 +/- 0.0137 m | 0.2128 +/- 0.0395 m | 80.8 +/- 1.0% | 41.3 +/- 3.5% | 39.5 +/- 25.2% | 0.652 / 0.572 |
| M4 | 0.0755 +/- 0.0006 m | 0.1755 +/- 0.0011 m | 78.5 +/- 0.5% | 46.7 +/- 1.1% | 24.4 +/- 0.6% | 0.691 / 0.588 |

EMPKPE uses the frozen G1 FK key-body set after subtracting pelvis translation. This preserves
root-orientation and articulated tracking error while excluding global root XY, which SONIC does
not directly track. Frequency values are the median active-joint Welch band-power retention for
each run, followed by mean +/- standard deviation over the three repeats.

For M2:

- All three 60 s runs remain stable. The problem is fidelity loss, not survival.
- Aggregate full-body energy retention is `46.8 +/- 2.7%`. The earlier median-per-joint definition gives `52.2 +/- 4.7%`; both show that roughly half of the dynamic energy is lost.
- Median-per-joint amplitude retention from the tracking-gap analysis is `90.6 +/- 1.0%`. The tracker largely reaches the pose range while suppressing speed and high-frequency expression.
- Low-frequency power retention is stable near `80.8%`, while mid-frequency retention is only `41.3%`. M2 runs r02/r03 retain only `22.4%/20.9%` of full-body 3--8 Hz power and `16.9%/18.3%` in the arms.
- M2 r01 reports `75.0%` high-frequency retention, but it is the same run with 140 ms lag and 1.35x jerk amplification. This is tracking oscillation/noise, not reliable preservation of intended high-frequency choreography; the median across the three M2 repeats is `22.4%`.
- Root-relative EMPKPE is `8.74 +/- 1.37 cm` on average and `21.28 +/- 3.95 cm` at P95. The r01 outlier is again visible (`10.68 cm` mean versus `7.83/7.72 cm` in r02/r03).
- Height-derived contact F1 is moderate and asymmetric (`0.652` left, `0.572` right); only `61.5%/70.2%` of target left/right contact transitions have an execution transition within 250 ms after lag compensation.
- BAS changes from approximately `0.262` in the resampled target to `0.295/0.229/0.157` in the three executions. The mean drop is `0.035`, but the run-to-run spread is too large for a robust retention claim.
- Onset-impact correlation drops by `0.013` on average, from an already weak target value near `0.026`. This cannot support a reliable timing-lag interpretation.
- Run r01 is an execution outlier: 140 ms tracking lag, 0.244 rad raw RMSE, and jerk P95 amplified to 1.35x target. Runs r02/r03 have 20 ms lag, about 0.168 rad raw RMSE, and reduce jerk P95 to about 0.57x.

Contact F1 and transition timing currently use a fixed FK foot-height detector. They are useful for
within-protocol diagnosis, but SONIC sim state and the standalone evaluation XML have a foot-height
offset. Contact/sliding must be validated against MuJoCo contact sensors before publication.

## 5. Current Conclusion

The available M2 PKLs are continuous, active, non-frozen, and not unusually discontinuous at C4 boundaries. Their dynamics are more conservative than M0 and close to M4. Automatic metrics do not yet prove aesthetic quality or stronger music correspondence. SONIC executes the trajectory stably and preserves most low-frequency pose variation, but removes about half of total dynamic energy, retains only about one fifth to one quarter of intended high-frequency power in normal runs, and introduces substantial run-dependent loss in beat/impact structure.

The strongest defensible statement is:

> M2 produces a stable, continuous 60 s reference that SONIC can execute, but current evidence does not establish superior music responsiveness, and the tracker substantially attenuates dynamic expression.

## 6. Next Required Evidence

1. Produce blinded M0/M2/M4 reference and execution videos, then collect separate ratings for naturalness, dance-likeness, expressiveness, long-term coherence, and music appropriateness.
2. Obtain the M2 checkpoint and expand to at least three songs and three generation seeds; the present song098 pack cannot test generalization.
3. Add tempo/phase, phrase-boundary and audio-motion retrieval metrics. BAS and onset correlation are insufficient on their own.
4. Validate foot contact and skating against the active SONIC MuJoCo contact state rather than standalone FK height alone.
5. Calibrate EMPKPE and frequency-retention engineering gates with GT references before using them as pass/fail thresholds.

## 7. Artifacts

- `reference_metrics.json/csv`: per-PKL reference quality and music metrics.
- `execution_metrics.json/csv`: per-run target, execution, retention, and tracking metrics.
- `route_summary.json`: all-route and matched-song098 aggregates.
- Analyzer: `eval/analyze_motion_music_execution.py`.
