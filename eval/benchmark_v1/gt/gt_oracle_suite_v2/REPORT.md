# Multi-dataset GT Oracle Suite

This report calibrates the common evaluation protocol on paired G1 motion and audio.
The default held-out suite combines AIST++ crossmodal test and FineDance cross-genre test.
It is a reference distribution, not a single aesthetic ground-truth score.

Scope: **test**; sequences: **38**.

## Dataset coverage

| dataset | sequences |
|---|---:|
| `aistpp` | 20 |
| `finedance` | 18 |

## Pooled reference distributions

| metric | direction | median | q10 | q90 | mean | std |
|---|---|---:|---:|---:|---:|---:|
| `motion_energy` | reference_only | 3.913826 | 1.100801 | 7.754012 | 4.570329 | 3.430492 |
| `joint_jerk_p95` | lower_is_better | 1014.331587 | 299.877318 | 1791.301471 | 1097.086147 | 652.470732 |
| `static_ratio` | lower_is_better | 0.000000 | 0.000000 | 0.096199 | 0.045969 | 0.120646 |
| `repeated_pose_ratio` | lower_is_better | 0.069485 | 0.000000 | 0.836934 | 0.219182 | 0.309156 |
| `fsr_proxy` | lower_is_better | 0.387546 | 0.170455 | 0.837797 | 0.434562 | 0.263273 |
| `pfc_proxy` | lower_is_better | 0.035837 | 0.007091 | 0.363268 | 0.112614 | 0.167531 |
| `root_height_min` | higher_is_better_within_valid_range | 0.738121 | 0.422633 | 0.849174 | 0.672895 | 0.234408 |
| `speed_corr` | higher_is_better | 0.059403 | 0.019915 | 0.132278 | 0.070869 | 0.056348 |
| `impact_corr` | higher_is_better | 0.082555 | 0.024438 | 0.177931 | 0.096199 | 0.058230 |
| `impact_abs_lag` | lower_is_better | 0.566667 | 0.190000 | 0.886667 | 0.546557 | 0.255074 |
| `bas` | higher_is_better | 0.247375 | 0.128954 | 0.557792 | 0.290358 | 0.175592 |
| `event_f1` | higher_is_better | 0.689044 | 0.539284 | 0.831016 | 0.678448 | 0.122591 |
| `event_precision` | higher_is_better | 0.681818 | 0.483537 | 0.874256 | 0.660359 | 0.142702 |
| `event_recall` | higher_is_better | 0.740698 | 0.516231 | 0.953333 | 0.734402 | 0.174556 |
| `event_timing_error` | lower_is_better | 0.096236 | 0.074862 | 0.125472 | 0.097127 | 0.021147 |
| `audio_bpm` | reference_only | 108.811399 | 85.299269 | 131.239206 | 109.608948 | 20.551463 |
| `motion_bpm` | reference_only | 128.571429 | 117.750000 | 168.545455 | 136.356499 | 23.035805 |
| `tempo_error` | lower_is_better | 26.668193 | 4.781250 | 75.744469 | 33.389492 | 27.745975 |
| `phase_error` | lower_is_better | 0.261451 | 0.220778 | 0.297674 | 0.263775 | 0.041621 |

## Use in generator and execution evaluation

The same metrics must be computed for generator reference and SONIC execution. The oracle suite supplies calibration ranges; it does not replace paired comparison, execution retention, or blinded human evaluation.

Raw values must be reported per dataset before any pooled summary because AIST++ and FineDance differ in style distribution, duration, and retargeting statistics.

For a metric with direction `higher_is_better`, a generated result is calibrated against the oracle quantiles directly. For `lower_is_better`, the inequality is reversed. `reference_only` metrics describe the data distribution and are not quality gates.
