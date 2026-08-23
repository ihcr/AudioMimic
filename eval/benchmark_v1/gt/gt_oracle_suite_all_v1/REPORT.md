# Multi-dataset GT Oracle Suite

This report calibrates the common evaluation protocol on paired G1 motion and audio.
The default held-out suite combines AIST++ crossmodal test and FineDance cross-genre test.
It is a reference distribution, not a single aesthetic ground-truth score.

Scope: **all**; sequences: **1611**.

## Dataset coverage

| dataset | sequences |
|---|---:|
| `aistpp` | 1408 |
| `finedance` | 203 |

## Pooled reference distributions

| metric | direction | median | q10 | q90 | mean | std |
|---|---|---:|---:|---:|---:|---:|
| `motion_energy` | reference_only | 4.808934 | 0.818876 | 11.175997 | 5.751346 | 4.622043 |
| `joint_jerk_p95` | lower_is_better | 1236.968128 | 290.335496 | 2578.914195 | 1380.852694 | 938.602216 |
| `static_ratio` | lower_is_better | 0.000000 | 0.000000 | 0.030641 | 0.020379 | 0.081939 |
| `repeated_pose_ratio` | lower_is_better | 0.080780 | 0.000000 | 0.745645 | 0.231599 | 0.293659 |
| `fsr_proxy` | lower_is_better | 0.566434 | 0.113027 | 0.898601 | 0.536863 | 0.287692 |
| `pfc_proxy` | lower_is_better | 0.141487 | 0.013520 | 0.694486 | 0.297302 | 0.673149 |
| `root_height_min` | higher_is_better_within_valid_range | 0.772925 | 0.392779 | 0.856392 | 0.689882 | 0.238468 |
| `speed_corr` | higher_is_better | 0.057866 | 0.017015 | 0.106816 | 0.060306 | 0.036256 |
| `impact_corr` | higher_is_better | 0.084377 | 0.029668 | 0.166420 | 0.091531 | 0.052741 |
| `impact_abs_lag` | lower_is_better | 0.533333 | 0.100000 | 0.933333 | 0.516966 | 0.305755 |
| `bas` | higher_is_better | 0.254905 | 0.136909 | 0.455985 | 0.280945 | 0.136254 |
| `event_f1` | higher_is_better | 0.702703 | 0.545455 | 0.836364 | 0.693665 | 0.120085 |
| `event_precision` | higher_is_better | 0.666667 | 0.500000 | 0.896552 | 0.675607 | 0.154050 |
| `event_recall` | higher_is_better | 0.741935 | 0.533333 | 0.933333 | 0.742640 | 0.150736 |
| `event_timing_error` | lower_is_better | 0.090113 | 0.053878 | 0.122268 | 0.089190 | 0.027100 |
| `audio_bpm` | reference_only | 117.453835 | 89.102909 | 161.499023 | 117.086064 | 21.196216 |
| `motion_bpm` | reference_only | 138.461538 | 112.500000 | 171.428571 | 139.162660 | 24.295532 |
| `tempo_error` | lower_is_better | 29.187414 | 4.953835 | 64.252349 | 31.770883 | 23.031962 |
| `phase_error` | lower_is_better | 0.263935 | 0.222019 | 0.316718 | 0.267392 | 0.039528 |

## Use in generator and execution evaluation

The same metrics must be computed for generator reference and SONIC execution. The oracle suite supplies calibration ranges; it does not replace paired comparison, execution retention, or blinded human evaluation.

Raw values must be reported per dataset before any pooled summary because AIST++ and FineDance differ in style distribution, duration, and retargeting statistics.

For a metric with direction `higher_is_better`, a generated result is calibrated against the oracle quantiles directly. For `lower_is_better`, the inequality is reversed. `reference_only` metrics describe the data distribution and are not quality gates.
