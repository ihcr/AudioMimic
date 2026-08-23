# M3 Music-Condition Causal Ablation: Aggregate v2

Formal generator-level evaluation: songs 012/065, seeds 1234/2345/3456,
conditions paired, wrong-song, +4 s shifted, and null. Total: 24 trajectories.

## Condition Summary

| condition | energy | jerk P95 | BAS | impact corr. | impact lag |
|---|---:|---:|---:|---:|---:|
| null | 2.444 +/- 0.218 | 794.148 +/- 71.616 | 0.274 +/- 0.024 | 0.025 +/- 0.022 | 0.578 +/- 0.301 |
| paired | 1.829 +/- 0.213 | 622.982 +/- 52.958 | 0.252 +/- 0.041 | 0.031 +/- 0.021 | 0.583 +/- 0.313 |
| shifted_4s | 1.723 +/- 0.287 | 606.257 +/- 86.429 | 0.245 +/- 0.046 | 0.034 +/- 0.029 | 0.656 +/- 0.249 |
| wrong | 1.624 +/- 0.535 | 571.624 +/- 150.071 | 0.271 +/- 0.021 | 0.029 +/- 0.029 | 0.594 +/- 0.321 |

## Paired Contrasts

A win uses the metric direction defined in the frozen evaluation map; `impact_abs_lag` uses absolute lag.

| control | metric | paired-control mean | paired wins / 6 |
|---|---|---:|---:|
| wrong | energy | 0.2047 | not ranked |
| wrong | jerk_p95 | 51.3587 | 3 / 6 |
| wrong | bas | -0.0193 | 2 / 6 |
| wrong | impact_corr | 0.0023 | 4 / 6 |
| wrong | impact_abs_lag | -0.0111 | 4 / 6 |
| shifted_4s | energy | 0.1058 | not ranked |
| shifted_4s | jerk_p95 | 16.7251 | 2 / 6 |
| shifted_4s | bas | 0.0067 | 3 / 6 |
| shifted_4s | impact_corr | -0.0032 | 3 / 6 |
| shifted_4s | impact_abs_lag | -0.0722 | 3 / 6 |
| null | energy | -0.6154 | not ranked |
| null | jerk_p95 | -171.1652 | 6 / 6 |
| null | bas | -0.0222 | 2 / 6 |
| null | impact_corr | 0.0061 | 5 / 6 |
| null | impact_abs_lag | 0.0056 | 2 / 6 |

## Interpretation

The sidecar is causally active because changing the music condition changes the generated trajectory.
This table does not yet establish correct music alignment: paired must win on event/phase metrics
against wrong, shifted, and null controls before making that claim.

Next: separate RMS-only and predicted-FMS-only controls, record sidecar magnitude, then run the
paired reference through the fixed SONIC execution protocol.
