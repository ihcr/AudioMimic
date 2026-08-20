# M0/M2/M4 vs SONIC Capability Baselines

Date: 2026-08-20

## Scope

Each route uses one fixed 60 s seed1234 execution reference repeated three times. Reference-quality summaries use the available song098 exports. Capability baselines use three repeats of one low/medium/high sequence and therefore remain diagnostic.

## Generator Reference

| Route | Nearest GT tier | Energy | Velocity P95 | Acceleration P95 | Jerk P95 | BAS music->motion | Impact corr. |
|---|---|---:|---:|---:|---:|---:|---:|
| M0 | low | 2.396 | 3.358 | 42.3 | 819.1 | 0.257 | 0.032 |
| M2 | low | 1.934 | 3.049 | 37.3 | 718.1 | 0.247 | 0.025 |
| M4 | low | 1.789 | 2.914 | 36.1 | 691.7 | 0.242 | 0.031 |

All routes are closest to the selected low-dynamics GT reference. M0 is the most dynamic; M4 is the most conservative. M0 is unconditional, so its BAS is an incidental baseline rather than evidence of music response.

## SONIC Execution

| Route | Success | Aligned RMSE | EMPKPE | Lag | Amplitude | Energy | 0--1 Hz | 1--3 Hz | 3--8 Hz | Arms 3--8 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 3/3 | 0.1763 | 0.0790 m | 20 ms | 0.912 | 0.468 | 0.802 | 0.427 | 0.227 | 0.204 |
| M2 | 3/3 | 0.1850 | 0.0874 m | 60 ms | 0.906 | 0.522 | 0.808 | 0.413 | 0.395 | 0.330 |
| M4 | 3/3 | 0.1678 | 0.0755 m | 20 ms | 0.872 | 0.503 | 0.785 | 0.467 | 0.244 | 0.227 |

## Baseline-Normalized Findings

| Route | RMSE / native-low | RMSE / GT-low | EMPKPE / native-low | Energy / native-low | Mid-band / native-low | Native gate checks | GT gate checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 1.57x | 1.14x | 1.25x | 0.75x | 0.65x | 3/7 | 4/7 |
| M2 | 1.64x | 1.20x | 1.39x | 0.83x | 0.63x | 4/7 | 4/7 |
| M4 | 1.49x | 1.09x | 1.20x | 0.80x | 0.71x | 4/7 | 4/7 |

M4 has the lowest execution error, followed by M0 and M2. All three preserve pose amplitude but retain only about half of median per-joint dynamic energy. Mid/high-band loss is stronger than either low-tier baseline. M2 high-band mean is inflated by one oscillatory run and must not be interpreted as superior detail retention.

The automatic music metrics do not establish an M2/M4 advantage over unconditional M0. No aesthetic or dance-beauty claim follows from these metrics; blinded human evaluation and broader song/seed coverage remain required.
