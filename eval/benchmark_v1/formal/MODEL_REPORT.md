# Existing Model Stage Results

This report consolidates already available artifacts. It is a progress table,
not a final model ranking: routes, songs, seeds and execution protocols are not
yet balanced across all models.

- M_ref rows: 11
- M_exec rows: 11
- SONIC repeat rows: 9

## M_ref

| route | sequence | energy | jerk P95 | impact corr | abs lag | BAS | source |
|---|---:|---:|---:|---:|---:|---:|---|
| M2 | 098 | 1.9131 | 718.81 | 0.0187 | -0.433 | 0.2524 | reference_metrics.json |
| M2 | 098 | 1.9501 | 713.76 | 0.0370 | 0.733 | 0.2419 | reference_metrics.json |
| M2 | 098 | 1.9385 | 721.69 | 0.0199 | 0.667 | 0.2465 | reference_metrics.json |
| M4 | 098 | 1.7592 | 681.73 | 0.0310 | 0.733 | 0.2610 | reference_metrics.json |
| M4 | 098 | 2.0743 | 775.42 | 0.0286 | -0.433 | 0.2404 | reference_metrics.json |
| M4 | 098 | 1.5340 | 618.10 | 0.0312 | 0.767 | 0.2251 | reference_metrics.json |
| M0 | 098 | 2.3052 | 812.89 | 0.0363 | 0.767 | 0.2219 | reference_metrics.json |
| M0 | 065 | 2.8716 | 885.40 | 0.0098 | -0.967 | 0.2530 | reference_metrics.json |
| M0 | 098 | 2.4868 | 825.32 | 0.0269 | 0.033 | 0.2920 | reference_metrics.json |
| M3 | 012 | 1.9945 | 655.24 | 0.0276 | -0.633 | 0.2433 | m3_012_pair_metrics.json |
| M3 | 065 | 1.9094 | 642.68 | 0.0138 | -0.867 | 0.2832 | m3_065_pair_metrics.json |

## M_exec

| route | sequence | condition | energy | impact corr | BAS | energy retention | BAS retention | extra lag |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| M0 | 098 | SONIC_repeat | 0.9944 | 0.0166 | 0.2033 | 0.4240 | 0.8830 | -1.080 |
| M0 | 098 | SONIC_repeat | 1.0144 | 0.0194 | 0.1937 | 0.4325 | 0.8414 | -1.040 |
| M0 | 098 | SONIC_repeat | 0.9900 | 0.0215 | 0.1973 | 0.4221 | 0.8568 | -1.100 |
| M2 | 098 | SONIC_repeat | 0.9839 | 0.0192 | 0.2938 | 0.5059 | 1.1200 | 0.440 |
| M2 | 098 | SONIC_repeat | 0.8722 | 0.0122 | 0.2300 | 0.4486 | 0.8767 | 0.880 |
| M2 | 098 | SONIC_repeat | 0.8738 | 0.0050 | 0.1574 | 0.4495 | 0.6000 | 0.820 |
| M4 | 098 | SONIC_repeat | 0.8105 | 0.0214 | 0.1825 | 0.4535 | 0.7536 | -1.020 |
| M4 | 098 | SONIC_repeat | 0.8050 | 0.0080 | 0.1838 | 0.4504 | 0.7590 | -1.020 |
| M4 | 098 | SONIC_repeat | 0.8057 | 0.0133 | 0.1656 | 0.4508 | 0.6836 | -1.060 |
| M3 | 012 | SONIC_corrected_measured | 0.9141 | 0.0324 | 0.3498 | 0.4583 | 1.4376 | 0.500 |
| M3 | 065 | SONIC_corrected_measured | 0.8038 | -0.0051 | 0.2677 | 0.4210 | 0.9452 | 1.833 |

Interpretation: compare M_ref against O-G1 for generator gap, and compare
M_exec against its paired M_ref for SONIC retention. The existing rows are
useful for pipeline debugging, but the formal claim still requires the planned
multi-song, multi-seed, tempo/style-stratified expansion.
