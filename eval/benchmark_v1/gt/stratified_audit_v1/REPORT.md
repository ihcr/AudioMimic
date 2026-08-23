# Stratified GT Benchmark Audit v1

The same music-motion metrics are reported separately for AIST++, FineDance,
tempo bands and available style/genre strata. Values are GT reference distributions,
not universal ideal scores.

## GT Strata (median)

| axis | stratum | n | BPM | BAS | reverse BAS | Beat F1 | Impact corr. | abs lag | tempo err | phase err | energy | jerk P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dataset | aistpp | 20 | 106.6 | 0.287 | 0.506 | 0.714 | 0.131 | 0.617 | 23.937 | 0.275 | 4.112 | 967.752 |
| dataset | finedance | 18 | 117.5 | 0.201 | 0.438 | 0.662 | 0.044 | 0.433 | 27.777 | 0.255 | 3.788 | 1161.186 |
| style | AIST_genre_BR | 2 | 107.7 | 0.277 | 0.440 | 0.674 | 0.110 | 0.650 | 20.905 | 0.262 | 5.494 | 1167.763 |
| style | AIST_genre_HO | 2 | 105.5 | 0.299 | 0.469 | 0.742 | 0.202 | 0.567 | 76.349 | 0.273 | 4.975 | 1284.388 |
| style | AIST_genre_JB | 2 | 129.2 | 0.172 | 0.403 | 0.836 | 0.108 | 0.551 | 20.722 | 0.252 | 5.839 | 1419.534 |
| style | AIST_genre_JS | 2 | 110.0 | 0.153 | 0.547 | 0.439 | 0.104 | 0.450 | 5.455 | 0.237 | 0.033 | 70.489 |
| style | AIST_genre_KR | 2 | 99.4 | 0.770 | 0.798 | 0.660 | 0.175 | 0.483 | 75.616 | 0.280 | 1.195 | 417.460 |
| style | AIST_genre_LH | 2 | 120.2 | 0.182 | 0.372 | 0.938 | 0.137 | 0.667 | 4.286 | 0.399 | 3.869 | 1015.792 |
| style | AIST_genre_LO | 2 | 99.4 | 0.327 | 0.606 | 0.736 | 0.081 | 0.617 | 22.788 | 0.273 | 1.155 | 284.045 |
| style | AIST_genre_MH | 2 | 110.0 | 0.197 | 0.392 | 0.844 | 0.163 | 0.533 | 14.329 | 0.192 | 9.917 | 1776.765 |
| style | AIST_genre_PO | 2 | 89.1 | 0.697 | 0.746 | 0.652 | 0.162 | 0.783 | 90.897 | 0.275 | 2.364 | 535.900 |
| style | AIST_genre_WA | 2 | 80.7 | 0.437 | 0.529 | 0.473 | 0.104 | 0.650 | 54.250 | 0.279 | 11.592 | 2279.118 |
| style | Classic+HanTang | 1 | 117.5 | 0.175 | 0.465 | 0.549 | 0.034 | 0.433 | 2.546 | 0.228 | 2.270 | 738.833 |
| style | Classic+ShenYun | 2 | 123.0 | 0.168 | 0.489 | 0.662 | 0.022 | 0.867 | 6.797 | 0.245 | 2.364 | 627.173 |
| style | Folk+Dai | 1 | 86.1 | 0.181 | 0.375 | 0.540 | 0.029 | 0.933 | 26.367 | 0.246 | 1.741 | 535.791 |
| style | Folk+Miao | 1 | 136.0 | 0.318 | 0.655 | 0.704 | 0.102 | 0.267 | 7.428 | 0.244 | 3.214 | 769.437 |
| style | Folk+Wei | 1 | 184.6 | 0.213 | 0.732 | 0.660 | 0.000 | 0.400 | 64.570 | 0.273 | 2.552 | 719.065 |
| style | Mix+Choreography | 1 | 136.0 | 0.242 | 0.588 | 0.615 | 0.067 | 0.600 | 15.999 | 0.244 | 3.105 | 784.072 |
| style | Mix+Korean | 3 | 99.4 | 0.248 | 0.393 | 0.694 | 0.069 | 0.200 | 35.035 | 0.265 | 6.461 | 1604.355 |
| style | Street+Breaking | 1 | 117.5 | 0.193 | 0.310 | 0.716 | 0.015 | 0.867 | 32.546 | 0.269 | 5.291 | 1413.767 |
| style | Street+Hiphop | 2 | 87.7 | 0.330 | 0.450 | 0.610 | 0.095 | 0.533 | 36.594 | 0.251 | 6.273 | 1650.558 |
| style | Street+Jazz | 1 | 83.4 | 0.190 | 0.252 | 0.611 | 0.069 | 0.633 | 66.646 | 0.256 | 4.486 | 916.808 |
| style | Street+Locking | 1 | 123.0 | 0.285 | 0.543 | 0.761 | 0.053 | 0.567 | 5.525 | 0.264 | 8.872 | 2169.311 |
| style | Street+Popping | 2 | 114.9 | 0.175 | 0.315 | 0.708 | 0.077 | 0.317 | 29.330 | 0.264 | 3.414 | 1185.643 |
| style | Street+Urban | 1 | 95.7 | 0.155 | 0.257 | 0.609 | 0.034 | 0.067 | 32.868 | 0.254 | 7.735 | 1791.117 |
| tempo | fast_>=120 | 11 | 129.2 | 0.213 | 0.540 | 0.718 | 0.069 | 0.600 | 10.547 | 0.246 | 3.322 | 929.689 |
| tempo | medium_90-120 | 19 | 107.7 | 0.248 | 0.453 | 0.705 | 0.085 | 0.533 | 29.187 | 0.263 | 3.555 | 1022.848 |
| tempo | slow_<90 | 8 | 84.7 | 0.383 | 0.453 | 0.583 | 0.099 | 0.700 | 55.323 | 0.265 | 5.340 | 1260.582 |

## Corruption Direction Checks

| dataset | check | expected | result | high-clean delta |
|---|---|---|---|---:|
| aistpp | jitter_jerk_p95_rad_s3 | increase | **PASS** | +1135.3794 |
| aistpp | lowpass_jerk_p95_rad_s3 | decrease | **PASS** | -2266.4634 |
| aistpp | freeze_static_ratio_below_0p05_rad_s | increase | **PASS** | +0.0986 |
| aistpp | freeze_G1BeatF1 | decrease | **PASS** | -0.0567 |
| finedance | jitter_jerk_p95_rad_s3 | increase | **PASS** | +858.4909 |
| finedance | lowpass_jerk_p95_rad_s3 | decrease | **PASS** | -2463.3446 |
| finedance | freeze_static_ratio_below_0p05_rad_s | increase | **PASS** | +0.0119 |
| finedance | freeze_G1BeatF1 | decrease | **WARN** | +0.0018 |

## Decision Rules

1. A metric is retained only when its intended response is stable within both datasets.
2. Tempo/style strata define calibration ranges; they are not pooled into one ideal score.
3. A style with fewer than three clips is descriptive only and cannot support a significance claim.
4. BAS remains a core beat-alignment metric and must be interpreted with Beat F1/coverage, onset response, lag, tempo and phase.
