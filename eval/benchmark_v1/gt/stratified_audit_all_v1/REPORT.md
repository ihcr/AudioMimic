# Stratified GT Benchmark Audit v1

The same music-motion metrics are reported separately for AIST++, FineDance,
tempo bands and available style/genre strata. Values are GT reference distributions,
not universal ideal scores.

## GT Strata (median)

| axis | stratum | n | BPM | BAS | reverse BAS | Beat F1 | Impact corr. | abs lag | tempo err | phase err | energy | jerk P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dataset | aistpp | 1408 | 112.3 | 0.264 | 0.513 | 0.706 | 0.092 | 0.533 | 29.815 | 0.267 | 5.069 | 1272.854 |
| dataset | finedance | 203 | 117.5 | 0.224 | 0.499 | 0.694 | 0.040 | 0.567 | 21.008 | 0.251 | 4.220 | 1103.685 |
| style | AIST_genre_BR | 141 | 112.3 | 0.307 | 0.471 | 0.667 | 0.106 | 0.533 | 37.653 | 0.264 | 8.722 | 1854.219 |
| style | AIST_genre_HO | 141 | 117.5 | 0.285 | 0.482 | 0.706 | 0.089 | 0.633 | 18.615 | 0.274 | 4.454 | 1150.750 |
| style | AIST_genre_JB | 141 | 120.2 | 0.226 | 0.475 | 0.718 | 0.074 | 0.500 | 23.377 | 0.267 | 6.588 | 1751.384 |
| style | AIST_genre_JS | 141 | 112.3 | 0.214 | 0.480 | 0.640 | 0.077 | 0.567 | 29.815 | 0.264 | 0.792 | 272.586 |
| style | AIST_genre_KR | 141 | 112.3 | 0.276 | 0.573 | 0.688 | 0.114 | 0.533 | 26.048 | 0.268 | 6.027 | 1542.349 |
| style | AIST_genre_LH | 141 | 117.5 | 0.292 | 0.572 | 0.741 | 0.129 | 0.500 | 32.468 | 0.274 | 4.403 | 1004.544 |
| style | AIST_genre_LO | 141 | 129.2 | 0.254 | 0.605 | 0.726 | 0.077 | 0.567 | 28.139 | 0.264 | 5.151 | 1374.890 |
| style | AIST_genre_MH | 141 | 112.3 | 0.298 | 0.556 | 0.765 | 0.107 | 0.467 | 30.897 | 0.258 | 7.854 | 1615.305 |
| style | AIST_genre_PO | 140 | 110.0 | 0.242 | 0.464 | 0.699 | 0.090 | 0.583 | 30.796 | 0.269 | 1.176 | 397.646 |
| style | AIST_genre_WA | 140 | 112.3 | 0.301 | 0.496 | 0.667 | 0.072 | 0.534 | 36.959 | 0.275 | 7.223 | 1743.051 |
| style | Classic+DunHuang | 2 | 120.3 | 0.220 | 0.491 | 0.696 | 0.005 | 0.767 | 13.266 | 0.257 | 3.643 | 915.245 |
| style | Classic+HanTang | 11 | 117.5 | 0.195 | 0.486 | 0.637 | 0.037 | 0.300 | 11.118 | 0.245 | 2.386 | 738.833 |
| style | Classic+Kun | 1 | 136.0 | 0.216 | 0.616 | 0.634 | -0.000 | 0.067 | 23.499 | 0.245 | 1.009 | 271.695 |
| style | Classic+ShenYun | 34 | 126.1 | 0.187 | 0.515 | 0.667 | 0.047 | 0.783 | 8.426 | 0.250 | 2.297 | 617.063 |
| style | Folk+Dai | 10 | 110.0 | 0.183 | 0.422 | 0.608 | 0.037 | 0.417 | 17.010 | 0.249 | 2.503 | 711.251 |
| style | Folk+Miao | 10 | 136.0 | 0.275 | 0.576 | 0.754 | 0.044 | 0.567 | 26.531 | 0.259 | 4.513 | 1078.028 |
| style | Folk+Wei | 8 | 117.5 | 0.215 | 0.487 | 0.672 | 0.000 | 0.417 | 13.671 | 0.252 | 3.337 | 959.520 |
| style | Mix+Chinese | 4 | 147.8 | 0.220 | 0.578 | 0.740 | 0.038 | 0.417 | 14.260 | 0.247 | 3.088 | 716.803 |
| style | Mix+Choreography | 6 | 136.0 | 0.239 | 0.583 | 0.742 | 0.065 | 0.467 | 15.491 | 0.263 | 3.287 | 864.571 |
| style | Mix+Korean | 35 | 123.0 | 0.241 | 0.508 | 0.719 | 0.042 | 0.500 | 23.037 | 0.251 | 5.076 | 1345.738 |
| style | Street+Breaking | 13 | 117.5 | 0.245 | 0.515 | 0.712 | 0.027 | 0.467 | 15.415 | 0.251 | 5.429 | 1436.131 |
| style | Street+Hiphop | 18 | 99.4 | 0.290 | 0.429 | 0.655 | 0.035 | 0.633 | 35.694 | 0.256 | 5.621 | 1524.427 |
| style | Street+Jazz | 19 | 112.3 | 0.223 | 0.446 | 0.700 | 0.042 | 0.567 | 23.428 | 0.252 | 5.186 | 1187.075 |
| style | Street+Locking | 3 | 123.0 | 0.271 | 0.543 | 0.774 | 0.037 | 0.800 | 21.008 | 0.254 | 7.967 | 2108.771 |
| style | Street+Popping | 21 | 107.7 | 0.268 | 0.490 | 0.705 | 0.054 | 0.533 | 42.334 | 0.254 | 3.416 | 1151.592 |
| style | Street+Urban | 6 | 125.6 | 0.215 | 0.506 | 0.716 | 0.019 | 0.550 | 31.832 | 0.255 | 6.435 | 1651.882 |
| style | jiewu+jazz | 2 | 117.5 | 0.249 | 0.483 | 0.742 | 0.015 | 0.467 | 18.444 | 0.250 | 3.031 | 727.337 |
| tempo | fast_>=120 | 727 | 129.2 | 0.245 | 0.567 | 0.750 | 0.086 | 0.535 | 20.462 | 0.264 | 4.682 | 1223.968 |
| tempo | medium_90-120 | 662 | 107.7 | 0.262 | 0.464 | 0.682 | 0.080 | 0.533 | 31.246 | 0.263 | 4.958 | 1240.226 |
| tempo | slow_<90 | 222 | 89.1 | 0.301 | 0.432 | 0.606 | 0.090 | 0.517 | 49.359 | 0.266 | 5.096 | 1241.599 |

## Corruption Direction Checks

| dataset | check | expected | result | high-clean delta |
|---|---|---|---|---:|
| n/a | GT-only full-dataset summary | n/a | **not run** | n/a |

## Decision Rules

1. A metric is retained only when its intended response is stable within both datasets.
2. Tempo/style strata define calibration ranges; they are not pooled into one ideal score.
3. A style with fewer than three clips is descriptive only and cannot support a significance claim.
4. BAS remains a core beat-alignment metric and must be interpreted with Beat F1/coverage, onset response, lag, tempo and phase.
