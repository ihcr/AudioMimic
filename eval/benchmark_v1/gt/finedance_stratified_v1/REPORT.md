# FineDance Dance-Style and Tempo-Stratified Metrics

This report stratifies the same music-motion evaluator by FineDance dance-style labels
and audio tempo. A sequence with multiple labels contributes to each applicable style;
style counts therefore overlap. All values are diagnostic distributions, not style rankings.

Tempo bins: slow `<90 BPM`, medium `90--130 BPM`, fast `>130 BPM`.

## Dance-style coverage

| style | sequence count |
|---|---:|
| Breaking | 13 |
| Chinese | 4 |
| Choreography | 6 |
| Classic | 48 |
| Dai | 10 |
| DunHuang | 2 |
| Folk | 28 |
| HanTang | 11 |
| Hiphop | 18 |
| Jazz | 19 |
| Korean | 35 |
| Kun | 1 |
| Locking | 3 |
| Miao | 10 |
| Mix | 45 |
| Popping | 21 |
| ShenYun | 34 |
| Street | 80 |
| Urban | 6 |
| Wei | 8 |
| jazz | 2 |
| jiewu | 2 |

## Style summary (full sequence)

| style | stage | n seq | BAS | Event F1 | Impact corr | Tempo error | Phase error |
|---|---|---:|---:|---:|---:|---:|---:|
| Breaking | source | 13 | 0.1302 | 0.4928 | 0.0317 | 11.1176 | 0.2523 |
| Breaking | g1_target | 13 | 0.2527 | 0.7395 | 0.0379 | 15.4147 | 0.2522 |
| Chinese | source | 4 | 0.1364 | 0.4797 | 0.0253 | 23.4277 | 0.2606 |
| Chinese | g1_target | 4 | 0.2312 | 0.7254 | 0.0398 | 30.5932 | 0.2467 |
| Choreography | source | 6 | 0.1224 | 0.4804 | 0.0085 | 15.9992 | 0.2636 |
| Choreography | g1_target | 6 | 0.2120 | 0.6862 | 0.0585 | 19.7769 | 0.2543 |
| Classic | source | 48 | 0.1146 | 0.4727 | 0.0293 | 12.5545 | 0.2490 |
| Classic | g1_target | 48 | 0.2024 | 0.6577 | 0.0433 | 16.7393 | 0.2487 |
| Dai | source | 10 | 0.1116 | 0.4357 | 0.0014 | 17.8808 | 0.2533 |
| Dai | g1_target | 10 | 0.1976 | 0.5881 | 0.0301 | 16.0418 | 0.2518 |
| DunHuang | source | 2 | 0.1252 | 0.4618 | 0.0680 | 7.0822 | 0.2428 |
| DunHuang | g1_target | 2 | 0.2233 | 0.6719 | 0.0155 | 1.8186 | 0.2484 |
| Folk | source | 28 | 0.1195 | 0.4666 | 0.0245 | 16.2243 | 0.2543 |
| Folk | g1_target | 28 | 0.2274 | 0.6610 | 0.0276 | 13.6586 | 0.2546 |
| HanTang | source | 11 | 0.1085 | 0.4240 | 0.0517 | 13.5375 | 0.2479 |
| HanTang | g1_target | 11 | 0.2028 | 0.5886 | 0.0413 | 11.5715 | 0.2475 |
| Hiphop | source | 18 | 0.1287 | 0.4419 | 0.0155 | 34.5257 | 0.2532 |
| Hiphop | g1_target | 18 | 0.2652 | 0.6535 | 0.0312 | 35.1022 | 0.2477 |
| Jazz | source | 19 | 0.1287 | 0.4681 | 0.0450 | 26.1144 | 0.2640 |
| Jazz | g1_target | 19 | 0.2333 | 0.6765 | 0.0461 | 16.2243 | 0.2453 |
| Korean | source | 35 | 0.1188 | 0.4887 | 0.0367 | 23.0375 | 0.2522 |
| Korean | g1_target | 35 | 0.2365 | 0.7131 | 0.0425 | 23.0375 | 0.2512 |
| Kun | source | 1 | 0.1040 | 0.4844 | 0.0466 | 15.9992 | 0.2272 |
| Kun | g1_target | 1 | 0.1786 | 0.6316 | 0.0121 | 35.9992 | 0.2448 |
| Locking | source | 3 | 0.1201 | 0.4767 | 0.0718 | 21.0077 | 0.2447 |
| Locking | g1_target | 3 | 0.2417 | 0.7874 | 0.0370 | 21.0077 | 0.2386 |
| Miao | source | 10 | 0.1232 | 0.4558 | 0.0170 | 21.8570 | 0.2618 |
| Miao | g1_target | 10 | 0.2771 | 0.7367 | 0.0430 | 24.3937 | 0.2585 |
| Mix | source | 45 | 0.1210 | 0.4885 | 0.0298 | 23.0375 | 0.2550 |
| Mix | g1_target | 45 | 0.2265 | 0.7005 | 0.0474 | 23.4992 | 0.2512 |
| Popping | source | 21 | 0.1252 | 0.4740 | 0.0129 | 25.6673 | 0.2549 |
| Popping | g1_target | 21 | 0.2562 | 0.6875 | 0.0497 | 30.7955 | 0.2548 |
| ShenYun | source | 34 | 0.1148 | 0.4769 | 0.0120 | 13.0504 | 0.2501 |
| ShenYun | g1_target | 34 | 0.2021 | 0.6673 | 0.0457 | 17.1645 | 0.2500 |
| Street | source | 80 | 0.1286 | 0.4754 | 0.0229 | 26.1144 | 0.2562 |
| Street | g1_target | 80 | 0.2405 | 0.6876 | 0.0378 | 26.9146 | 0.2502 |
| Urban | source | 6 | 0.1333 | 0.5277 | 0.0095 | 33.5648 | 0.2562 |
| Urban | g1_target | 6 | 0.2274 | 0.7189 | 0.0239 | 21.9903 | 0.2464 |
| Wei | source | 8 | 0.1262 | 0.4851 | 0.0351 | 16.2243 | 0.2516 |
| Wei | g1_target | 8 | 0.2317 | 0.6690 | 0.0039 | 11.1176 | 0.2626 |
| jazz | source | 2 | 0.1216 | 0.4898 | 0.0420 | 21.0077 | 0.2398 |
| jazz | g1_target | 2 | 0.2107 | 0.7337 | 0.0199 | 11.1176 | 0.2515 |
| jiewu | source | 2 | 0.1216 | 0.4898 | 0.0420 | 21.0077 | 0.2398 |
| jiewu | g1_target | 2 | 0.2107 | 0.7337 | 0.0199 | 11.1176 | 0.2515 |

## Tempo summary (full sequence)

| tempo bin | stage | n seq | BAS | Event F1 | Impact corr | Tempo error | Phase error |
|---|---|---:|---:|---:|---:|---:|---:|
| slow | source | 13 | 0.1115 | 0.4211 | 0.0307 | 39.4685 | 0.2582 |
| slow | g1_target | 13 | 0.2027 | 0.5515 | 0.0374 | 30.8971 | 0.2474 |
| medium | source | 134 | 0.1220 | 0.4701 | 0.0299 | 18.9715 | 0.2505 |
| medium | g1_target | 134 | 0.2315 | 0.6764 | 0.0391 | 16.4325 | 0.2499 |
| fast | source | 56 | 0.1186 | 0.4890 | 0.0172 | 17.7080 | 0.2580 |
| fast | g1_target | 56 | 0.2235 | 0.7329 | 0.0436 | 26.8358 | 0.2523 |

Use the common core metrics for every style. Style-specific claims require adequate sample
counts and should be supplemented with style/emotion retrieval or blinded human evaluation.
A style with very few sequences is retained for transparency but must not be used for a strong claim.
