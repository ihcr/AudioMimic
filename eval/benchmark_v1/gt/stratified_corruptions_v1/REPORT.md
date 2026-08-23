# GT Motion Corruption Calibration

- Source manifest: `/home/tianhup/AudioMimic/eval/benchmark_v1/gt/stratified_corruptions_v1/corruption_manifest.json`
- Dataset: AIST++ retargeted G1, declared crossmodal test split.
- Purpose: verify metric sensitivity; this is not a model ranking result.

## Aggregate Response

| variant | severity | n | energy | jerk p95 | static ratio | G1BAS | beat F1 | repeat similarity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | clean | 38 | 5.2175 | 2576.8990 | 0.0476 | 0.2595 | 0.2316 | 0.4323 |
| freeze | high | 38 | 5.2887 | 2593.7198 | 0.1051 | 0.2404 | 0.2025 | 0.3034 |
| freeze | low | 38 | 5.4570 | 2660.7959 | 0.0610 | 0.2489 | 0.2142 | 0.3324 |
| freeze | medium | 38 | 5.5078 | 2656.5533 | 0.0748 | 0.2465 | 0.2107 | 0.3180 |
| jitter | high | 38 | 5.3993 | 3581.1207 | 0.0000 | 0.2553 | 0.2260 | 0.3965 |
| jitter | low | 38 | 5.2248 | 2632.1203 | 0.0007 | 0.2621 | 0.2395 | 0.4267 |
| jitter | medium | 38 | 5.2621 | 2881.8795 | 0.0000 | 0.2569 | 0.2261 | 0.4153 |
| lowpass | high | 38 | 1.5597 | 217.1761 | 0.0550 | 0.2567 | 0.2243 | 0.5296 |
| lowpass | low | 38 | 4.1482 | 1312.8261 | 0.0494 | 0.2595 | 0.2350 | 0.4733 |
| lowpass | medium | 38 | 3.0006 | 618.8286 | 0.0527 | 0.2594 | 0.2395 | 0.5062 |
| repeat | high | 38 | 5.7968 | 2781.9233 | 0.0498 | 0.2437 | 0.2086 | 0.4037 |
| repeat | low | 38 | 6.1282 | 2833.6933 | 0.0450 | 0.2562 | 0.2209 | 0.3792 |
| repeat | medium | 38 | 5.9553 | 2785.4151 | 0.0444 | 0.2592 | 0.2285 | 0.3562 |

## Directional Checks

| check | result | observed |
|---|---|---|
| jitter increases jerk p95 | **PASS** | mean delta +1004.221652 |
| lowpass decreases jerk p95 | **PASS** | mean delta -2359.722881 |
| freeze increases static ratio | **PASS** | mean delta +0.057537 |

## Interpretation

- Jitter should primarily increase jerk and perturbation magnitude.
- Low-pass corruption should reduce high-frequency jerk and usually reduce energy.
- Freeze corruption should increase the static ratio and reduce local energy.
- Repeat corruption is reported with a self-similarity diagnostic; it is not treated as a single universal quality score.
- Beat metrics are diagnostic only here. They should not be used as the sole dance-quality criterion.
