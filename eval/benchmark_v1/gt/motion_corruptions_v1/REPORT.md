# GT Motion Corruption Calibration

- Source manifest: `/home/tianhup/AudioMimic/eval/benchmark_v1/gt/motion_corruptions_v1/corruption_manifest.json`
- Dataset: AIST++ retargeted G1, declared crossmodal test split.
- Purpose: verify metric sensitivity; this is not a model ranking result.

## Aggregate Response

| variant | severity | n | energy | jerk p95 | static ratio | G1BAS | beat F1 | repeat similarity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | clean | 20 | 5.3125 | 2478.0347 | 0.0669 | 0.2931 | 0.2666 | 0.4738 |
| freeze | high | 20 | 5.4046 | 2496.4796 | 0.1655 | 0.2581 | 0.2098 | 0.2588 |
| freeze | low | 20 | 5.7317 | 2615.5992 | 0.0898 | 0.2714 | 0.2310 | 0.2969 |
| freeze | medium | 20 | 5.8044 | 2607.0204 | 0.1136 | 0.2680 | 0.2253 | 0.2782 |
| jitter | high | 20 | 5.4901 | 3634.6706 | 0.0000 | 0.2957 | 0.2753 | 0.4326 |
| jitter | low | 20 | 5.3196 | 2556.2474 | 0.0009 | 0.2923 | 0.2584 | 0.4682 |
| jitter | medium | 20 | 5.3588 | 2877.2160 | 0.0000 | 0.2868 | 0.2572 | 0.4555 |
| lowpass | high | 20 | 1.5482 | 211.5714 | 0.0779 | 0.2979 | 0.2569 | 0.5770 |
| lowpass | low | 20 | 4.1969 | 1229.8002 | 0.0697 | 0.2951 | 0.2719 | 0.5162 |
| lowpass | medium | 20 | 3.0330 | 586.4496 | 0.0749 | 0.2977 | 0.2853 | 0.5554 |
| repeat | high | 20 | 6.1688 | 2807.6575 | 0.0709 | 0.2657 | 0.2255 | 0.4543 |
| repeat | low | 20 | 6.8684 | 2910.9462 | 0.0621 | 0.2856 | 0.2448 | 0.3719 |
| repeat | medium | 20 | 6.4880 | 2815.3371 | 0.0610 | 0.2910 | 0.2588 | 0.3592 |

## Directional Checks

| check | result | observed |
|---|---|---|
| jitter increases jerk p95 | **PASS** | mean delta +1156.635825 |
| lowpass decreases jerk p95 | **PASS** | mean delta -2266.463370 |
| freeze increases static ratio | **PASS** | mean delta +0.098617 |

## Interpretation

- Jitter should primarily increase jerk and perturbation magnitude.
- Low-pass corruption should reduce high-frequency jerk and usually reduce energy.
- Freeze corruption should increase the static ratio and reduce local energy.
- Repeat corruption is reported with a self-similarity diagnostic; it is not treated as a single universal quality score.
- Beat metrics are core for music-beat alignment, but are diagnostic for generic dance quality and should not be used as the sole overall-quality criterion.
