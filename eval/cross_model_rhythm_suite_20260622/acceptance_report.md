# Cross-Model Rhythm Suite Acceptance Report

## Status

`completed` for `EXP-20260617-finedance-g1-beat8d-motion-beatness_r01_b64_acc8` and `EXP-20260622-cross-model-rhythm-suite`.

Latest verified checkpoint: `runs/train/EXP-20260617-finedance-g1-beat8d-motion-beatness_r01_b64_acc8/weights/train-1000.pt`.

Verdict: reject `beat8d_beatness_1000_pred` as a mainline checkpoint, accept the new rhythm suite as the default ckpt500/ckpt1000 gate for V6a/contact-support-aware runs.

## What I Checked

- Training process state: no live `train.py`, `accelerate`, or G1 eval process remained after the checks; the tmux pane is idle at shell prompt.
- Checkpoint evidence: `train-1000.pt` exists and the training log records `[MODEL SAVED at Epoch 1000]`.
- Eval evidence: training log records full eval completion for `ckpt1000_pred_controls`, `oracle_controls`, `zero_beatness`, and `zero_all_controls`.
- Cross-model eval evidence: `eval/cross_model_rhythm_suite_20260622/summary.json` has 8 rows; every row has `3265` motions, `metrics.json`, and `failure_panel.json`.
- Baseline set: included Librosa35, 1D GaussianBeat, Wav2CLIP/STFT r02, V3, V3b, V5, 8D-only, and 8D+motion_beatness.

## Rhythm And Distribution

| model | G1BAS ↑ | G1FK ↑ | F1 ↑ | Rec ↑ | Density ↑ | Unmatched ↓ | Dist ↓ | Div ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| librosa35_2000 | 0.2413 | 0.2544 | 0.2139 | 0.1757 | 0.5350 | 0.6834 | 9.2544 | 11.3661 |
| gaussian_beat_1d_1000 | 0.2072 | 0.2311 | 0.1913 | 0.1546 | 0.5341 | 0.7117 | 9.2000 | 20.5369 |
| wav2clip_stft_r02_2000 | 0.2237 | 0.2377 | 0.1979 | 0.1607 | 0.5414 | 0.7051 | 8.9113 | 12.8445 |
| v3_r03_1000_pred | 0.2337 | 0.2286 | 0.2050 | 0.1615 | 0.4761 | 0.6654 | 6.0560 | 18.4838 |
| v3b_1500_pred | 0.2435 | 0.2429 | 0.2106 | 0.1687 | 0.5149 | 0.6754 | 5.7822 | 14.0929 |
| v5_yaw_1000_pred | 0.2247 | 0.2604 | 0.2340 | 0.1903 | 0.5280 | 0.6441 | 3.9908 | 16.5609 |
| beat8d_1000_auto | 0.2169 | 0.2298 | 0.1934 | 0.1551 | 0.5178 | 0.7041 | 9.1177 | 14.4888 |
| beat8d_beatness_1000_pred | 0.2445 | 0.2629 | 0.2269 | 0.1882 | 0.5488 | 0.6633 | 10.2321 | 14.3124 |

## Robot And Naturalness

| model | FootF1 ↑ | ContactBeat ↑ | NearSupport ↑ | NoSupport ↓ | HighLift ↓ | WristJerk ↓ | FootJerk ↓ | FootSlide ↓ | Ground ↓ | Drift ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| librosa35_2000 | 0.2042 | 0.9426 | 0.9966 | 0.0010 | 0.0032 | 968.6 | 1128.3 | 0.5306 | 0.0352 | 0.2022 |
| gaussian_beat_1d_1000 | 0.1961 | 0.8517 | 0.9777 | 0.0235 | 0.0304 | 1171.9 | 1289.1 | 0.5964 | 0.0803 | 0.2709 |
| wav2clip_stft_r02_2000 | 0.2029 | 0.9292 | 0.9963 | 0.0014 | 0.0048 | 1058.2 | 1173.3 | 0.5525 | 0.0408 | 0.2726 |
| v3_r03_1000_pred | 0.2028 | 0.3901 | 0.8623 | 0.1843 | 0.0755 | 664.0 | 889.7 | 0.7373 | 0.0483 | 0.0656 |
| v3b_1500_pred | 0.2023 | 0.5643 | 0.9620 | 0.0577 | 0.0382 | 1132.2 | 1049.5 | 0.7549 | 0.0517 | 0.2810 |
| v5_yaw_1000_pred | 0.2142 | 0.4196 | 0.8727 | 0.1790 | 0.1112 | 1974.1 | 1010.9 | 0.8261 | 0.0757 | 0.7159 |
| beat8d_1000_auto | 0.1952 | 0.8316 | 0.9710 | 0.0308 | 0.0271 | 1180.0 | 1333.6 | 0.6480 | 0.1580 | 0.2894 |
| beat8d_beatness_1000_pred | 0.2178 | 0.8565 | 0.9790 | 0.0218 | 0.0230 | 1594.9 | 1480.1 | 0.6865 | 0.2168 | 0.3045 |

## Findings

- `beat8d_beatness_1000_pred` is a rhythm-control success but an acceptance failure: it leads `G1FK=0.2629` and beats 8D-only on F1/recall, but fails `Dist`, `Ground`, and endpoint jerk.
- `v5_yaw_1000_pred` is the clearest metric-hacking risk: old metrics rank it highly, but new metrics expose wrist-heavy rhythm, poor beat-time foot contact, high foot lift, high wrist jerk, and high root drift.
- `librosa35_2000` and `wav2clip_stft_r02_2000` remain useful naturalness anchors: they are not rhythm winners, but their support/contact metrics are much cleaner than V3/V5.
- Across all rows, `Density` remains around `0.476-0.549` and `Unmatched` remains high at `0.644-0.712`, so the rhythm bottleneck is both low event coverage and off-target generated events.

## Next Action

Stop `beat8d_beatness_1000_pred` at ckpt1000; do not extend to 1500. Use this acceptance report and the new suite as the default gate for V6a/contact-support-aware training.
