# Cross-Model Rhythm Suite Analysis

Artifacts generated from saved motions on 2026-06-22. Each row uses 3265 test motions and the same FK/reference path.

| Model | FKBAS | BeatF1 | Recall | Density | Unmatched | WristF1 | FootF1 | ContactBeat | NearSupport | NoSupport | HighLift | WristJerk | FootJerk | Dist | Div | Pen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| librosa35_2000 | 0.254 | 0.214 | 0.176 | 0.535 | 0.683 | 0.216 | 0.204 | 0.943 | 0.997 | 0.001 | 0.003 | 968.603 | 1128.335 | 9.254 | 11.366 | 0.035 |
| gaussian_beat_1d_1000 | 0.231 | 0.191 | 0.155 | 0.534 | 0.712 | 0.186 | 0.196 | 0.852 | 0.978 | 0.023 | 0.030 | 1171.864 | 1289.061 | 9.200 | 20.537 | 0.080 |
| wav2clip_stft_r02_2000 | 0.238 | 0.198 | 0.161 | 0.541 | 0.705 | 0.192 | 0.203 | 0.929 | 0.996 | 0.001 | 0.005 | 1058.150 | 1173.285 | 8.911 | 12.844 | 0.041 |
| v3_r03_1000_pred | 0.229 | 0.205 | 0.162 | 0.476 | 0.665 | 0.210 | 0.203 | 0.390 | 0.862 | 0.184 | 0.076 | 663.992 | 889.652 | 6.056 | 18.484 | 0.048 |
| v3b_1500_pred | 0.243 | 0.211 | 0.169 | 0.515 | 0.675 | 0.215 | 0.202 | 0.564 | 0.962 | 0.058 | 0.038 | 1132.150 | 1049.545 | 5.782 | 14.093 | 0.052 |
| v5_yaw_1000_pred | 0.260 | 0.234 | 0.190 | 0.528 | 0.644 | 0.262 | 0.214 | 0.420 | 0.873 | 0.179 | 0.111 | 1974.132 | 1010.937 | 3.991 | 16.561 | 0.076 |
| beat8d_1000_auto | 0.230 | 0.193 | 0.155 | 0.518 | 0.704 | 0.192 | 0.195 | 0.832 | 0.971 | 0.031 | 0.027 | 1180.010 | 1333.564 | 9.118 | 14.489 | 0.158 |
| beat8d_beatness_1000_pred | 0.263 | 0.227 | 0.188 | 0.549 | 0.663 | 0.226 | 0.218 | 0.856 | 0.979 | 0.022 | 0.023 | 1594.927 | 1480.054 | 10.232 | 14.312 | 0.217 |

## Rankings

- Old rhythm winners by `G1BeatF1`: v5_yaw_1000_pred, beat8d_beatness_1000_pred, librosa35_2000, v3b_1500_pred, v3_r03_1000_pred
- Old FK BAS winners by `G1FKBAS`: beat8d_beatness_1000_pred, v5_yaw_1000_pred, librosa35_2000, v3b_1500_pred, wav2clip_stft_r02_2000
- New safest foot-contact-on-beat: librosa35_2000, wav2clip_stft_r02_2000, beat8d_beatness_1000_pred, gaussian_beat_1d_1000, beat8d_1000_auto
- New lowest no-near-support rate: librosa35_2000, wav2clip_stft_r02_2000, beat8d_beatness_1000_pred, gaussian_beat_1d_1000, beat8d_1000_auto
- New lowest ground penetration: librosa35_2000, wav2clip_stft_r02_2000, v3_r03_1000_pred, v3b_1500_pred, v5_yaw_1000_pred

## Interpretation

- `v5_yaw_1000_pred` is the clearest old-metric trap: it leads `G1BeatF1` and has very low `G1Dist`, but it is wrist-heavy and unstable around support: `WristBeatF1=0.262`, `FootBeatF1=0.214`, `FootContactOnBeat=0.420`, `NoNearSupport=0.179`, `FootHighLift=0.111`, `WristJerk=1974`.
- `beat8d_beatness_1000_pred` proves predicted beatness is rhythm-active: relative to `beat8d_1000_auto`, `G1FKBAS` rises `0.230 -> 0.263`, `G1BeatF1` rises `0.193 -> 0.227`, and recall rises `0.155 -> 0.188`. But it fails robot quality through `G1GroundPenetration=0.217` and high endpoint jerk, so it should not be promoted.
- `v3_r03_1000_pred` looks attractive on old quality/diversity (`G1Dist=6.056`, `G1Div=18.484`), but the new support metrics explain the render caveat: `FootContactOnBeat=0.390`, `NoNearSupport=0.184`, `FootHighLift=0.076`.
- `librosa35_2000` and `wav2clip_stft_r02_2000` are not rhythm winners, but they are the clean-support references: near-support is about `0.996` and no-near-support is around `0.001`. This gives a realistic target for V6 support/contact gates.
- Across all models, `G1BeatDensityRatio` stays around `0.476-0.549`, so the system underproduces FK motion-beat events relative to audio beats. At the same time `G1UnmatchedMotionBeatRate` remains high (`0.644-0.712`), meaning the bottleneck is not only too few events; many generated events are also off-target.

## Bottom Line

The new suite explains substantially more than the old metric set. The old set can rank checkpoints by rhythm score and coarse motion distribution, but it cannot separate wrist-only rhythm hacks, missing support on beats, high-lift/hovering feet, penetration, or unmatched event density. The cross-model result supports using the new suite as the default gate before training or accepting V6-style checkpoints.
