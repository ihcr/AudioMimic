# O-Human to O-G1 Formal GT Results

O-Human is the original SMPL/SMPLH paired motion. O-G1 is the GMR-retargeted G1 motion
and the training domain of the diffusion generator. This table measures GMR changes before
any generator or SONIC execution result is introduced.

| dataset | n | BAS human | BAS G1 | delta | Beat F1 human | Beat F1 G1 | delta | activity corr | root corr | activity ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aistpp | 20 | 0.3340 | 0.3580 | +0.0240 | 0.6750 | 0.6884 | +0.0134 | 0.6990 | 0.9270 | 0.8172 |
| finedance | 18 | 0.1199 | 0.2443 | +0.1244 | 0.4591 | 0.6461 | +0.1869 | 0.0687 | 0.0938 | 0.7787 |

## FineDance music-pairing integrity check

This is a dataset-integrity diagnostic, separate from GMR loss. The same-ID
audio is compared with a cyclic wrong-song control and all other test songs.

- paired-vs-wrong best-lag correlation margin: **0.0736**
- paired-vs-wrong event-F1 margin: **0.0270**
- paired correlation top-1 rate among wrong songs: **0.2222**

The paired distribution is a positive reference and the wrong-song
distribution is a negative control; neither is a perfect human-quality score.
Source: `/home/tianhup/AudioMimic/eval/benchmark_v1/gt/finedance_music_pairing_v1/pairing_metrics.json`.

## AIST++ music-pairing integrity check

This is a dataset-integrity diagnostic, separate from GMR loss. The same-ID
audio is compared with a cyclic wrong-song control and all other test songs.

- paired-vs-wrong best-lag correlation margin: **0.0381**
- paired-vs-wrong event-F1 margin: **0.0133**
- paired correlation top-1 rate among wrong songs: **0.3000**

The paired distribution is a positive reference and the wrong-song
distribution is a negative control; neither is a perfect human-quality score.
Source: `/home/tianhup/AudioMimic/eval/benchmark_v1/gt/aist_music_pairing_v1/pairing_metrics.json`.
