# M3 Music-Condition Causal Ablation: 8 s Pilot

Date: 2026-08-22

## Question

Does M3 merely accept a music tensor, or does changing that tensor causally change the
generated dance? If it changes the dance, is the paired condition consistently better
aligned with the target song than wrong, shifted, or null conditions?

## Controlled protocol

All variants use the same:

- M3 checkpoint and frozen unconditional parent;
- motion sequence `012` and its exact K64 motion history;
- start frame 0 and H8/C4 runtime;
- sampling seed within each four-way set;
- 8 s / 240-frame rollout.

Only the music condition changes:

| Variant | Condition |
|---|---|
| paired012 | sequence 012 cache at the correct clock |
| wrong065 | sequence 065 cache with the 012 K64 motion history |
| shifted012_4s | sequence 012 cache advanced by 120 motion frames (4 s) |
| null | exact `MRT2Condition.null()`, bypassing both M3 music sidecars |

All variants are scored against the unshifted sequence-012 audio cropped from 4.267 s.

## Results

### Sampling seed 1234

| Variant | RMSE from paired (rad) | Normalized effect | BAS | Impact corr. | Energy |
|---|---:|---:|---:|---:|---:|
| paired012 | 0.000 | 0.000 | 0.279 | 0.047 | 0.638 |
| wrong065 | 0.310 | 0.803 | 0.220 | 0.080 | 1.570 |
| shifted012_4s | 0.353 | 0.914 | 0.190 | 0.080 | 0.790 |
| null | 0.358 | 0.926 | 0.290 | 0.053 | 1.321 |

### Sampling seed 2345

| Variant | RMSE from paired (rad) | Normalized effect | BAS | Impact corr. | Energy |
|---|---:|---:|---:|---:|---:|
| paired012 | 0.000 | 0.000 | 0.214 | 0.118 | 0.711 |
| wrong065 | 0.326 | 0.842 | 0.250 | 0.154 | 1.215 |
| shifted012_4s | 0.304 | 0.785 | 0.255 | 0.127 | 1.107 |
| null | 0.249 | 0.643 | 0.194 | 0.141 | 1.137 |

Two-seed mean paired-distance RMSE is 0.318 rad for wrong music, 0.328 rad for shifted
music, and 0.303 rad for null music. These are large changes relative to the paired motion
scale, so the sidecar is not being ignored.

Two-seed mean BAS is 0.246 for paired, 0.235 for wrong, 0.222 for shifted, and 0.242 for
null. The paired margin is small and is not consistent per seed. Mean impact correlation is
0.083 for paired, lower than wrong (0.117), shifted (0.104), and null (0.097).

## Conclusion

The pilot supports **causal sensitivity**: changing or removing the music condition changes
the generated trajectory substantially under fixed motion history and sampling noise.

It does not yet support **correct music alignment**: paired music is not consistently better
than the controls, and the correlation values are weak or unstable on an 8 s window. The
current evidence therefore supports "music-conditioned" but not yet the stronger claim
"music-responsive dance with superior paired alignment."

## Next formal experiment

Run at least 30-60 s, 3 generation seeds, and both available held-out songs, then add a
blinded paired-music preference study. SONIC should only be run after the generator-level
paired condition shows a repeatable advantage; tracker evaluation cannot repair a weak
music-to-motion mapping.
