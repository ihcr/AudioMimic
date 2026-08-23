# M3 Music-Condition Causal Ablation: Formal 30 s Round

Date: 2026-08-22

The reproducible per-trajectory rerun is in `song012/analysis_v2/` and
`song065/analysis_v2/`; the machine-generated aggregate is in
`aggregate_v2/REPORT.md`.

## Design

- Songs: held-out test-cache sequences 012 and 065.
- Sampling seeds: 1234, 2345, 3456.
- Conditions: paired, wrong song, same song shifted by +4 s, and exact null sidecar.
- Duration: 30 s (896 committed frames because duration is quantized to 8-frame C4 blocks).
- Total: 24 generated trajectories.
- Within each song/seed block, checkpoint, parent, K64 history, start frame, runtime, and
  random seed are identical. Only the music condition changes.
- Every trajectory is scored against the unshifted target song associated with its fixed
  K64 motion history.

## Aggregate results

Mean +/- sample standard deviation over 2 songs x 3 seeds:

| Condition | RMSE from paired (rad) | Normalized effect | Energy | BAS | Impact corr. |
|---|---:|---:|---:|---:|---:|
| Paired | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.829 +/- 0.213 | 0.252 +/- 0.041 | 0.031 +/- 0.021 |
| Wrong | 0.169 +/- 0.043 | 0.500 +/- 0.141 | 1.624 +/- 0.535 | 0.271 +/- 0.021 | 0.029 +/- 0.029 |
| Shifted +4 s | 0.174 +/- 0.081 | 0.514 +/- 0.243 | 1.723 +/- 0.287 | 0.245 +/- 0.046 | 0.034 +/- 0.029 |
| Null | 0.163 +/- 0.015 | 0.479 +/- 0.036 | 2.444 +/- 0.218 | 0.274 +/- 0.024 | 0.025 +/- 0.022 |

The condition changes the trajectory by roughly half of the paired motion scale. This is
strong evidence that the trained M3 sidecar is active and causally affects generation.

## Paired-condition advantage

Paired-minus-control differences over the six matched song/seed blocks:

| Metric | Control | Mean paired-control | Paired wins | Wilcoxon p |
|---|---|---:|---:|---:|
| BAS | Wrong | -0.019 | 2/6 | 0.438 |
| BAS | Shifted | +0.007 | 3/6 | 1.000 |
| BAS | Null | -0.022 | 2/6 | 0.438 |
| Impact correlation | Wrong | +0.002 | 4/6 | 0.563 |
| Impact correlation | Shifted | -0.003 | 3/6 | 0.688 |
| Impact correlation | Null | +0.006 | 5/6 | 0.219 |

No automatic alignment comparison is significant. Paired music has lower mean BAS than
wrong and null music, while onset-impact correlations are weak in every condition.

## Interpretation

1. **Music sensitivity is supported.** M3 does not ignore its music sidecar.
2. **Correct music alignment is not supported.** The paired condition does not reliably
   outperform wrong, shifted, or null controls on the current automatic metrics.
3. **The sidecar changes dynamics strongly.** Null has substantially higher energy and jerk,
   indicating that the sidecar often regularizes or suppresses the unconditional parent rather
   than consistently placing motion accents on the correct music events.
4. This result is generator-level. Running all 24 controls through SONIC would test tracker
   retention, but it cannot establish a music mapping that is absent at the reference level.

## Consequence for the paper

The current model supports the claim "music-conditioned streaming generation" in the
mechanistic sense, but not yet the stronger claim "the robot dances better to the paired
music." Before the main SONIC matrix, improve or retrain the music control objective and
evaluate with event response, phase, beat precision/recall, and blinded paired-audio ratings.

The next diagnostic should separate RMS and predicted-FMS contributions (`RMS-only`,
`FMS-only`, both, null) and measure sidecar output magnitude. This will identify whether the
problem comes from high-level history conditioning, future-event prediction, or the adapter's
motion response.
