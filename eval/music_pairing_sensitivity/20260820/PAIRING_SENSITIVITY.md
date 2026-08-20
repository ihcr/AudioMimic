# Music Pairing Sensitivity: Existing M0/M2/M4 Exports

Date: 2026-08-20

This is a fixed-trajectory pairing audit. It does not regenerate motion under shuffled or silent conditions and therefore cannot establish causal use of music by the generator.

| Source | Route | N | Impact corr. paired/null | Impact rank | BAS M2M paired/null | BAS rank | BAS paired-wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| execution | M0 | 3 | -0.001/0.005 | 40.2% | 0.198/0.237 | 3.4% | -0.012 |
| execution | M2 | 3 | 0.006/0.007 | 41.4% | 0.227/0.245 | 24.1% | +0.001 |
| execution | M4 | 3 | 0.006/0.006 | 48.3% | 0.176/0.219 | 9.2% | -0.044 |
| reference | M0 | 2 | 0.012/0.001 | 70.7% | 0.257/0.253 | 46.6% | +0.013 |
| reference | M2 | 3 | -0.007/0.003 | 26.4% | 0.247/0.265 | 25.3% | -0.005 |
| reference | M4 | 3 | -0.016/0.003 | 17.2% | 0.242/0.253 | 33.3% | +0.001 |

A high rank means the correct audio clock scores above most circular shifts. It is a temporal-alignment diagnostic, not a calibrated p-value. M0 is the unconditional negative control; similar M0 and M2/M4 ranks weaken a music-conditioning claim.

The next causal experiment must regenerate each seed with paired, time-shifted, shuffled, and silent music using the same M2 checkpoint and sampling noise.
