# Formal Model Summary

This is a descriptive summary of currently available artifacts. It is not a
final ranking because M0/M2/M4 are primarily song098 repeats while M3 uses
songs 012/065 and a different pilot collection protocol.

## 1. Dance quality

These metrics describe movement quality without assuming that more motion is always better.

| route/stage | n | energy | amplitude | vel P95 | acc P95 | jerk P95 | static | repeat | C4 vel jump |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0/M_ref | 3 | 2.5545 | 0.7442 | 3.49 | 43.56 | 841.20 | 0.0039 | 0.0015 | 1.153 |
| M0/M_exec | 3 | 0.9996 | 0.6342 | 2.20 | 25.35 | 648.95 | 0.0106 | 0.0000 | 0.443 |
| M2/M_ref | 3 | 1.9339 | 0.6510 | 3.05 | 37.32 | 718.09 | 0.0002 | 0.0000 | 1.020 |
| M2/M_exec | 3 | 0.9100 | 0.6872 | 2.04 | 28.94 | 886.09 | 0.0103 | 0.0000 | 0.583 |
| M3/M_ref | 2 | 1.9520 | 0.7302 | 2.98 | 35.04 | 648.96 | 0.0000 | 0.0000 | 0.982 |
| M3/M_exec | 2 | 0.8589 | 0.6035 | 1.93 | 23.32 | 491.65 | 0.0028 | 0.0000 | 0.885 |
| M4/M_ref | 3 | 1.7892 | 0.6440 | 2.91 | 36.10 | 691.75 | 0.0015 | 0.0000 | 0.935 |
| M4/M_exec | 3 | 0.8071 | 0.6277 | 1.95 | 22.34 | 568.62 | 0.0126 | 0.0000 | 0.419 |

## 2. Physical and execution quality

FSR is a foot-sliding proxy, PFC is a foot-contact proxy, penetration measures
ground violation, and root height is a stability diagnostic.

| route/stage | n | FSR proxy | PFC proxy | foot contact | penetration | root height min | root displacement |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0/M_ref | 3 | 0.8049 | 0.0778 | 0.6930 | 0.0057 | 0.7097 | 1.2165 |
| M0/M_exec | 3 | 0.5640 | 0.1106 | 0.3419 | 0.0074 | 0.6370 | 0.9874 |
| M2/M_ref | 3 | 0.7860 | 0.0651 | 0.7635 | 0.0054 | 0.7114 | 0.6161 |
| M2/M_exec | 3 | 0.4823 | 0.1180 | 0.3299 | 0.0053 | 0.6328 | 0.8614 |
| M3/M_ref | 2 | 0.8088 | 0.0853 | 0.6755 | 0.0014 | 0.7516 | 2.6209 |
| M3/M_exec | 2 | 0.5901 | 0.0389 | 0.9710 | 0.0006 | 0.6584 | 1.6019 |
| M4/M_ref | 3 | 0.8017 | 0.0570 | 0.8092 | 0.0081 | 0.7462 | 0.8476 |
| M4/M_exec | 3 | 0.5893 | 0.0927 | 0.3890 | 0.0090 | 0.6584 | 1.5749 |

## 3. Music and beat alignment

BAS is only one item in this suite. We also report speed/impact correlation,
response lag and both BAS directions. Beat F1, onset precision/recall, tempo
error, phase error, semantic retrieval and human preference are reported according
to artifact availability. The legacy M0/M2/M4 execution logs were exported to
portable measured-motion PKLs, so their event metrics are included below; semantic
retrieval and human preference still require a separate calibrated protocol.

| route/stage | n | speed corr | impact corr | impact lag (s) | BAS M->A | BAS A->M | audio beats | motion beats |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0/M_ref | 3 | -0.0107 | 0.0244 | -0.056 | 0.2556 | 0.4552 | 107.3 | 60.3 |
| M0/M_exec | 3 | -0.0176 | 0.0192 | -0.313 | 0.1981 | 0.3476 | 99.0 | 56.3 |
| M2/M_ref | 3 | -0.0169 | 0.0252 | 0.322 | 0.2469 | 0.3860 | 99.0 | 63.3 |
| M2/M_exec | 3 | -0.0260 | 0.0121 | -0.067 | 0.2271 | 0.3784 | 99.0 | 58.7 |
| M3/M_ref | 2 | -0.0012 | 0.0207 | -0.750 | 0.2633 | 0.4173 | 103.0 | 65.0 |
| M3/M_exec | 2 | -0.0043 | 0.0136 | 0.417 | 0.3087 | 0.4802 | 103.0 | 64.0 |
| M4/M_ref | 3 | -0.0135 | 0.0303 | 0.356 | 0.2422 | 0.3945 | 99.0 | 60.7 |
| M4/M_exec | 3 | -0.0178 | 0.0142 | -0.293 | 0.1773 | 0.3367 | 99.0 | 52.0 |

### Extended beat event suite

| route/stage | n | event P | event R | event F1 | tempo error | phase error |
|---|---:|---:|---:|---:|---:|---:|
| M0/M_ref | 3 | 0.6349 | 0.7348 | 0.6803 | 35.0361 | 0.2613 |
| M0/M_exec | 3 | 0.6227 | 0.7609 | 0.6849 | 40.9237 | 0.2558 |
| M2/M_ref | 3 | 0.6176 | 0.7508 | 0.6777 | 39.6270 | 0.2531 |
| M2/M_exec | 3 | 0.6227 | 0.7340 | 0.6738 | 39.0775 | 0.2493 |
| M3/M_ref | 2 | 0.6676 | 0.7630 | 0.7074 | 25.3708 | 0.2543 |
| M3/M_exec | 2 | 0.6151 | 0.7143 | 0.6551 | 32.6968 | 0.2428 |
| M4/M_ref | 3 | 0.6270 | 0.7677 | 0.6902 | 40.9237 | 0.2606 |
| M4/M_exec | 3 | 0.6226 | 0.7441 | 0.6780 | 32.4841 | 0.2535 |

## 4. SONIC retention

| route | n | energy retention | amplitude ratio | jerk ratio | FSR change | BAS retention | extra lag (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 3 | 0.4262 | 0.9940 | 0.5330 | -0.2240 | 0.8604 | -1.073 |
| M2 | 3 | 0.4680 | 1.0484 | 0.8272 | -0.2836 | 0.8656 | 0.713 |
| M3 | 2 | 0.4396 | 0.8291 | 0.7570 | -0.2187 | 1.1914 | 1.167 |
| M4 | 3 | 0.4516 | 1.0332 | 0.5568 | -0.1920 | 0.7320 | -1.033 |

## Interpretation

The current table is suitable for checking the evaluator and identifying
tracker retention patterns. It is not evidence that one route is better than
another: song identity, training/sampling seed, audio alignment and SONIC
repeat protocol are not yet balanced. The next formal claim requires the
72-cell expansion matrix to be populated with the same protocol.
