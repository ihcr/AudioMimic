# Model Expansion Status

## Current artifact boundary

The latest `Musics2Dance-prior-dev` release currently contains the pure
Commit Forcing d16/q0/codec closure. That release is unconditional with respect
to music. It can provide an M0-style generator baseline, but it cannot
re-generate the old M2 or M3 music-conditioned routes.

The existing M2/M3 files in `onlinegeneratedmotion/` and `eval/mrt2_comparison/`
are offline generated PKL artifacts. They can be evaluated and replayed through
SONIC, but they are not a current music-conditioned inference checkpoint.

## Formal collection state

The target matrix is frozen in:

- `eval/benchmark_v1/formal/EXPANSION_MATRIX.csv`
- `eval/benchmark_v1/formal/EXPANSION_MATRIX.json`

It contains 72 cells:

```text
4 routes x 3 songs (012/065/098) x 3 sampling seeds x {M_ref, M_exec}
```

Current balanced-cell status:

| stage | available | pending |
|---|---:|---:|
| `M_ref` | 5 | 31 |
| `M_exec` | 3 | 33 |

The M3 012/065 pilot and M2 artifacts from different training seeds remain in
`MODEL_REPORT.md`, but are not silently counted as balanced sampling-seed runs.

## Required next acquisition

1. Obtain or publish the music-conditioned M2/M3 inference checkpoint together
   with its feature-cache contract, training seed, sampling seed and audio
   alignment convention.
2. Generate `M_ref` for songs 012, 065 and 098 with sampling seeds 1234, 2345
   and 3456. Keep the same checkpoint, duration, FPS and start-time convention.
3. Replay each exact `M_ref` through SONIC with the frozen initialization,
   `packet_mode`, reference FPS and timing protocol. Record `M_exec` and SONIC
   feedback in separate run directories.
4. Run the frozen evaluator and update the matrix only when both motion and
   audio alignment metadata are present.

No new claim about music-conditioned generation should be made until steps 1-4
are complete. The current M2/M3 PKL results support pipeline diagnostics, not
the final multi-song online-generation claim.
