# Formal Music-to-G1 Four-Layer Benchmark

The generator directly outputs G1 motion. GMR is only used to create O-G1 GT.

- `O_Human`: original SMPL/SMPLH paired motion/audio.
- `O_G1`: GMR/retargeted G1 target and generator training domain.
- `M_ref`: generator-produced G1 reference.
- `M_exec`: SONIC-executed G1 motion.

GT records: 38

Existing model records: 5

Only records with all required artifacts and a fixed shared clock may enter the formal result table.

The existing model-stage summary is in `MODEL_REPORT.md`,
`model_stage_results.csv`, and `model_stage_results.json`. It contains 11
`M_ref` rows and 11 `M_exec` rows, including 9 M0/M2/M4 SONIC repeats and the
paired M3 012/065 results. These are diagnostic progress results, not a
balanced model ranking.

Route-level descriptive means are in `MODEL_SUMMARY.md`; they are diagnostic
only and must not be used as a cross-route ranking until the 72-cell matrix is
balanced.

The extended beat event, tempo, phase, and correlation report is available in
`model_music_extended/REPORT.md` and `model_music_extended/REPORT_ZH.md`.

The dataset-level GT calibration report, including the Chinese version, is in
`../benchmark_v1/gt/gt_oracle_suite_v2/REPORT.md` and
`../benchmark_v1/gt/gt_oracle_suite_v2/REPORT_ZH.md`.

The target collection matrix is frozen in `EXPANSION_MATRIX.csv` and
`EXPANSION_MATRIX.json`: 4 routes x 3 songs (012/065/098) x 3 sampling seeds
(1234/2345/3456) x 2 stages. It currently contains 72 target cells, with 5
balanced reference cells and 3 execution cells available. M3's existing pilot
and M2's alternate training-seed artifacts remain separate until their seed
metadata and execution repeats are aligned.
