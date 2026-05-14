# EXP-20260511-finedance-g1-librosa-fullctx

Status: finished
Owner: Yukun
Created: 2026-05-12
Last Updated: 2026-05-12 UTC

## Evidence Scope

This spec was created as a truth-first sync of the current dirty `diffusion` worktree, not as an original experiment proposal. Claims below are grounded in local diffs, sbatch scripts, `sacct`, validation logs, and metric JSON files. Where intent is inferred from run names or code changes, it is marked as inferred.

## Research Question

For retargeted FineDance G1 dance generation, do low-dimensional Librosa-style music features become more useful when rebuilt from full-song context, and does dropping the binary music-beat channel (`baseline34`) help when beat conditioning already supplies a separate distance signal?

## Hypothesis

Inferred from the current code diff and run names:

- Full-song-context Librosa features may avoid per-slice feature artifacts from extracting each 5-second window in isolation.
- `baseline34` tests whether removing the binary beat one-hot channel improves or clarifies the comparison when `beat_rep=distance` is already used as an explicit conditioning signal.
- Success should not be judged by beat scores alone; G1 distance, contact, penetration, and smoothness must stay competitive.

## Baseline Or Control

- FineDance G1 Jukebox FKBeat anchor: `slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2/eval/metrics.json`
- Completed Librosa35 controls:
  - `slurm/pipelines/20260510-finedance-g1-librosa35-baseline-1000/eval/metrics.json`
  - `slurm/pipelines/20260510-finedance-g1-librosa35-motiondist-cond-1000/eval/metrics.json`
  - `slurm/pipelines/20260511-finedance-g1-librosa35-baseline-2000/eval/metrics.json`
  - `slurm/pipelines/20260511-finedance-g1-librosa35-motiondist-cond-2000/eval/metrics.json`

## Intervention

- Add `baseline34` feature support across data creation, FineDance preparation, validation, full-song eval, training config, and Slurm submission.
- Add `data/rebuild_finedance_librosa_features.py` to rebuild FineDance Librosa features from source full-song audio and then slice them by prepared clip stems.
- Launch three full-context FineDance-G1 Librosa runs:
  - Librosa35, no beat conditioning, 2000 epochs.
  - Librosa35, motion-distance beat conditioning, 2000 epochs.
  - Librosa34, motion-distance beat conditioning, 2000 epochs.

## Invariant Controls

- Dataset: `data/finedance_g1_fkbeats`
- Motion format: `g1`
- Test set size observed in completed evals: `3265` valid/scored files.
- Validation for full-context runs sampled `64` files per split and passed.
- Feature cache mode in active sbatch scripts: `memmap` with `float16`.
- G1 FK model for FK-enabled eval: `third_party/unitree_g1_description/g1_29dof_rev_1_0.xml`
- Root quaternion order: `xyzw`

## Data And Cache Contract

- Full-context runs use separate processed cache directories:
  - `data/finedance_g1_librosa35_fullctx_baseline_dataset_backups`
  - `data/finedance_g1_librosa35_fullctx_motiondist_cond_dataset_backups`
  - `data/finedance_g1_librosa34_fullctx_motiondist_cond_dataset_backups`
- Validation logs report processed cache version `v4` and tensor cache version `v5`.
- Validation logs report:
  - Librosa35 baseline: `baseline_feats`, train `47817`, test `3265`, no beat files.
  - Librosa35 motiondist: `baseline_feats`, train `47817`, test `3265`, beat files present for all clips.
  - Librosa34 motiondist: `baseline34_feats`, train `47817`, test `3265`, beat files present for all clips.
- Do not reuse older tensor caches across feature type or full-context changes.

## Implementation Scope

- Branch/worktree: `diffusion` at commit `3655a98`, with uncommitted changes.
- Dirty files observed during sync:
  - `EDGE.py`
  - `data/audio_extraction/baseline_features.py`
  - `data/create_dataset.py`
  - `data/prepare_finedance_dataset.py`
  - `data/prepare_finedance_g1_dataset.py`
  - `data/validate_preprocessed_data.py`
  - `eval/run_full_song_eval.py`
  - `submit_training_pipeline.py`
  - `tests/test_beat_features.py`
  - `tests/test_validate_preprocessed_data.py`
  - `data/rebuild_finedance_librosa_features.py` untracked

## Completed Runs

Observed by `sacct` on 2026-05-12 UTC. All listed training and evaluation jobs completed with `ExitCode 0:0`.

| Run | Validation | Training | Evaluation | Evidence |
|---|---|---|---|---|
| `20260511-finedance-g1-librosa35-fullctx-baseline-2000` | `4556030` completed | `4556031` completed | `4556032` completed | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-baseline-2000/` |
| `20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000` | `4556033` completed | `4556034` completed | `4556035` completed | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/` |
| `20260511-finedance-g1-librosa34-fullctx-motiondist-cond-2000` | `4556036` completed | `4556037` completed | `4556038` completed | `slurm/pipelines/20260511-finedance-g1-librosa34-fullctx-motiondist-cond-2000/` |
| `20260511-finedance-g1-librosa35-fullctx-lbeat-robotloss-b020-kin040-cap1-scratch-1000` | `4556026` completed | `4556028` completed | `4556029` completed | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-lbeat-robotloss-b020-kin040-cap1-scratch-1000/` |
| `20260511-finedance-g1-jukebox-lbeat-robotloss-b020-kin040-cap1-scratch-1000` | `4545745` completed | `4545749` completed | `4545750` completed | `slurm/pipelines/20260511-finedance-g1-jukebox-lbeat-robotloss-b020-kin040-cap1-scratch-1000/` |

Completion check:

```bash
sacct -j 4556027,4556028,4556029,4556031,4556032,4556034,4556035,4556037,4556038,4545749,4545750 --format=JobID,JobName,State,Elapsed,ExitCode
```

## Completed Control Metrics

| Run | Feature / Conditioning | Key Result |
|---|---|---|
| `20260510-finedance-g1-librosa35-baseline-1000` | Librosa35, no beats | `G1BAS=0.2157`, `G1RoboPerformBAS=0.4115`, `G1Dist=6.8242`, `RootSmoothnessJerkMean=738.3078` |
| `20260510-finedance-g1-librosa35-motiondist-cond-1000` | Librosa35, `beat_rep=distance` | `G1BAS=0.2337`, `G1RoboPerformBAS=0.4521`, `G1FKBAS=0.2475`, `G1FKRoboPerformBAS=0.4264`, `G1BeatF1=0.2056`, `G1FootSliding=0.5595`, `G1Dist=9.8051` |
| `20260511-finedance-g1-librosa35-baseline-2000` | Librosa35, no beats | `G1BAS=0.2158`, `G1RoboPerformBAS=0.4095`, `G1Dist=7.5650`, `RootSmoothnessJerkMean=776.5635` |
| `20260511-finedance-g1-librosa35-motiondist-cond-2000` | Librosa35, `beat_rep=distance` | `G1BAS=0.2522`, `G1RoboPerformBAS=0.4664`, `G1FKBAS=0.2705`, `G1FKRoboPerformBAS=0.4395`, `G1BeatF1=0.2243`, `G1FootSliding=0.5798`, `G1Dist=11.2301` |
| `20260509-finedance-g1-fkbeatdistance-1000-r2` | Jukebox FKBeat anchor | `G1BAS=0.3046`, `G1RoboPerformBAS=0.5416`, `G1FKBAS=0.3192`, `G1FKRoboPerformBAS=0.5142`, `G1BeatF1=0.2910`, `G1Dist=8.6295` |

Interim reading from older sliced-context controls: Librosa35 motion-distance conditioning improves beat metrics over the Librosa35 no-beat control, but the sliced-context 2000-epoch conditioned run worsens `G1Dist` relative to the no-beat Librosa35 run and still trails the Jukebox FKBeat anchor on rhythm metrics.

## Completed Full-Context Metrics

All rows below evaluate the same `3265` FineDance-G1 test clips. Higher is better for BAS/F1/diversity. Lower is better for distance, sliding, penetration, and violations.

| Run | G1BAS | G1FKBAS | G1BeatF1 | G1RoboPerformBAS | G1FKRoboPerformBAS | G1Dist | G1Div | G1FootSliding | G1GroundPenetration | RootHeightViolationRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GT/reference `data/finedance_g1_fkbeats/test/motions_sliced` | 0.2130 | 0.2160 | 0.1770 | 0.4091 | 0.4000 | 0.0000 | 15.4889 | 0.2176 | 0.0392 | 0.0018 |
| Jukebox FKBeat no-lbeat `20260509-finedance-g1-fkbeatdistance-1000-r2` | 0.3046 | 0.3192 | 0.2910 | 0.5416 | 0.5142 | 8.6295 | 12.2347 | 0.5477 | 0.0406 | 0.0000 |
| Jukebox relative lbeat only `20260509-finedance-g1-relative-lbeat-scratch-lam010-1000-r2` | 0.3523 | 0.3511 | 0.2961 | 0.5023 | 0.4940 | 39.5323 | 16.4214 | 1.3512 | 0.0666 | 0.0002 |
| Jukebox lbeat + robot loss `20260511-finedance-g1-jukebox-lbeat-robotloss-b020-kin040-cap1-scratch-1000` | 0.3978 | 0.7218 | 0.7140 | 0.5699 | 0.7781 | 28.8001 | 13.6672 | 0.4741 | 0.0827 | 0.0044 |
| Librosa35 full-context baseline `20260511-finedance-g1-librosa35-fullctx-baseline-2000` | 0.2246 | 0.2421 | 0.1947 | 0.4223 | 0.4088 | 9.5547 | 12.4967 | 0.5871 | 0.0758 | 0.0000 |
| Librosa35 full-context motiondist `20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000` | 0.2498 | 0.2653 | 0.2255 | 0.4666 | 0.4510 | 8.9239 | 12.3020 | 0.5246 | 0.0323 | 0.0000 |
| Librosa34 full-context motiondist `20260511-finedance-g1-librosa34-fullctx-motiondist-cond-2000` | 0.2459 | 0.2629 | 0.2180 | 0.4610 | 0.4311 | 12.0596 | 12.1899 | 0.6234 | 0.0239 | 0.0000 |
| Librosa35 full-context lbeat + robot loss `20260511-finedance-g1-librosa35-fullctx-lbeat-robotloss-b020-kin040-cap1-scratch-1000` | 0.2123 | 0.6298 | 0.6220 | 0.5065 | 0.6932 | 14.8613 | 14.5804 | 0.5856 | 0.1715 | 0.0288 |

## Metric Interpretation

- `G1BAS` and `G1FKBAS` are paper-style music-to-motion beat alignment. `G1BAS` detects motion beats from the direct G1 root/joint speed curve. `G1FKBAS` detects motion beats from FK keypoint speed. A run can therefore score high on `G1FKBAS` while direct `G1BAS` stays low if it creates FK keypoint slowdowns near music beats without producing corresponding global/root or motor-speed minima.
- `G1RoboPerformBAS` and `G1FKRoboPerformBAS` reverse the direction: generated motion beats are compared to nearby music beats. These are useful for checking whether the model creates extra or misplaced movement minima.
- `G1BeatF1`, precision, recall, and timing are stricter FK beat matching statistics. They are rhythm metrics, not motion-quality metrics.
- `G1Dist` is the closest G1-native distribution-distance proxy to FID in this branch. It computes summary features from generated and reference G1 motions, normalizes by reference statistics, and reports the distance between generated and reference feature centers. Lower means the generated distribution is closer to the reference distribution.
- `G1Div` is the G1-native diversity metric. It is average pairwise distance among generated normalized G1 distribution features. Compare it against `G1ReferenceDiv=15.4889`; higher is more diverse, but high diversity with bad `G1Dist` can just mean unrealistic spread.
- `G1FootSliding`, `G1GroundPenetration`, root-height violations, root drift, and smoothness/jerk are robot-quality and contact-plausibility metrics. They should gate any beat-score improvement.
- GT/reference rows should be included in future metric comparisons whenever the matching reference split is available. The FineDance-G1 GT row above uses augmented runtime reference pickles with `audio_path` attached so BAS/FK BAS/F1 are computed on the same audio-beat path as generated outputs. Evidence: `slurm/pipelines/20260511-finedance-g1-librosa-fullctx-gt/eval_with_audio/metrics.json`, Slurm `4571181`.
- GT beat scores are not an upper bound. Generated models can score higher than GT by over-aligning motion minima to music beats, so GT should be used as context alongside `G1Dist`, contact, penetration, and smoothness.

## Evaluation Plan

When full-context eval jobs finish, compare:

1. Full-context Librosa35 baseline vs completed Librosa35 baseline controls.
2. Full-context Librosa35 motiondist vs completed Librosa35 motiondist controls.
3. Full-context Librosa34 motiondist vs full-context Librosa35 motiondist.
4. Best Librosa full-context run vs Jukebox FineDance G1 FKBeat anchor.

Required metrics:

- Rhythm: `G1BAS`, `G1RoboPerformBAS`, `G1FKBAS`, `G1FKRoboPerformBAS`, `G1BeatF1`
- Motion/contact: `G1Dist`, `G1FootSliding`, `G1GroundPenetration`, `RootSmoothnessJerkMean`
- Validity: `BadFileCount`, `FiniteMotionRate`, `num_valid_motion_files`

## Current Conclusion

The best balanced Librosa run is `20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000`. It improves over the full-context Librosa35 no-beat baseline on rhythm, distribution distance, foot sliding, ground penetration, root drift, and joint-range violations.

Dropping the binary music beat from Librosa35 did not help. `baseline34 + motiondist` is slightly worse than `baseline35 + motiondist` on rhythm and substantially worse on `G1Dist` and foot sliding, although it has slightly lower ground penetration.

Full-song-context feature rebuilding helped the 35D motion-distance run's motion quality versus the earlier sliced-context 35D motion-distance run: `G1Dist` improved from `11.2301` to `8.9239` and foot sliding improved from `0.5798` to `0.5246`, while rhythm stayed roughly similar.

Jukebox lbeat + robot loss is better than the older Jukebox relative-lbeat-only FineDance scratch run on both rhythm and most motion-quality checks: `G1FKBAS 0.3511 -> 0.7218`, `G1BeatF1 0.2961 -> 0.7140`, `G1Dist 39.5323 -> 28.8001`, and `G1FootSliding 1.3512 -> 0.4741`. However, it is still not the best deployable motion-quality checkpoint because `G1Dist=28.8001`, ground penetration is worse than FKBeat no-lbeat, and it has nonzero root-height violations.

The high `G1FKBAS` but low direct `G1BAS` pattern in `Librosa35 full-context lbeat + robot loss` means the loss is producing FK-visible keypoint beat minima, but those beats are not as consistently expressed in the direct G1/root-joint speed curve. Treat it as a rhythm shaping result, not proof of better overall dance quality.

## Paper Comparison Caveat

The current local results show that Librosa35 is weaker than Jukebox for this G1-retargeted EDGE setup on rhythm metrics, even after full-song-context rebuilding:

- GT/reference: `G1BAS=0.2130`, `G1FKBAS=0.2160`, `G1BeatF1=0.1770`, `G1Dist=0.0000`.
- Best balanced Librosa35 motion-distance: `G1BAS=0.2498`, `G1FKBAS=0.2653`, `G1BeatF1=0.2255`, `G1Dist=8.9239`.
- Jukebox FKBeat no-lbeat anchor: `G1BAS=0.3046`, `G1FKBAS=0.3192`, `G1BeatF1=0.2910`, `G1Dist=8.6295`.

This does not directly contradict DGFM/FineDance-style papers. DGFM compares music features inside its own SMPL/FineDance diffusion architecture and reports that handcrafted features can have better FID/diversity than Jukebox, but its best setup is Wav2CLIP + STFT, not Librosa35 alone. Lodge and FineDance use Librosa35 with different architectures, representation, long-context design, and evaluation protocols. For this branch, the confirmed conclusion is narrower: Librosa35 alone, plugged into the EDGE G1-retargeted architecture, is not matching Jukebox on beat alignment and only comes close on `G1Dist`.

## Qualitative Render Jobs

Requested on 2026-05-12 UTC: generate matched G1 robot videos for three FineDance examples across:

- Jukebox FKBeat no-lbeat/no-robot-loss: `runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt`
- Jukebox lbeat + robot loss: `runs/train/finedance_g1_jukebox_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt`
- Librosa35 full-context lbeat + robot loss: `runs/train/finedance_g1_librosa35_fullctx_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt`

The first full-song attempt was cancelled because FineDance songs are too long for quick qualitative comparison:

- Cancelled job: `4570014`
- Cancelled script: `slurm/pipelines/20260512-finedance-g1-full-song-renders/render_fullsong_three_models.sbatch`

Corrected 30-second render job:

- Source songs: FineDance `012`, `063`, `143`, trimmed to `30.00s`
- Input clips: `slurm/pipelines/20260512-finedance-g1-30s-renders/input_wavs/`
- Script: `slurm/pipelines/20260512-finedance-g1-30s-renders/render_30s_three_models.sbatch`
- Failed first 30-second job: `4570052`; it produced the first render set but exited during post-render metrics because `run_full_song_eval.py` expected `data/finedance_g1_fkbeats/test/motions` while this FineDance tree uses `motions_sliced`.
- Fix: added `--skip_metrics` render-only mode to `eval/run_full_song_eval.py` and enabled it in the 30-second render script.
- Completed Slurm job: `4570136`, `COMPLETED`, elapsed `00:03:59`, exit code `0:0`
- Log paths: `slurm/pipelines/20260512-finedance-g1-30s-renders/render_30s_three_models.4570136.out` and `.err`
- Render roots:
  - `slurm/pipelines/20260512-finedance-g1-30s-renders/jukebox_fkbeat_no_lbeat/renders/`
  - `slurm/pipelines/20260512-finedance-g1-30s-renders/jukebox_lbeat_robotloss/renders/`
  - `slurm/pipelines/20260512-finedance-g1-30s-renders/librosa35_lbeat_robotloss/renders/`
- Output check: `9` non-empty MP4 files, three per checkpoint; metadata JSON files report `900` generated frames for each 30-second clip.

## Next Action

Inspect the three matched 30-second videos before using beat scores alone to choose a checkpoint. Keep `20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000` as the low-dimensional Librosa baseline for the next ablation unless qualitative videos show a surprising failure.
