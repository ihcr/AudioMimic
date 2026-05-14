# Dataset Preparation Notes

This document records the nonstandard dataset preparation paths for this branch.
The original EDGE AIST++ setup still follows the README. The notes below cover
supplementary datasets that need conversion before training.

## FineDance Supplement

FineDance can be added as extra training data for the beat-conditioned diffusion
experiments. The prepared output uses the same body-only clip layout as the AIST
tree used by EDGE:

- `motions_sliced`
- `wavs_sliced`
- `jukebox_feats`
- `beat_feats`

The converter accepts the downloaded FineDance SMPLH motion payloads with either
`315` columns of 6D rotations or the older `159` column axis-angle layout. It
drops hand-only joints and writes EDGE-compatible body motion with root position
and 24 body joints. FineDance source clips are 30 fps; prepared motion clips are
stored in the 60 fps container expected by EDGE.

Use the shared raw FineDance folder when it is present:

```bash
../../.venv311/bin/python data/prepare_finedance_dataset.py \
  --finedance_root /lus/lfs1aip2/projects/u6ed/yukun/EDGE/data/finedance \
  --output_root data/finedance_aistpp \
  --feature_type jukebox \
  --extract_beats \
  --build_mixed \
  --aist_root data \
  --mixed_output_root data/aist_finedance
```

The prepared FineDance-only tree is:

```text
data/finedance_aistpp
```

The mixed training tree is:

```text
data/aist_finedance
```

The mixed tree trains on AIST train plus FineDance train. The normal `test`
split remains AIST-only for comparison with earlier EDGE runs. FineDance test
clips are kept separately under `finedance_test`.

## Verified Local Snapshot

The 2026-05-03 completed preparation produced these counts:

| Dataset | Split | Clips |
| --- | --- | ---: |
| FineDance only | train | 47,817 |
| FineDance only | test | 3,265 |
| AIST + FineDance | train | 65,550 |
| AIST + FineDance | test | 186 |
| AIST + FineDance | finedance_test | 3,265 |

For each split above, `motions_sliced`, `wavs_sliced`, `jukebox_feats`, and
`beat_feats` had matching file counts. Metadata files were written at:

- `data/finedance_aistpp/metadata.json`
- `data/aist_finedance/metadata.json`

## Failure Modes To Check First

- The downloaded file named `data/finedance.zip` can actually be a RAR5 archive.
  Check it with `file` and extract it with a RAR-capable tool such as `7zz`.
- Do not assume the older `159` column FineDance motion format. The local
  downloaded bundle used `315` columns.
- If a Jukebox extraction job fails at the final clip, check the one-item batch
  path. The extractor should duplicate the final single audio item internally and
  save only the first returned feature.
- If a Slurm helper script is run by file path, make sure it adds the repo root
  to `sys.path` before importing repo modules.

## Training With The Mixed Tree

After the mixed tree is prepared, use the dedicated preset wrapper:

```bash
scripts/train_aist_finedance_beatdistance.sh aist_finedance_beatdistance_run
```

That wrapper submits the `aist_finedance_beatdistance` preset through the normal
pipeline launcher.

## G1 AIST Full Retarget With FK Beats

The corrected G1 AIST data should be prepared into a new tree, separate from the
older partial `data/g1_aistpp` and proxy-beat `data/g1_aistpp_full` trees.

Use this raw source:

```text
/projects/u6ed/yukun/EDGE/aist-g1-retargeted
```

Use this prepared output:

```text
data/g1_aistpp_full_fkbeats
```

The preparation command reads `root_pos`, `root_rot`, and `dof_pos`, applies the
official EDGE AIST train/test split, skips the AIST ignore list for train, slices
5-second windows at 30 fps with 0.5-second stride, writes compatibility `pos`
and `q` fields, reuses unchanged AIST wavs and Jukebox features, and extracts G1
motion beats from Unitree G1 forward kinematics.

Run long preparation on a compute node:

```bash
srun --partition=workq --time=02:00:00 --ntasks=1 --cpus-per-task=8 --mem=32G bash -lc \
  'cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion && \
   source ../../.venv311/bin/activate && \
   python data/prepare_g1_aist_dataset.py \
     --g1_motion_dir /projects/u6ed/yukun/EDGE/aist-g1-retargeted \
     --aist_data_root data \
     --output_root data/g1_aistpp_full_fkbeats \
     --feature_type jukebox \
     --clean \
     --extract_beats \
     --g1_motion_beat_source fk \
     --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml \
     --g1_root_quat_order xyzw'
```

Validate the tree before training:

```bash
srun --partition=workq --time=00:30:00 --ntasks=1 --cpus-per-task=4 --mem=16G bash -lc \
  'cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion && \
   source ../../.venv311/bin/activate && \
   python data/validate_preprocessed_data.py \
     --data_path data/g1_aistpp_full_fkbeats \
     --processed_data_dir data/g1_aistpp_full_fkbeats_dataset_backups \
     --feature_type jukebox \
     --motion_format g1 \
     --feature_cache_mode memmap \
     --feature_cache_dtype float32 \
     --sample_count 64 \
     --use_beats \
     --beat_rep distance'
```

The 2026-05-04 verified local snapshot produced these counts:

| Dataset | Split | Source sequences | Clips |
| --- | --- | ---: | ---: |
| G1 AIST FK-beat | train | 952 | 17,733 |
| G1 AIST FK-beat | test | 20 | 186 |

Additional metadata from `data/g1_aistpp_full_fkbeats/metadata.json`:

- raw G1 files: `1,408`
- official train names: `980`
- ignored train names: `28`
- official test names: `20`
- unused files outside the official EDGE split: `391`
- FK model: `third_party/unitree_g1_description/g1_29dof_rev_1_0.xml`
- root quaternion order: `xyzw`

The matching fresh processed-cache path is:

```text
data/g1_aistpp_full_fkbeats_dataset_backups
```

The submitted 2026-05-04 training pipeline is:

```text
slurm/pipelines/20260504-g1-aist-beatdistance-fkbeats
```

It uses the `g1_beatdistance_fkbeats` preset, trains with beat conditioning but
no auxiliary beat loss, and runs FK-enabled G1 evaluation after training.

The follow-up cautious beat-loss fine-tune uses:

```text
g1_beatdistance_fkbeats_lbeat
```

That preset keeps the same prepared tree and cache, trains a G1-native beat
estimator from normalized 38-D robot motion with FK-derived `motion_dist`
targets, then fine-tunes from:

```text
runs/train/g1_aist_beatdistance_fkbeats/weights/train-2000.pt
```

It uses a weak delayed beat loss (`lambda_beat=0.01`) with a soft cap so the
auxiliary term stays bounded without becoming fully gradient-dead when capped.

The 2026-05-05 submitted pipeline is:

```text
slurm/pipelines/20260505-g1-aist-beatdistance-fkbeats-lbeat
```

The chain validates the FK-beat tree, trains the G1 beat estimator, fine-tunes
the diffusion checkpoint, then runs FK-enabled G1 evaluation. The validation
stage must pass `--feature_cache_mode memmap --feature_cache_dtype float32`;
without those flags the validator will correctly reject the current memmap
tensor caches as non-matching legacy caches.

## G1 Lbeat Experiment Results

The G1 beat-loss path is separate from SMPL `Lbeat`. The normalized estimator
uses `relative_distance = clamp(motion_dist / motion_spacing, 0, 1)` and a
sigmoid output head, then diffusion transforms `cond["beat_target"]` into the
same target space before applying the auxiliary beat loss.

The first normalized-target scale check showed why old weights are not directly
comparable: raw FK `motion_dist` had mean-square scale about `13.7263`, while
relative distance was about `0.10598`, a roughly `129.5x` MSE-scale drop.

Completed G1 FK-beat runs to use as current anchors. `G1BAS` and `G1FKBAS`
are paper-style music-to-motion scores. `G1RoboPerformBAS` and
`G1FKRoboPerformBAS` are RoboPerform-style motion-to-music scores rescored on
2026-05-10.

| Run | Path | Decision | Key metrics |
| --- | --- | --- | --- |
| FKBeat no lbeat | `slurm/pipelines/20260504-g1-aist-beatdistance-fkbeats` | stable default | `G1BAS=0.3320`, `G1RoboPerformBAS=0.5894`, `G1FKBAS=0.3497`, `G1FKRoboPerformBAS=0.5502`, `G1BeatF1=0.3333`, `G1FootSliding=0.6112`, `G1Dist=6.87` |
| Raw lbeat fine-tune | `slurm/pipelines/20260505-g1-aist-beatdistance-fkbeats-lbeat` | weak rhythm gain, not clearly better | `G1BAS=0.3530`, `G1RoboPerformBAS=0.6065`, `G1FKBAS=0.3538`, `G1FKRoboPerformBAS=0.5538`, `G1BeatF1=0.3289`, `G1FootSliding=0.6577`, `G1Dist=7.02` |
| Relative lbeat scratch `lambda_beat=0.2` | `slurm/pipelines/20260506-g1-aist-fkbeats-lbeat-relative-scratch-lam020-cap1c` | rejected | `G1BAS=0.6580`, `G1RoboPerformBAS=0.5006`, `G1FKBAS=0.8232`, `G1FKRoboPerformBAS=0.8792`, `G1BeatF1=0.8716`, `G1FootSliding=3.0399`, `G1Dist=56.26` |
| Relative lbeat fine-tune `lambda_beat=0.2` | `slurm/pipelines/20260507-g1-aist-fkbeats-lbeat-relative-finetune-lam020-cap1` | best lbeat direction so far, not a clean replacement | `G1BAS=0.4484`, `G1RoboPerformBAS=0.6211`, `G1FKBAS=0.4673`, `G1FKRoboPerformBAS=0.5978`, `G1BeatF1=0.4372`, `G1FootSliding=0.7102`, `G1GroundPenetration=0.0979`, `G1Dist=7.59` |

Completed retargeted FineDance G1 runs:

| Run | Path | Decision | Key metrics |
| --- | --- | --- | --- |
| FineDance GT/reference | `slurm/pipelines/20260511-finedance-g1-librosa-fullctx-gt/eval_with_audio/metrics.json` | reference context for future comparisons | `G1BAS=0.2130`, `G1RoboPerformBAS=0.4091`, `G1FKBAS=0.2160`, `G1FKRoboPerformBAS=0.4000`, `G1BeatF1=0.1770`, `G1FootSliding=0.2176`, `G1GroundPenetration=0.0392`, `G1Dist=0.00` |
| FineDance FKBeat no lbeat | `slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2` | cleaner FineDance G1 anchor | `G1BAS=0.3046`, `G1RoboPerformBAS=0.5416`, `G1FKBAS=0.3192`, `G1FKRoboPerformBAS=0.5142`, `G1BeatF1=0.2910`, `G1Dist=8.63` |
| FineDance relative lbeat scratch `lambda_beat=0.1` | `slurm/pipelines/20260509-finedance-g1-relative-lbeat-scratch-lam010-1000-r2` | rejected as a deployable checkpoint | `G1BAS=0.3523`, `G1RoboPerformBAS=0.5023`, `G1FKBAS=0.3511`, `G1FKRoboPerformBAS=0.4940`, `G1BeatF1=0.2961`, `G1Dist=39.53` |
| FineDance Jukebox lbeat + robot loss | `slurm/pipelines/20260511-finedance-g1-jukebox-lbeat-robotloss-b020-kin040-cap1-scratch-1000` | strong rhythm, still not clean motion | `G1BAS=0.3978`, `G1RoboPerformBAS=0.5699`, `G1FKBAS=0.7218`, `G1FKRoboPerformBAS=0.7781`, `G1BeatF1=0.7140`, `G1FootSliding=0.4741`, `G1GroundPenetration=0.0827`, `G1Dist=28.80` |
| FineDance Librosa35 full-context motiondist | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000` | best balanced Librosa run | `G1BAS=0.2498`, `G1RoboPerformBAS=0.4666`, `G1FKBAS=0.2653`, `G1FKRoboPerformBAS=0.4510`, `G1BeatF1=0.2255`, `G1FootSliding=0.5246`, `G1GroundPenetration=0.0323`, `G1Dist=8.92` |
| FineDance Librosa35 full-context lbeat + robot loss | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-lbeat-robotloss-b020-kin040-cap1-scratch-1000` | FK beat gains but quality regressions | `G1BAS=0.2123`, `G1RoboPerformBAS=0.5065`, `G1FKBAS=0.6298`, `G1FKRoboPerformBAS=0.6932`, `G1BeatF1=0.6220`, `G1FootSliding=0.5856`, `G1GroundPenetration=0.1715`, `G1Dist=14.86` |

Interpretation: normalized lbeat fine-tuning can improve G1 rhythm without the
scratch-run root-motion collapse, but the current loss still buys beat score by
worsening contact quality. Before treating any lbeat checkpoint as the deployable
G1 model, compare beat metrics against foot sliding, ground penetration, root
velocity, root jerk, and distribution distance.

## G1 Full-Song Audio Sources

G1 generation still runs through 5-second model windows internally. That does
not mean the final render should be a 5-second clip. Full-song evaluation
creates enough overlapping windows to cover the selected audio, stitches the
generated robot motion, and then renders one `.mp4` for the stitched motion.

There are three different audio surfaces, and they should not be mixed up:

- Raw full AIST music, for example
  `/projects/u6ed/yukun/Music2Dance/Code/aist_plusplus_datasets/audio/mHO5.mp3`.
  Use this with `--full_music_dir` when the goal is a video for the whole song.
- Choreography-trimmed test wavs under a prepared dataset, for example
  `data/g1_aistpp_full_fkbeats/test/wavs`. Use these only when the goal is to
  match the existing trimmed test choreography span.
- 5-second sliced wavs under `wavs_sliced`. These are model input windows and
  metric/eval artifacts, not whole-song audio.

For whole-song qualitative renders, use:

```bash
python -m eval.run_full_song_eval \
  --data_path data/g1_aistpp_full_fkbeats \
  --full_music_dir /projects/u6ed/yukun/Music2Dance/Code/aist_plusplus_datasets/audio \
  --motion_format g1 \
  --render \
  --g1_render_backend mujoco \
  --g1_root_quat_order xyzw
```

Do not combine `--full_music_dir` with `--use_precomputed_test_slices`; those
precomputed slices were derived from the shorter choreography-trimmed wav tree.
