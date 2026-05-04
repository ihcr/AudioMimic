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
     --g1_root_quat_order wxyz'
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
- root quaternion order: `wxyz`

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
