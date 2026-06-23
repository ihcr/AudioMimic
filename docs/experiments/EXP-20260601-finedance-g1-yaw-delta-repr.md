# EXP-20260601-finedance-g1-yaw-delta-repr

## Question

Can a yaw-only local root trajectory representation keep the v4 benefit of removing sudden yaw spikes while eliminating the long-render roll/pitch drift that made the robot lie sideways and float?

## Motivation

V4 `g1_root_delta` confirmed that local root deltas are useful: short-eval root angular max dropped from the v3b tail near `92 rad/s` to about `16-25 rad/s`. The 90s render then exposed the missing piece: integrating full SO(3) root deltas also integrates small roll/pitch errors, so root-up drift compounds over long samples.

This ablation removes that failure mode at the representation level instead of clamping render output or adding another patch loss.

## Representation

New motion format:

```text
g1_yaw_delta
```

Per-frame channels:

```text
root_local_delta_xy: [2]
root_height: [1]
root_delta_yaw_sin_cos: [2]
g1_dof_pos: [29]
total: 34
```

Semantics:

- Frame 0 root xy and yaw deltas are canonical/ignored, matching the existing root-delta canonical start.
- Translation is represented as local horizontal xy delta under the previous yaw.
- Root yaw is represented as wrapped per-frame delta using `[sin(delta_yaw), cos(delta_yaw)]`.
- Root roll/pitch are not integrated and are not model outputs in this ablation.
- Decoding emits native G1 payloads with upright root quaternions in `xyzw` order.

This matches the long-horizon motion-generation convention of keeping heading/yaw as the integrated root orientation while treating roll/pitch as non-trajectory state. It also matches the robotics distinction between full floating-base state and planar `base_footprint` style navigation/yaw frames.

## Conditions

Keep the V3b condition stack unchanged:

- Semantic: `Wav2CLIP`.
- Controls: `GaussianBeat`, root-local `motion_intensity`, root-local `motion_beatness`.
- Feature type: `wav2clip_local_motion_intensity_beatness`.
- No STFT, Librosa35, or Jukebox in the main run.

## Cache

Do not reuse v3b/v4 processed caches. New processed cache:

```text
data/finedance_g1_wav2clip_local_motion_intensity_beatness_yawdelta_dataset_backups
```

The raw feature caches stay unchanged because the condition features are unchanged. Only the processed motion tensor/normalizer cache changes.

## Implementation

- `dataset/motion_representation.py`: add `g1_yaw_delta` encode/decode and `motion_repr_dim(...) == 34`.
- `args.py`, `train.py`, eval entry points, and validation: accept `g1_yaw_delta`.
- `model/diffusion.py`: long-render stitching decodes integrated local-root formats by concatenating root-delta windows before decode.
- Tests cover roundtrip, roll/pitch discard, canonical first-frame decode, dataset loading, render payload decode, CLI defaults, and mixed-checkpoint render format loading.

## Training Plan

Train from scratch. Do not finetune v3b or v4 because the target representation and normalizer differ.

Command:

```bash
mkdir -p setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr
export MUJOCO_GL=egl
PYTHONUNBUFFERED=1 .venv311/bin/python -m accelerate.commands.launch train.py \
  --feature_type wav2clip_local_motion_intensity_beatness \
  --feature_fusion linear \
  --motion_format g1_yaw_delta \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_wav2clip_local_motion_intensity_beatness_yawdelta_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260601-finedance-g1-yaw-delta-repr_r01 \
  --render_dir renders/EXP-20260601-finedance-g1-yaw-delta-repr \
  --batch_size 256 \
  --gradient_accumulation_steps 2 \
  --epochs 2000 \
  --save_interval 50 \
  --full_eval_interval 500 \
  --full_eval_root eval \
  --full_eval_variants auto \
  --full_eval_batch_size 32 \
  --full_eval_diagnostic_count 8 \
  --wandb_pj_name EDGE \
  --wandb_log_interval 1 \
  --mixed_precision bf16 \
  --train_num_workers 2 \
  --test_num_workers 2 \
  --lambda_acc 0.02 \
  --lambda_acc_final 0.1 \
  --lambda_acc_warmup_start_epoch 500 \
  --lambda_acc_warmup_epochs 500 \
  --lambda_g1_foot 0.05 \
  --g1_kin_loss_warmup_epochs 100 \
  --g1_kin_loss_max_fraction 0.25 \
  --lambda_g1_root_angular 0.0 \
  --lambda_motion_intensity 0.05 \
  --motion_intensity_norm_p05 0.16980750858783722 \
  --motion_intensity_norm_p95 2.8171334266662598 \
  --lambda_motion_beatness 0.02 \
  --motion_beatness_warmup_start_epoch 100 \
  --motion_beatness_warmup_epochs 400 \
  --motion_beatness_max_fraction 0.1 \
  --lambda_energy_pred 1.0 \
  --energy_teacher_forcing_epochs 100 \
  --energy_pred_mix_prob 0.5 \
  --energy_smoothness_weight 0.1 \
  --skip_train_sample_render \
  --g1_mujoco_gl egl \
  2>&1 | tee -a setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr/train_r01_20260601.log
```

## Evaluation Plan

The training hook must run full eval every 500 epochs:

- `pred_controls`
- `oracle_controls`
- `flat_intensity`
- `zero_beatness`
- `zero_all_controls`

At 500 and 1000, also render a matched long comparison against v3b 1500/2000 and Librosa35 2000 before declaring success.

Main gates:

- Upright/root-up: `RootUpZP01 >= 0.90`, `RootTiltGt60DegRate == 0`, `RootInvertedRate == 0` on full eval and long render.
- Root yaw stability: `RootAngularVelocityP99 <= 5.0 rad/s`, `RootAngularVelocityMax < 30 rad/s` unless isolated and visually acceptable.
- Rhythm: target at least v4 1500 level, `G1FKBAS >= 0.235`, `BeatF1 >= 0.200` by 1500.
- Quality: preserve v4 distribution gains, `G1Dist <= 4.5` after 1000 if rhythm does not collapse.
- Anti-average: `G1Div >= 14.0`, `JointPositionRangeMean >= 1.20`.
- Contact: do not let foot sliding regress beyond v4; track separately because yaw-only root does not directly solve contact.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-06-01 | implementation | passed | source changes in `dataset/motion_representation.py`, `model/diffusion.py`, `args.py`, `train.py`, `eval/run_g1_dataset_eval.py`, `eval/run_full_song_eval.py`, `data/validate_preprocessed_data.py`, `submit_training_pipeline.py`; tests `.venv311/bin/python -m unittest tests.test_g1_motion_format tests.test_phase0_cli_and_preprocess tests.test_render_g1_checkpoint_comparison` | Adds `g1_yaw_delta`, keeps old `g1`/`g1_root_delta` checkpoints compatible. Test result: `Ran 47 tests ... OK`. |
| 2026-06-01 | cache validation | passed | `.venv311/bin/python data/validate_preprocessed_data.py --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_wav2clip_local_motion_intensity_beatness_yawdelta_dataset_backups --feature_type wav2clip_local_motion_intensity_beatness --motion_format g1_yaw_delta --sample_count 64` | Validation passed for train `47817` clips and test `3265` clips. Structured feature dirs validated as `wav2clip_stft_beat_feats+gaussian_beat_feats+motion_control_v3_local_feats`; processed cache dir exists and has no stale yaw-delta caches. |
| 2026-06-01 | r01 launch | stopped by user | tmux `m2d_train_yaw_delta_v5` stopped and session killed; log `setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr/train_r01_20260601.log`; run dir `runs/train/EXP-20260601-finedance-g1-yaw-delta-repr_r01`; W&B run `9u5w1e5m`; no durable checkpoint expected | Launched with direct `.venv311/bin/python` because `.venv311/bin/activate` is absent in this checkout. User requested stopping this run to prioritize `EXP-20260601-finedance-g1-beat-features-8d`. Ctrl-C interrupted training during epoch `6/2000` around batch `118/186`; no `save_interval=50` checkpoint had been reached. |
| 2026-06-11 | r02 relaunch | running | tmux `m2d_train_yaw_delta_v5`; log `setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr/train_r02_b128_acc4_20260611.log`; run dir `runs/train/EXP-20260601-finedance-g1-yaw-delta-repr_r02_b128_acc4`; W&B run `50lo8tdd` | Relaunched from scratch because r01 had no durable checkpoint. Another user's `vlm_server.py` was using about `12.6GB` GPU memory, so r02 uses `--batch_size 128 --gradient_accumulation_steps 4` to preserve effective batch size `512` while reducing model peak memory. Full eval every 500 remains active. Epoch 1 completed in `96.00s`, `498.11 samples/s`, model peak CUDA `9140.82 MB`, ETA about `53h18m`; training continued into epoch 2. |
| 2026-06-12 | r02 interruption and loader fix | fixed, resumed | crash log `setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr/train_r02_b128_acc4_20260611.log`; latest durable checkpoint `runs/train/EXP-20260601-finedance-g1-yaw-delta-repr_r02_b128_acc4/weights/train-250.pt`; source `dataset/dance_dataset.py`; tests `.venv311/bin/python -m unittest tests.test_g1_motion_format tests.test_phase0_cli_and_preprocess tests.test_validate_preprocessed_data`; full cache scan command in agent turn | r02 crashed during epoch `279/2000` with `TypeError: unhashable type: 'NpzFile'` from a dataloader worker while lazily reading structured `.npz` controls. Full scan of `51082` `motion_control_v3_local_feats` files found `bad=0`, so this was not upstream feature-cache corruption. Fixed loader to copy Wav2CLIP/Gaussian arrays out of mmap and to raise path-bearing explicit errors for `KeyError`, `TypeError`, or `BadZipFile` when reading motion-control npz files. |
| 2026-06-12 | r02 resume250 | running | tmux `m2d_train_yaw_delta_v5`; log `setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr/train_r02_resume250_b128_acc4_20260612.log`; run dir `runs/train/EXP-20260601-finedance-g1-yaw-delta-repr_r02_resume250_b128_acc4`; W&B run `w6vd8xvh`; checkpoint input `runs/train/EXP-20260601-finedance-g1-yaw-delta-repr_r02_b128_acc4/weights/train-250.pt` | Relaunched with `--checkpoint train-250.pt`, `--epoch_offset 250`, and `--epochs 1750`, so the global counter resumed at `Train 251/2000`. Same effective batch size `512`, full eval interval `500`, and five eval variants remain active. |
| 2026-06-12 | r02 resume250 npz-worker fix | fixed, running | crash pane/log showed epoch `278/2000` failure on `data/finedance_g1_fkbeats/train/motion_control_v3_local_feats/049_slice124.npz` with `TypeError: '_SharedFile' object is not callable`; source `dataset/dance_dataset.py`; focused test `.venv311/bin/python -m unittest tests.test_phase2_dataset_and_estimator.DatasetBeatSchemaTests.test_structured_motion_controls_are_loaded_from_memmap_store`; regression tests `.venv311/bin/python -m unittest tests.test_g1_motion_format tests.test_phase0_cli_and_preprocess tests.test_validate_preprocessed_data`; real stores `data/finedance_g1_wav2clip_local_motion_intensity_beatness_yawdelta_dataset_backups/feature_stores/{train,test}_wav2clip_local_motion_intensity_beatness_motion_control_memmap_float32_v1.npy`; 220-batch multi-worker smoke; tmux `m2d_train_yaw_delta_v5`; log `setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr/train_r02_resume250_controlstore_b128_acc4_20260612.log`; run dir `runs/train/EXP-20260601-finedance-g1-yaw-delta-repr_r02_resume250_controlstore_b128_acc4`; W&B run `53gngvmx` | The second crash proved the issue was not a single corrupt npz; direct and multiprocessing reads of `049_slice124.npz` succeeded, but DataLoader workers could still hit a numpy/zipfile shared-handle failure while reading npz members. Long-term fix: pack `motion_intensity` and `motion_beatness` into a small memmap control store during dataset setup, then workers slice the `.npy` store instead of opening per-sample `.npz`. Relaunched from `train-250.pt` with `--no_cache` once to overwrite old tensor dataset pickles and verified live training at `Train 252/2000`; full eval every 500 remains active. |
| 2026-06-14 | ckpt1000 full eval | passed, needs long render | checkpoint `runs/train/EXP-20260601-finedance-g1-yaw-delta-repr_r02_resume250_controlstore_b128_acc4/weights/train-1000.pt`; manual full eval log `setup_logs/EXP-20260601-finedance-g1-yaw-delta-repr/eval_ckpt1000_full_20260614.log`; metrics dirs `eval/EXP-20260601-finedance-g1-yaw-delta-repr_r02_resume250_controlstore_b128_acc4/ckpt1000_{pred_controls,oracle_controls,flat_intensity,zero_beatness,zero_all_controls}` | Training reached checkpoint 1000, then the training-triggered 1000 `pred_controls` eval was interrupted by `^C` at about `640/3265` clips, so the agent reran all five full eval variants. `pred_controls` passes the main 1000 gates: `G1FKBAS=0.2604`, `G1BeatF1=0.2340`, `G1Dist=3.9908`, `G1Div=16.5609`, `JointPositionRangeMean=1.2879`, `RootUpZP01=1.0`, `RootTiltGt60DegRate=0`, `RootInvertedRate=0`, `RootAngularVelocityP99=2.9524`, `RootAngularVelocityMax=14.9566`. Compared with v3b, v5 sharply reduces root spin tails while improving FK rhythm and distribution. Main regressions/risks are `G1FootSliding=0.8384`, `G1GroundPenetration=0.0757`, and larger `RootDriftMean=0.7159` / `RootFlatRangeMean=0.9684`. Condition sensitivity is real: `zero_beatness` drops to `G1FKBAS=0.2139`, `G1BeatF1=0.1828`; `zero_all_controls` has collapsed diversity (`G1Div=8.2816`, `JointPositionRangeMean=0.0837`) despite deceptively high FK beat scores, so beat score alone remains insufficient. |
| 2026-06-14 | qualitative render and jitter/contact diagnostic | failed quality gate | user-observed v5 render issue: more visible hand jitter than v3b and frequent lifted/hovering feet; diagnostic `eval/EXP-20260601-finedance-g1-yaw-delta-repr_r02_resume250_controlstore_b128_acc4/v5_jitter_foot_hacking_diagnostic_20260614.json` over 512 matched clips against v3b 1500, v3b 2000, and Librosa35 2000 | V5 fixes sudden turning but shifts the failure mode into endpoint/contact behavior. Against v3b 1500, v5 has `1.67x` wrist FK jerk p95, `2.86x` no-near-support rate, `2.48x` high-lift frame rate, and only `0.78x` contact-proxy rate. Against v3b 2000, the same ratios are `1.64x`, `3.78x`, `3.17x`, and `0.73x`. Against Librosa35, v5 has `2.62x` wrist FK jerk p95, `84x` no-near-support rate, `30x` high-lift frame rate, and `0.55x` contact-proxy rate. This is objective/metric mismatch rather than data-cache corruption: the current motion-control target gives wrists `70%` of weighted speed mass, the FK rhythm metric can be improved by endpoint speed valleys, and the foot loss penalizes horizontal motion on target-contact frames but does not force predicted feet to preserve support height/contact. |

## Current Conclusion

R02 resume250-controlstore reached checkpoint 1000 and has complete 500/1000 full eval metrics. V5 is validated as a representation ablation, not as a final model: yaw-only local root removes v4's roll/pitch integration failure and avoids the v3b extreme root-spin tail, while improving FK rhythm and maintaining healthy diversity. The matched qualitative render and endpoint/contact diagnostic expose a new failure mode: hand/wrist jitter and hovering/high-lift feet. This looks like objective/metric mismatch: the model can satisfy rhythm/intensity through high-frequency endpoint motion and poor support contact because the current motion-control target is wrist-heavy and the current contact loss does not enforce predicted support height. Do not resume v5 as the mainline without addressing support/contact and endpoint smoothness.

## Next Action

Design a v6 follow-up that keeps the successful yaw-only root representation but rebuilds the motion-control target and quality gates: contact-aware/support-aware beatness, reduced/capped wrist contribution, explicit support/contact supervision or representation, FK endpoint smoothness diagnostics, and 500-epoch full eval gates for wrist jerk, no-support rate, high-lift rate, contact-proxy rate, root spin, rhythm, distribution, and contact quality. Treat v5 `train-1000.pt` only as a possible short diagnostic finetune seed; the clean ablation should rebuild caches and train under the corrected objective.
