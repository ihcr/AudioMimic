# EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness

Status: needs_decision
Owner: yukun
Created: 2026-05-26
Last Updated: 2026-05-30

## Research Question

Can v3 keep its improved amplitude and diversity while removing the root-yaw exploit that caused sudden turns in long G1 renders?

## Hypothesis

The v3 `motion_intensity` and `motion_beatness` controls are useful, but their world-frame FK speed target can be satisfied by spinning root heading. Rebuilding those controls in the root-local frame should make the controls reward local body motion rather than global yaw. A light root angular velocity/acceleration guard should catch the remaining fast-turn tail without turning this into a render-only patch.

## Baseline Or Control

- Main predecessor: `EXP-20260524-finedance-g1-wav2clip-intensity-beatness` r03 `train-1000.pt`.
- Mixed-signal predecessor: `EXP-20260522-finedance-g1-wav2clip-motion-energy-beat` r05 `train-1000.pt`.
- Calm but lower-diversity reference: Librosa35 2000.
- Diversity reference: GaussianBeat 1000.

## Intervention

New feature type:

```text
wav2clip_local_motion_intensity_beatness
```

The condition schema stays the same as v3:

```python
cond = {
    "semantic": {"wav2clip": Tensor[B, 150, 512]},
    "control": {
        "gaussian_beat": Tensor[B, 150, 1],
        "motion_intensity": Tensor[B, 150, 1],
        "motion_beatness": Tensor[B, 150, 1],
    },
}
```

The semantic/control encoder remains compact and shared with v3. The change is the frame definition and robot-quality guard:

- `motion_intensity` and `motion_beatness` are generated from weighted FK keypoints after subtracting root position and rotating into the inverse root heading frame.
- Training resolves `--motion_energy_frame auto` to `root_local` for this feature type, so the global intensity and beatness valley losses use the same frame as the cache.
- `lambda_g1_root_angular` adds a decoded G1 root angular velocity/acceleration excess loss against GT margins, capped as an auxiliary loss.
- Full eval runs every 500 epochs by default and includes root angular metrics already added to `eval/g1_metrics.py`.

## Data And Cache Contract

New cache:

- Metadata: `data/finedance_g1_fkbeats/motion_control_v3_local_metadata.json`.
- Train/test features: `data/finedance_g1_fkbeats/{train,test}/motion_control_v3_local_feats/*.npz`.
- Processed tensor cache: `data/finedance_g1_wav2clip_local_motion_intensity_beatness_dataset_backups`.

Required `.npz` fields are the same as v3:

- `motion_intensity_envelope`: `float32 [150, 1]`.
- `motion_beatness_envelope`: `float32 [150, 1]`.
- `weighted_fk_speed`: `float32 [150]`.
- `smoothed_weighted_fk_speed`: `float32 [150]`.
- `audio_beat_frames`: `int64 [N]`.
- `intensity_peaks`: `float32 [N]`.
- `beatness_peaks`: `float32 [N]`.

Cache command:

```bash
PATH=/home/tianhup/Desktop/Musics2Dance/.venv311/bin:$PATH \
PYTHONUNBUFFERED=1 \
python -m data.audio_extraction.motion_control_v3_local_features \
  --data_path data/finedance_g1_fkbeats \
  --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml \
  --g1_root_quat_order xyzw \
  --batch_size 512 \
  --device cuda
```

Any formula, keypoint, frame, sigma, or normalization change requires deleting `motion_control_v3_local_feats`, `motion_control_v3_local_metadata.json`, and the processed cache before retraining.

## Training Plan

Start from scratch; do not finetune v3 r03 as the mainline because the feature semantics changed from world-frame to root-local.

Template command:

```bash
PATH=/home/tianhup/Desktop/Musics2Dance/.venv311/bin:$PATH \
PYTHONUNBUFFERED=1 \
accelerate launch train.py \
  --feature_type wav2clip_local_motion_intensity_beatness \
  --feature_fusion linear \
  --motion_format g1 \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_wav2clip_local_motion_intensity_beatness_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01 \
  --render_dir renders/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness \
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
  --lambda_g1_root_angular 0.02 \
  --g1_root_angular_velocity_margin 3.141592653589793 \
  --g1_root_angular_acceleration_margin 18.84955592153876 \
  --g1_root_angular_acceleration_weight 0.25 \
  --g1_root_angular_max_fraction 0.05 \
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
  --energy_smoothness_weight 0.1
```

Use tmux with live output:

```bash
tmux new -s m2d_train_wav2clip_local_intensity_beatness
```

Inside tmux, run the command with `2>&1 | tee -a setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_20260526.log`.

## Evaluation Plan

Full eval checkpoints: `500`, `1000`, `1500`, `2000`.

Variants per checkpoint:

- `pred_controls`
- `oracle_controls`
- `flat_intensity`
- `zero_beatness`
- `zero_all_controls`

Acceptance at 1000:

- Root angular tail should be much lower than v3 r03 1000 on the same long render and full-eval saved motions.
- `pred_controls` should keep v3-like anti-average motion: `G1Div` and joint range should not collapse toward Librosa35/STFT.
- `G1Dist` should stay closer to v3/r05 than GaussianBeat/Librosa35.
- `zero_beatness` should still move rhythm metrics, proving the beatness condition remains used.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-26 | v3b implementation | passed | source changes in `feature_config.py`, `data/audio_extraction/motion_control_v3_local_features.py`, `dataset/dance_dataset.py`, `model/diffusion.py`, `EDGE.py`, `args.py`; focused tests `.venv311/bin/python -m unittest tests.test_feature_config_and_fusion tests.test_motion_control_v2_features tests.test_g1_motion_format tests.test_phase0_cli_and_preprocess tests.test_motion_energy_condition_variants tests.test_render_g1_checkpoint_comparison tests.test_g1_eval_metrics` | Implements new feature type, separate local cache/metadata names, root-local training frame resolution, root angular training loss, W&B logs, and full-eval-compatible variants. Test result: `Ran 57 tests ... OK`. |
| 2026-05-26 | motion-control v3 local cache | passed | metadata `data/finedance_g1_fkbeats/motion_control_v3_local_metadata.json`; log `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_motion_control_v3_local_cache_20260526.log`; feature dirs `data/finedance_g1_fkbeats/{train,test}/motion_control_v3_local_feats` | Generated `47817` train and `3265` test `.npz` files with `coordinate_frame=root_local`. Normalization: intensity p05 `0.1698075086`, p95 `2.8171334267`; beatness p05 `0.0022732092`, p95 `0.1142841130`; peak count `435660`. |
| 2026-05-26 | r01 launch | stopped at sample render | log `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_20260526.log`; run dir `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01`; W&B run `0jvleht4`; checkpoint `weights/train-50.pt` | Launched from scratch with `feature_type=wav2clip_local_motion_intensity_beatness`, `motion_energy_frame=root_local`, `lambda_g1_root_angular=0.02`, and full eval every 500. Epoch 1 completed in `78.48s`; epoch 50 completed and saved `train-50.pt`, then `Generating Sample` failed because MuJoCo initialized with `MUJOCO_GL=None` (`gladLoadGL error`). This was a training sample-render environment failure, not a model/loss failure. |
| 2026-05-26 | MuJoCo sample-render guard | patched | source `train.py`, `args.py`, `EDGE.py`; test `.venv311/bin/python -m py_compile args.py train.py EDGE.py && .venv311/bin/python -m unittest tests.test_phase0_cli_and_preprocess`; memory note `/home/tianhup/.codex/memories/extensions/ad_hoc/notes/20260526T214857Z-musics2dance-mujoco-training-render.md` | Training parser now accepts `--g1_render_backend`, `--g1_render_width`, `--g1_render_height`, and `--g1_mujoco_gl`; `train.py` sets `MUJOCO_GL` before launching G1 training; training sample renders pass the explicit G1 render args into `render_sample`. |
| 2026-05-26 | r01 resume from 50 | stopped after ckpt500 eval | log `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_resume50_20260526.log`; run dir `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume50`; W&B run `ncos55nv`; checkpoint `weights/train-500.pt`; eval root `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume50` | Reached global epoch 500 and completed all configured full-eval variants. Continued to epoch 533, then stopped on `zipfile.BadZipFile: Bad CRC-32 for file 'motion_beatness_envelope.npy'` while a DataLoader worker read a motion-control `.npz`. Full `unzip -t` and NumPy replay over all `51082` v3 local `.npz` files found `0` persistent corrupt files (`setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_bad_npz_20260527.txt`, `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_bad_npz_numpy_20260527.txt`), so this looks like a transient npz read failure or an error path that needs better filename reporting before resuming long runs. |
| 2026-05-27 | motion-control npz diagnostics | patched | source `dataset/dance_dataset.py`; test `.venv311/bin/python -m unittest tests.test_feature_config_and_fusion tests.test_motion_control_v2_features tests.test_motion_energy_condition_variants tests.test_phase0_cli_and_preprocess` | Structured motion-control loading now copies intensity/beatness arrays while the `.npz` is open and re-raises `zipfile.BadZipFile` with the exact feature-cache path plus rebuild instructions. Test result: `Ran 37 tests ... OK`. This does not change model inputs. |
| 2026-05-27 | r01 resume from 500 | stopped at sample render | launcher `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_resume500_20260527.sh`; log `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_resume500_20260527.log`; run dir `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume500`; checkpoint `weights/train-550.pt`; W&B run `r4p4vhle` | Reached global epoch `550`, saved `train-550.pt`, and completed DDIM sample generation. It then hung for more than three hours in the training sample-render path: log stopped after `Generating Sample` / `DDIM sampling: 100%`, GPU utilization was `0%`, and process wait channel was `nvkms_open_common`. This is another MuJoCo/NVIDIA render hang, not a training-loss or data-loader stall. Killed the stuck tmux and resumed from `train-550.pt`. |
| 2026-05-27 | skip train sample render | patched | source `args.py`, `EDGE.py`; test `.venv311/bin/python -m py_compile args.py EDGE.py && .venv311/bin/python -m unittest tests.test_phase0_cli_and_preprocess` | Added explicit `--skip_train_sample_render`. Checkpoint saves and scheduled full eval still run, but periodic preview sample renders can be skipped so MuJoCo preview generation cannot block training. Test result: `Ran 21 tests ... OK`. |
| 2026-05-27 | r01 resume from 550 | stopped for render after ckpt1550 | tmux `m2d_train_wav2clip_local_intensity_beatness`; launcher `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_resume550_20260527.sh`; log `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_resume550_20260527.log`; run dir `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550`; checkpoint input `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume500/weights/train-550.pt`; W&B run `h6p3lad8` | Relaunched with `--skip_train_sample_render`, `--checkpoint train-550.pt`, `--epoch_offset 550`, and `--epochs 1450`, so global training resumed at epoch `551/2000`. It reached global epoch `1550`; training was intentionally interrupted after `train-1550.pt` to free the GPU for requested long MuJoCo renders. |
| 2026-05-28 | ckpt1000 full eval | passed, training continuing | checkpoint `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1000.pt`; eval root `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550`; W&B run `h6p3lad8` | Completed all configured full-eval variants at global epoch 1000: `pred_controls`, `oracle_controls`, `flat_intensity`, `zero_beatness`, and `zero_all_controls`. Training continued past checkpoint 1000 toward the required 1500 eval. No phase diagnostic file was generated under this eval root, so phase audit remains pending. |
| 2026-05-28 | ckpt1500 full eval | passed | checkpoint `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt`; eval root `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550`; W&B run `h6p3lad8` | Completed all configured full-eval variants at global epoch 1500. Pred controls improved rhythm and robot distance versus 1000 (`G1FKBAS=0.2429`, `BeatF1=0.2106`, `G1Dist=5.7822`, `G1FootSliding=0.7639`) but diversity fell (`G1Div=14.0929`), so checkpoint 2000 is useful but diversity remains the main watch item. |
| 2026-05-28 | three motion-energy MuJoCo comparison renders | invalidated, deleted | deleted the bad short MuJoCo comparison render directories, their log, and the superseded no-render final-hard-overlap probe | These 40s renders used the wrong duration and were contaminated by the final hard-overlap overwrite in `long_ddim_sample`, which created discontinuities at 75-frame slice boundaries. Boundary diagnosis on `012` showed bad root-position jumps near `0.96 m/frame` for r05/v3 and `0.82 m/frame` for v3b; after removing final hard-overlap overwrite, a no-render probe dropped the same check to about `0.035` for r05 and `0.064` for v3b. Do not use these deleted artifacts for qualitative judgment. Clean 90s MuJoCo comparisons are pending a GPU-safe render window. |
| 2026-05-28 | r01 resume from 1550 | completed | launcher `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_resume1550_20260528.sh`; log `setup_logs/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_train_r01_resume1550_20260528.log`; run dir `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550`; checkpoint input `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1550.pt`; W&B run `qe15imkz` | Relaunched with `--skip_train_sample_render`, `--checkpoint train-1550.pt`, `--epoch_offset 1550`, and `--epochs 450`, so global training continued at epoch `1551/2000`. It reached global epoch `2000` and completed all configured full-eval variants. |
| 2026-05-29 | ckpt2000 full eval | passed, needs decision | checkpoint `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/weights/train-2000.pt`; eval root `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550`; W&B run `qe15imkz` | Completed `pred_controls`, `oracle_controls`, `flat_intensity`, `zero_beatness`, and `zero_all_controls`. Pred controls regressed versus 1500 on rhythm (`G1FKBAS 0.2429 -> 0.2378`, `BeatF1 0.2106 -> 0.2031`) while recovering some diversity (`G1Div 14.0929 -> 14.6534`). Root rotation was not fixed: `RootAngularVelocityP99=6.0926`, `Max=92.2423`, `GtPiRate=0.1044`, and `Gt2PiRate=0.0230`. |
| 2026-05-30 | root-rotation representation audit | passed, no-op for requested 6D ablation | source `dataset/motion_representation.py`; dataset callsite `dataset/dance_dataset.py`; checkpoint configs `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/weights/train-2000.pt`, `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r03_resume550/weights/train-1000.pt`, and `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2/weights/train-1000.pt` | Current G1 training already encodes raw root quaternions into 6D rotation before normalization/model input (`G1_REPR_DIM=38`). v3b, v3, and r05 checkpoint configs all report `motion_format=g1` and `repr_dim=38`, so launching a "direct 6D" ablation would repeat the existing representation. The unresolved rotation tail should be treated as an absolute-root/trajectory/behavior issue, not a direct-quaternion neural-output issue. |

## Checkpoint 500 Full Eval

Metric paths:

- Predicted controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume50/ckpt500_pred_controls/metrics.json`
- Oracle controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume50/ckpt500_oracle_controls/metrics.json`
- Flat intensity: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume50/ckpt500_flat_intensity/metrics.json`
- Zero beatness: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume50/ckpt500_zero_beatness/metrics.json`
- Zero all controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume50/ckpt500_zero_all_controls/metrics.json`

Summary:

| Variant | G1FKBAS | Beat F1 | G1Dist | G1Div | Joint Range | Root Flat Range | Root Ang P99 | Foot Sliding | Ground Pen. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pred_controls | 0.2162 | 0.1859 | 6.7673 | 18.3761 | 1.2426 | 0.5395 | 6.5271 | 0.7780 | 0.0926 |
| oracle_controls | 0.2162 | 0.1880 | 6.7049 | 19.0556 | 1.2430 | 0.5389 | 6.5821 | 0.7923 | 0.1322 |
| flat_intensity | 0.2280 | 0.1920 | 6.3309 | 18.9561 | 1.1885 | 0.5136 | 5.9875 | 0.7778 | 0.0799 |
| zero_beatness | 0.2184 | 0.1861 | 7.4575 | 19.7421 | 1.2769 | 0.5536 | 7.0158 | 0.8497 | 0.0877 |
| zero_all_controls | 0.2576 | 0.2035 | 18.0584 | 7.2961 | 0.0932 | 0.1059 | 1.1276 | 0.2170 | 0.0088 |

Interpretation:

- The main predicted-control model has not collapsed: `G1Div=18.38` and joint range `1.24` stay close to r05/v3 amplitude, not Librosa35/STFT's low-diversity regime.
- Root-local controls did not yet solve the root-turn tail at checkpoint 500. `RootAngularVelocityP99=6.53 rad/s`, about `2pi`, and max is still around `93 rad/s`. The max tail remains too high.
- The beatness condition is not measurably controlling rhythm yet. `zero_beatness` is almost identical to `pred_controls` on `G1FKBAS` and `Beat F1`, and oracle controls do not give a rhythm upper bound over predicted controls.
- `zero_all_controls` has better beat metrics only because motion amplitude collapses (`G1Div=7.30`, joint range `0.093`, `G1Dist=18.06`). Treat that row as a collapse diagnostic, not a better model.
- `flat_intensity` is slightly better than pred controls on rhythm and distance at this checkpoint, which suggests the current predicted intensity may still inject noisy amplitude timing before the denoiser has learned the local-control semantics.

## Checkpoint 1000 Full Eval

Metric paths:

- Predicted controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1000_pred_controls/metrics.json`
- Oracle controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1000_oracle_controls/metrics.json`
- Flat intensity: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1000_flat_intensity/metrics.json`
- Zero beatness: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1000_zero_beatness/metrics.json`
- Zero all controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1000_zero_all_controls/metrics.json`

Summary:

| Variant | G1FKBAS | Beat F1 | Precision | Recall | G1Dist | G1Div | Joint Range | Root Flat Range | Root Ang P99 | Foot Sliding | Ground Pen. | Root Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pred_controls | 0.2375 | 0.2079 | 0.3224 | 0.1647 | 6.4878 | 16.0776 | 1.2471 | 0.7239 | 5.9723 | 0.8364 | 0.0525 | 0.3333 |
| oracle_controls | 0.2380 | 0.2054 | 0.3218 | 0.1638 | 6.4266 | 16.5323 | 1.2111 | 0.7033 | 6.2269 | 0.8320 | 0.1016 | 0.3257 |
| flat_intensity | 0.2406 | 0.2021 | 0.3080 | 0.1627 | 6.4978 | 16.5885 | 1.2139 | 0.7042 | 5.8360 | 0.8495 | 0.0797 | 0.3239 |
| zero_beatness | 0.2087 | 0.1787 | 0.2953 | 0.1390 | 6.3898 | 16.5647 | 1.2178 | 0.6943 | 6.2972 | 0.8719 | 0.0949 | 0.3183 |
| zero_all_controls | 0.2632 | 0.2065 | 0.2834 | 0.1744 | 18.5323 | 7.5285 | 0.1062 | 0.1333 | 0.9863 | 0.2393 | 0.0123 | 0.0706 |

Interpretation:

- From 500 to 1000, predicted-control rhythm improved clearly: `G1FKBAS` `0.2162 -> 0.2375`, `Beat F1` `0.1859 -> 0.2079`, precision `0.2997 -> 0.3224`, and recall `0.1454 -> 0.1647`.
- `zero_beatness` now significantly hurts rhythm: `G1FKBAS` drops from `0.2375` to `0.2087`, and `Beat F1` drops from `0.2079` to `0.1787`. Unlike checkpoint 500, this proves the model is using the beatness control by checkpoint 1000.
- Oracle controls do not improve over predicted controls, so the current bottleneck is not mainly the two-head control predictor. The denoiser/objective is the likely ceiling.
- Compared with v3 r03 checkpoint 1000, v3b checkpoint 1000 is slightly better on rhythm (`G1FKBAS 0.2375` vs `0.2286`, `Beat F1 0.2079` vs `0.2050`) but worse on distribution and robot quality (`G1Dist 6.4878` vs `6.0560`, `G1Div 16.0776` vs `18.4838`, foot sliding `0.8364` vs `0.7462`, root drift `0.3333` vs `0.0656`).
- Compared with r05 checkpoint 1000 predicted energy, v3b is much better on rhythm (`Beat F1 0.2079` vs `0.1866`) but worse on distance/diversity (`G1Dist 6.4878` vs `5.3518`, `G1Div 16.0776` vs `18.0232`).
- Compared with STFT r02 2000 and Librosa35 2000, v3b has better distribution/diversity than both (`G1Dist 6.4878` vs about `9`, `G1Div 16.0776` vs `12.8445`/`11.3661`) and is competitive with STFT rhythm, but it is still below Librosa35 on rhythm (`G1FKBAS 0.2375` vs `0.2544`, `Beat F1 0.2079` vs `0.2139`).
- The root-turn problem is only partially improved. `RootAngularVelocityP99` improved from checkpoint 500 (`6.5271 -> 5.9723`) and is below about `2pi`, but the maximum is still about `94 rad/s`. The tail has not disappeared.
- Root behavior shifted into drift/travel instead of only spin: `RootFlatRangeMean` and `RootDriftMean` increased from checkpoint 500 (`0.5395 -> 0.7239`, `0.2332 -> 0.3333`). If this persists at 1500, the next objective should address root path/drift/foot contact, not only angular velocity.
- `zero_all_controls` again has high beat metrics only by collapsing motion amplitude and distribution (`G1Dist 18.5323`, `G1Div 7.5285`, joint range `0.1062`). Do not treat that row as a good model.

## Checkpoint 1500 Full Eval

Metric paths:

- Predicted controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1500_pred_controls/metrics.json`
- Oracle controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1500_oracle_controls/metrics.json`
- Flat intensity: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1500_flat_intensity/metrics.json`
- Zero beatness: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1500_zero_beatness/metrics.json`
- Zero all controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/ckpt1500_zero_all_controls/metrics.json`

Summary:

| Variant | G1FKBAS | Beat F1 | Precision | Recall | G1Dist | G1Div | Joint Range | Root Flat Range | Root Ang P99 | Foot Sliding | Ground Pen. | Root Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pred_controls | 0.2429 | 0.2106 | 0.3225 | 0.1687 | 5.7822 | 14.0929 | 1.1876 | 0.6249 | 5.9071 | 0.7639 | 0.0517 | 0.2810 |
| oracle_controls | 0.2408 | 0.2048 | 0.3125 | 0.1651 | 5.7889 | 14.5547 | 1.1505 | 0.5961 | 5.9815 | 0.7546 | 0.0531 | 0.2700 |
| flat_intensity | 0.2441 | 0.2097 | 0.3175 | 0.1677 | 5.7437 | 14.6435 | 1.1584 | 0.5941 | 5.7654 | 0.7752 | 0.0993 | 0.2629 |
| zero_beatness | 0.2171 | 0.1869 | 0.2990 | 0.1468 | 5.6800 | 14.6801 | 1.1766 | 0.6013 | 6.0851 | 0.7877 | 0.0643 | 0.2690 |
| zero_all_controls | 0.2659 | 0.2096 | 0.2863 | 0.1770 | 18.4837 | 6.5687 | 0.0807 | 0.0978 | 2.2745 | 0.2648 | 0.0107 | 0.0387 |

Interpretation:

- Predicted-control rhythm keeps improving, but only modestly from checkpoint 1000: `G1FKBAS 0.2375 -> 0.2429`, `Beat F1 0.2079 -> 0.2106`, and recall `0.1647 -> 0.1687`.
- `zero_beatness` still hurts rhythm strongly (`Beat F1 0.2106 -> 0.1869`), so the beatness condition remains used at checkpoint 1500.
- Robot distribution quality improves over checkpoint 1000 on `G1Dist` (`6.4878 -> 5.7822`), foot sliding (`0.8364 -> 0.7639`), ground penetration (`0.0525 -> 0.0517`), root drift (`0.3333 -> 0.2810`), and root flat range (`0.7239 -> 0.6249`).
- The main regression is diversity/amplitude: `G1Div` drops from `16.0776` to `14.0929`, and `JointPositionRangeMean` drops from `1.2471` to `1.1876`. This is still above Librosa35/STFT-style collapse, but it is moving in the wrong direction for the anti-average goal.
- Oracle controls still do not beat predicted controls on rhythm, so the predictor is not the main ceiling.
- `zero_all_controls` again has high beat scores only through severe motion collapse (`G1Dist=18.4837`, `G1Div=6.5687`, joint range `0.0807`), so it remains a failure-mode row.

## Checkpoint 2000 Full Eval

Metric paths:

- Predicted controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/ckpt2000_pred_controls/metrics.json`
- Oracle controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/ckpt2000_oracle_controls/metrics.json`
- Flat intensity: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/ckpt2000_flat_intensity/metrics.json`
- Zero beatness: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/ckpt2000_zero_beatness/metrics.json`
- Zero all controls: `eval/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/ckpt2000_zero_all_controls/metrics.json`

Summary:

| Variant | G1FKBAS | Beat F1 | Precision | Recall | G1Dist | G1Div | Joint Range | Root Flat Range | Root Ang P99 | Root Ang Max | >pi Rate | >2pi Rate | Foot Sliding | Ground Pen. | Root Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pred_controls | 0.2378 | 0.2031 | 0.3112 | 0.1628 | 5.8714 | 14.6534 | 1.2272 | 0.6124 | 6.0926 | 92.2423 | 0.1044 | 0.0230 | 0.7648 | 0.0540 | 0.2837 |
| oracle_controls | 0.2361 | 0.2024 | 0.3112 | 0.1623 | 5.9301 | 15.2899 | 1.1948 | 0.5846 | 6.0756 | 93.8078 | 0.0988 | 0.0209 | 0.7624 | 0.0466 | 0.2721 |
| flat_intensity | 0.2476 | 0.2090 | 0.3138 | 0.1686 | 5.9642 | 15.0815 | 1.1843 | 0.5784 | 6.0170 | 93.6440 | 0.0983 | 0.0219 | 0.7680 | 0.0438 | 0.2652 |
| zero_beatness | 0.2130 | 0.1804 | 0.2889 | 0.1409 | 5.9289 | 15.3520 | 1.2166 | 0.5884 | 6.2800 | 94.2201 | 0.1019 | 0.0229 | 0.7832 | 0.0769 | 0.2699 |
| zero_all_controls | 0.2692 | 0.2120 | 0.2889 | 0.1804 | 18.6338 | 6.8975 | 0.0737 | 0.0782 | 2.9852 | 93.7263 | 0.0306 | 0.0082 | 0.2733 | 0.0123 | 0.0315 |

Interpretation:

- Checkpoint 2000 is not better than checkpoint 1500 as a main model. Rhythm regressed (`G1FKBAS 0.2429 -> 0.2378`, `Beat F1 0.2106 -> 0.2031`, recall `0.1687 -> 0.1628`) while `G1Dist`, foot sliding, ground penetration, and root drift stayed roughly flat or slightly worse.
- The main gain from 1500 to 2000 is partial recovery of diversity/range (`G1Div 14.0929 -> 14.6534`, joint range `1.1876 -> 1.2272`), but it does not recover the earlier v3/r05 diversity band.
- Beatness remains active: `zero_beatness` drops `BeatF1` from `0.2031` to `0.1804`, so the condition path is still used.
- Oracle controls do not improve rhythm, so the predictor is not the main bottleneck.
- Rotation is still unresolved. `RootAngularVelocityP99=6.0926` and `GtPiRate=0.1044` are slightly worse than 1500, and `RootAngularVelocityMax=92.2423` remains in the same extreme-tail regime as previous checkpoints.
- A quick saved-motion audit over `3265` pred-control motions found `172` clips with root angular max above `30 rad/s`, `46` above `60 rad/s`, and `5` above `90 rad/s`. The worst `Max` rows are mostly single-frame spikes, but the worst `P99` rows show sustained multi-frame high-speed turning, so this is not just one isolated metric artifact.
- `zero_all_controls` lowers most root angular distribution metrics only by collapsing motion (`G1Dist=18.6338`, `G1Div=6.8975`, joint range `0.0737`), so the apparent stability there is not acceptable.

## Invalidated Render Comparison

The three 40s MuJoCo comparisons on music `012`, `109`, and `137` were invalidated and deleted. They were too short for the intended qualitative comparison and exposed a generation bug: final hard-overlap overwrite in `long_ddim_sample` produced discontinuities at the 75-frame slice boundaries. Boundary diagnosis on the bad 40s `012` motions found root-position jumps around `0.96 m/frame` for r05/v3 and `0.82 m/frame` for v3b. After reverting the final hard-overlap overwrite, the same no-render boundary probe dropped to about `0.035` for r05 and `0.064` for v3b.

Clean 90s comparisons should use the same model set below, `--feature_source extract`, real MuJoCo through `xvfb-run` plus `MUJOCO_GL=glfw`, and a post-render boundary-jump audit before reporting the videos as valid.

Models in each comparison:

- `r05_pred_energy_1000`: `wav2clip_motion_energy_beat`, `pred_energy`, checkpoint `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2/weights/train-1000.pt`.
- `v3_pred_controls_1000`: `wav2clip_motion_intensity_beatness`, `pred_controls`, checkpoint `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r03_resume550/weights/train-1000.pt`.
- `v3b_local_pred_controls_1000`: `wav2clip_local_motion_intensity_beatness`, `pred_controls`, checkpoint `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1000.pt`.

Required validation for the clean replacement:

- Manifest check: `out_length=90.0`, `sample_size=35`, `feature_source=extract`, `audio_source=extract`, expected feature types and condition variants.
- Media check: comparison videos are about `90s`, H.264 video, AAC stereo `48000` Hz audio.
- Motion check: saved motions must not show the previous 75-frame boundary jumps; root-position boundary max should be in the old 90s/fixed-probe range, not near `0.8-1.0 m/frame`.
- Render route: direct EGL can hit the recurring `nvkms_open_common` hang, so prefer `xvfb-run -a -s '-screen 0 1280x1024x24' env MUJOCO_GL=glfw python -m eval.render_g1_checkpoint_comparison ... --g1_render_backend mujoco --g1_render_width 640 --g1_render_height 480`.

## Root-Rotation Representation Audit

The requested "switch root rotation directly to 6D" ablation is already represented by the current G1 pipeline.

Evidence:

- `dataset/motion_representation.py` defines `G1_REPR_DIM = 3 + 6 + G1_DOF_DIM`.
- `encode_g1_motion(...)` takes raw root quaternions, converts them through `quaternion_to_matrix(...)`, then stores `matrix_to_rotation_6d(...)` in the model-facing tensor.
- `decode_g1_motion(...)` converts the model's 6D output back to a quaternion only for FK, metrics, saved motion, and MuJoCo/render compatibility.
- `dataset/dance_dataset.py` calls `encode_g1_motion(root_pos, local_q[:, :, :4], local_q[:, :, 4:])` for G1 training.
- v3b checkpoint 2000, v3 checkpoint 1000, and r05 checkpoint 1000 all store `repr_dim=38`, matching `3 root position + 6 root rotation + 29 dof`.

Conclusion: the saved motions and renderer payloads contain quaternions because G1 FK/MuJoCo need them, but the denoiser is already trained on 6D root rotation. A duplicate 6D ablation would not test a new hypothesis or save the rotation-tail problem.

## Current Conclusion

Checkpoint 2000 completed, but checkpoint 1500 is the stronger current v3b checkpoint for rhythm. Neither 1500 nor 2000 fixes the sudden root-rotation problem. The root-local control signal and soft root-angular guard improved some aggregate robot metrics compared with early v3b, but they did not eliminate the extreme yaw tail. The 2026-05-30 representation audit confirms this is not caused by directly training root quaternions: current G1 training already uses 6D root rotation internally. This run should be treated as evidence that the current pipeline needs a deeper root-trajectory representation redesign rather than more epochs, a duplicate 6D run, or another small loss on the same absolute-root output.

## Next Action

Do not continue training this exact pipeline past 2000, and do not launch a duplicate "6D root rotation" run because the current representation is already 6D. Use checkpoint 1500 as the best current v3b qualitative candidate and checkpoint 2000 as a completed diagnostic point.

The next informative ablation should be a v4 root-delta/local-trajectory representation: train root translation as local velocity or delta position, train root heading/rotation as relative delta rather than absolute pose, keep the G1 dof representation unchanged, keep `--skip_train_sample_render`, keep W&B, and keep full eval every 500 epochs. That tests whether the current absolute-root trajectory output is the source of the sudden-turn tail. Phase diagnostic is still pending for checkpoints 1000/1500/2000.
