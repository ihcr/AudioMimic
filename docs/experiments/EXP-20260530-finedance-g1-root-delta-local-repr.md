# EXP-20260530-finedance-g1-root-delta-local-repr

Status: running
Owner: yukun
Created: 2026-05-30
Last Updated: 2026-06-01

## Research Question

Does replacing absolute G1 root pose diffusion with a local root-delta trajectory representation reduce the sudden high-speed root-turn tail seen in v3/v3b, without sacrificing the Wav2CLIP+GaussianBeat motion-intensity/beatness gains?

## Hypothesis

The previous v3/v3b root-rotation issue is not direct quaternion output: current G1 training already uses 6D root rotation internally. The likely failure is absolute global root trajectory generation. A local-delta representation should make the denoiser predict per-frame local translation and relative root rotation, then integrate them at decode time, reducing global heading discontinuities without adding another hand-tuned root-angular loss.

## Baseline Or Control

- Main control: `EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness` v3b.
- Best current v3b qualitative/rhythm checkpoint: `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt`.
- Completed v3b diagnostic checkpoint: `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume1550/weights/train-2000.pt`.
- Main v3b 2000 metrics: `G1FKBAS=0.2378`, `BeatF1=0.2031`, `G1Dist=5.8714`, `G1Div=14.6534`, `RootAngularVelocityP99=6.0926`, `RootAngularVelocityMax=92.2423`, `G1FootSliding=0.7648`, `RootDrift=0.2837`.

## Intervention

New motion format:

```text
g1_root_delta
```

Model-facing representation keeps the G1 dimensionality at 38:

```text
root_local_delta_xy: [2]
root_height: [1]
root_delta_rot_6d: [6]
dof_pos: [29]
```

Encoding:

- First-frame root xy is canonicalized to zero.
- First-frame root rotation is canonical identity.
- For frame `t > 0`, root xy displacement is represented in the previous root-local frame.
- For frame `t > 0`, root rotation is represented as relative rotation from `t-1` to `t`, then converted to 6D.
- Root height stays absolute per frame.
- G1 dof positions are unchanged.

Decoding:

- Integrate local xy deltas from canonical origin.
- Integrate relative root rotations from canonical identity heading.
- Convert decoded root rotation back to native G1 quaternion payload for FK, metrics, saved motions, and MuJoCo.

## Invariant Controls

- Feature type: `wav2clip_local_motion_intensity_beatness`.
- Condition schema: Wav2CLIP semantic plus GaussianBeat, motion intensity, and motion beatness controls.
- Feature cache: reuse `data/finedance_g1_fkbeats/{train,test}/motion_control_v3_local_feats`.
- Model architecture, control encoder, and predictor heads remain v3b-compatible.
- Full eval variants remain `pred_controls`, `oracle_controls`, `flat_intensity`, `zero_beatness`, and `zero_all_controls`.
- W&B remains enabled.
- Training sample renders remain disabled with `--skip_train_sample_render`.

## Data And Cache Contract

Raw dataset and motion-control feature cache are unchanged. The processed tensor cache must be new because the motion representation and normalizer semantics changed:

```text
data/finedance_g1_wav2clip_local_motion_intensity_beatness_rootdelta_dataset_backups
```

Do not reuse old `g1` processed or tensor caches for this experiment. If the root-delta formula changes, delete this processed cache before relaunching.

## Implementation Scope

- `dataset/motion_representation.py`: add `g1_root_delta` encode/decode and G1-family motion-format helpers.
- `dataset/dance_dataset.py`: route G1-family formats through G1 loading and encode by selected motion format.
- `model/diffusion.py`: decode G1-family samples by motion format for FK losses, motion controls, saved motions, and long generation.
- `args.py`, `train.py`, and eval entry points: accept `g1_root_delta` and preserve MuJoCo EGL protection.
- Focused tests cover representation roundtrip, canonical first frame, dataset loading, render payload decode, CLI/condition/render paths.

## Training Plan

Run from scratch. Do not finetune v3b checkpoints because the target representation and normalizer are different.

Command:

```bash
source .venv311/bin/activate
export MUJOCO_GL=egl
PYTHONUNBUFFERED=1 accelerate launch train.py \
  --feature_type wav2clip_local_motion_intensity_beatness \
  --feature_fusion linear \
  --motion_format g1_root_delta \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_wav2clip_local_motion_intensity_beatness_rootdelta_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260530-finedance-g1-root-delta-local-repr_r01 \
  --render_dir renders/EXP-20260530-finedance-g1-root-delta-local-repr \
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
  2>&1 | tee -a setup_logs/EXP-20260530-finedance-g1-root-delta-local-repr_train_r01_20260530.log
```

## Evaluation Plan

Full eval every 500 epochs:

- `ckpt500_pred_controls`, `ckpt500_oracle_controls`, `ckpt500_flat_intensity`, `ckpt500_zero_beatness`, `ckpt500_zero_all_controls`.
- Repeat at 1000, 1500, and 2000.

Main gates:

- Root rotation: `RootAngularVelocityP99`, `RootAngularVelocityMax`, `GtPiRate`, `Gt2PiRate`, and saved-motion tail counts.
- Rhythm: `G1BAS`, `G1FKBAS`, `G1BeatF1`, precision, recall.
- Anti-average: `G1Div`, `JointPositionStdMean`, `JointPositionRangeMean`, `RootFlatRangeMean`.
- Robot quality: `G1Dist`, foot sliding, ground penetration, root drift, root jerk.

Acceptance at 1000:

- `RootAngularVelocityP99 < 4.5 rad/s`.
- Saved-motion audit has far fewer `RootAngularVelocityMax > 60 rad/s` clips than v3b 2000's `46/3265`.
- `G1FKBAS >= 0.235` and `BeatF1 >= 0.200`.
- `G1Div >= 14.5`.
- `G1Dist <= 6.5`.
- No obvious sudden-turn bursts in matched 90s comparison renders.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-30 | implementation | passed | source changes in `dataset/motion_representation.py`, `dataset/dance_dataset.py`, `model/diffusion.py`, `args.py`, `train.py`, `eval/run_g1_dataset_eval.py`, `eval/run_full_song_eval.py`, `data/validate_preprocessed_data.py`; tests `.venv311/bin/python -m py_compile ...` and `.venv311/bin/python -m unittest tests.test_g1_motion_format tests.test_phase0_cli_and_preprocess tests.test_feature_config_and_fusion tests.test_motion_energy_condition_variants tests.test_render_g1_checkpoint_comparison` | Adds `g1_root_delta`, keeps old `g1` checkpoints compatible, and disables training sample render for launch. Test result: `Ran 53 tests ... OK`. |
| 2026-05-30 | r01 first launch attempt | failed fast | log `setup_logs/EXP-20260530-finedance-g1-root-delta-local-repr_train_r01_20260530.log` | The tmux pane exited immediately with `accelerate: command not found` because the shell did not expose the accelerate console script. No training state, checkpoint, or cache was produced by this failed attempt. |
| 2026-05-30 | r01 launch | running | tmux `m2d_train_root_delta_v4`; log `setup_logs/EXP-20260530-finedance-g1-root-delta-local-repr_train_r01_20260530.log`; run dir `runs/train/EXP-20260530-finedance-g1-root-delta-local-repr_r01`; W&B run `ybzxni4p` | Relaunched with `.venv311/bin/python -m accelerate.commands.launch train.py`, `MUJOCO_GL=egl`, `--motion_format g1_root_delta`, `--lambda_g1_root_angular 0.0`, `--skip_train_sample_render`, and full eval every 500 epochs. The log reached dataset loading and epoch `1/2000`; GPU utilization was active, so training is running rather than stuck in MuJoCo preview. |
| 2026-05-31 | r01 checkpoint 500 full eval | completed, training continuing | checkpoint `runs/train/EXP-20260530-finedance-g1-root-delta-local-repr_r01/weights/train-500.pt`; eval root `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt500_*`; log reached training epoch 623 after eval | Main `pred_controls`: `G1FKBAS=0.2019`, `BeatF1=0.1747`, `G1Dist=4.9064`, `G1Div=19.3670`, `JointPositionRangeMean=1.2709`, `RootFlatRangeMean=0.2604`, `RootAngularVelocityP99=4.8891`, `RootAngularVelocityMax=24.6522`, `G1FootSliding=0.8929`, `RootDrift=0.1913`. Root max tail is fixed versus v3b, but rhythm and foot sliding are not yet acceptable. |
| 2026-06-01 | r01 checkpoint 1000 full eval | completed, training continuing | checkpoint `runs/train/EXP-20260530-finedance-g1-root-delta-local-repr_r01/weights/train-1000.pt`; eval root `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1000_*` | Main `pred_controls`: `G1FKBAS=0.2195`, `BeatF1=0.1826`, `G1Dist=3.4906`, `G1Div=15.3983`, `JointPositionRangeMean=1.2269`, `RootFlatRangeMean=0.3169`, `RootAngularVelocityP99=4.0905`, `RootAngularVelocityMax=16.7639`, `G1FootSliding=0.8327`, `RootDrift=0.2355`. This is the best root-stability checkpoint so far and passes the root/div/dist gates, but rhythm remains below target. |
| 2026-06-01 | r01 checkpoint 1500 full eval | completed, training continuing | checkpoint `runs/train/EXP-20260530-finedance-g1-root-delta-local-repr_r01/weights/train-1500.pt`; eval root `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1500_*`; log reached training epoch 1505 after eval | Main `pred_controls`: `G1FKBAS=0.2347`, `BeatF1=0.2025`, `G1Dist=3.3720`, `G1Div=14.4285`, `JointPositionRangeMean=1.2640`, `RootFlatRangeMean=0.3406`, `RootAngularVelocityP99=4.8527`, `RootAngularVelocityMax=16.7173`, `G1FootSliding=0.8544`, `RootDrift=0.2421`. Rhythm largely recovers, but root P99 and foot sliding worsen relative to 1000. |

## Checkpoint 500 Full Eval

Metric paths:

- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt500_pred_controls/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt500_oracle_controls/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt500_flat_intensity/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt500_zero_beatness/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt500_zero_all_controls/metrics.json`

| Variant | G1FKBAS | BeatF1 | Beat P/R | G1Dist | G1Div | JointRange | RootFlatRange | RootAngP99 | RootAngMax | FootSliding | GroundPen | RootDrift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pred_controls | 0.2019 | 0.1747 | 0.2912 / 0.1343 | 4.9064 | 19.3670 | 1.2709 | 0.2604 | 4.8891 | 24.6522 | 0.8929 | 0.0590 | 0.1913 |
| oracle_controls | 0.2022 | 0.1751 | 0.2925 / 0.1347 | 4.6458 | 19.8071 | 1.2424 | 0.2525 | 4.8446 | 20.5286 | 0.8784 | 0.0825 | 0.1834 |
| flat_intensity | 0.2109 | 0.1824 | 0.2948 / 0.1420 | 4.8630 | 20.3748 | 1.2535 | 0.2708 | 5.0150 | 22.7404 | 0.9391 | 0.0741 | 0.1972 |
| zero_beatness | 0.1995 | 0.1714 | 0.2876 / 0.1322 | 5.0600 | 20.1533 | 1.2651 | 0.2725 | 5.4259 | 24.1379 | 0.9631 | 0.1209 | 0.1960 |
| zero_all_controls | 0.2489 | 0.2007 | 0.2900 / 0.1655 | 18.3533 | 9.9566 | 0.1029 | 0.1967 | 0.6662 | 1.9957 | 0.2022 | 0.0165 | 0.1742 |

Saved-motion audit for `pred_controls`:

- `num_valid_files=3265`, `bad_files=0`.
- `RootAngularVelocityMax > 30/60/90 rad/s`: `0/0/0` clips. This is the strongest evidence that root-delta removed the previous rare extreme sudden-turn tail.
- `RootAngularVelocityMax > 10/20 rad/s`: `226/10` clips.
- `RootAngularVelocityP99 > 4.5/5.0/6.0 rad/s`: `1499/1301/916` clips. The remaining issue is frequent moderate-fast turning, not the old extreme one-frame spikes.

Interpretation:

- The root-delta representation is doing the important thing it was designed to do: `RootAngularVelocityMax=24.65`, versus v3b 500 around `92.94` and v3b 2000 around `92.24`.
- The acceptance gate is not passed at 500. `RootAngularVelocityP99=4.8891` is still above the `<4.5` target; `G1FKBAS=0.2019` and `BeatF1=0.1747` are below the `0.235/0.200` rhythm targets.
- Distribution quality is strong: `G1Dist=4.9064`, `G1Div=19.3670`, and `JointPositionRangeMean=1.2709` are better than v3b 500/2000 on anti-average and distribution metrics.
- Root travel is more controlled than v3b: `RootFlatRangeMean=0.2604` and `RootDrift=0.1913`.
- Contact is mixed: ground penetration is acceptable, but `G1FootSliding=0.8929` is worse than v3b 500/2000. This needs attention if it persists at 1000.
- `pred_controls` and `oracle_controls` are very close, so the control predictor is not the main 500-epoch bottleneck.
- `zero_beatness` changes rhythm only slightly at 500, so beatness usage is still weak this early. Recheck at 1000 before changing weights.
- `zero_all_controls` has high beat scores only because it collapses motion amplitude (`G1Div=9.9566`, `JointPositionRangeMean=0.1029`, `G1Dist=18.3533`); do not treat that as a real win.

## Checkpoint 1000 And 1500 Full Eval

Metric paths:

- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1000_pred_controls/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1000_oracle_controls/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1000_flat_intensity/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1000_zero_beatness/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1000_zero_all_controls/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1500_pred_controls/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1500_oracle_controls/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1500_flat_intensity/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1500_zero_beatness/metrics.json`
- `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1500_zero_all_controls/metrics.json`

| Epoch/Variant | G1FKBAS | BeatF1 | Beat P/R | G1Dist | G1Div | JointRange | RootFlatRange | RootAngP99 | RootAngMax | FootSliding | GroundPen | RootDrift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 pred_controls | 0.2195 | 0.1826 | 0.2868 / 0.1447 | 3.4906 | 15.3983 | 1.2269 | 0.3169 | 4.0905 | 16.7639 | 0.8327 | 0.0798 | 0.2355 |
| 1000 oracle_controls | 0.2246 | 0.1906 | 0.2943 / 0.1530 | 3.4060 | 15.9288 | 1.2038 | 0.3016 | 4.1616 | 15.9542 | 0.8511 | 0.0876 | 0.2243 |
| 1000 flat_intensity | 0.2352 | 0.1969 | 0.2956 / 0.1584 | 3.3597 | 15.8506 | 1.1932 | 0.3239 | 4.1297 | 15.7798 | 0.8705 | 0.1095 | 0.2376 |
| 1000 zero_beatness | 0.2169 | 0.1852 | 0.2941 / 0.1458 | 3.4733 | 15.9149 | 1.1993 | 0.3172 | 4.2597 | 17.2239 | 0.8467 | 0.0710 | 0.2342 |
| 1000 zero_all_controls | 0.2505 | 0.2015 | 0.2870 / 0.1673 | 18.2227 | 9.9242 | 0.0989 | 0.3258 | 0.9794 | 2.2449 | 0.3368 | 0.0143 | 0.2399 |
| 1500 pred_controls | 0.2347 | 0.2025 | 0.3123 / 0.1616 | 3.3720 | 14.4285 | 1.2640 | 0.3406 | 4.8527 | 16.7173 | 0.8544 | 0.0614 | 0.2421 |
| 1500 oracle_controls | 0.2384 | 0.2074 | 0.3183 / 0.1654 | 3.2142 | 15.0293 | 1.2420 | 0.3318 | 4.9775 | 18.1763 | 0.8649 | 0.0838 | 0.2372 |
| 1500 flat_intensity | 0.2555 | 0.2187 | 0.3222 / 0.1785 | 3.0727 | 14.8469 | 1.2218 | 0.3433 | 4.8466 | 18.4670 | 0.8912 | 0.0596 | 0.2427 |
| 1500 zero_beatness | 0.2167 | 0.1854 | 0.2955 / 0.1456 | 3.3564 | 15.0363 | 1.2449 | 0.3383 | 4.9910 | 18.4615 | 0.8705 | 0.0695 | 0.2416 |
| 1500 zero_all_controls | 0.2549 | 0.2047 | 0.2878 / 0.1713 | 17.4809 | 7.7663 | 0.1106 | 0.2096 | 0.9284 | 2.6264 | 0.3375 | 0.0149 | 0.1687 |

Saved-motion audit:

- 1000 `pred_controls`: `RootAngularVelocityMax > 30/60/90 rad/s = 0/0/0`, `RootAngularVelocityP99 > 4.5/6.0 rad/s = 1135/609` clips.
- 1500 `pred_controls`: `RootAngularVelocityMax > 30/60/90 rad/s = 0/0/0`, `RootAngularVelocityP99 > 4.5/6.0 rad/s = 1440/863` clips.
- The extreme sudden-turn tail remains solved through 1500, but the moderate-fast turning population grows again from 1000 to 1500.

Comparison to main controls:

| Checkpoint | G1FKBAS | BeatF1 | G1Dist | G1Div | JointRange | RootFlatRange | RootAngP99 | RootAngMax | FootSliding | RootDrift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4 root-delta 1000 pred | 0.2195 | 0.1826 | 3.4906 | 15.3983 | 1.2269 | 0.3169 | 4.0905 | 16.7639 | 0.8327 | 0.2355 |
| v4 root-delta 1500 pred | 0.2347 | 0.2025 | 3.3720 | 14.4285 | 1.2640 | 0.3406 | 4.8527 | 16.7173 | 0.8544 | 0.2421 |
| v3b 1000 pred | 0.2375 | 0.2079 | 6.4878 | 16.0776 | 1.2471 | 0.7239 | 5.9723 | 94.1768 | 0.8364 | 0.3333 |
| v3b 1500 pred | 0.2429 | 0.2106 | 5.7822 | 14.0929 | 1.1876 | 0.6249 | 5.9071 | 93.9078 | 0.7639 | 0.2810 |
| v3b 2000 pred | 0.2378 | 0.2031 | 5.8714 | 14.6534 | 1.2272 | 0.6124 | 6.0926 | 92.2423 | 0.7648 | 0.2837 |
| v3 1000 pred | 0.2286 | 0.2050 | 6.0560 | 18.4838 | 1.2945 | 0.2282 | n/a | n/a | 0.7462 | 0.0656 |
| Wav2CLIP+STFT r02 2000 | 0.2377 | 0.1979 | 8.9113 | 12.8445 | n/a | n/a | n/a | n/a | 0.5572 | 0.2726 |
| Librosa35 2000 | 0.2544 | 0.2139 | 9.2544 | 11.3661 | n/a | n/a | n/a | n/a | 0.5349 | 0.2022 |
| GaussianBeat 1000 | 0.2311 | 0.1913 | 9.2000 | 20.5369 | n/a | n/a | n/a | n/a | 0.6015 | 0.2709 |

Interpretation:

- 1000 is the cleanest root-quality checkpoint: it passes `RootAngularVelocityP99 < 4.5`, has no `>30 rad/s` clips, and has the best `G1Dist`, but rhythm is too low (`G1FKBAS=0.2195`, `BeatF1=0.1826`).
- 1500 is the best balanced checkpoint so far: `BeatF1=0.2025` passes the 1000 acceptance target and `G1FKBAS=0.2347` is effectively at the `0.235` threshold, but `RootAngularVelocityP99=4.8527` fails the root-quality gate and `G1Div=14.4285` is slightly below `14.5`.
- Compared with v3b, v4 root-delta is much better on distribution distance and root max spikes, while only slightly behind on rhythm. This confirms the representation change is valuable.
- Compared with v3, v4 1500 is close on `BeatF1` but lower on diversity and worse on foot sliding; v3's low root drift is not an apples-to-apples win because v3 did not have the same root-tail audit in its metric schema.
- Compared with older STFT/Librosa35 baselines, v4 1500 has far better `G1Dist` and better diversity than Librosa/STFT, but Librosa35 still has higher beat score and much lower foot sliding.
- `flat_intensity` at 1500 reaches the best beat numbers (`G1FKBAS=0.2555`, `BeatF1=0.2187`) and better `G1Dist=3.0727`, but it worsens foot sliding. This suggests predicted intensity is not ideal yet, and the model may respond to a more stable amplitude prior, but flat intensity is not a deployment answer.
- `zero_beatness` at 1500 drops `G1FKBAS` from `0.2347` to `0.2167` and `BeatF1` from `0.2025` to `0.1854`, so beatness finally matters by 1500.
- `zero_all_controls` still scores deceptively high on beat metrics while collapsing motion (`G1Div=7.7663`, `JointRange=0.1106`, `G1Dist=17.4809`), so it remains a diagnostic of metric weakness rather than a candidate.

## Current Conclusion

Root-delta is validated as a representation-level fix for the rare extreme root-turn tail: checkpoints 1000 and 1500 both have `0` saved motions above `30/60/90 rad/s`, unlike v3b. The unresolved tradeoff is that rhythm improves from 1000 to 1500 while moderate-fast root turning and foot sliding worsen. Checkpoint 1500 is currently the best balanced candidate for qualitative render, while checkpoint 1000 is the cleanest root-stability reference.

## 2026-06-01 Matched Long Render

Rendered a 90s same-music comparison for the current qualitative audit:

- Music: `data/finedance/music_wav/012.wav`
- Slice: `slice_start=3`, `out_length=90`, `seed=1234`, `sample_size=35`
- Feature route: `--feature_source extract`, manifest `audio_source=extract`
- Renderer: MuJoCo via `xvfb-run` and `MUJOCO_GL=glfw`; no stick fallback
- Output: `renders/EXP-20260530-finedance-g1-root-delta-local-repr/checkpoint_comparison_012_90s_seed1234_extract_v4_1500_v3b_1500_librosa35/comparison.mp4`
- Manifest: `renders/EXP-20260530-finedance-g1-root-delta-local-repr/checkpoint_comparison_012_90s_seed1234_extract_v4_1500_v3b_1500_librosa35/manifest.json`
- Logs:
  - `setup_logs/EXP-20260530-finedance-g1-root-delta-local-repr/render_012_90s_v4_1500_v3b_1500_librosa35_20260601_rerun2.log`
  - `setup_logs/EXP-20260530-finedance-g1-root-delta-local-repr/render_012_90s_v4_1500_v3b_1500_librosa35_20260601_manifest_refresh.log`

Compared models:

| Label | Checkpoint | Feature type | Variant | Motion format |
|---|---|---|---|---|
| `v4_rootdelta_1500` | `runs/train/EXP-20260530-finedance-g1-root-delta-local-repr_r01/weights/train-1500.pt` | `wav2clip_local_motion_intensity_beatness` | `pred_controls` | `g1_root_delta` |
| `v3b_local_1500` | `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt` | `wav2clip_local_motion_intensity_beatness` | `pred_controls` | `g1` |
| `librosa35_2000` | `runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt` | `baseline` | `auto` | `g1` |

Validation:

- `manifest.json` records `feature_source=extract`, `audio_source=extract`, `g1_render_backend=mujoco`, `g1_mujoco_gl=glfw`, and checkpoint-inferred mixed motion formats.
- `ffprobe` confirms `comparison.mp4` is H.264/yuv420p, `1920x536`, 90.000s, with AAC stereo 48kHz audio.
- Per-model MP4s are H.264/yuv420p, `640x480`, 90.000s, with AAC stereo 48kHz audio.
- Renderer script now supports mixed `g1_root_delta` and legacy `g1` checkpoints in one comparison by reading each checkpoint config's `motion_format`.

## 2026-06-01 Root Tilt Failure

The 90s render exposed a severe v4 failure that was not visible from `RootAngularVelocityMax` alone: the v4 robot lies sideways and floats during the long sample. This is not a MuJoCo renderer or video-composition issue. The saved v4 motion itself contains accumulated root roll/pitch drift.

Audit artifacts:

- Long-render tilt audit: `renders/EXP-20260530-finedance-g1-root-delta-local-repr/checkpoint_comparison_012_90s_seed1234_extract_v4_1500_v3b_1500_librosa35/root_tilt_audit_20260601.json`
- Short full-eval tilt audit: `eval/EXP-20260530-finedance-g1-root-delta-local-repr_r01/ckpt1500_pred_controls/root_tilt_audit_20260601.json`

Long render results:

| Label | RootUpZMean | RootUpZP01 | RootUpZMin | Tilt >60 Frames | Inverted Frames |
|---|---:|---:|---:|---:|---:|
| `v4_rootdelta_1500` | 0.4971 | -0.1313 | -0.2221 | 1276 / 2700 | 168 / 2700 |
| `v3b_local_1500` | 0.9848 | 0.9095 | 0.8745 | 0 / 2700 | 0 / 2700 |
| `librosa35_2000` | 0.9964 | 0.9790 | 0.9690 | 0 / 2700 | 0 / 2700 |

Short 150-frame eval also shows the issue, but less dramatically:

- `ckpt1500_pred_controls` files: 3265
- Global `RootUpZMin=-0.4344`
- Mean `RootUpZP01=0.8304`
- Mean `RootTiltGt60DegRate=0.00468`
- Clips with any `>60deg` tilt: 108
- Clips with any inverted root: 5

Interpretation:

- The v4 full-SO(3) `g1_root_delta` representation fixes rare one-frame root angular velocity spikes, but it removes the absolute upright/root alignment anchor and lets small roll/pitch errors accumulate as a long-sequence random walk.
- Long-sample stitching decodes the concatenated relative rotations continuously, so roll/pitch drift compounds over 90s. This is exactly the wrong failure mode for deployment.
- This is a representation failure, not a render-only bug. Do not hide it by clamping root orientation only at render time except as a temporary diagnostic.
- Evaluator now includes root-up/tilt metrics (`RootUpZMin`, `RootUpZP01`, `RootUpZMean`, `RootTiltGt30DegRate`, `RootTiltGt60DegRate`, `RootInvertedRate`) so future full evals catch this.

Training action:

- Stopped local tmux training after checkpoint `train-1550.pt` to avoid wasting GPU on the invalid v4 line.
- Preserved checkpoints through `runs/train/EXP-20260530-finedance-g1-root-delta-local-repr_r01/weights/train-1550.pt`.

## Next Action

Do not continue `r01` to 2000 as the main line. Design the next ablation as a representation-level fix: preserve root-local translation and yaw continuity, but do not integrate unconstrained roll/pitch over long sequences. Candidate v5 direction is yaw-only root delta plus bounded absolute/root-upright roll-pitch residual, with root-up metrics as acceptance gates and matched long renders required before declaring improvement.
