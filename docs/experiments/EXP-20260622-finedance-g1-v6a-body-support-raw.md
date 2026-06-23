# EXP-20260622-finedance-g1-v6a-body-support-raw

## Question

Can a clean raw diffusion ablation fix v5's wrist-jitter and hovering-foot failure by separating body intensity, support beatness, upper beatness, and support contact controls, while keeping the successful `g1_yaw_delta` root representation?

## Status

`rejected`

Implementation is complete on local 4090 clone branch `codex/wav2clip-stage-20260526`, base commit `1364508`. Focused unit tests for the new schema, model heads, feature cache fields, and eval variants passed on 2026-06-22. V6a feature cache has been built and validated.

Run `r01_b256_acc2` completed the planned 500 epochs and all scheduled full-eval variants. The run did not crash, but checkpoint 500 fails the V6a acceptance gate on rhythm, support/contact, grounding, and visible motion quality. Do not continue this raw V6a design to 1000.

Decision after fixed 90s render review: reject V6a as a mainline design. The issue is not merely undertraining. The visible failure is structural: motions are unreasonable, feet hover/floats, and support behavior remains unnatural even when oracle controls are supplied. This raw-control design can increase motion/diversity, but it does not give the diffusion model a strong enough feasible-motion prior or contact-consistent support manifold.

## Hypothesis

V5 failed because the old `motion_beatness` target was wrist-heavy and the pipeline had no explicit support/contact condition. V6a should remove that shortcut without becoming another hand-written loss stack:

- `body_intensity`: torso + feet activity, no wrists; local max/activity envelope.
- `support_beatness`: torso + feet rhythm, no wrists; local minimum + shoulder contrast gated by near-support.
- `upper_beatness`: wrist/upper rhythm; local minimum + shoulder contrast, no support gate.
- `support_contact`: left/right contact probability condition, aligned with eval contact thresholds.

If V6a still shows hand jitter or hovering feet after these semantic fixes, the next step is V6b robot-native latent/prior work, not more raw-loss patching.

## Baselines And Controls

- Primary comparison: `EXP-20260601-finedance-g1-yaw-delta-repr` v5 `train-1000.pt`, because it fixed root spinning but failed naturalness/contact.
- Naturalness reference: v3b 1500 and Librosa35 2000 from `EXP-20260622-cross-model-rhythm-suite`.
- Keep invariant:
  - `motion_format=g1_yaw_delta`.
  - `predict_epsilon=False`.
  - Wav2CLIP semantic condition plus GaussianBeat timing condition.
  - No STFT, Librosa35, or Jukebox in the V6a mainline.
  - No old `--use_beats` or beat-distance loss as the main mechanism.

## Implementation Contract

Feature type:

```text
wav2clip_body_support_beatness
```

Feature cache:

```text
data/finedance_g1_fkbeats/{train,test}/motion_control_v4_support_feats/*.npz
data/finedance_g1_fkbeats/motion_control_v4_support_metadata.json
```

Processed cache:

```text
data/finedance_g1_v6a_body_support_beatness_dataset_backups
```

Each `.npz` must include:

```text
body_intensity_envelope          float32 [150,1]
support_beatness_envelope        float32 [150,1]
upper_beatness_envelope          float32 [150,1]
support_contact                  float32 [150,2]
body_weighted_fk_speed           float32 [150]
support_weighted_fk_speed        float32 [150]
upper_weighted_fk_speed          float32 [150]
audio_beat_frames                int64 [N]
body_intensity_peaks             float32 [N]
support_beatness_peaks           float32 [N]
upper_beatness_peaks             float32 [N]
lowest_foot_heights              float32 [150,2]
```

Normalization is per stream using train split p05/p95. Any formula change requires rebuilding both the feature cache and processed/tensor dataset caches.

## Condition Schema

```python
cond = {
    "semantic": {
        "wav2clip": Tensor[B, 150, 512],
    },
    "control": {
        "gaussian_beat": Tensor[B, 150, 1],
        "body_intensity": Tensor[B, 150, 1],
        "support_beatness": Tensor[B, 150, 1],
        "upper_beatness": Tensor[B, 150, 1],
        "support_contact": Tensor[B, 150, 2],
    },
}
```

Architecture:

- One compact typed `ControlEncoder`, control input dim `6`.
- Shared `Wav2CLIP + GaussianBeat` predictor stem with four heads:
  - `body_intensity` MSE.
  - `support_beatness` MSE.
  - `upper_beatness` MSE.
  - `support_contact` BCE logits.
- `control_summary_projection` input: `[body_intensity_mean, support_beatness_mean, upper_beatness_mean, left_contact_mean, right_contact_mean]`.

Training policy:

- Epoch 1-100: denoiser uses GT controls.
- Epoch 101+: each sample uses 50% GT / 50% detached predicted controls.
- `upper_beatness` has predictor loss only in the first pass; no upper beatness motion-side auxiliary loss.
- `body_intensity` global log mean speed loss uses torso/ankle body-support weights.
- `support_beatness` local valley loss uses body/support keypoints only and separate support-beatness p05/p95 normalization.

## Commands

Feature/cache rebuild:

```bash
PYTHONUNBUFFERED=1 .venv311/bin/python -m data.audio_extraction.motion_control_v2_features \
  --data_path data/finedance_g1_fkbeats \
  --device cuda \
  --batch_size 512 \
  --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml \
  --g1_root_quat_order xyzw \
  --support_v6a \
  --force \
  2>&1 | tee -a setup_logs/EXP-20260622-finedance-g1-v6a-body-support-raw_feature_cache.log
```

Training command, with normalization values from `motion_control_v4_support_metadata.json`:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=online

PYTHONUNBUFFERED=1 .venv311/bin/python -m accelerate.commands.launch train.py \
  --feature_type wav2clip_body_support_beatness \
  --feature_fusion linear \
  --motion_format g1_yaw_delta \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_v6a_body_support_beatness_dataset_backups \
  --exp_name EXP-20260622-finedance-g1-v6a-body-support-raw_r01_b256_acc2 \
  --render_dir renders/EXP-20260622-finedance-g1-v6a-body-support-raw \
  --epochs 500 \
  --save_interval 50 \
  --batch_size 256 \
  --gradient_accumulation_steps 2 \
  --mixed_precision bf16 \
  --lambda_energy_pred 1.0 \
  --energy_teacher_forcing_epochs 100 \
  --energy_pred_mix_prob 0.5 \
  --energy_smoothness_weight 0.1 \
  --lambda_motion_intensity 0.05 \
  --lambda_motion_beatness 0.02 \
  --motion_beatness_warmup_start_epoch 50 \
  --motion_beatness_warmup_epochs 200 \
  --motion_beatness_max_fraction 0.10 \
  --lambda_acc 0.02 \
  --lambda_acc_final 0.10 \
  --lambda_acc_warmup_start_epoch 0 \
  --lambda_acc_warmup_epochs 500 \
  --lambda_g1_fk 0.0 \
  --lambda_g1_fk_vel 0.0 \
  --lambda_g1_fk_acc 0.0 \
  --lambda_g1_foot 0.0 \
  --lambda_g1_root_angular 0.05 \
  --g1_root_angular_max_fraction 0.05 \
  --motion_energy_frame root_local \
  --motion_intensity_norm_p05 0.06964919716119766 \
  --motion_intensity_norm_p95 1.7272565364837646 \
  --motion_beatness_norm_p05 0.001247384469024837 \
  --motion_beatness_norm_p95 0.06870578974485397 \
  --g1_mujoco_gl egl \
  --skip_train_sample_render \
  --full_eval_interval 500 \
  --full_eval_batch_size 16 \
  --full_eval_variants pred_controls,oracle_controls,flat_body_intensity,zero_support_beatness,zero_upper_beatness,zero_support_contact,zero_all_controls \
  --wandb_pj_name Musics2Dance \
  2>&1 | tee -a setup_logs/EXP-20260622-finedance-g1-v6a-body-support-raw_train_r01_b256_acc2.log
```

## Evaluation Plan

At epoch 500 run full eval variants:

```text
pred_controls
oracle_controls
flat_body_intensity
zero_support_beatness
zero_upper_beatness
zero_support_contact
zero_all_controls
```

Use the cross-model rhythm suite metrics:

- Rhythm: `G1FKBAS`, `G1BeatF1`, precision/recall, phase diagnostics.
- Anti-average/amplitude: `G1Div`, `JointPositionStdMean`, `JointPositionRangeMean`, `RootFlatRangeMean`.
- Robot quality: `G1Dist`, `G1FootSliding`, `G1GroundPenetration`, root drift/jerk.
- Contact/support: `G1FootContactOnBeatRate`, `G1NearSupportOnBeatRate`, `G1NoNearSupportRate`, `G1FootHighLiftRate`.
- Endpoint naturalness: `G1WristJerkMean`, `G1FootJerkMean`, `G1WristBeatF1`, `G1WristDominanceRatio`.

500 preliminary pass:

- `G1FKBAS` near or above v3b1500 and `G1BeatF1` not below v3b.
- `G1WristJerkMean <= 1.2x` v3b.
- `G1NoNearSupportRate <= 0.06`.
- `G1FootHighLiftRate <= 0.04`.
- `G1GroundPenetration <= 0.08`.
- `G1Dist <= 5.5`.
- `G1Div >= 14.0`.
- `zero_support_beatness` changes rhythm/support metrics.
- `zero_upper_beatness` changes wrist/upper rhythm metrics.
- `zero_support_contact` changes contact/support metrics.

If 500 passes or is close, continue to 1000. If rhythm improves but hand jitter or foot hover remains, stop raw-loss patching and start V6b prior.

## Artifacts

- Feature cache log: `setup_logs/EXP-20260622-finedance-g1-v6a-body-support-raw_feature_cache.log`.
- Training log: `setup_logs/EXP-20260622-finedance-g1-v6a-body-support-raw_train_r01_b256_acc2.log`.
- Active tmux: `m2d_train_v6a_body_support_r01`.
- W&B run: `https://wandb.ai/realroboticslab_tianhu/Musics2Dance/runs/uezc61ld`.
- Run dir: `runs/train/EXP-20260622-finedance-g1-v6a-body-support-raw_r01_b256_acc2/`.
- Full eval root: `eval/EXP-20260622-finedance-g1-v6a-body-support-raw_r01_b256_acc2/`.
- Render root: `renders/EXP-20260622-finedance-g1-v6a-body-support-raw/`.
- Matched V6a/V3b/Librosa render: `renders/EXP-20260622-finedance-g1-v6a-body-support-raw/checkpoint_comparison_012_90s_seed1234_extract_v6a500_v3b1500_librosa35/comparison.mp4`.

## Current Evidence

Feature cache built:

```text
metadata: data/finedance_g1_fkbeats/motion_control_v4_support_metadata.json
train written: 47817 / 47817
test written: 3265 / 3265
body_intensity p05/p95: 0.06964919716119766 / 1.7272565364837646
support_beatness p05/p95: 0.001247384469024837 / 0.06870578974485397
upper_beatness p05/p95: 0.0027791191823780537 / 0.14906735718250275
```

Validation:

```bash
.venv311/bin/python data/validate_preprocessed_data.py \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_v6a_body_support_beatness_dataset_backups \
  --feature_type wav2clip_body_support_beatness \
  --motion_format g1_yaw_delta \
  --feature_cache_mode off \
  --feature_cache_dtype float32 \
  --sample_count 16
```

Result: `Preprocessed data validation passed`.

Training launch:

```text
r01_b128_acc4 was started first and interrupted during epoch 1 before any checkpoint because measured speed was about 120s/epoch.
r01_b256_acc2 keeps the same effective batch size 512, uses more 4090 memory, and completed epoch 1 successfully.
epoch 1: 103.65s, 1.79 batches/s, 461.32 samples/s, peak CUDA memory 16780.26 MB.
ETA after epoch 1: 14h22m03s, estimated finish 2026-06-22 23:56:12 local time.
```

Focused tests passed:

```bash
.venv311/bin/python -m unittest \
  tests.test_feature_config_and_fusion \
  tests.test_motion_energy_condition_variants \
  tests.test_motion_control_v2_features \
  tests.test_v6a_body_support_dataset
```

Result: `Ran 23 tests ... OK`.

Checkpoint 500 training/eval status:

```text
train-500.pt saved: 2026-06-22 23:51 local time
epoch 500: 103.02s, 1.81 batches/s, 464.16 samples/s, peak CUDA memory 17050.20 MB
training elapsed: 14h18m46s
process state checked 2026-06-23: no active train/eval process, GPU idle
W&B run: https://wandb.ai/realroboticslab_tianhu/Musics2Dance/runs/uezc61ld
```

Full eval completed for all planned variants:

```text
pred_controls
oracle_controls
flat_body_intensity
zero_support_beatness
zero_upper_beatness
zero_support_contact
zero_all_controls
```

Main checkpoint-500 metrics:

| variant | G1FKBAS | BeatF1 | Dist | Div | ContactBeat | NearSupport | NoSupport | HighLift | Ground | WristJerk | RootDrift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pred_controls | 0.2214 | 0.1891 | 6.8711 | 17.9140 | 0.2886 | 0.7634 | 0.3274 | 0.1995 | 0.1190 | 649.6 | 0.8353 |
| oracle_controls | 0.2203 | 0.1879 | 6.2433 | 18.4137 | 0.3193 | 0.7774 | 0.3054 | 0.1884 | 0.0693 | 644.5 | 0.8281 |
| zero_support_contact | 0.2141 | 0.1866 | 8.7105 | 20.0473 | 0.2234 | 0.6182 | 0.4695 | 0.2989 | 0.1455 | 643.5 | 1.1112 |
| zero_all_controls | 0.2045 | 0.1773 | 13.5157 | 14.0225 | 0.6562 | 0.9759 | 0.0291 | 0.0028 | 0.0123 | 334.8 | 0.4600 |

Interpretation:

- The run is technically healthy: no traceback, no MuJoCo/EGL failure, no OOM, no bad files, and all eval variants completed.
- `pred_controls` misses the 500 gate: `G1FKBAS=0.2214` and `BeatF1=0.1891` are below v3b1500/librosa35 anchors, `Dist=6.8711` is above the target `<=5.5`, and support/contact is still poor.
- Root spinning is improved: `RootAngularVelocityP99=1.6544`, no tilt/inversion, and yaw remains bounded.
- Motion amplitude/diversity is strong: `G1Div=17.9140`, `JointPositionRangeMean=1.4554`.
- Contact/support is the blocker: `NoSupport=0.3274`, `HighLift=0.1995`, and `Ground=0.1190` fail the acceptance thresholds by a wide margin.
- The control sensitivity is mixed. Zeroing support contact strongly worsens support/contact and distribution, so contact control is active. Zeroing support beatness only weakly changes rhythm, so support beatness is not strong enough yet.

Matched render:

```text
path: renders/EXP-20260622-finedance-g1-v6a-body-support-raw/checkpoint_comparison_012_90s_seed1234_extract_v6a500_v3b1500_librosa35/comparison.mp4
manifest: renders/EXP-20260622-finedance-g1-v6a-body-support-raw/checkpoint_comparison_012_90s_seed1234_extract_v6a500_v3b1500_librosa35/manifest.json
music: data/finedance/music_wav/012.wav
duration: 90.006s
layout: grid2x2, GT + v6a_500 + v3b_1500 + librosa35_2000
feature_source: extract
backend: MuJoCo, MUJOCO_GL=egl
encoding: H.264 video, AAC stereo 48kHz audio
```

Render/debug note:

- The first 2026-06-23 matched render was invalid for GT inspection: `feature_source=extract` created temporary 2.5s-stride audio slices named `012_slice3..37`, but GT lookup incorrectly resolved those names directly against the dataset's 0.5s-stride `motions_sliced` cache. That made GT use `012_slice3..37` instead of the time-aligned `012_slice15,20,25...185`, so the GT tile looked like repeated/slow stitched motion.
- Fixed `eval.render_g1_checkpoint_comparison` so extract-mode GT lookup maps by time into cached motion slices while cache-mode lookup keeps the selected cached index unchanged.
- The canonical matched render path above was overwritten after the fix. Current manifest verification: GT source motion indices are `15,20,25...185`, unique diff `5`; saved GT motion shape is `root_pos=(2700,3)` and `dof_pos=(2700,29)`; ffprobe reports H.264 `1280x960` video duration `90.000s` plus AAC stereo audio duration `90.005s`.
- Visual spot-check contact sheet: `renders/EXP-20260622-finedance-g1-v6a-body-support-raw/checkpoint_comparison_012_90s_seed1234_extract_v6a500_v3b1500_librosa35/gt_timebase_check.jpg`.
- Validation: `.venv311/bin/python -m unittest tests.test_render_g1_checkpoint_comparison`.

## Rejection Analysis

Fixed-render qualitative review rejects V6a. The V6a tile shows unreasonable motion and persistent floating/hovering feet. That failure matches the quantitative robot diagnostics instead of contradicting them:

| model | FKBAS ↑ | BeatF1 ↑ | Dist ↓ | Div ↑ | ContactBeat ↑ | NearSupport ↑ | NoSupport ↓ | HighLift ↓ | Ground ↓ | FootSlide ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6a pred 500 | 0.221 | 0.189 | 6.871 | 17.914 | 0.289 | 0.763 | 0.327 | 0.199 | 0.119 | 0.959 |
| v6a oracle 500 | 0.220 | 0.188 | 6.243 | 18.414 | 0.319 | 0.777 | 0.305 | 0.188 | 0.069 | 0.937 |
| v6a zero all | 0.204 | 0.177 | 13.516 | 14.023 | 0.656 | 0.976 | 0.029 | 0.003 | 0.012 | 0.385 |
| v3b 1500 | 0.243 | 0.211 | 5.782 | 14.093 | 0.564 | 0.962 | 0.058 | 0.038 | 0.052 | 0.755 |
| librosa35 2000 | 0.254 | 0.214 | 9.254 | 11.366 | 0.943 | 0.997 | 0.001 | 0.003 | 0.035 | 0.531 |

Why this is not just undertraining:

- Oracle controls do not fix the failure. `oracle_controls` still has `NoSupport=0.305`, `HighLift=0.188`, and `FootSlide=0.937`, so the main issue is not the control predictor being too immature.
- Zeroing all controls is physically safer but motion-poor. `zero_all_controls` sharply improves support/contact metrics while destroying quality/distribution (`Dist=13.516`, `Div=14.023`), which means the learned controls push amplitude without enforcing feasible support.
- V6a improved one target but failed the design target. Wrist jerk is lower than v3b, but feet/support become worse. That is a failure of the raw-control formulation, not a checkpoint-selection issue.
- Continuing to 1000 would spend compute on a design that already violates the intended long-term direction: fewer hand-written patches, stronger robot-native prior/feasibility, and contact-consistent motion manifolds.

## Next Action

1. Do not resume `EXP-20260622-finedance-g1-v6a-body-support-raw_r01_b256_acc2`.
2. Archive V6a as a negative ablation: contact/support conditions are active, but raw controls do not produce natural robot support behavior.
3. Move the next mainline to V6b prior/feasibility work rather than adding another small hand-written loss stack to V6a.
