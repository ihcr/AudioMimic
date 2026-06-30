# EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion

## Question

Can explicit condition-use training make the accepted V6b-A G1 motion latent
prior respond to music controls, and does Wav2CLIP semantic audio add value
beyond dense rhythm controls?

## Status

`rejected`

## Decision

Run V6b-C as a paired two-route ablation. Both routes train the same
music/control-to-latent diffusion architecture over the frozen accepted V6b-A
latent space. The only intended difference is whether a Wav2CLIP semantic tower
is enabled.

```text
r01_control_only:
  dense rhythm/control tower only

r02_wav2clip_control:
  Wav2CLIP semantic tower + same dense rhythm/control tower
```

Submit both routes as separate Slurm jobs once the shared implementation passes
smoke tests. They can run at the same time if two GPUs are available; otherwise
Slurm can queue them and the comparison remains valid because all settings are
held fixed.

## Motivation

V6b-B beat8d-only latent diffusion proved that the V6b-A prior helps robot
quality, but it failed the condition-sensitivity gate. At ckpt1500,
`real_beat8d`, `shifted_beat8d`, and `random_beat8d` were nearly tied on
rhythm metrics, while `zero_beat8d` achieved higher beat scores through an
invalid low-quality mode. The next run should not merely add a larger condition;
it must make condition use measurable during training and eval.

## Shared Controls

Frozen prior and decoder:

```text
runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/weights/train-500.pt
```

Data roots:

```text
data/finedance_g1_fkbeats
data/finedance_g1_v6b_motion_prior_dataset_backups
```

New latent cache root:

```text
data/finedance_g1_v6bc_music_control_latent_dataset_backups/
```

Motion target:

```text
normalized g1_yaw_delta motion
  -> frozen V6b-A encoder
  -> normalized latent [N, 75, 128]
```

Dense rhythm/control input:

```text
data/finedance_g1_fkbeats/{train,test}/beat_features_8d_feats/*.npy
```

Use the full 8D beat feature initially, but treat it as a control stream rather
than as a semantic music stream:

```text
beat_pulse
gaussian_beat
dist_to_prev_beat_norm
dist_to_next_beat_norm
beat_phase_sin
beat_phase_cos
beat_interval_norm
onset_strength_norm
```

The dense features make this route different from V6b-B only when paired with
the new architecture and condition-use losses.

Semantic input for the Wav2CLIP route only:

```text
data/finedance_g1_fkbeats/{train,test}/wav2clip_stft_beat_feats/*.npy[:, :512]
```

Do not use STFT channels in V6b-C. The route being tested is Wav2CLIP semantic
audio plus dense beat/control, not a return to the full Wav2CLIP/STFT concat
baseline.

## Architecture

Shared latent denoiser:

```text
noisy latent [B, 75, 128]
  -> latent projection + timestep embedding
  -> transformer latent denoiser blocks
  -> predicted diffusion noise [B, 75, 128]
```

Control tower:

```text
dense beat/control [B, 150, 8]
  -> linear projection
  -> 2-layer transformer encoder
  -> control tokens [B, 150, H]
  -> cross-attention memory and pooled FiLM/AdaLN signal
```

Optional semantic tower:

```text
Wav2CLIP [B, 150, 512]
  -> linear projection
  -> 2-layer transformer encoder
  -> semantic tokens [B, 150, H]
  -> cross-attention memory
```

Fusion:

```text
latent self-attention
  + cross-attention to semantic tokens when enabled
  + cross-attention to control tokens
  + control-conditioned FiLM/AdaLN or block scale/shift from pooled control
```

Keep the two towers separate before fusion. Do not collapse Wav2CLIP and beat
controls into one concatenated feature at the input.

## Training Objective

Base denoising loss:

```text
loss_noise = MSE(pred_noise(real_condition), true_noise)
```

Condition-use ranking loss for both routes:

```text
mse_real = MSE(pred_noise(real_control), true_noise)
mse_shift = MSE(pred_noise(shifted_control), true_noise)
mse_random = MSE(pred_noise(random_control), true_noise)

loss_control_use =
  relu(margin + mse_real - mse_shift)
  + relu(margin + mse_real - mse_random)
```

For `r02_wav2clip_control`, add a weaker semantic-use diagnostic loss:

```text
mse_real_full = MSE(pred_noise(real_wav2clip, real_control), true_noise)
mse_random_sem = MSE(pred_noise(random_wav2clip, real_control), true_noise)

loss_semantic_use = relu(semantic_margin + mse_real_full - mse_random_sem)
```

Total:

```text
r01_control_only:
  loss = loss_noise + control_rank_weight * loss_control_use

r02_wav2clip_control:
  loss = loss_noise
       + control_rank_weight * loss_control_use
       + semantic_rank_weight * loss_semantic_use
```

Initial weights:

```text
control_rank_weight = 0.10
control_rank_margin = 0.02
semantic_rank_weight = 0.03
semantic_rank_margin = 0.01
cond_drop_prob = 0.10
```

The semantic ranking term is deliberately weaker because Wav2CLIP may encode
style/genre/phrase information rather than exact frame-level beat timing.

## Diagnostics During Training

Log these every epoch:

```text
train/loss_noise
train/loss_control_use
train/loss_semantic_use
train/delta_shift_control = mse_shift - mse_real
train/delta_random_control = mse_random - mse_real
train/delta_random_semantic = mse_random_sem - mse_real_full
```

Early fail signal:

```text
delta_shift_control ~= 0 and delta_random_control ~= 0 through epoch 100
```

If the deltas do not become positive early, the model is still not using the
condition and should not be trusted just because the denoising loss decreases.

## Evaluation

Run full eval every 500 epochs. Both routes must run the same core variants:

```text
real
shifted_control
random_control
zero_control
zero_all
```

For `r02_wav2clip_control`, also run semantic sensitivity variants:

```text
random_semantic
zero_semantic
random_semantic_real_control
real_semantic_random_control
```

Every variant decodes through frozen V6b-A and runs G1 metrics, including:

```text
G1BeatF1
G1FKBAS
G1FKRoboPerformBAS
G1Dist
G1Div
G1NoNearSupportRate
G1FootHighLiftRate
G1GroundPenetration
G1FootSliding
G1WristJerkMean
G1FootJerkMean
failure_panel.json
```

## Acceptance Criteria

Do not accept either route on denoising loss alone.

At ckpt500, a route is promising only if:

- `real` beats `shifted_control` and `random_control` on `G1BeatF1`,
  `G1FKBAS`, and `G1FKRoboPerformBAS`;
- `zero_control` and `zero_all` do not win rhythm metrics through quality
  collapse;
- robot quality stays in the V6b-B ckpt1500 quality range or improves:
  low no-support/high-lift rates, no ground/foot-slide explosion, and no
  endpoint jerk regression;
- validation deltas show the real condition predicts the denoising target
  better than corrupted conditions.

Compare routes as follows:

```text
If both fail condition sensitivity:
  the architecture/loss still does not solve weak conditioning.

If control_only succeeds and wav2clip_control does not:
  Wav2CLIP is introducing interference; continue dense rhythm/control without it.

If wav2clip_control succeeds and control_only does not:
  semantic audio is necessary; keep Wav2CLIP and refine semantic/control fusion.

If both succeed:
  choose the route with the better rhythm-quality Pareto and use the other as an
  ablation in the report.
```

## Implementation Scope

Reuse the compact V6b-B latent-diffusion entrypoints and extend them into one
shared V6b-C codepath:

```text
dataset/g1_latent_beat_dataset.py
model/g1_latent_diffusion.py
train_g1_latent_diffusion.py
eval/run_g1_latent_diffusion_eval.py
scripts/slurm_train_g1_latent_diffusion.sh
tests/test_g1_latent_diffusion.py
```

The legacy V6b-B dataset/model class names remain available for old artifacts.
The new V6b-C path is selected by `G1MusicControlLatentDataset`,
`G1MusicControlLatentDenoiser`, and the `--use_wav2clip_semantic` flag. The
Slurm launcher uses run suffixes:

```text
r01_control_only
r02_wav2clip_control
```

## Launch Plan

Smoke tests:

```bash
RUN_SUFFIX=r01_control_only USE_WAV2CLIP_SEMANTIC=0 EPOCHS=1 FULL_EVAL_INTERVAL=0 \
  scripts/slurm_train_g1_latent_diffusion.sh

RUN_SUFFIX=r02_wav2clip_control USE_WAV2CLIP_SEMANTIC=1 EPOCHS=1 FULL_EVAL_INTERVAL=0 \
  scripts/slurm_train_g1_latent_diffusion.sh
```

Main paired comparison:

```bash
RUN_SUFFIX=r01_control_only USE_WAV2CLIP_SEMANTIC=0 EPOCHS=500 FULL_EVAL_INTERVAL=500 \
  scripts/slurm_train_g1_latent_diffusion.sh

RUN_SUFFIX=r02_wav2clip_control USE_WAV2CLIP_SEMANTIC=1 EPOCHS=500 FULL_EVAL_INTERVAL=500 \
  scripts/slurm_train_g1_latent_diffusion.sh
```

Keep all other hyperparameters identical between routes unless a smoke test
exposes a shape or memory issue.

## Current Conclusion

Reject both V6b-C routes as mainline candidates. The full ckpt1500 eval is now
complete for `r01_control_only` and `r02_wav2clip_control`; the final Wav2CLIP
route summary is:

```text
eval/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r02_wav2clip_control/ckpt1500_v6bc_variants/summary.json
```

Against the previous V6b-B beat8d route, V6b-C does not solve the acceptance
problem. `r01_control_only` is worse than V6b-B on real-condition rhythm at
ckpt1500 and remains condition-insensitive. `r02_wav2clip_control` has the best
real-condition intermediate point at ckpt1000 (`G1BeatF1=0.2049`,
`G1FKBAS=0.2517`, `G1NoNearSupportRate=0.0609`), but corrupted conditions still
win rhythm metrics. By ckpt1500, r02 regresses to `G1BeatF1=0.1920`,
`G1FKBAS=0.2403`, `G1Dist=11.0950`, `G1NoNearSupportRate=0.1631`, and
`G1WristJerkMean=5069.6`.

The key failure mode is unchanged: `real` does not beat shifted/random/zero
conditions on the rhythm gates, and many corrupted variants achieve higher
beat scores only through robot-quality collapse. For r02 ckpt1500,
`zero_control` reaches `G1BeatF1=0.2146` and `G1FKRoboPerformBAS=0.4227`, but
with `G1Dist=2.07e8`, `G1GroundPenetration=1.20e5`, and wrist/foot jerk around
`4.18e9`. That is not usable conditioning; it is the same metric-hacking
failure in a more violent form.

2026-06-30 qualitative render review strengthens the rejection. The fresh
V6b-B and V6b-C comparison videos look globally unnatural and chaotic rather
than merely weakly synchronized; the Wav2CLIP V6b-C route shows especially
severe jerk and non-human/non-feasible motion. Treat this as a latent-generator
architecture failure, not as an isolated checkpoint choice. The next route
should not continue V6b-C training or only add richer audio features. It should
first make generated latents stay on a motion-feasible manifold and couple
conditioning to decoded beat/contact/support quality.

Live follow commands:

```bash
tail -f slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r01_control_only/train_5417035.out
tail -f slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r02_wav2clip_control/train_5417036.out
tail -f slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r02_wav2clip_control/eval_ckpt1500_mw8_5431982.out
```

Curated tee log fallbacks:

```bash
tail -f setup_logs/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/train_r01_control_only.log
tail -f setup_logs/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/train_r02_wav2clip_control.log
```

## Implementation Log

| Date | Stage | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-06-29 | implementation | passed | `dataset/g1_latent_beat_dataset.py`, `model/g1_latent_diffusion.py`, `train_g1_latent_diffusion.py`, `eval/run_g1_latent_diffusion_eval.py`, `scripts/slurm_train_g1_latent_diffusion.sh`, `tests/test_g1_latent_diffusion.py` | Shared V6b-C dataset/model/train/eval/Slurm path implemented with control-only and Wav2CLIP+control modes. |
| 2026-06-29 | unit test | passed | `.venv311/bin/python -m unittest tests.test_g1_latent_diffusion` | Synthetic cache tests cover control-only and semantic cache schemas, denoiser forward/ranking stats, DDIM sample shape, and eval corruption helpers. |
| 2026-06-29 | syntax and launch dry-run | passed | `.venv311/bin/python -m py_compile dataset/g1_latent_beat_dataset.py model/g1_latent_diffusion.py train_g1_latent_diffusion.py eval/run_g1_latent_diffusion_eval.py`; `USE_WAV2CLIP_SEMANTIC=0 scripts/slurm_train_g1_latent_diffusion.sh --dry-run`; `USE_WAV2CLIP_SEMANTIC=1 scripts/slurm_train_g1_latent_diffusion.sh --dry-run` | Dry-runs wrote route-specific sbatch files under `slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/` without submitting jobs. |
| 2026-06-29 | smoke launch | failed/cancelled | `RUN_SUFFIX=r01_control_only USE_WAV2CLIP_SEMANTIC=0 EPOCHS=1 FULL_EVAL_INTERVAL=0 SAVE_INTERVAL=1 scripts/slurm_train_g1_latent_diffusion.sh`; Slurm job `5417019`; log `slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r01_control_only/train_5417019.out` | r01 reached the end of epoch 1, then exposed a trainer bug: `evaluate_loss()` referenced undefined `args`. |
| 2026-06-29 | smoke launch | cancelled | `RUN_SUFFIX=r02_wav2clip_control USE_WAV2CLIP_SEMANTIC=1 EPOCHS=1 FULL_EVAL_INTERVAL=0 SAVE_INTERVAL=1 scripts/slurm_train_g1_latent_diffusion.sh`; Slurm job `5417020`; log `slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r02_wav2clip_control/train_5417020.out` | Cancelled at user request to proceed directly with 1500-epoch parallel training. |
| 2026-06-29 | trainer fix | passed | `.venv311/bin/python -m py_compile train_g1_latent_diffusion.py` | Fixed `evaluate_loss()` to receive explicit loss config instead of referencing out-of-scope `args`. |
| 2026-06-29 | main launch | running | `RUN_SUFFIX=r01_control_only USE_WAV2CLIP_SEMANTIC=0 EPOCHS=1500 FULL_EVAL_INTERVAL=500 SAVE_INTERVAL=100 TIME_LIMIT=12:00:00 scripts/slurm_train_g1_latent_diffusion.sh`; Slurm job `5417035`; log `slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r01_control_only/train_5417035.out`; run dir `runs/train/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r01_control_only/` | `squeue` showed `RUNNING` on `nid011159`; log reached `epoch 1/1500`. |
| 2026-06-29 | main launch | running | `RUN_SUFFIX=r02_wav2clip_control USE_WAV2CLIP_SEMANTIC=1 EPOCHS=1500 FULL_EVAL_INTERVAL=500 SAVE_INTERVAL=100 TIME_LIMIT=12:00:00 scripts/slurm_train_g1_latent_diffusion.sh`; Slurm job `5417036`; log `slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r02_wav2clip_control/train_5417036.out`; run dir `runs/train/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r02_wav2clip_control/` | `squeue` showed `RUNNING` on `nid011171`; log reached model initialization. |
| 2026-06-29 | r01 main run | completed | Slurm job `5417035` completed (`COMPLETED`, exit `0:0`, elapsed `07:33:34`); checkpoint `runs/train/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r01_control_only/weights/train-1500.pt`; full eval `eval/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r01_control_only/ckpt1500_v6bc_variants/summary.json` | Training was stable, but ckpt1500 does not pass the condition-sensitivity gate. `real` has `G1BeatF1=0.1894`, `G1FKBAS=0.2353`, `G1FKRoboPerformBAS=0.4104`, while `random_control` is equal or better on all three. |
| 2026-06-29 | r02 main run | needs_eval | Slurm job `5417036` timed out (`TIMEOUT`, elapsed `12:00:10`) after saving `runs/train/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r02_wav2clip_control/weights/train-1500.pt`; complete summaries exist for ckpt500 and ckpt1000; ckpt1500 has only `real` and `shifted_control` metrics so far | Failure layer is scheduler time limit during full eval, not training. The final checkpoint exists and should be evaluated without resuming training. |
| 2026-06-30 | eval resume optimization | passed | `eval/run_g1_latent_diffusion_eval.py`, `train_g1_latent_diffusion.py`; `.venv311/bin/python -m py_compile eval/run_g1_latent_diffusion_eval.py train_g1_latent_diffusion.py`; helper smoke for `_load_existing_variant_metrics` | Added `--resume_existing_metrics` so reruns reuse completed `<output_dir>/<variant>/metrics.json` files after checking the checkpoint path. Future training-triggered full eval commands include this flag. |
| 2026-06-30 | r02 ckpt1500 eval repair | cancelled/replaced | Slurm job `5430913` was cancelled after completing `real`, `shifted_control`, `random_control`, and `zero_control`; pending resume job `5431955` was cancelled before start and replaced by metric-parallel job `5431982`; output target `eval/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r02_wav2clip_control/ckpt1500_v6bc_variants/summary.json` | The resumed eval should skip completed variants and only compute the remaining ckpt1500 variants. |
| 2026-06-30 | eval metric parallelism | passed/completed | `eval/g1_metrics.py`, `eval/run_g1_latent_diffusion_eval.py`, `train_g1_latent_diffusion.py`; `.venv311/bin/python -m py_compile eval/g1_metrics.py eval/run_g1_latent_diffusion_eval.py train_g1_latent_diffusion.py`; 2-clip serial-vs-2-worker FK metric smoke; Slurm job `5431982` completed in `00:11:56`; log `slurm/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion/r02_wav2clip_control/eval_ckpt1500_mw8_5431982.out`; summary `eval/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r02_wav2clip_control/ckpt1500_v6bc_variants/summary.json` | Added `--metric_workers` for per-motion beat/FK metric parallelism without changing metric definitions. Cancelled pending job `5431955` and resubmitted ckpt1500 resume with `--metric_workers 8`; completed variants were reused. |
| 2026-06-30 | final acceptance check | rejected | Compared V6b-B beat8d ckpt1000/1500, V6b-C r01 ckpt500/1000/1500, and V6b-C r02 ckpt500/1000/1500 summaries | Best real-condition V6b-C point is r02 ckpt1000, but corrupted conditions still win rhythm metrics; ckpt1500 r02 regresses and zero/shifted variants win through severe motion collapse. Do not resume either V6b-C route as the next mainline. |
