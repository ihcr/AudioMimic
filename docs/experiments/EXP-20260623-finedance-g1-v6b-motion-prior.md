# EXP-20260623-finedance-g1-v6b-motion-prior

## Question

Can a standalone robot-native G1 motion prior learn a compact, reconstructive latent manifold that preserves FineDance-G1 motion quality, support/contact behavior, and amplitude before any music conditioning is added?

## Status

`ready`

V6a is rejected as a raw-control diffusion ablation: it increased activity but still produced visibly unreasonable motion, floating feet, and poor support behavior even with oracle controls. V6b starts with Stage A only: train and evaluate a deterministic continuous G1 motion autoencoder on GT G1 clips. Do not integrate this into the music-conditioned EDGE diffusion path until reconstruction quality passes the gates below.

r01 was user-stopped on 2026-06-23 during repository cleanup and migration preparation before checkpoint 100. It is not an accepted training result and there is no checkpoint to resume; relaunch the same experiment from scratch after the new SSH server is validated.

## Hypothesis

The current raw diffusion pipeline is being asked to learn both music alignment and the feasible G1 motion manifold at once. A motion prior should first learn a lower-dimensional robot-native latent space where decoded samples stay upright, preserve foot support, and avoid the V6a/V5 jitter and hovering-foot failures. If this first stage fails on reconstruction, music-to-latent generation should not be started.

## Baselines And Controls

- Negative control: `EXP-20260622-finedance-g1-v6a-body-support-raw` checkpoint 500, rejected for floating/unreasonable motion.
- Motion representation invariant: `g1_yaw_delta`, 150 frames, 34 channels.
- Data invariant: GT only from `data/finedance_g1_fkbeats/{train,test}/motions_sliced/*.pkl`.
- No music features, Wav2CLIP, GaussianBeat, STFT, Librosa35, Jukebox, or beat losses in V6b-A.
- No generated V3/V5/V6a outputs in the prior training set.

## Implementation Contract

New cache:

```text
data/finedance_g1_v6b_motion_prior_dataset_backups/
```

Each split cache stores normalized `g1_yaw_delta` motion plus contact/support labels derived from decoded yaw-delta FK feet:

```text
motion                    float32 [N,150,34]
contact                   float32 [N,150,2]
near_support              float32 [N,150,2]
lowest_foot_heights       float32 [N,150,2]
ground                    float32 [N]
source_paths              list[str]
stems                     list[str]
```

Normalizer is train-only mean/std over encoded motion channels and is saved with checkpoints. Cache metadata records motion format, frame count, repr dim, contact thresholds, FK model path, root quaternion order, and cache version. Any change to representation, contact semantics, or normalizer requires rebuilding this cache.

Model:

- `G1MotionAutoencoder`, independent of `EDGE.py`.
- Conv1D encoder, temporal downsample 2, latent `[B,75,128]`.
- Conv/upsample decoder to `[B,150,34]`.
- Contact head predicts `[B,150,2]` logits.
- `prior_type=ae` is the r01 default. `prior_type=vae` is available for the next ablation with KL warmup, but is not the main first result.

Loss:

```text
motion MSE             1.0
velocity MSE           0.5
acceleration MSE       0.1
FK keypoint MSE        0.5
contact BCE            0.1
contact height         0.2
contact sliding        0.1
KL                     0.0 for r01 AE
```

## Commands

Smoke:

```bash
PYTHONUNBUFFERED=1 .venv311/bin/python -m train_g1_motion_prior \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \
  --exp_name EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128_smoke \
  --motion_format g1_yaw_delta \
  --prior_type ae \
  --latent_dim 128 \
  --epochs 1 \
  --cache_limit_per_split 64 \
  --data_len 64 \
  --eval_data_len 16 \
  --eval_max_clips 16 \
  --full_eval_interval 0 \
  --wandb_mode disabled
```

Main r01 on Isambard:

```bash
PARTITION=workq \
TIME_LIMIT=24:00:00 \
CPUS_PER_TASK=8 \
MEMORY=64G \
GPUS=1 \
WANDB_MODE=online \
scripts/slurm_train_g1_motion_prior.sh
```

The launcher writes the generated `.sbatch` file under:

```text
slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r01_ae_s2_latent128/
```

Local 4090 fallback, only for non-Slurm machines:

```bash
tmux new -s m2d_v6b_prior_r01
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
PYTHONUNBUFFERED=1 .venv311/bin/python -m train_g1_motion_prior \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \
  --exp_name EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128 \
  --motion_format g1_yaw_delta \
  --prior_type ae \
  --latent_dim 128 \
  --temporal_downsample 2 \
  --batch_size 256 \
  --epochs 500 \
  --learning_rate 2e-4 \
  --weight_decay 0.02 \
  --save_interval 100 \
  --eval_interval 50 \
  --full_eval_interval 500 \
  --wandb_pj_name Musics2Dance \
  2>&1 | tee -a setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior_train_r01.log
```

Manual eval:

```bash
PYTHONUNBUFFERED=1 .venv311/bin/python -m eval.run_g1_motion_prior_eval \
  --checkpoint runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128/weights/train-500.pt \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \
  --output_dir eval/EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128/ckpt0500_reconstruction \
  --motion_format g1_yaw_delta \
  --enable_fk_metrics \
  --diagnostic_count 8 \
  --render_count 8
```

## Evaluation And Acceptance

Full reconstruction eval writes:

```text
metrics.json
gt_baseline/metrics.json
reconstruction_metrics.json
motion_audit.json
failure_panel.json
paper_report.md
render_manifest.json
```

100% reconstruction quality is not required, but checkpoint 500 should satisfy:

- `loss/fk_mpjpe <= 0.05`.
- Contact F1 >= `0.85`.
- `G1NoNearSupportRate`, `G1FootHighLiftRate`, and `G1GroundPenetration` <= GT yaw-delta baseline + `0.03`.
- `G1FootSliding` <= GT yaw-delta baseline * `1.25 + 0.05`.
- `JointPositionRangeMean >= 0.90 * GT`.
- Fixed render pairs show no floating body, no lying pose, no repeated stitch artifacts, and no hand jitter visibly worse than the target.

If r01 passes, continue with V6b-B music-to-latent generation. If r01 fails, inspect whether failure comes from representation capacity, contact loss balance, or AE bottleneck before adding any music condition.

## Current Evidence

Implementation is complete on branch `codex/wav2clip-stage-20260526`.

Validated:

```bash
.venv311/bin/python -m unittest tests.test_g1_motion_prior
PYTHONUNBUFFERED=1 .venv311/bin/python -m train_g1_motion_prior \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \
  --exp_name EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128_smoke_ampcheck \
  --motion_format g1_yaw_delta \
  --prior_type ae \
  --latent_dim 128 \
  --epochs 1 \
  --cache_limit_per_split 64 \
  --data_len 64 \
  --eval_data_len 16 \
  --eval_max_clips 16 \
  --full_eval_interval 0 \
  --wandb_mode disabled
```

Smoke passed with `loss=1.368964`, `val=2.135168`; checkpoint and eval CLI were also verified using the smoke checkpoint.

r01 launched locally in tmux and was then stopped before the first checkpoint:

```text
tmux: m2d_v6b_prior_r01
log: setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior_train_r01.log
run dir: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128/
W&B: https://wandb.ai/realroboticslab_tianhu/Musics2Dance/runs/08zxd1ss
```

Launch evidence:

```text
cache encoded train/test: 47817 / 3265 clips
contact cache: 187 train batches at about 34 batch/s, 13 test batches at about 35 batch/s
training: epoch 1 started, about 7.5 train batches/s after warmup
GPU memory: about 2.3GB on RTX 4090
```

Stop evidence:

```text
tmux m2d_v6b_prior_r01: absent
train_g1_motion_prior process: absent
GPU compute processes: only /usr/bin/anydesk
last log progress: epoch 27/500, batch 111/187
last completed epoch: epoch 26/500, loss=0.013763, time=25s, eta=3h18m45s
saved files: config.json and W&B run file only; no weights/*.pt checkpoint
processed cache: data/finedance_g1_v6b_motion_prior_dataset_backups, about 1.2GB locally
```

Next action: migrate the repo and required data/cache artifacts to Isambard, validate the environment through Slurm, then relaunch r01 from scratch with the Isambard command above. Checkpoint 100 is the first reconstruction sanity gate; checkpoint 500 should automatically run full reconstruction eval.
