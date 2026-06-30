# EXP-20260623-finedance-g1-v6b-motion-prior

## Question

Can a standalone robot-native G1 motion prior learn a compact, reconstructive latent manifold that preserves FineDance-G1 motion quality, support/contact behavior, and amplitude before any music conditioning is added?

## Status

`finished`

V6a is rejected as a raw-control diffusion ablation: it increased activity but still produced visibly unreasonable motion, floating feet, and poor support behavior even with oracle controls. V6b starts with Stage A only: train and evaluate a deterministic continuous G1 motion autoencoder on GT G1 clips. Do not integrate this into the music-conditioned EDGE diffusion path until reconstruction quality passes the gates below.

r01 was user-stopped on 2026-06-23 during repository cleanup and migration preparation before checkpoint 100. It is not an accepted training result and there is no checkpoint to resume; relaunch the same experiment from scratch after the new SSH server is validated.

r02 completed 500 epochs and full reconstruction eval on Isambard/GH200 on
2026-06-25. Core reconstruction and aggregate support-contact gates pass, 8
stick reconstruction/target render pairs were generated, and the user visually
confirmed on 2026-06-26 that recon GIFs look good. V6b-A is accepted as the
motion-prior stage for the next music-to-latent experiment.

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

Isambard migration/setup validation completed on 2026-06-24:

```text
repo: /lus/lfs1aip2/projects/u6og/yukunwang.u6og/Musics2Dance
branch: codex/wav2clip-stage-20260526
commit at launch: cee3457
python: .venv311/bin/python
torch: 2.12.1+cu126
torchaudio: 2.11.0+cu126
gpu smoke: Slurm job 5366093, device=cuda, NVIDIA GH200 120GB
unit tests: Slurm job 5366092, tests.test_g1_motion_prior, completed 0:0
data validation: setup_logs/validate_finedance_g1_fkbeats_isambard_20260624.log
symlink migration fix: old /home/tianhup links under data/finedance_g1_fkbeats reduced to 0
```

Validation evidence:

```text
Preprocessed data validation passed.
train: count=47817 sampled=64 feature_dir=wav2clip_stft_beat_feats
test: count=3265 sampled=64 feature_dir=wav2clip_stft_beat_feats
smoke run: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_isambard_smoke_20260624/weights/train-1.pt
```

Formal Isambard r01 launch attempt on 2026-06-24:

```bash
PARTITION=workq \
TIME_LIMIT=24:00:00 \
CPUS_PER_TASK=8 \
MEMORY=64G \
GPUS=1 \
WANDB_MODE=online \
scripts/slurm_train_g1_motion_prior.sh
```

This submitted Slurm job `5366101`, but a pre-start W&B check found no API key
configured on Isambard:

```text
wandb_check_failed UsageError No API key configured.
```

Job `5366101` was still `PENDING (Priority)` and was cancelled before it
started. The run was relaunched with W&B offline so training can actually start
while preserving local W&B run files for later sync:

```bash
PARTITION=workq \
TIME_LIMIT=24:00:00 \
CPUS_PER_TASK=8 \
MEMORY=64G \
GPUS=1 \
WANDB_MODE=offline \
scripts/slurm_train_g1_motion_prior.sh
```

Launch record:

```text
Cancelled pre-start job: 5366101
Active Slurm job: 5366104
initial Slurm state: PENDING (Priority)
sbatch: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r01_ae_s2_latent128/train_r01_ae_s2_latent128.sbatch
Slurm output: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r01_ae_s2_latent128/train_5366104.out
tee log: setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior/train_r01_ae_s2_latent128.log
run dir: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128/
checkpoint schedule: train-100.pt first sanity checkpoint; train-500.pt first full acceptance gate
```

Queue diagnosis on 2026-06-24:

```text
06:22 UTC: job 5366104 still PENDING, Reason=Priority, StartTime=Unknown, no Slurm output file yet.
No dependency, no node constraint, no script-side failure evidence.
Requested resources: workq, 1 node, 1 GPU, 8 CPU, 64G.
TimeLimit was reduced in place from 24:00:00 to 08:00:00 with scontrol to improve backfill chances.
06:24 UTC: job remained PENDING, Reason=Priority, TimeLimit=08:00:00.
07:46 UTC: deeper Slurm probe showed user-visible pending queue still only had job 5366104, but many nodes are hidden as allocated/planned/reserved by scheduler policy. Job priority is 1 with all `sprio` components 0 under account `brics.u6og` and QOS `normal`. `squeue --start` estimated `2026-06-24T23:00:01` on `nid011315`. Attempting `--qos=workq_qos` failed with `Invalid qos specification`; attempting to force a visible idle node failed with `Requested node configuration is not available`. Conclusion: pending state is scheduler/account priority, not a repo or sbatch script failure.
```

Job `5366104` eventually started on 2026-06-24 at `15:56:24Z` on `nid011042`
but failed after 10 seconds:

```text
Slurm state: FAILED, ExitCode=1:0
Slurm output: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r01_ae_s2_latent128/train_5366104.out
tee log: setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior/train_r01_ae_s2_latent128.log
failure: wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
checkpoint result: no run weights directory was created
```

Root cause: `scripts/slurm_train_g1_motion_prior.sh` exported
`WANDB_MODE=offline`, but the training script reads `--wandb_mode` from argparse
and defaulted to `online`. The launcher now passes `--wandb_mode "$WANDB_MODE"`
explicitly.

Relaunch on 2026-06-25:

```bash
PARTITION=workq \
TIME_LIMIT=08:00:00 \
CPUS_PER_TASK=8 \
MEMORY=64G \
GPUS=1 \
WANDB_MODE=offline \
scripts/slurm_train_g1_motion_prior.sh
```

Current launch record:

```text
Failed job: 5366104
Active Slurm job: 5375000
submitted: 2026-06-25T03:22:47Z
started: 2026-06-25T03:24:41Z on nid010512
queue wait: 1m54s
initial Slurm state: PENDING (Priority), then RUNNING
sbatch: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r01_ae_s2_latent128/train_r01_ae_s2_latent128.sbatch
Slurm output: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r01_ae_s2_latent128/train_5375000.out
tee log: setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior/train_r01_ae_s2_latent128.log
run dir: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128/
local W&B run: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128/wandb/offline-run-20260625_032449-2ymgkgby
startup evidence: train=47817 test=3265 device=cuda start_epoch=0, epoch 1/500 active
time limit: 8h; attempted extension to 12h was denied by Slurm permissions
checkpoint schedule: train-100.pt first sanity checkpoint; train-500.pt first full acceptance gate
```

Performance diagnosis on 2026-06-25 showed r01 was still using conservative
4090-era settings on a GH200:

```text
r01 job: 5375000
node: nid010512
GPU: NVIDIA GH200 120GB
settings: batch_size=256, num_workers=0, mixed_precision=fp16
observed memory: 2675 / 97871 MiB
observed GPU util: about 37%
epoch time: about 1m03s-1m05s
logged ETA: 8h47m-8h59m, exceeding the 8h Slurm time limit before full eval
checkpoint result: no weights/*.pt files yet
action: cancelled r01 after 6m17s before checkpoint 100
```

The Slurm launcher now exposes `NUM_WORKERS`, `CACHE_BATCH_SIZE`,
`MIXED_PRECISION`, and `LEARNING_RATE` so Isambard/GH200 runs do not inherit the
small local-4090 defaults by accident.

GH200 r02 launch:

```bash
RUN_SUFFIX=r02_gh200_b1024_w8_bf16 \
PARTITION=workq \
TIME_LIMIT=08:00:00 \
CPUS_PER_TASK=16 \
MEMORY=96G \
GPUS=1 \
WANDB_MODE=offline \
BATCH_SIZE=1024 \
NUM_WORKERS=8 \
CACHE_BATCH_SIZE=1024 \
MIXED_PRECISION=bf16 \
scripts/slurm_train_g1_motion_prior.sh
```

Current r02 launch record:

```text
Cancelled conservative job: 5375000
Active Slurm job: 5375012
initial Slurm state: PENDING (Priority)
sbatch: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r02_gh200_b1024_w8_bf16/train_r02_gh200_b1024_w8_bf16.sbatch
Slurm output: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r02_gh200_b1024_w8_bf16/train_5375012.out
tee log: setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior/train_r02_gh200_b1024_w8_bf16.log
run dir: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/
live log command: tail -f slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r02_gh200_b1024_w8_bf16/train_5375012.out
checkpoint schedule: train-100.pt first sanity checkpoint; train-500.pt first full acceptance gate
```

r02 completion on 2026-06-25:

```text
Slurm job: 5375012
state: COMPLETED, ExitCode=0:0
elapsed: 2h42m34s
node: nid010842
run dir: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/
checkpoint: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/weights/train-500.pt
eval dir: eval/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/ckpt0500_reconstruction/
offline W&B: runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/wandb/offline-run-20260625_033226-ct0yxcbk
```

Checkpoint schedule artifacts:

```text
train-100.pt
train-200.pt
train-300.pt
train-400.pt
train-500.pt
```

Core full-eval evidence:

```text
reconstruction_metrics.json:
  loss/fk_mpjpe: 0.01848082480007985
  contact_f1: 0.9018347542099712
  contact_precision: 0.9166743086271593
  contact_recall: 0.8880828767786713
  loss/total: 0.04495035447807809

metrics.json:
  FiniteMotionRate: 1.0
  BadFileCount: 0
  G1Dist: 0.20292945206165314
  G1Div: 15.449141553110469
  JointPositionRangeMean: 1.2003774638397542
  RootUpZP01: 1.0
  RootTiltGt60DegRate: 0.0
  RootInvertedRate: 0.0

gt_baseline/metrics.json:
  G1Div: 15.488929905236299
  JointPositionRangeMean: 1.196966360078962
  RootUpZP01: 1.0
  RootTiltGt60DegRate: 0.0
```

Gate notes:

```text
Passed:
  loss/fk_mpjpe <= 0.05
  contact_f1 >= 0.85
  FiniteMotionRate = 1.0 and BadFileCount = 0
  JointPositionRangeMean is slightly above GT
  root stays upright under reported root metrics

Still pending:
  Aggregate G1NoNearSupportRate, G1FootHighLiftRate, G1GroundPenetration,
  and G1FootSliding were not written to this eval's metrics.json or
  motion_audit.json, despite failure_panel.json containing the corresponding
  ranked diagnostic groups. Run or patch the support-contact scorer before
  final acceptance.
  Review diagnostic PNGs and any render outputs before starting music-to-latent.
```

Next action: run/inspect the missing aggregate support-contact gates and review
diagnostics. If those pass, accept V6b-A and start V6b-B music-to-latent.

Support-contact eval fix and rerun on 2026-06-25:

```text
code fix: eval/g1_metrics.py now computes FK foot/support diagnostics even when
  a reconstruction motion has no audio_path; audio beat metrics remain zero in
  that case instead of skipping FK metrics entirely.
test: .venv311/bin/python -m unittest \
  tests.test_g1_eval_metrics.G1MetricTests.test_fk_metrics_are_optional_and_reported_when_enabled \
  tests.test_g1_eval_metrics.G1MetricTests.test_fk_support_metrics_are_reported_without_audio_path
test result: passed
cancelled GPU eval job: 5376762, still PENDING (Priority), replaced by CPU-only rerun
support-fix CPU eval Slurm job: 5376862
state: COMPLETED, ExitCode=0:0
elapsed: 6m33s
node: nid010309
live log command: tail -f slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r02_gh200_b1024_w8_bf16/eval_supportfix_cpu_5376862.out
Slurm output: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r02_gh200_b1024_w8_bf16/eval_supportfix_cpu_5376862.out
tee log: setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior/eval_ckpt0500_supportfix_cpu.log
target eval dir: eval/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/ckpt0500_reconstruction/
```

Support-contact gate results after the rerun:

```text
reconstruction_metrics.json:
  loss/fk_mpjpe: 0.01847166114958833
  contact_f1: 0.9018776665785411
  contact_precision: 0.9167504979311781
  contact_recall: 0.8880920994336535
  loss/total: 0.04494969418092672

metrics.json vs gt_baseline/metrics.json:
  G1NoNearSupportRate: 0.08073506891271057 vs 0.0818601327207759, PASS <= GT+0.03
  G1FootHighLiftRate: 0.05123226135783562 vs 0.051346605410923944, PASS <= GT+0.03
  G1GroundPenetration: 0.14956742525100708 vs 0.1570272445678711, PASS <= GT+0.03
  G1FootSliding: 0.550388793186905 vs 0.5339775725134119, PASS <= GT*1.25+0.05
  JointPositionRangeMean: 1.200363968220321 vs 1.196966360078962, PASS >= 0.90*GT
  G1Div: 15.448837210716418 vs 15.488929905236299
  G1Dist: 0.2028449922800064
  RootUpZP01: 1.0 vs 1.0
  RootTiltGt60DegRate: 0.0 vs 0.0
  RootInvertedRate: 0.0 vs 0.0
  num_fk_scored_files: 3265 vs 3265
```

Render-pair completion on 2026-06-25:

```text
render Slurm job: 5377316
state: COMPLETED, ExitCode=0:0
elapsed: 2m38s
node: nid010309
live log command: tail -f slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r02_gh200_b1024_w8_bf16/render_pairs_stick_5377316.out
Slurm output: slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r02_gh200_b1024_w8_bf16/render_pairs_stick_5377316.out
tee log: setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior/render_pairs_stick8.log
manifest: eval/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/ckpt0500_reconstruction/render_manifest.json
render pairs dir: eval/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/ckpt0500_reconstruction/render_pairs/
rendered pairs: 8 target/reconstruction pairs for stems 012_slice0, 012_slice1, 012_slice10, 012_slice100, 012_slice101, 012_slice102, 012_slice103, 012_slice104
backend: stick
layout: separate GIFs; comparison_video is empty because ffmpeg is not on PATH in this environment
sanity check: 16 GIF files are present, all 480x640 and 2.8MB-3.36MB; sampled target/recon first frames are nonblank and upright
```

Current conclusion: all numeric V6b-A reconstruction, contact, support,
uprightness, range, and diversity gates pass for checkpoint 500. The eval dir
also contains diagnostic PNGs, failure-panel PNGs, and 8 reconstruction/target
stick render pairs. Basic render sanity passed, and the user confirmed the recon
GIFs look good on 2026-06-26. V6b-A is finished; next step is V6b-B
music-to-latent.
