# EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion

## Question

Can a beat-only music condition generate usable G1 dance by denoising inside the
accepted V6b-A motion-prior latent space, instead of denoising raw G1 motion?

## Status

`rejected`

V6b-A is accepted and frozen. V6b-B starts with the cleanest music-to-latent
ablation: use only inference-available `beat_features_8d`, do not add Wav2CLIP,
motion-derived controls, oracle beatness, or InfoNCE. This isolates whether the
new G1 latent prior fixes the robot-quality failures seen in raw 8D-beat runs.

Current scheduler state:

- 1-epoch smoke job `5386482` completed successfully on 2026-06-26
  (`COMPLETED`, exit `0:0`, elapsed `00:00:10`).
- Initial 8h main r01 job `5386787` was cancelled before start because it stayed
  `PENDING (Priority)` without a start estimate.
- r01 chunk job `5386890` completed successfully on 2026-06-26 (`COMPLETED`,
  exit `0:0`, elapsed `00:05:50`), built the full latent cache, and saved
  checkpoints through epoch 100.
- r01 resume job `5386925` completed successfully on 2026-06-26 (`COMPLETED`,
  exit `0:0`, elapsed `00:41:07`), resumed from `train-100.pt`, trained to
  epoch 500, and ran the initial four-variant eval.
- Audio-metrics repair job `5387716` launched on 2026-06-26 after the first
  ckpt500 eval completed with missing `audio_path` in generated motion files,
  but its wrapper hid an import-path failure behind `tee`.
- Audio-metrics repair job `5387734` fixed the import path and used
  `set -o pipefail`. It patched generated/target pkl metadata in place and
  recomputed audio-backed metrics for `real_beat8d` and `shifted_beat8d`, then
  was cancelled manually at the user's direction after `00:14:52`.
- Audio-metrics repair job `5387853` completed successfully on 2026-06-26
  (`COMPLETED`, exit `0:0`, elapsed `00:14:49`) and repaired the remaining
  `random_beat8d` and `zero_beat8d` variants against
  `data/finedance_aistpp/test/wavs_sliced`.
- Undertraining-check continuation job `5415037` was submitted on 2026-06-29,
  started at `2026-06-29T02:36:22` on `nid010813`, and completed successfully
  at `2026-06-29T04:20:10` (`COMPLETED`, exit `0:0`, elapsed `01:43:48`).
  It resumed from `train-500.pt` with optimizer state, trained to
  `train-1500.pt`, and launched full four-variant eval at epochs 1000 and 1500.

Smoke Slurm log:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/smoke/smoke_5386482.out
```

Cancelled 8h main r01 Slurm log path:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/r01_beat8d_only/train_5386787.out
```

Completed r01 chunk100 Slurm log:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/r01_beat8d_only_chunk100/train_5386890.out
```

Completed r01 resume100-to-500 Slurm log:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/r01_beat8d_only_resume100_to500/train_5386925.out
```

Failed audio-metrics repair Slurm log:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/audio_metrics_repair/repair_5387716.out
```

Cancelled partial audio-metrics repair Slurm log:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/audio_metrics_repair/repair2_5387734.out
```

Completed remaining-variants audio-metrics repair Slurm log:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/audio_metrics_repair/repair3_5387853.out
```

Completed 500-to-1500 continuation Slurm log:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/r01_beat8d_only/train_5415037.out
```

## Hypothesis

The old 8D beat-conditioned raw diffusion path proved that beat structure is
useful, but it also failed acceptance through distribution/contact/endpoint
quality. If V6b-A learned a useful G1-feasible manifold, then a beat8d-only
latent diffusion model should preserve support/contact and uprightness better
than raw 8D beat diffusion while still showing real condition sensitivity.

## Baselines And Controls

- Accepted frozen prior: `EXP-20260623-finedance-g1-v6b-motion-prior`, r02
  checkpoint `runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/weights/train-500.pt`.
- Negative raw-motion reference: `EXP-20260617-finedance-g1-beat8d-motion-beatness`.
  It improved rhythm but was rejected because `G1Dist`, ground penetration, and
  endpoint jerk/contact quality failed.
- Condition input: `data/finedance_g1_fkbeats/{train,test}/beat_features_8d_feats/*.npy`.
- No Wav2CLIP, STFT, Jukebox, InfoNCE, motion intensity, motion beatness,
  support beatness, contact oracle, or generated V3/V5/V6a outputs.
- Decoder is frozen V6b-A; no joint training until this ablation is understood.

## Implementation Contract

New cache:

```text
data/finedance_g1_v6b_beat8d_latent_dataset_backups/
```

Cache source:

```text
normalized g1_yaw_delta motion -> frozen V6b-A encoder -> latent [N,75,128]
beat_features_8d -> condition [N,150,8]
```

The cache stores train-normalized latents, beat features, source paths, stems,
latent normalizer, and metadata with the V6b-A checkpoint fingerprint. Rebuild
the cache if the V6b-A checkpoint, latent semantics, beat8d extractor, split,
or cache limit changes.

Model:

- `G1Beat8DLatentDenoiser`: transformer latent denoiser.
- Input latent shape: `[B,75,128]`.
- Condition shape: `[B,150,8]`.
- Beat condition is encoded separately and cross-attended from latent tokens.
- Classifier-free condition dropout is enabled for future guidance sweeps.
- `G1LatentDiffusion` trains epsilon prediction over normalized V6b-A latents.

## Commands

Smoke:

```bash
PYTHONUNBUFFERED=1 .venv311/bin/python -m train_g1_latent_diffusion \
  --data_path data/finedance_g1_fkbeats \
  --motion_prior_processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \
  --latent_processed_data_dir data/finedance_g1_v6b_beat8d_latent_dataset_backups_smoke \
  --prior_checkpoint runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/weights/train-500.pt \
  --exp_name EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_smoke \
  --cache_limit_per_split 64 \
  --data_len 64 \
  --eval_data_len 16 \
  --epochs 1 \
  --batch_size 16 \
  --cache_batch_size 32 \
  --num_workers 0 \
  --hidden_dim 128 \
  --num_layers 1 \
  --ff_size 256 \
  --diffusion_steps 100 \
  --eval_interval 1 \
  --full_eval_interval 0 \
  --wandb_mode disabled
```

Main Isambard launch:

```bash
RUN_SUFFIX=r01_beat8d_only \
PARTITION=workq \
TIME_LIMIT=08:00:00 \
CPUS_PER_TASK=16 \
MEMORY=160G \
GPUS=1 \
WANDB_MODE=offline \
BATCH_SIZE=1024 \
NUM_WORKERS=12 \
CACHE_BATCH_SIZE=2048 \
MIXED_PRECISION=bf16 \
scripts/slurm_train_g1_latent_diffusion.sh
```

Resume 500-to-1500 undertraining check:

```bash
RUN_SUFFIX=r01_beat8d_only \
PARTITION=workq \
TIME_LIMIT=04:00:00 \
CPUS_PER_TASK=16 \
MEMORY=160G \
GPUS=1 \
WANDB_MODE=offline \
BATCH_SIZE=1024 \
NUM_WORKERS=12 \
CACHE_BATCH_SIZE=2048 \
MIXED_PRECISION=bf16 \
CHECKPOINT=runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/weights/train-500.pt \
RESUME_OPTIMIZER=1 \
EPOCHS=1500 \
SAVE_INTERVAL=100 \
EVAL_INTERVAL=25 \
FULL_EVAL_INTERVAL=500 \
scripts/slurm_train_g1_latent_diffusion.sh
```

The initial 8h job stayed pending with `Reason=Priority`, so r01 was switched
to a shorter backfill-friendly chunk:

```bash
sbatch --job-name=m2d_v6b_lat_c100 --partition=workq --time=02:00:00 \
  --cpus-per-task=16 --mem=160G --gres=gpu:1 \
  --output=slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/r01_beat8d_only_chunk100/train_%j.out \
  --wrap='cd /lus/lfs1aip2/projects/u6og/yukunwang.u6og/Musics2Dance && ... train_g1_latent_diffusion --epochs 100 --save_interval 25 --full_eval_interval 0'
```

The launcher writes:

```text
slurm/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/r01_beat8d_only/
setup_logs/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion/train_r01_beat8d_only.log
runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/
```

## Evaluation And Acceptance

Every full eval decodes generated latent samples through the frozen V6b-A
decoder and runs four variants:

```text
real_beat8d
shifted_beat8d
random_beat8d
zero_beat8d
```

Acceptance requires:

- `real_beat8d` beats shifted/random/zero on `G1FKBAS`, `G1BeatF1`, and recall.
- `real_beat8d` does not win by metric hacking: check body-part response,
  `G1UnmatchedMotionBeatRate`, `G1FootContactOnBeatRate`, and failure panels.
- Robot quality is not worse than old raw beat8d winners:
  `G1GroundPenetration`, `G1NoNearSupportRate`, `G1FootHighLiftRate`,
  `G1FootSliding`, wrist/foot jerk, root drift, and root-up metrics must stay
  within the V6b-A/GT-informed gates.
- `G1Div`, `G1Dist`, and `JointPositionRangeMean` do not collapse into a
  near-static prior sample.
- Fixed renders show no floating body, lying pose, repeated stitch artifacts, or
  hand/foot jitter visibly worse than the target.

Do not accept a checkpoint on beat scores alone.

## Current Evidence

Implementation added:

```text
dataset/g1_latent_beat_dataset.py
model/g1_latent_diffusion.py
train_g1_latent_diffusion.py
eval/run_g1_latent_diffusion_eval.py
scripts/slurm_train_g1_latent_diffusion.sh
tests/test_g1_latent_diffusion.py
```

Focused tests passed locally. 1-epoch smoke job `5386482` completed and wrote
`runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_smoke/weights/train-1.pt`.
Initial 8h main r01 job `5386787` was cancelled before start due to priority
wait. r01 chunk job `5386890` used H200/GH200-scaled settings: batch size
`1024`, cache batch size `2048`, 12 workers, `bf16`, max 100 epochs, checkpoint
every 25 epochs, and no full eval in the chunk. It completed in `00:05:50` with
epoch-100 `loss=0.094927`, `val=0.091748`, and checkpoint
`runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/weights/train-100.pt`.
Resume job `5386925` completed in `00:41:07` with checkpoint
`runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/weights/train-500.pt`
and initial eval summary
`eval/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/ckpt0500_beat8d_variants/summary.json`.
The initial eval had `num_audio_beats=0` because generated pkl files lacked
`audio_path`; it is not valid for rhythm acceptance. Repair job `5387716`
exposed an import-path issue in the standalone repair script. Repair job
`5387734` recomputed audio-backed metrics for `real_beat8d` and
`shifted_beat8d`, then was cancelled after `00:14:52`. Early repaired evidence
does not show strong condition sensitivity: `real_beat8d` has
`G1BeatF1=0.1999`, `G1FKBAS=0.2493`, and `G1FKRoboPerformBAS=0.4134`;
`shifted_beat8d` has `G1BeatF1=0.2017`, `G1FKBAS=0.2488`, and
`G1FKRoboPerformBAS=0.4136`. Job `5387853` was submitted to complete only the
remaining `random_beat8d` and `zero_beat8d` repairs.

Job `5387853` completed in `00:14:49`, making all four variants audio-backed
with `num_fk_audio_beats=27286`.

Repaired ckpt500 summary:

| variant | G1FKBAS ↑ | G1FKRobo ↑ | BeatF1 ↑ | Recall ↑ | Dist ↓ | Div ↑ | NoSupport ↓ | HighLift ↓ | Ground ↓ | WristJerk ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_beat8d | 0.2493 | 0.4134 | 0.1999 | 0.1666 | 8.2038 | 32.6942 | 0.3722 | 0.2127 | 0.1461 | 2315.5 |
| shifted_beat8d | 0.2488 | 0.4136 | 0.2017 | 0.1682 | 8.6677 | 32.6416 | 0.4082 | 0.2214 | 0.1721 | 2351.6 |
| random_beat8d | 0.2463 | 0.4067 | 0.1974 | 0.1639 | 8.2878 | 32.6095 | 0.3725 | 0.2101 | 0.1883 | 2326.3 |
| zero_beat8d | 0.2673 | 0.4136 | 0.2094 | 0.1774 | 61.9175 | 25.3159 | 0.8759 | 0.7587 | 0.3349 | 11718.4 |

Interim verdict at ckpt500: reject V6b-B r01 as a mainline model at this
checkpoint. The latent prior keeps the real/shifted/random samples diverse, but
the real condition does not cleanly beat shifted/random on BeatF1 or
RoboPerform BAS. The zero-condition run gets the highest beat metrics only
while collapsing into severe robot-quality failure.

Because training loss was still decreasing at epoch 500, job `5415037`
continued the same run to epoch 1500 as an undertraining check. The decision
criterion was not whether denoising loss kept dropping; it was whether
ckpt1000/ckpt1500 made `real_beat8d` clearly separate from `shifted_beat8d` and
`random_beat8d` on BeatF1, `G1FKBAS`, and `G1FKRoboPerformBAS` without letting
`zero_beat8d` win through robot-quality collapse.

Final undertraining-check result: reject V6b-B r01 as a mainline model. Training
longer improved support/contact, jerk, distribution distance, and root speed,
but it did not make the model use the real beat condition. By epoch 1500,
`real_beat8d`, `shifted_beat8d`, and `random_beat8d` are still nearly tied on
the rhythm metrics, while `zero_beat8d` gets the highest beat scores through a
clearly invalid low-diversity/poor-quality mode.

Final artifacts:

```text
runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/weights/train-1500.pt
eval/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/ckpt1000_beat8d_variants/summary.json
eval/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/ckpt1500_beat8d_variants/summary.json
```

Key comparison:

| checkpoint / variant | G1FKBAS ↑ | G1FKRobo ↑ | BeatF1 ↑ | Recall ↑ | Dist ↓ | Div ↑ | NoSupport ↓ | HighLift ↓ | Ground ↓ | FootSlide ↓ | WristJerk ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ckpt500 real | 0.2493 | 0.4134 | 0.1999 | 0.1666 | 8.2038 | 32.6942 | 0.3722 | 0.2127 | 0.1461 | 1.1564 | 2315.5 |
| ckpt500 shifted | 0.2488 | 0.4136 | 0.2017 | 0.1682 | 8.6677 | 32.6416 | 0.4082 | 0.2214 | 0.1721 | 1.1687 | 2351.6 |
| ckpt500 random | 0.2463 | 0.4067 | 0.1974 | 0.1639 | 8.2878 | 32.6095 | 0.3725 | 0.2101 | 0.1883 | 1.1458 | 2326.3 |
| ckpt500 zero | 0.2673 | 0.4136 | 0.2094 | 0.1774 | 61.9175 | 25.3159 | 0.8759 | 0.7587 | 0.3349 | 1.2818 | 11718.4 |
| ckpt1000 real | 0.2404 | 0.4114 | 0.1985 | 0.1628 | 5.2799 | 26.3567 | 0.1501 | 0.0676 | 0.1094 | 0.8177 | 1308.9 |
| ckpt1000 shifted | 0.2451 | 0.4150 | 0.2004 | 0.1657 | 5.5552 | 26.8464 | 0.1703 | 0.0815 | 0.1541 | 0.8098 | 1334.4 |
| ckpt1000 random | 0.2404 | 0.4102 | 0.1940 | 0.1590 | 5.3505 | 26.2734 | 0.1449 | 0.0658 | 0.1222 | 0.8274 | 1313.3 |
| ckpt1000 zero | 0.2780 | 0.4198 | 0.2135 | 0.1836 | 68.5360 | 13.6342 | 0.4611 | 0.2432 | 0.1258 | 0.6417 | 2767.1 |
| ckpt1500 real | 0.2403 | 0.4125 | 0.1959 | 0.1615 | 5.1801 | 22.2497 | 0.0938 | 0.0359 | 0.1172 | 0.7972 | 1139.1 |
| ckpt1500 shifted | 0.2386 | 0.4147 | 0.1966 | 0.1601 | 5.1709 | 22.7358 | 0.1155 | 0.0455 | 0.1291 | 0.7941 | 1169.6 |
| ckpt1500 random | 0.2401 | 0.4124 | 0.1960 | 0.1605 | 5.1548 | 22.4456 | 0.0966 | 0.0379 | 0.1228 | 0.7945 | 1143.4 |
| ckpt1500 zero | 0.2748 | 0.4168 | 0.2138 | 0.1827 | 69.9584 | 10.5916 | 0.5349 | 0.5088 | 0.0888 | 1.1367 | 2770.1 |

The most important trend is the split between quality and control use:

- Quality improves from ckpt500 to ckpt1500: `real_beat8d` drops from
  `NoSupport=0.3722` to `0.0938`, `HighLift=0.2127` to `0.0359`, and
  `WristJerk=2315.5` to `1139.1`.
- Rhythm/control does not improve: `real_beat8d` falls from
  `BeatF1=0.1999` at ckpt500 to `0.1959` at ckpt1500, and remains tied with
  shifted/random.
- `zero_beat8d` remains a metric-hacking failure mode: it has the highest
  `BeatF1` and `G1FKBAS`, but `Dist=69.9584`, `Div=10.5916`, and much worse
  no-support/high-lift/foot-slide behavior.

Next action: stop this exact beat8d-only mainline. Move to the documented V6b-C
direction: semantic music encoder plus GaussianBeat or beat-distance controls,
separate semantic/control encoders, and an explicit condition-sensitivity
objective such as beat-alignment or real-vs-shift/random ranking.
