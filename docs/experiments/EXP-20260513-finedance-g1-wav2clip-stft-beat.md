# EXP-20260513-finedance-g1-wav2clip-stft-beat

Status: blocked
Owner: yukun
Created: 2026-05-13
Last Updated: 2026-05-14

## Research Question

Can the G1 FineDance pipeline train directly on a lighter non-Jukebox music stack while keeping the current Transformer diffusion backbone fixed?

## Hypothesis

`Wav2CLIP + STFT + GaussianBeat` should be a stronger first replacement for FineDance's 35-D Librosa feature than using handcrafted features alone. The two fusion variants test whether raw stream-normalized concat is enough or whether a learned per-stream adapter is needed to stop the 512-D Wav2CLIP stream from dominating STFT and beat channels.

## Baseline Or Control

- Current FineDance+G1 Librosa35 preset: `g1_finedance_librosa35_lbeat_robotloss`.
- This run intentionally isolates the feature/backbone interface first: no Mamba, no FSQ, no new denoiser loss, and no lbeat estimator.

## Intervention

Train two FineDance+G1 runs from the same feature cache:

1. `r01_concat_norm`: `feature_type=wav2clip_stft_beat`, `feature_fusion=concat_norm`.
2. `r02_stream_adapter`: `feature_type=wav2clip_stft_beat`, `feature_fusion=stream_adapter`.

Both use the current Transformer `DanceDecoder`, G1 motion format, 5-second horizon, 30 FPS features, and the same FineDance/G1 train-test tree.

## Invariant Controls

- Branch: `wav2clip-stft-beat`; on a new server, clone this branch directly as
  the repo root rather than depending on an existing EDGE worktree.
- Dataset: `data/finedance_g1_fkbeats`.
- Migration source data: HF keeps compact raw `data/finedance/`, retargeted
  `data/finedance-g1-retargeted/`, and checkpoints only. On the 4090 server,
  rebuild `motions_sliced`, `wavs_sliced`, `baseline_feats`, `jukebox_feats`,
  `beat_feats`, and `wav2clip_stft_beat_feats` locally with
  `scripts/bootstrap_finedance_g1_4090.sh`.
- New feature cache: write `wav2clip_stft_beat_feats` in this worktree, not into the diffusion worktree.
- Train clips: `47817`; test clips: `3265`.
- Backbone: current Transformer diffusion, no Mamba or hybrid block.
- Beat/alignment signal: GaussianBeat is conditioning only; no extra beat supervision in this first run.
- Batch and optimizer: batch size `512`, gradient accumulation `1`, learning rate `2e-4`.
- Checkpoint cadence: 500 epochs, save every 50.

## Data And Cache Contract

- Feature shape: `(150, 706)` with stream dims `512 + 193 + 1`.
- Data path: `data/finedance_g1_fkbeats`.
- r01 processed/tensor cache: `data/finedance_g1_wav2clip_stft_beat_concat_norm_dataset_backups`.
- r02 processed/tensor cache: `data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups`.
- Cache invalidation needed: yes. Feature semantics and width differ from Librosa35 and Jukebox.

## Training Or Execution Plan

- Environment: `source .venv311/bin/activate` from the branch repo root.
- Feature preprocess script: `slurm/EXP-20260513-finedance-g1-wav2clip-stft-beat/preprocess_features.sbatch`.
- Superseded single-process preprocess job: `4576095` was cancelled after measuring roughly 2.5 clips/sec.
- Active preprocess array job: `4576163_[0-7]`.
- r01 pipeline preset: `g1_finedance_wav2clip_stft_beat_concat_norm`.
- r01 pipeline jobs: validate `4576167` completed, train `4576168` cancelled before running, eval `4576169` cancelled.
- r02 pipeline preset: `g1_finedance_wav2clip_stft_beat_stream_adapter`.
- r02 pipeline jobs: validate `4576164` completed, train `4576165` completed, eval `4576166` cancelled.
- Both train pipelines were intended to start after feature extraction succeeds; r02 completed, while r01 was cancelled before training because the account hit a Slurm CPU-minute/GPU-credit policy limit.

## Evaluation Plan

- Use G1 dataset eval after each `train-500.pt`.
- Metrics: `metrics.json`, `g1_table.json`, `motion_audit.json`, and `paper_report.md`.
- Enable FK metrics against the Unitree G1 MJCF.
- First comparison is r01 versus r02. Compare against Librosa35/G1 only after both runs finish cleanly.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-13 | data link | passed | `data/finedance_g1_fkbeats/{train,test}` | Source G1 FineDance tree is visible in this worktree through subdirectory symlinks. |
| 2026-05-13 | focused tests | passed | `python -m unittest tests.test_feature_config_and_fusion tests.test_wav2clip_stft_beat_features tests.test_validate_preprocessed_data tests.test_submit_training_pipeline tests.test_phase4_to_6_beat_integration` | 102 tests passed. |
| 2026-05-13 | G1 loss smoke | passed | synthetic `EDGE('wav2clip_stft_beat', motion_format='g1', feature_fusion=...)` loss | `concat_norm` loss `5.3715`; `stream_adapter` loss `5.3404`; both finite with G1 repr dim `38`. |
| 2026-05-13 | feature preprocess | cancelled | Slurm job `4576095`; log `slurm/EXP-20260513-finedance-g1-wav2clip-stft-beat/preprocess_features_4576095.out` | Superseded by the 8-way array after measuring roughly 2.5 clips/sec. |
| 2026-05-13 | r01_concat_norm pipeline | cancelled | jobs `4576104` -> `4576106` -> `4576108` | Superseded by the array-dependent resubmit. |
| 2026-05-13 | r02_stream_adapter pipeline | cancelled | jobs `4576105` -> `4576107` -> `4576109` | Superseded by the array-dependent resubmit. |
| 2026-05-13 | feature preprocess speedup | passed | Slurm array `4576163_[0-7]`; old jobs `4576095`, `4576104`-`4576109` cancelled | Extraction sharded across 8 array tasks; all array tasks completed. |
| 2026-05-13 | r01_concat_norm resubmit | blocked | jobs `4576167` -> `4576168` -> `4576169` | Validation completed; train was cancelled before running with `Reason=AssocGrpCPUMinutesLimit`; eval was cancelled. |
| 2026-05-13 | r02_stream_adapter resubmit | partial | jobs `4576164` -> `4576165` -> `4576166` | Validation and train completed; eval was cancelled after the quota block appeared. |
| 2026-05-13 | r02_stream_adapter train | passed | `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt`; Slurm `4576165` | Completed 500 epochs in `03:15:16`; final epoch logged finite losses and `[MODEL SAVED at Epoch 500]`. |
| 2026-05-13 | scheduler repair | blocked | `squeue`, `sacct`, `sbatch --test-only` | r01 train `4576168` was reduced from 24h to 4h but still hit `Reason=AssocGrpCPUMinutesLimit`. Fresh 1h, 30m, and 15m GPU test submissions also failed with the same policy error. User has only `normal` QoS; `interactive_qos` is invalid. Queued evals `4576166` and `4576169` were cancelled to prioritize training. |
| 2026-05-14 | migration pack-up | blocked | no active jobs in `squeue`; `sacct -j 4576168` shows `CANCELLED by 0` | Feature cache is complete, r02 checkpoint exists, but r01/eval must move to another server or account. |

## Results

- Metric files: none yet.
- Checkpoints: `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt`.
- Feature cache: `data/finedance_g1_fkbeats/train/wav2clip_stft_beat_feats` has `47817/47817`; `data/finedance_g1_fkbeats/test/wav2clip_stft_beat_feats` has `3265/3265`.
- Current conclusion: feature extraction and validation passed; r02 trained cleanly. r01 is not blocked by code or data, but by the account-level Slurm CPU-minute quota.
- Curated Slurm evidence committed for migration:
  `docs/experiments/artifacts/EXP-20260513-finedance-g1-wav2clip-stft-beat/`.

## Next Action

On the 4090 server, use the compact-HF plus local-rebuild flow in
`docs/NEW_SERVER_SETUP.md`. Do not wait on old Slurm job `4576168`; it was
cancelled on the previous account. HF should hold raw FineDance, retargeted G1
motions, and checkpoints only; rebuild feature folders locally.

```bash
cd /path/to/EDGE-wav2clip
source .venv311/bin/activate
python submit_training_pipeline.py \
  --preset g1_finedance_wav2clip_stft_beat_concat_norm \
  --train_name EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm \
  --run_id EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm \
  --skip_preprocess \
  --train_time 04:00:00 \
  --eval_time 02:00:00
```

If you reuse the generated sbatch scripts instead of regenerating the pipeline,
submit them from the branch repo root:

```bash
sbatch --time=04:00:00 \
  slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm/train.sbatch
```

After r01 finishes, resubmit both eval scripts with short walltimes:

```bash
sbatch --time=02:00:00 \
  slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm/evaluate.sbatch
sbatch --time=02:00:00 \
  slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/evaluate.sbatch
```
