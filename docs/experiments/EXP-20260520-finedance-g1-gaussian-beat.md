# EXP-20260520-finedance-g1-gaussian-beat

Status: finished
Owner: yukun
Created: 2026-05-20
Last Updated: 2026-05-21

## Research Question

How much of the FineDance+G1 conditioning signal comes from the explicit Gaussian beat channel alone when Wav2CLIP and STFT are removed?

## Hypothesis

Pure `GaussianBeat` should test whether frame-level rhythm timing alone can drive plausible G1 motion. It is expected to be weaker than `Wav2CLIP + STFT + GaussianBeat` for style and motion diversity, but useful as a lower-bound ablation for rhythm-conditioned generation.

## Baseline Or Control

- Primary control: `EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm`.
- Secondary anchor: `EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter`.
- This run keeps the current Transformer diffusion backbone, G1 motion format, data split, losses, optimizer, effective batch, and checkpoint cadence unchanged from the local r01 setup, while extending the schedule to 1000 epochs.

## Intervention

Train one FineDance+G1 run with `feature_type=gaussian_beat` and `feature_fusion=linear`.

The feature tensor is `(150, 1)`: the final GaussianBeat channel sliced from the already validated `(150, 706)` `wav2clip_stft_beat_feats` cache. This intentionally removes Wav2CLIP semantics and STFT texture while keeping the exact beat curve used by EXP-20260513.

## Invariant Controls

- Branch: `wav2clip-stft-beat`; local clone at `/home/tianhup/Desktop/Musics2Dance`.
- Git commit at setup: `60309cd` plus local experiment changes.
- Dataset: `data/finedance_g1_fkbeats`.
- Train clips: `47817`; test clips: `3265`.
- Motion format: G1, 5-second horizon, 30 FPS.
- Backbone: current Transformer `DanceDecoder`.
- Beat/alignment supervision: none; `GaussianBeat` is conditioning only, not `use_beats`.
- Effective batch: local 4090 microbatch `256` with gradient accumulation `2`, matching effective batch `512`.
- Optimizer: learning rate `2e-4`, weight decay `0.02`.
- Schedule: 1000 epochs, save every 50.
- Robot losses: unchanged from EXP-20260513 r01, with only `lambda_g1_kin=1.0` active and warmup fraction `0.0`.

## Data And Cache Contract

- Feature type: `gaussian_beat`.
- Feature shape: `(150, 1)`.
- Feature cache:
  - `data/finedance_g1_fkbeats/train/gaussian_beat_feats`
  - `data/finedance_g1_fkbeats/test/gaussian_beat_feats`
- Processed/tensor cache: `data/finedance_g1_gaussian_beat_dataset_backups`.
- Feature cache dtype: `float16` memmap.
- Cache invalidation: required because feature width/semantics differ from `wav2clip_stft_beat`.

## Training Or Execution Plan

- Environment: `source .venv311/bin/activate`.
- Run ID: `EXP-20260520-finedance-g1-gaussian-beat_r01_linear`.
- The local 4090 is reserved for this run while it is active. Do not run another training concurrently.
- Launcher: `setup_logs/finedance_g1_4090/launch_gaussian_beat_after_r01.sh`.
- tmux watcher: `m2d_train_gaussian_beat_after_r01`.
- Log path: `setup_logs/finedance_g1_4090/train_gaussian_beat_r01_linear_1000_20260520.log`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv311/bin/python train.py \
  --feature_type gaussian_beat \
  --motion_format g1 \
  --lambda_beat 0.0 \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_gaussian_beat_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260520-finedance-g1-gaussian-beat_r01_linear \
  --render_dir renders \
  --batch_size 256 \
  --gradient_accumulation_steps 2 \
  --epochs 1000 \
  --epoch_offset 0 \
  --save_interval 50 \
  --ema_interval 1 \
  --learning_rate 0.0002 \
  --weight_decay 0.02 \
  --train_num_workers 2 \
  --test_num_workers 2 \
  --mixed_precision bf16 \
  --feature_cache_mode memmap \
  --feature_cache_dtype float16 \
  --feature_fusion linear \
  --lambda_g1_fk 0.0 \
  --lambda_g1_fk_vel 0.0 \
  --lambda_g1_fk_acc 0.0 \
  --lambda_g1_foot 0.0 \
  --lambda_g1_kin 1.0 \
  --g1_kin_loss_warmup_epochs 0 \
  --g1_kin_loss_max_fraction 0.0 \
  --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml \
  --g1_root_quat_order xyzw
```

## Evaluation Plan

- Evaluate after `train-1000.pt` using the same G1 dataset eval path as EXP-20260513.
- Metrics: `metrics.json`, `g1_table.json`, `motion_audit.json`, and `paper_report.md`.
- Compare against EXP-20260513 r01/r02 first, then Librosa35/G1 if needed.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-20 | implementation | passed | `feature_config.py`, `data/audio_extraction/gaussian_beat_features.py`, `submit_training_pipeline.py` | Added official `gaussian_beat` feature type, a derivation script, and a pipeline preset. |
| 2026-05-20 | focused tests | passed | `.venv311/bin/python -m unittest tests.test_feature_config_and_fusion tests.test_gaussian_beat_features tests.test_validate_preprocessed_data`; `.venv311/bin/python -m unittest tests.test_submit_training_pipeline` | 13 feature/validation tests and 46 pipeline tests passed. |
| 2026-05-20 | feature derivation | passed | `data/finedance_g1_fkbeats/{train,test}/gaussian_beat_feats` | Derived from the final channel of `wav2clip_stft_beat_feats`; counts are `47817` train and `3265` test. |
| 2026-05-20 | validation | passed | `.venv311/bin/python data/validate_preprocessed_data.py --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_gaussian_beat_dataset_backups --feature_type gaussian_beat --motion_format g1 --feature_cache_mode memmap --feature_cache_dtype float16 --sample_count 64` | Validation passed with G1 root height ranges matching EXP-20260513 and `beat_count=0`. |
| 2026-05-20 | queued launch watcher | running | tmux `m2d_train_gaussian_beat_after_r01`; script `setup_logs/finedance_g1_4090/launch_gaussian_beat_after_r01.sh` | Watcher is waiting for EXP-20260513 r01 process to exit and for `train-500.pt` to exist before launching this training. |
| 2026-05-20 | r01_linear 1000-epoch launch | running | script `setup_logs/finedance_g1_4090/launch_gaussian_beat_after_r01.sh`; log `setup_logs/finedance_g1_4090/train_gaussian_beat_r01_linear_1000_20260520.log`; tmux `m2d_train_gaussian_beat_after_r01`; W&B run `0gw0qic8` | User requested pure GaussianBeat training for 1000 epochs. Previous wav2clip r01 checkpoint exists and no active GaussianBeat process/checkpoint was found, so the watcher script was updated from 500 to 1000 epochs and relaunched. Training passed epoch `1/1000` and was active at epoch `2/1000`; `scripts/training_progress.py` estimated ~`47.74s/epoch`, ETA `13h14m52s`, finish `2026-05-21 11:37:13`. |
| 2026-05-21 | r01_linear 1000-epoch train | passed | `runs/train/EXP-20260520-finedance-g1-gaussian-beat_r01_linear/weights/train-1000.pt`; log `setup_logs/finedance_g1_4090/train_gaussian_beat_r01_linear_1000_20260520.log` | Training completed and all 50-epoch checkpoints through `train-1000.pt` exist. |
| 2026-05-21 | full G1 dataset eval | passed | `setup_logs/finedance_g1_4090/eval_gaussian_beat_1000_20260521.log`; outputs `eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/` | Evaluated all `3265` FineDance+G1 test clips with `--batch_size 32`, FK metrics enabled, seed `1234`. |
| 2026-05-21 | benchmark comparison | passed | `eval/EXP-20260520-finedance-g1-gaussian-beat/comparison_g1_metrics.md`; `eval/EXP-20260520-finedance-g1-gaussian-beat/comparison_g1_metrics.json` | Compared GaussianBeat 1000 against wav2clip r01 500, wav2clip r02 500, and Librosa35 2000. |

## Results

- Metric files:
  - `eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/metrics.json`
  - `eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/g1_table.json`
  - `eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/motion_audit.json`
  - `eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/paper_report.md`
  - `eval/EXP-20260520-finedance-g1-gaussian-beat/comparison_g1_metrics.md`
  - `eval/EXP-20260520-finedance-g1-gaussian-beat/comparison_g1_metrics.json`
- Checkpoint:
  - `runs/train/EXP-20260520-finedance-g1-gaussian-beat_r01_linear/weights/train-1000.pt`
- Feature cache: ready and validated.
- Full-test G1 metric comparison:

| Model | Files | G1BAS | G1RoboPerformBAS | G1FKBAS | G1FKRoboPerformBAS | G1BeatF1 | G1Dist | G1Div | G1FootSliding | G1GroundPenetration | RootDriftMean | RootHeightViolationRate | ReferenceRangeViolationRate | JointSmoothnessJerkMean | RootSmoothnessJerkMean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gaussian_beat_1000` | 3265 | 0.2072 | 0.4210 | 0.2311 | 0.4199 | 0.1913 | 9.2000 | 20.5369 | 0.6015 | 0.0803 | 0.2709 | 0.0009 | 0.0117 | 441.7651 | 1069.8000 |
| `wav2clip_r01_500` | 3265 | 0.2162 | 0.4355 | 0.2235 | 0.4329 | 0.1924 | 11.8843 | 17.4055 | 0.8473 | 0.0751 | 0.3429 | 0.0775 | 0.0161 | 589.4798 | 1094.4722 |
| `wav2clip_r02_500` | 3265 | 0.2137 | 0.4372 | 0.2211 | 0.4346 | 0.1905 | 9.9181 | 16.3441 | 0.7382 | 0.0689 | 0.2807 | 0.0085 | 0.0096 | 524.9584 | 1018.0622 |
| `librosa35_baseline_2000` | 3265 | 0.2413 | 0.4730 | 0.2544 | 0.4504 | 0.2139 | 9.2544 | 11.3661 | 0.5349 | 0.0352 | 0.2022 | 0.0000 | 0.0006 | 438.5067 | 886.3016 |

- Current conclusion: pure GaussianBeat is a strong beat-only lower-bound. It beats both 500-epoch wav2clip variants on `G1Dist`, foot sliding, root drift, root-height violation, and joint jerk, and is essentially tied with Librosa35 on `G1Dist`. It does not beat Librosa35 overall: Librosa35 remains stronger on rhythm, contact/grounding, root drift, range violations, and root smoothness.

## Next Action

Keep this as the beat-only baseline. The promising follow-up is a hybrid feature experiment that keeps GaussianBeat's motion-quality signal while adding a better semantic or texture stream.
