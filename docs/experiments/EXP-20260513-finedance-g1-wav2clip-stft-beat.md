# EXP-20260513-finedance-g1-wav2clip-stft-beat

Status: finished
Owner: yukun
Created: 2026-05-13
Last Updated: 2026-05-22

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
- Batch and optimizer: old Slurm target used batch size `512`, gradient accumulation `1`, learning rate `2e-4`. On the local 24GB RTX 4090, use microbatch `256` with gradient accumulation `2` to keep the effective batch size `512`.
- Initial checkpoint cadence: 500 epochs, save every 50.
- 2026-05-20 continuation target: resume the stronger `r02_stream_adapter`
  checkpoint from epoch 500 to epoch 2000, saving every 50.

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
- For long continuations, do not decide epoch sufficiency from train loss alone. Evaluate checkpoint sweeps at meaningful anchors such as `1000`, `1500`, and `2000`; continue past `2000` only if G1 motion-quality or rhythm metrics are still improving without contact/root degradation.

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
| 2026-05-20 | 4090 local setup | running | branch `wav2clip-stft-beat`, commit `60309cd`; tmux session `m2d_bootstrap_4090`; logs `setup_logs/finedance_g1_4090/` | Created repo-local `.venv311` with conda Python 3.10 because system `python3.11` is unavailable and `python3.10 -m venv` lacks `ensurepip`. Installed CUDA PyTorch and requirements with `--no-build-isolation --use-deprecated=legacy-resolver` for the legacy Jukebox stack. Downloaded compact HF artifacts plus Jukebox `vqvae.pth.tar` and `prior_level_2.pth.tar`; local bootstrap is rebuilding prepared data/features and will run focused validation. |
| 2026-05-20 | r01 local watcher | running | tmux session `m2d_train_r01_after_bootstrap`; launcher `setup_logs/finedance_g1_4090/launch_r01_after_bootstrap.sh`; log `setup_logs/finedance_g1_4090/train_r01_concat_norm_after_bootstrap_20260520_0153.log` | Watcher waits for `m2d_bootstrap_4090` to finish, requires `[bootstrap_4090] Done`, checks `wav2clip_stft_beat_feats` counts `47817/3265`, then launches local `python train.py ... --feature_fusion concat_norm` for `EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm`. |
| 2026-05-20 | 4090 speed optimization | running | stopped default Jukebox extraction at `1662/47817`; launcher `setup_logs/finedance_g1_4090/bootstrap_wav2clip_fast.sh`; patched `data/audio_extraction/wav2clip_stft_beat_features.py`; shard logs `setup_logs/finedance_g1_4090/extract_wav2clip_train_shard*.log` | Default bootstrap was not optimized for the immediate r01 goal: Jukebox ran at about `1.87 clips/s` with low GPU utilization and is not required for `wav2clip_stft_beat` training. Switched to a Wav2CLIP-only fast path: reuse completed source baseline features, prepare the G1 tree without beat metadata, extract Wav2CLIP/STFT/GaussianBeat with 4 shards, then validate the concat_norm cache. Added an explicit librosa frame compatibility patch for the old `wav2clip` package under librosa `0.11.0`; smoke extraction passed before relaunch. |
| 2026-05-20 | r01 local train launch | running | tmux session `m2d_train_r01_after_bootstrap`; log `setup_logs/finedance_g1_4090/train_r01_concat_norm_after_bootstrap_20260520_0153.log`; W&B run `crtohf53` failed then restarted | First local launch with microbatch `512` hit CUDA OOM on the 24GB 4090. Installed missing `p_tqdm`, added it to `requirements-new-server.txt`, then relaunched with `batch_size=256`, `gradient_accumulation_steps=2`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This preserves effective batch `512`; epoch 1 progressed with ~14.2GB GPU memory and ~99% GPU utilization. |
| 2026-05-20 | r01 resume after render dependency | running | checkpoint `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm2/weights/train-50.pt`; tmux session `m2d_train_r01_resume50`; log `setup_logs/finedance_g1_4090/train_r01_concat_norm_resume50_20260520.log`; W&B run `tkwu81du` | Training reached epoch 50 and saved `train-50.pt`, then exited during sample rendering because `imageio_ffmpeg` was missing. Installed `imageio-ffmpeg`, added it to `requirements-new-server.txt`, and resumed from `train-50.pt` with `--epoch_offset 50 --epochs 450`, keeping effective batch `256x2=512`. Resume is running from epoch 51. |
| 2026-05-20 | r01 4090 speed/progress audit | running | `nvidia-smi`; `scripts/training_progress.py setup_logs/finedance_g1_4090/train_r01_concat_norm_resume50_20260520.log` | RTX 4090 is compute-bound at ~100% GPU utilization, ~14.4GB/24.6GB memory, ~320W, and ~49.2s/epoch (~970 samples/s). H200-style microbatch `512` previously OOMed, so `256x2` is the current stable 4090 setting. Added total ETA fields to future training logs and a log parser for the current process. At epoch `283/500`, ETA was about `2h58m`, estimated finish `2026-05-20 20:14:48`. |
| 2026-05-20 | r01_concat_norm 4090 train | passed | `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm_resume50/weights/train-500.pt`; log `setup_logs/finedance_g1_4090/train_r01_concat_norm_resume50_20260520.log`; W&B run `tkwu81du` | Resume completed the remaining epochs and saved the final checkpoint. |
| 2026-05-20 | full eval and comparison | passed | `setup_logs/finedance_g1_4090/eval_wav2clip_compare_20260520.sh`; log `setup_logs/finedance_g1_4090/eval_wav2clip_compare_20260520.log`; outputs `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/` | Evaluated r01, r02, and the previous Librosa35 2000-epoch baseline over all `3265` FineDance+G1 test clips with `--batch_size 32`; comparison writer produced `comparison_g1_metrics.md` and `.json`. |
| 2026-05-20 | r02_stream_adapter continuation to 2000 | partial | script `setup_logs/finedance_g1_4090/launch_wav2clip_r02_to2000_after_gaussian.sh`; tmux `m2d_train_wav2clip_r02_to2000_after_gaussian`; log `setup_logs/finedance_g1_4090/train_wav2clip_r02_stream_adapter_resume500_to2000_20260520.log`; checkpoint `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter2/weights/train-600.pt` | User asked to continue the better Wav2CLIP variant to 2000 epochs. The watcher launched after GaussianBeat finished and saved `train-550.pt` and `train-600.pt`. It was interrupted during epoch 630 so GaussianBeat could be evaluated. |
| 2026-05-21 | r02_stream_adapter resume600 to 2000 | running | script `setup_logs/finedance_g1_4090/launch_wav2clip_r02_resume600_to2000_after_eval.sh`; tmux `m2d_train_wav2clip_r02_resume600_to2000`; log `setup_logs/finedance_g1_4090/train_wav2clip_r02_stream_adapter_resume600_to2000_20260521.log`; W&B run `firfyuhd` | Resumed from `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter2/weights/train-600.pt` with `--epoch_offset 600 --epochs 1400`, targeting effective epoch `2000`. The new run name is `EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter_resume600_to2000` to avoid the auto-incremented train directory ambiguity. |
| 2026-05-21 | future W&B epoch logging | passed | `EDGE.py`, `args.py`, `tests/test_phase0_cli_and_preprocess.py`, `tests/test_phase4_to_6_beat_integration.py` | Future training runs keep W&B enabled by default and log epoch-level losses, progress, ETA, throughput, CUDA memory, and checkpoint markers through `--wandb_log_interval` default `1`. This does not affect the already-running r02 process. |
| 2026-05-21 | r02 epoch-sufficiency eval watcher | queued | script `setup_logs/finedance_g1_4090/eval_r02_resume600_checkpoint_sweep_after_train_20260521.sh`; tmux `m2d_eval_r02_epoch_sweep_after_train`; log `setup_logs/finedance_g1_4090/eval_r02_resume600_checkpoint_sweep_after_train_20260521.log` | Watcher waits for active training to finish, then evaluates r02 continuation checkpoints at `1000`, `1500`, and `2000` and writes `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/comparison_r02_resume600_epoch_sweep.md`. |
| 2026-05-22 | r02_stream_adapter resume600 to 2000 train | passed | `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter_resume600_to2000/weights/train-2000.pt`; log `setup_logs/finedance_g1_4090/train_wav2clip_r02_stream_adapter_resume600_to2000_20260521.log`; W&B run `firfyuhd` | Continuation completed at effective epoch `2000` after `19h19m43s`; final timing line reported `49.61s/epoch`, `963.92 samples/s`, and `peak_cuda_memory_mb=12789.28`. |
| 2026-05-22 | r02 checkpoint sweep eval | passed | `setup_logs/finedance_g1_4090/eval_r02_resume600_checkpoint_sweep_after_train_20260521.log`; `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/comparison_r02_resume600_epoch_sweep.md` | Evaluated r02 continuation checkpoints at `1000`, `1500`, and `2000` against r01 500, r02 500, GaussianBeat 1000, and Librosa35 2000. |
| 2026-05-22 | fixed 40s qualitative render | passed | `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234/comparison.mp4`; manifest `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234/manifest.json`; script `eval/render_g1_checkpoint_comparison.py` | Rendered `wav2clip_r02_2000`, `gaussian_beat_1000`, and `librosa35_2000` on the same `data/finedance/music_wav/012.wav` 40s slice window with seed `1234`, slice start `14`. Librosa35 required checkpoint-inferred beat conditioning. The bundled ffmpeg lacks `drawtext`, so the script now builds a Pillow label banner before ffmpeg hstack/vstack composition. |
| 2026-05-22 | low-amplitude motion audit | passed | cache-faithful render `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234_cache/comparison.mp4`; audit `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234_cache/amplitude_audit.json`; renderer patch `eval/render_g1_checkpoint_comparison.py` | The first qualitative renderer recomputed features from temporary audio slices, which did not match the training/eval cache. The renderer now supports `--feature_source cache` and loads cached `wavs_sliced`, `*_feats`, and `beat_feats`. The amplitude issue persists under cache-faithful inputs: `dof_std_mean` is `0.3342` for GaussianBeat, `0.1708` for Wav2CLIP r02, `0.1635` for Librosa35, and `0.3769` for the stitched reference. |
| 2026-05-22 | future amplitude metrics | passed | `eval/g1_metrics.py`; `eval/write_g1_metric_comparison.py`; tests `.venv311/bin/python -m unittest tests.test_g1_eval_metrics tests.test_render_g1_checkpoint_comparison tests.test_feature_config_and_fusion` | Future G1 benchmark outputs include `JointPositionStdMean`, `JointPositionRangeMean`, and `RootFlatRangeMean` so low-amplitude averaged motion is visible in metric tables, not only in rendered videos. |
| 2026-05-22 | five-song inference-style qualitative renders | passed | `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_{001,003,010,014,022}_40s_seed1234_extract_stick/comparison.mp4`; log `setup_logs/finedance_g1_4090/render_five_inference_extract_mujoco_overwrite_stick_20260522.log` | Generated five additional 40s three-model comparisons on different full-song music inputs. Each manifest reports `feature_source=extract`, `audio_source=extract`, `sample_size=15`, and three rendered models. The first pass used CPU/stick while NVIDIA probing was slow; after 4090 recovered, the same `*_extract_stick` directories were overwritten with GPU inference and `--g1_render_backend mujoco --g1_mujoco_gl egl`. |
| 2026-05-22 | VSCode audio compatibility rewrap | passed | `ffprobe` on `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_{001,003,010,014,022}_40s_seed1234_extract_stick/{comparison.mp4,videos/*/*.mp4}`; test `.venv311/bin/python -m unittest tests.test_render_g1_checkpoint_comparison` | Rewrapped the five comparison videos and their 15 per-model videos in place with H.264 video copied and AAC audio converted to stereo 48kHz for VSCode/Chromium preview compatibility. Updated future render/mux code to write stereo 48kHz AAC by default. |

## Results

- Metric files:
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/r01_concat_norm/metrics.json`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/r02_stream_adapter/metrics.json`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/baseline_librosa35_2000/metrics.json`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/comparison_g1_metrics.md`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/comparison_g1_metrics.json`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/r02_stream_adapter_resume600_to2000_ckpt1000/metrics.json`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/r02_stream_adapter_resume600_to2000_ckpt1500/metrics.json`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/r02_stream_adapter_resume600_to2000_ckpt2000/metrics.json`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/comparison_r02_resume600_epoch_sweep.md`
  - `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/comparison_r02_resume600_epoch_sweep.json`
- Qualitative render:
  - Command: `.venv311/bin/python -m eval.render_g1_checkpoint_comparison`
  - Comparison video: `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234/comparison.mp4`
  - Manifest: `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234/manifest.json`
  - Individual videos and saved G1 motion pickles are under `videos/{wav2clip_r02_2000,gaussian_beat_1000,librosa35_2000}/` and `motions/{wav2clip_r02_2000,gaussian_beat_1000,librosa35_2000}/` in the same render directory.
- Cache-faithful qualitative render and amplitude audit:
  - Command: `.venv311/bin/python -m eval.render_g1_checkpoint_comparison --feature_source cache --music data/finedance/music_wav/012.wav --out_length 40 --seed 1234 --slice_start 14 --output_dir renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234_cache --overwrite`
  - Comparison video: `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234_cache/comparison.mp4`
  - Manifest: `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234_cache/manifest.json`
  - Amplitude audit: `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234_cache/amplitude_audit.json`
- Five full-song inference-style MuJoCo renders:
  - Command shape: `.venv311/bin/python -m eval.render_g1_checkpoint_comparison --feature_source extract --music data/finedance/music_wav/<music>.wav --out_length 40 --seed 1234 --g1_render_backend mujoco --g1_mujoco_gl egl --output_dir renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_<music>_40s_seed1234_extract_stick --overwrite`
  - Music IDs: `001`, `003`, `010`, `014`, `022`.
  - Comparison videos: `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_{001,003,010,014,022}_40s_seed1234_extract_stick/comparison.mp4`.
  - Manifests: `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_{001,003,010,014,022}_40s_seed1234_extract_stick/manifest.json`.
  - Verification: `ffprobe` reports 40.000-40.008s for all comparison videos; all comparison and per-model mp4 files now have AAC stereo 48kHz audio.
- Checkpoints:
  - `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm_resume50/weights/train-500.pt`
  - `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt`
  - continuation checkpoint before GaussianBeat eval: `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter2/weights/train-600.pt`
  - final continuation checkpoint: `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter_resume600_to2000/weights/train-2000.pt`
- Feature cache: `data/finedance_g1_fkbeats/train/wav2clip_stft_beat_feats` has `47817/47817`; `data/finedance_g1_fkbeats/test/wav2clip_stft_beat_feats` has `3265/3265`.
- First-stage full-test G1 metric comparison:

| Model | Files | G1BAS | G1RoboPerformBAS | G1FKBAS | G1FKRoboPerformBAS | G1BeatF1 | G1Dist | G1Div | G1FootSliding | G1GroundPenetration | RootHeightViolationRate | JointSmoothnessJerkMean | RootSmoothnessJerkMean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `wav2clip_r01_concat_norm` | 3265 | 0.2162 | 0.4355 | 0.2235 | 0.4329 | 0.1924 | 11.8843 | 17.4055 | 0.8473 | 0.0751 | 0.0775 | 589.4798 | 1094.4722 |
| `wav2clip_r02_stream_adapter` | 3265 | 0.2137 | 0.4372 | 0.2211 | 0.4346 | 0.1905 | 9.9181 | 16.3441 | 0.7382 | 0.0689 | 0.0085 | 524.9584 | 1018.0622 |
| `librosa35_baseline_2000` | 3265 | 0.2413 | 0.4730 | 0.2544 | 0.4504 | 0.2139 | 9.2544 | 11.3661 | 0.5349 | 0.0352 | 0.0000 | 438.5067 | 886.3016 |

- r02 continuation checkpoint sweep:

| Model | Files | G1BAS | G1RoboPerformBAS | G1FKBAS | G1FKRoboPerformBAS | G1BeatF1 | G1Dist | G1Div | G1FootSliding | G1GroundPenetration | RootDriftMean | RootHeightViolationRate | ReferenceRangeViolationRate | JointSmoothnessJerkMean | RootSmoothnessJerkMean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `wav2clip_r02_500` | 3265 | 0.2137 | 0.4372 | 0.2211 | 0.4346 | 0.1905 | 9.9181 | 16.3441 | 0.7382 | 0.0689 | 0.2807 | 0.0085 | 0.0096 | 524.9584 | 1018.0622 |
| `wav2clip_r02_1000` | 3265 | 0.2164 | 0.4306 | 0.2281 | 0.4233 | 0.1905 | 8.4780 | 12.8282 | 0.6144 | 0.0421 | 0.2763 | 0.0000 | 0.0013 | 513.3883 | 963.5223 |
| `wav2clip_r02_1500` | 3265 | 0.2158 | 0.4285 | 0.2341 | 0.4222 | 0.1950 | 8.9353 | 12.8244 | 0.5826 | 0.0347 | 0.2758 | 0.0000 | 0.0011 | 522.7130 | 976.1589 |
| `wav2clip_r02_2000` | 3265 | 0.2237 | 0.4321 | 0.2377 | 0.4245 | 0.1979 | 8.9113 | 12.8445 | 0.5572 | 0.0408 | 0.2726 | 0.0000 | 0.0011 | 529.8643 | 939.8722 |
| `gaussian_beat_1000` | 3265 | 0.2072 | 0.4210 | 0.2311 | 0.4199 | 0.1913 | 9.2000 | 20.5369 | 0.6015 | 0.0803 | 0.2709 | 0.0009 | 0.0117 | 441.7651 | 1069.8000 |
| `librosa35_2000` | 3265 | 0.2413 | 0.4730 | 0.2544 | 0.4504 | 0.2139 | 9.2544 | 11.3661 | 0.5349 | 0.0352 | 0.2022 | 0.0000 | 0.0006 | 438.5067 | 886.3016 |

- Current conclusion: r02 continuation fixed the weak 500-epoch motion-quality result and is now the best Wav2CLIP/STFT/GaussianBeat checkpoint. `train-2000.pt` has the best r02 rhythm metrics and foot sliding, while `train-1000.pt` has the lowest `G1Dist`. Against Librosa35 2000, r02 2000 wins on `G1Dist` but still loses on rhythm, diversity, contact/grounding, root drift, range violations, and smoothness. The cache-faithful 40s audit confirms a separate low-amplitude issue: Wav2CLIP r02 and Librosa35 produce much lower joint-motion amplitude than GaussianBeat and the reference. This is not a normalizer mismatch; it points to conditional guidance/objective collapse where richer conditioning steers the denoiser toward safer averaged motion.
- Curated Slurm evidence committed for migration:
  `docs/experiments/artifacts/EXP-20260513-finedance-g1-wav2clip-stft-beat/`.

## Next Action

Use `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter_resume600_to2000/weights/train-2000.pt` as the current Wav2CLIP/STFT/GaussianBeat checkpoint. For full-song inference qualitative comparison, use `--feature_source extract` so features are recomputed from the target audio; use `--feature_source cache` only when deliberately reproducing dataset cached inputs. Next, start a targeted amplitude/energy-preservation follow-up rather than blindly extending epochs: try inference guidance sweeps only as diagnostics, and train with a motion-energy or joint-velocity/range preservation objective plus stream/dropout changes that keep the beat channel from being washed out by richer features.
