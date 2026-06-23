# EXP-20260601-finedance-g1-beat-features-8d

Status: finished
Owner: yukun
Created: 2026-06-01
Last Updated: 2026-06-11

## Research Question

Does replacing the prior 1-D `gaussian_beat` condition with an 8-D beat-structure feature improve beat timing/control while preserving the strong motion-quality profile of the old G1 GaussianBeat baseline?

## Hypothesis

The 1-D GaussianBeat baseline showed that beat-only conditioning can produce a useful lower-bound G1 generator, but prior condition-sensitivity results showed weak exact beat-timing use. Adding pulse, previous/next beat distance, beat phase, local interval, and onset-strength channels should provide denser timing structure without reintroducing Wav2CLIP or STFT semantics. Success requires better rhythm metrics without sacrificing diversity, contact, root behavior, or range.

## Baseline Or Control

- Primary control: `EXP-20260520-finedance-g1-gaussian-beat_r01_linear`, checkpoint `runs/train/EXP-20260520-finedance-g1-gaussian-beat_r01_linear/weights/train-1000.pt`.
- Sensitivity reference: `EXP-20260522-gaussian-beat-condition-ablation`, which showed that real, shifted, and random GaussianBeat conditions were too similar on benchmark scores.
- Rich-feature references: Wav2CLIP/STFT/GaussianBeat and later Wav2CLIP motion-control experiments are comparison anchors only; this experiment intentionally removes those streams.

## Intervention

Train a new FineDance+G1 old-`g1` run with:

```text
feature_type=beat_features_8d
feature_fusion=linear
motion_format=g1
```

The feature tensor is `(150, 8)` with channel order:

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

This is a music-conditioning feature only. It does not enable `--use_beats`, does not add beat-estimator supervision, and does not use Wav2CLIP/STFT or structured motion-control predictor heads.

## Invariant Controls

- Branch/worktree: `codex/wav2clip-stage-20260526` branch on `/home/tianhup/Desktop/Musics2Dance`.
- Dataset: `data/finedance_g1_fkbeats`.
- Motion format: legacy `g1`, not v4/v5 root-delta/yaw-delta.
- Train/test split: FineDance+G1 train `47817`, test `3265`.
- Horizon/FPS: 150 frames, 30 FPS.
- Backbone: normal `DanceDecoder` with linear condition projection.
- Batch/effective batch: local 4090 microbatch `256`, gradient accumulation `2`, effective batch `512`.
- Optimizer and robot losses match the prior GaussianBeat baseline unless explicitly recorded in the run log.
- W&B remains enabled.

## Data And Cache Contract

- Feature cache:
  - `data/finedance_g1_fkbeats/train/beat_features_8d_feats`
  - `data/finedance_g1_fkbeats/test/beat_features_8d_feats`
- Processed/tensor cache:
  - `data/finedance_g1_beat_features_8d_dataset_backups`
- Feature cache dtype for training: `float16` memmap.
- Rebuild the feature cache and processed/tensor cache if channel order, beat extraction, normalization, or feature width changes.

## Implementation Scope

- Add `feature_type=beat_features_8d` with dim `8`.
- Add `data/audio_extraction/beat_features_8d_features.py`.
- Wire `data/create_dataset.py`, `test.py`, and the optional submit pipeline preset.
- Add focused unit tests for extractor shape/channel order/no-beat behavior, feature config validation, preprocessing validation, and custom-music feature resolution.

## Training Plan

Feature extraction:

```bash
.venv311/bin/python -m data.audio_extraction.beat_features_8d_features \
  data/finedance_g1_fkbeats/train/wavs_sliced \
  data/finedance_g1_fkbeats/train/beat_features_8d_feats

.venv311/bin/python -m data.audio_extraction.beat_features_8d_features \
  data/finedance_g1_fkbeats/test/wavs_sliced \
  data/finedance_g1_fkbeats/test/beat_features_8d_feats
```

Validation:

```bash
.venv311/bin/python data/validate_preprocessed_data.py \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_beat_features_8d_dataset_backups \
  --feature_type beat_features_8d \
  --motion_format g1 \
  --feature_cache_mode memmap \
  --feature_cache_dtype float16 \
  --sample_count 64
```

Training:

```bash
source .venv311/bin/activate
export MUJOCO_GL=egl
PYTHONUNBUFFERED=1 .venv311/bin/python -m accelerate.commands.launch train.py \
  --feature_type beat_features_8d \
  --feature_fusion linear \
  --motion_format g1 \
  --lambda_beat 0.0 \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_beat_features_8d_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260601-finedance-g1-beat-features-8d_r01_linear \
  --render_dir renders/EXP-20260601-finedance-g1-beat-features-8d \
  --batch_size 256 \
  --gradient_accumulation_steps 2 \
  --epochs 1000 \
  --save_interval 50 \
  --full_eval_interval 500 \
  --full_eval_root eval \
  --full_eval_variants auto \
  --full_eval_batch_size 32 \
  --wandb_pj_name EDGE \
  --wandb_log_interval 1 \
  --mixed_precision bf16 \
  --feature_cache_mode memmap \
  --feature_cache_dtype float16 \
  --lambda_g1_kin 1.0 \
  --g1_kin_loss_warmup_epochs 0 \
  --g1_kin_loss_max_fraction 0.0 \
  --skip_train_sample_render \
  --g1_mujoco_gl egl \
  2>&1 | tee -a setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_r01_20260601.log
```

## Evaluation Plan

- Full eval every 500 epochs, primarily `ckpt500` and `ckpt1000`.
- Compare against `gaussian_beat_1000`, Librosa35 2000, Wav2CLIP/STFT anchors, and current Wav2CLIP motion-control references.
- Primary rhythm metrics: `G1BAS`, `G1FKBAS`, `G1BeatF1`, beat precision, beat recall.
- Quality gates: `G1Dist`, `G1Div`, foot sliding, ground penetration, root drift, root angular/root-up metrics where available, joint/root range, and qualitative matched renders.
- Do not call the run successful from beat metrics alone if diversity, root behavior, range, or contact regresses.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-06-01 | spec | ready | this file; `docs/experiments/INDEX.md` | Preparing implementation and cache generation for the 8D beat-only ablation. |
| 2026-06-01 | implementation | passed | source changes in `feature_config.py`, `data/audio_extraction/beat_features_8d_features.py`, `data/create_dataset.py`, `test.py`, `submit_training_pipeline.py`; tests `.venv311/bin/python -m py_compile ...`, `.venv311/bin/python -m unittest tests.test_beat_features_8d_features tests.test_feature_config_and_fusion tests.test_validate_preprocessed_data tests.test_phase0_cli_and_preprocess`, `.venv311/bin/python -m unittest tests.test_submit_training_pipeline`; smoke extraction on one real wav produced `(150, 8)` finite `float32` | Added `feature_type=beat_features_8d`, extractor, CLI/test/pipeline wiring, and focused coverage. |
| 2026-06-01 | r01 pipeline launch | running | tmux `m2d_train_beat_features_8d`; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_r01_20260601.log`; feature cache target `data/finedance_g1_fkbeats/{train,test}/beat_features_8d_feats`; run dir target `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear` | User requested stopping v5 and running this first. Pipeline starts with train/test feature extraction, then validation, then 1000-epoch training. Initial train feature extraction reached `1054/47817` files after launch. |
| 2026-06-01 | r01 training | running | tmux `m2d_train_beat_features_8d`; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_r01_20260601.log`; W&B run `1s1l6y7o` at `https://wandb.ai/realroboticslab_tianhu/EDGE/runs/1s1l6y7o`; local W&B dir `wandb/run-20260601_114714-1s1l6y7o`; train features `47817`, test features `3265`; loaded train `(47817, 150, 3)/(47817, 150, 33)`, test `(3265, 150, 3)/(3265, 150, 33)`; `train_epoch=3` complete at `4.05` batch/s, ETA `2026-06-02 00:33:45` | Feature extraction completed, validation reached training without errors, and the 1000-epoch run is active with live tmux output. First planned durable checkpoint remains epoch 50; full eval remains scheduled at epochs 500 and 1000. |
| 2026-06-02 | r01 ckpt500 eval | running | checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear/weights/train-500.pt`; metrics `eval/EXP-20260601-finedance-g1-beat-features-8d_r01_linear/ckpt500_auto/metrics.json`; report `eval/EXP-20260601-finedance-g1-beat-features-8d_r01_linear/ckpt500_auto/paper_report.md`; latest log shows `train_epoch=653` complete, checkpoint `train-650.pt`, ETA `2026-06-02 00:37:35` | Midpoint metrics: `G1BAS=0.2131`, `G1FKBAS=0.2275`, `G1BeatF1=0.1911`, precision/recall `0.2961/0.1533`, `G1Dist=8.4624`, `G1Div=15.4475`, foot sliding `0.6910`, ground penetration `0.1178`, root drift `0.2655`, root-up p01 `0.8570`. Versus GaussianBeat 1000, paper-style `G1BAS` improves by `+0.0060` and `G1Dist` improves by `-0.7376`, but FK beat align drops `-0.0036`, F1 is flat `-0.0002`, recall drops `-0.0013`, diversity drops `-5.0894`, and foot/ground contact regress. Not a win at ckpt500; wait for ckpt1000 before final conclusion. |
| 2026-06-02 | r01 interruption check | blocked | no `tmux` server at `/tmp/tmux-1005/default`; no active train/accelerate process for `beat_features_8d`; log mtime `2026-06-01 20:12:00 +0100`; latest complete log line `train_epoch=653`; raw tail ends mid `Train 654/1000`; latest checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear/weights/train-650.pt` mtime `2026-06-01 20:09:17 +0100`; only full eval remains `ckpt500_auto` | Training appears to have been interrupted by machine shutdown after checkpoint 650. It is resumable from `train-650.pt`; no ckpt1000 or ckpt1000 eval exists yet. |
| 2026-06-03 | r01 result recheck | blocked | no `tmux` server; no new checkpoint beyond `train-650.pt`; eval dir still only contains `ckpt500_auto/metrics.json` and `ckpt500_auto/paper_report.md`; train log mtime remains `2026-06-01 20:12:00 +0100` and ends during `Train 654/1000` | No resume occurred after the shutdown. Current result is still the ckpt500 midpoint only; no final ckpt1000 result exists. |
| 2026-06-03 | completion watcher | stopped | tmux `m2d_watch_beat_features_8d` was started briefly, then stopped per user request; watcher script removed | User clarified to resume training directly and not use a watcher. |
| 2026-06-03 | r01 resume650 | running | tmux `m2d_train_beat_features_8d_resume650`; checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear/weights/train-650.pt`; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_resume650_20260603.log`; W&B run `qpa3hrpv` at `https://wandb.ai/realroboticslab_tianhu/EDGE/runs/qpa3hrpv`; command uses `--checkpoint .../train-650.pt --epoch_offset 650 --epochs 350`; pane shows `Train 651/1000` active | Resumed with training state restored from checkpoint and global epoch labels continuing to 1000. Full eval should trigger at global epoch 1000. |
| 2026-06-03 | r01 resume650 failure | blocked | no active tmux/train process; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_resume650_20260603.log`; W&B run `qpa3hrpv`; latest resume checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear2/weights/train-700.pt`; original checkpoint remains `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear/weights/train-650.pt`; log reaches complete `train_epoch=738` and crashes during `Train 739/1000`; traceback `ValueError: too many values to unpack (expected 0)` in `diffusion.ema.update_model_average(...)` via `torch.nn.Module.named_parameters` | Resume did not reach 1000. The run directory incremented to `_r01_linear2`, so `train-700.pt` was saved there. There is still no ckpt1000 or ckpt1000 eval; only validated quality metrics remain from `ckpt500_auto`. |
| 2026-06-03 | EMA resume fix | passed | code change in `model/diffusion.py`; test `tests/test_phase4_to_6_beat_integration.py::EmaTests`; commands `.venv311/bin/python -m py_compile model/diffusion.py EDGE.py` and `.venv311/bin/python -m unittest tests.test_phase4_to_6_beat_integration.EmaTests tests.test_phase4_to_6_beat_integration.AccumulationTrainingHelperTests tests.test_phase4_to_6_beat_integration.CheckpointRestoreTests` | Fixed the resume crash path by caching EMA model/current model parameter pairs after the first traversal, then reusing parameter references for later EMA updates. This avoids repeated recursive `parameters()` traversal during long resumed runs while preserving the same EMA update math. |
| 2026-06-03 | r01 resume700 after EMA fix | failed | tmux `m2d_train_beat_features_8d_resume700`; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_resume700_ema_cache_20260603.log`; checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear2/weights/train-700.pt`; W&B run `wspo4nbz`; log reached `Train 712/1000`; traceback `TypeError: 'AISTPPDataset' object is not callable` at `dataset/dance_dataset.py:465`, `feature = self._load_feature(idx)` inside a DataLoader worker | EMA crash did not recur, but the resumed worker failed on feature loading because an instance-level `_load_feature` attribute shadowed the class method. No new checkpoint beyond `train-700.pt`. |
| 2026-06-03 | dataset loader fix | passed | code change in `dataset/dance_dataset.py`; test `tests/test_phase2_dataset_and_estimator.py::DatasetBeatSchemaTests.test_getitem_loads_feature_when_instance_attribute_shadows_loader_method`; commands `.venv311/bin/python -m py_compile dataset/dance_dataset.py model/diffusion.py EDGE.py` and `.venv311/bin/python -m unittest tests.test_phase2_dataset_and_estimator.DatasetBeatSchemaTests.test_getitem_loads_feature_when_instance_attribute_shadows_loader_method tests.test_phase2_dataset_and_estimator.DatasetBeatSchemaTests.test_getitem_can_load_music_from_memmap_feature_cache tests.test_phase2_dataset_and_estimator.DatasetBeatSchemaTests.test_memmap_feature_cache_reopens_after_dataset_pickle_roundtrip tests.test_phase4_to_6_beat_integration.EmaTests` | Changed `AISTPPDataset.__getitem__` to call the feature loader from the class (`type(self)._load_feature(self, idx)`), so DataLoader workers are not broken if an instance attribute shadows the method. |
| 2026-06-03 | r01 resume700 loader-fix | running | tmux `m2d_train_beat_features_8d_resume700_loaderfix`; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_resume700_loaderfix_20260603.log`; checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_linear2/weights/train-700.pt`; run dir target `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_resume700_loader-fix`; W&B run `pq0es7c1` at `https://wandb.ai/realroboticslab_tianhu/EDGE/runs/pq0es7c1`; command uses `--checkpoint .../train-700.pt --epoch_offset 700 --epochs 300`; pane shows `Train 714/1000` active after passing the prior loader crash point at `Train 712/1000` | Resumed again after both EMA and dataset-loader fixes. Full eval should trigger at global epoch 1000; no final ckpt1000 result exists yet. Next durable checkpoint should be global epoch 750. |
| 2026-06-10 | r01 resume700 loader-fix status check | blocked | no tmux server; no active train/accelerate process; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_resume700_loaderfix_20260603.log`; run dir `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_resume700_loader-fix`; valid checkpoints `train-750.pt` and `train-800.pt`; attempted `train-850.pt` is corrupt (`416956416` bytes vs normal `1125960638` bytes; `torch.load` fails with `failed finding central directory`); repo filesystem `/dev/nvme0n1p2` is `99%` used with `23G` free and repo `runs/` alone is `216G`; eval still only has completed `ckpt500_auto/metrics.json`; an attempted `ckpt800_auto` eval was interrupted after producing 96 partial motion files and no `metrics.json` | Training reached a complete `train_epoch=850` log line, then crashed while saving `train-850.pt` with `RuntimeError: basic_ios::clear: iostream error` / `unexpected pos ...`. This is a checkpoint write/storage failure, not a numerical training collapse. Latest usable resume/eval checkpoint is `train-800.pt`; resume requires freeing disk or moving old runtime artifacts first. |
| 2026-06-10 | checkpoint save hardening | passed | code change in `EDGE.py`; command `.venv311/bin/python -m py_compile EDGE.py` | Added atomic checkpoint saving through a temporary file plus `os.replace`, with failed temporary files removed and disk free/total space reported in the raised error. This does not create space, but prevents future failed saves from leaving a corrupt file under the final checkpoint name. |
| 2026-06-10 | r01 resume800-to1000 b128 | finished | tmux `m2d_train_beat_features_8d_resume800_to1000_b128`; log `setup_logs/EXP-20260601-finedance-g1-beat-features-8d_train_resume800_to1000_b128_20260610.log`; W&B run `c7ghpim3` at `https://wandb.ai/realroboticslab_tianhu/EDGE/runs/c7ghpim3`; checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_resume800_to1000_b128/weights/train-1000.pt`; metrics `eval/EXP-20260601-finedance-g1-beat-features-8d_r01_resume800_to1000_b128/ckpt1000_auto/metrics.json`; report `eval/EXP-20260601-finedance-g1-beat-features-8d_r01_resume800_to1000_b128/ckpt1000_auto/paper_report.md` | Completed epoch 1000 and full eval on all `3265` test clips with `FiniteMotionRate=1.0` and `BadFileCount=0`. Final metrics: `G1BAS=0.2169`, `G1FKBAS=0.2298`, `G1BeatF1=0.1934`, precision/recall `0.2979/0.1551`, `G1Dist=9.1177`, `G1Div=14.4888`, foot sliding `0.6561`, ground penetration `0.1580`, root drift `0.2894`, root-up p01 `0.9020`. |

| 2026-06-15 | 90s 2x2 cache-audio render fix | passed | code/test changes in `eval/render_g1_checkpoint_comparison.py` and `tests/test_render_g1_checkpoint_comparison.py`; regenerated render `renders/EXP-20260601-finedance-g1-beat-features-8d/checkpoint_comparison_012_90s_seed1234_cache_8d_1d_v3b1500/comparison.mp4`; manifest records `audio_source=cache`, `slice_start=3`, `slice_step=5`, `effective_stride_seconds=2.5`, `comparison_layout=grid2x2` | The repeated-music render bug came from selecting consecutive cached wav slices spaced by 0.5s for a long stitched comparison that expects 2.5s slice stride. Long cache-backed comparison renders must step cached slices by 5, or use extracted full-song slices. Do not use older cache renders that have `slice_step` missing or equal to 1 for 90s comparison videos. |

## Current Conclusion

R01 finished, but it is not a candidate to promote as a main/default model. Compared with the old 1-D GaussianBeat 1000 baseline, the 8-D beat-only condition improves paper-style `G1BAS` (`0.2169` vs `0.2072`) and beat precision (`0.2979` vs `0.2889`), but the primary FK beat gate is slightly worse (`G1FKBAS=0.2298` vs `0.2311`) and beat recall is essentially unchanged (`0.1551` vs `0.1546`). The cost is large: diversity drops sharply (`G1Div=14.4888` vs `20.5369`), foot sliding worsens (`0.6561` vs `0.6015`), ground penetration roughly doubles (`0.1580` vs `0.0803`), and root drift worsens (`0.2894` vs `0.2709`). From ckpt500 to ckpt1000, rhythm improves only marginally while diversity and contact/root quality do not recover.

Against the strongest current Wav2CLIP-family checkpoint, v3b local predicted controls at checkpoint 1500, the 8-D run is clearly behind on the primary rhythm and distribution gates: `G1BAS=0.2169` vs `0.2435`, `G1FKBAS=0.2298` vs `0.2429`, `G1BeatF1=0.1934` vs `0.2106`, precision/recall `0.2979/0.1551` vs `0.3225/0.1687`, and `G1Dist=9.1177` vs `5.7822`. It has slightly higher `G1Div` (`14.4888` vs `14.0929`), lower foot sliding (`0.6561` vs `0.7639`), and lower root angular p99 (`3.8288` vs `5.9071`), but these do not compensate for much worse rhythm, distribution quality, and ground penetration (`0.1580` vs `0.0517`). The result is useful as a completed beat-only ablation and negative control, not as a practical replacement for richer music features.

## Next Action

Do not spend more GPU on this exact 8-D setup. Keep the checkpoint and metrics as a completed lower-bound beat-only ablation. If beat-only conditioning remains scientifically interesting, run a small condition-sensitivity follow-up for `real`, `shift_p10`, `random`, `constant_mean`, and `uncond` to test whether the model actually uses the extra beat-structure channels. For practical model quality, continue with richer features or the v3b/v5 line rather than replacing Wav2CLIP/Librosa conditioning with this 8-D feature alone.
