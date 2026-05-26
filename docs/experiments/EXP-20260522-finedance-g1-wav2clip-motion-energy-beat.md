# EXP-20260522-finedance-g1-wav2clip-motion-energy-beat

Status: needs_decision
Owner: yukun
Created: 2026-05-22
Last Updated: 2026-05-24

## Research Question

Can a structured condition pipeline using Wav2CLIP semantics plus explicit beat-timed motion-energy control reduce the low-amplitude averaged-motion failure seen in richer audio-conditioned G1 models?

## Hypothesis

The current Wav2CLIP/STFT/GaussianBeat and Librosa35 checkpoints can follow dataset-level music structure but often choose conservative low-amplitude motion. A smaller, structured condition should help the denoiser keep useful semantic audio context without washing out the rhythm/control signal:

- `SemanticEncoder`: Wav2CLIP only.
- `ControlEncoder`: GaussianBeat plus BeatEnergyEnvelope.
- `EnergyPredictor`: Wav2CLIP plus GaussianBeat predicts BeatEnergyEnvelope for inference.
- `EnergyFiLM`: pooled BeatEnergyEnvelope directly modulates timestep conditioning.

The BeatEnergyEnvelope is a training-derived motion-intensity target, not a music feature. It is meant to teach the model when high-energy dance motion is expected for a beat, then let the predictor infer that control signal from audio at inference time.

## Baseline Or Control

- Main baseline: `EXP-20260513-finedance-g1-wav2clip-stft-beat` r02 stream-adapter continuation at epoch 2000.
- Lower-bound amplitude reference: `EXP-20260520-finedance-g1-gaussian-beat` epoch 1000.
- Legacy handcrafted reference: FineDance+G1 Librosa35 2000.
- Diagnostic ablation: `EXP-20260522-gaussian-beat-condition-ablation`, which showed pure GaussianBeat is weak as exact timing control but useful as an auxiliary rhythm probe.

## Intervention

Add feature type `wav2clip_motion_energy_beat` with a structured condition dict:

```python
cond = {
    "semantic": {"wav2clip": Tensor[B, 150, 512]},
    "control": {
        "gaussian_beat": Tensor[B, 150, 1],
        "beat_energy_envelope": Tensor[B, 150, 1],
    },
}
```

Model design:

- Semantic encoder: input 512, hidden 512, 2 layers, 8 heads.
- Control encoder: input 2, hidden 256, 2 layers, 4 heads, output 512.
- Energy predictor: input Wav2CLIP plus GaussianBeat, hidden 256, 2 layers, output `[B, 150, 1]`.
- Denoiser conditioning: `memory = semantic_tokens + control_tokens + time_tokens`; FiLM/time vector is `time_embed + semantic_hidden + control_hidden + energy_hidden`.

Training policy:

- Train from scratch; no checkpoint migration from concat or stream-adapter models.
- Target 2000 epochs, with checkpoints every 50 and evaluation anchors at 500, 1000, 1500, and 2000.
- Epochs 1-100: denoiser sees GT BeatEnergyEnvelope.
- Epochs 101+: per sample, 50% GT envelope and 50% detached predicted envelope.
- Keep ordinary classifier-free guidance and branch dropout; defer branch-specific CFG to a later experiment.
- Do not use STFT, lbeat estimator, beat estimator loss, or full FK tracking loss in this mainline.
- Keep a light G1 foot/contact auxiliary term only if smoke training remains stable.
- Add global motion-energy preservation loss: compare generated global motion energy against the selected BeatEnergyEnvelope mean with `lambda_motion_energy=0.05`.
- Add energy predictor loss: MSE to GT envelope plus `0.1` temporal smoothness.

## Invariant Controls

- Worktree: `/home/tianhup/Desktop/Musics2Dance`.
- Branch: `wav2clip-stft-beat`.
- Git commit at spec time: `64425f4` plus local uncommitted experiment changes.
- Environment: repo-local `.venv311`; do not use shared `yukun` Conda.
- Hardware: direct-attached RTX 4090, no Slurm.
- Dataset: `data/finedance_g1_fkbeats`.
- Train/test clips: expected `47817` / `3265`.
- Motion format: Unitree G1, 150 frames, 30 FPS, 38-D encoded motion.
- Backbone: current Transformer diffusion denoiser.
- Batch policy: start with 4090-stable microbatch `256` and `gradient_accumulation_steps=2`.
- W&B: keep enabled unless explicitly disabled by environment/user.

## Data And Cache Contract

Existing reused caches:

- Wav2CLIP source: first 512 dims of `data/finedance_g1_fkbeats/{train,test}/wav2clip_stft_beat_feats/*.npy`.
- GaussianBeat source: `data/finedance_g1_fkbeats/{train,test}/gaussian_beat_feats/*.npy`.

New motion-energy cache:

- Metadata: `data/finedance_g1_fkbeats/motion_energy_metadata.json`.
- Train features: `data/finedance_g1_fkbeats/train/motion_energy_feats/*.npz`.
- Test features: `data/finedance_g1_fkbeats/test/motion_energy_feats/*.npz`.
- Required fields:
  - `beat_energy_envelope`: `float32 [150, 1]`
  - `weighted_fk_speed`: `float32 [150]`
  - `audio_beat_frames`: `int64 [N]`
  - `beat_energy_peaks`: `float32 [N]`

BeatEnergyEnvelope generation:

- Compute weighted G1 FK keypoint speed from GT motion.
- Keypoints and weights:
  - `left_wrist_yaw_link`: `0.35`
  - `right_wrist_yaw_link`: `0.35`
  - `left_ankle_roll_link`: `0.10`
  - `right_ankle_roll_link`: `0.10`
  - `torso_link`: `0.10`
- Exclude root position and pelvis.
- For each audio beat frame, take local max of weighted FK speed in beat +/- 6 frames.
- Normalize by train global p05-p95 into `[0, 1]`, clamp.
- Expand beat peaks into a 150-frame Gaussian envelope with `sigma=5` frames.
- Keep `energy_peak_mode=p75` as an ablation option, not the main run.

Cache invalidation:

- New feature semantics require a new processed/tensor cache directory: `data/finedance_g1_wav2clip_motion_energy_beat_dataset_backups`.
- If motion-energy metadata or envelope generation changes, delete both `motion_energy_feats` and this processed/tensor cache before retraining.

## Training Or Execution Plan

Motion-energy cache command:

```bash
.venv311/bin/python -m data.audio_extraction.motion_energy_features \
  --data_path data/finedance_g1_fkbeats \
  --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml \
  --g1_root_quat_order xyzw \
  --batch_size 512 \
  --device cuda
```

Training command shape:

```bash
accelerate launch train.py \
  --feature_type wav2clip_motion_energy_beat \
  --feature_fusion linear \
  --motion_format g1 \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_wav2clip_motion_energy_beat_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r01_structured \
  --render_dir renders/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat \
  --batch_size 256 \
  --gradient_accumulation_steps 2 \
  --epochs 2000 \
  --save_interval 50 \
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
  --lambda_motion_energy 0.05 \
  --motion_energy_norm_p05 0.16682735085487366 \
  --motion_energy_norm_p95 2.747065782546997 \
  --lambda_energy_pred 1.0 \
  --energy_teacher_forcing_epochs 100 \
  --energy_pred_mix_prob 0.5 \
  --energy_smoothness_weight 0.1
```

Use local `tmux` for the full run:

- Session: `m2d_train_wav2clip_motion_energy_beat`.
- Log: `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_r01_structured_20260522.log`.
- Run dir: `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r01_structured/`.

## Evaluation Plan

For checkpoints `500`, `1000`, `1500`, and `2000`, run full G1 dataset evaluation by default. Sample renders do not count as evaluation evidence. If the only GPU is occupied by training, record the checkpoint as eval-pending and run the eval at the next GPU-safe window.

For each 500-epoch checkpoint, evaluate:

- `full_pred_energy`
- `oracle_gt_energy`
- `flat_energy`
- `shifted_energy_p10`
- `no_control`

Metrics:

- Rhythm: `G1BAS`, `G1FKBAS`, `G1BeatF1`, `G1RoboPerformBAS`, `G1FKRoboPerformBAS`.
- Motion quality: `G1Dist`, `G1Div`, `G1FootSliding`, `G1GroundPenetration`, `RootDriftMean`, jerk metrics.
- Anti-average amplitude: `JointPositionStdMean`, `JointPositionRangeMean`, `RootFlatRangeMean`.
- Paired condition sensitivity.
- 40s qualitative render against the same music and seed as the previous three-model comparison.

Expected output root: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/`.

Required checkpoint eval paths:

- `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt0500_full/`
- `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_full/`
- `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1500_full/`
- `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt2000_full/`

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-22 | spec | spec | this file; index row | Design accepted. Next step is implementation and cache generation. |
| 2026-05-22 | motion-energy cache | passed | `data/finedance_g1_fkbeats/motion_energy_metadata.json`; `data/finedance_g1_fkbeats/{train,test}/motion_energy_feats`; log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/motion_energy_cache_20260522.log` | Generated `47817` train and `3265` test `.npz` files. Train normalization stats: p05 `0.1668273509`, p95 `2.7470657825`, `435660` beat peaks. |
| 2026-05-22 | implementation smoke | passed | `feature_config.py`; `dataset/dance_dataset.py`; `model/model.py`; `model/diffusion.py`; `EDGE.py`; tests `tests.test_feature_config_and_fusion`, `tests.test_motion_energy_features`, `tests.test_phase4_to_6_beat_integration`, `tests.test_phase7_eval_and_presets` | Added structured condition loading, Wav2CLIP semantic encoder, GaussianBeat+BeatEnergyEnvelope control encoder, energy predictor, normalized motion-energy loss, lambda-acc ramp, and W&B energy metrics. Real dataset smoke returned nested condition shapes `[150,512]`, `[150,1]`, `[150,1]`. Batch-256 4090 forward/backward passed with peak CUDA memory `14928.77 MB`; normalized motion-energy contribution was `0.1207` and energy-predictor contribution was `0.1308`. |
| 2026-05-23 | r01_structured launch | stopped | tmux `m2d_train_wav2clip_motion_energy_beat`; log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_r01_structured_20260522.log`; run dir `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r01_structured`; W&B run `1csxc8sr` | Stopped before first checkpoint for speed audit. Stable epochs were around `105-110s/epoch`, much slower than prior Wav2CLIP r02. |
| 2026-05-23 | speed audit | passed | `model/diffusion.py`; tests `tests.test_feature_config_and_fusion`, `tests.test_motion_energy_features`, `tests.test_phase4_to_6_beat_integration`, `tests.test_phase7_eval_and_presets` | Kept model architecture and condition tokens unchanged. Optimized only duplicate hot-path G1 FK work: `model_out` FK is reused between G1 foot loss and motion-energy preservation loss, and zero-weight FK/FK-vel/FK-acc diagnostic losses are skipped for the current command. This preserves the active loss terms. |
| 2026-05-23 | r04_aligned_token_fusion | stopped | log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_r04_aligned_token_fusion_20260523.log`; W&B run `1k82dhpz` | This run briefly tested summing same-frame semantic/control tokens to reduce decoder cross-attention length. It improved early speed but changed the model/conditioning architecture, so it was stopped and reverted before any checkpoint. Do not use this as the main run. |
| 2026-05-23 | r05_same_model_fk_reuse | interrupted | log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_r05_same_model_fk_reuse_20260523.log`; run dir `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse`; W&B run `jq6v42kh`; checkpoint `weights/train-50.pt` | Reached epoch 50 and saved a valid `1.2G` checkpoint. Stable speed after epoch 1 was `89.72s/epoch` mean, `89.28s/epoch` median, last-10 mean `89.29s/epoch`, peak CUDA memory `16438 MB`. Training then crashed during sample video render because MuJoCo imported before `MUJOCO_GL` was set and failed with `gladLoadGL error`; one sample `.pkl` was written before the video failure. |
| 2026-05-23 | r05_resume50 | running | tmux `m2d_train_wav2clip_motion_energy_beat`; log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_r05_resume50_20260523.log`; checkpoint input `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse/weights/train-50.pt`; continuation run dir `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2`; W&B run `wvsdt0ly` | Resumed with `MUJOCO_GL=egl`, `--checkpoint train-50.pt`, `--epoch_offset 50`, and `--epochs 1950`, so global training continues at epoch 51/2000. A direct EGL renderer smoke test passed before relaunch. |
| 2026-05-24 | r05_resume50 status | running | checkpoint `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2/weights/train-1000.pt`; log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_r05_resume50_20260523.log`; sample renders under `renders/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2/` | Training passed epoch 1000 and was active around epoch 1014. Last stable epoch summaries around 991-1010 remained finite, with `~89-92s/epoch`, `~519-536 samples/s`, peak CUDA memory `16654 MB`, and ETA around `2026-05-25 14:00`. No formal full-dataset eval metrics exist yet for this checkpoint; a CPU smoke eval was abandoned because it was too slow and would not be comparable. |
| 2026-05-24 | r05 500/1000 full eval backlog | pending | checkpoints `train-500.pt`, `train-1000.pt`; required eval dirs `r05_ckpt0500_full`, `r05_ckpt1000_full` | User clarified that future long runs need full eval every 500 epochs. Because the single 4090 is currently occupied by active training, do not launch competing full eval immediately. Backfill full eval for 500 and 1000 at the next GPU-safe window, then run 1500 and 2000 full eval as those checkpoints land. |
| 2026-05-24 | stop at 1000 and eval | passed | stopped tmux `m2d_train_wav2clip_motion_energy_beat`; checkpoint `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2/weights/train-1000.pt`; eval log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/eval_r05_ckpt1000_full_20260524.log`; metrics `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_full/metrics.json`; report `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_full/paper_report.md`; comparison `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/comparison_r05_ckpt1000.md` | User requested stopping the active run to inspect 1000 performance. Full eval completed on all `3265` clips with FK metrics and 8 diagnostic renders. Compared against Wav2CLIP/STFT r02 1000/2000, GaussianBeat 1000, and Librosa35 2000. |
| 2026-05-24 | beat-score diagnosis | passed | GT/reference audit `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/reference_gt_beat_audit.json`; phase audit `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/beat_phase_diagnostic_full.json`; eval code `eval/eval_bas_bap.py`, `eval/g1_metrics.py`; train log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/train_r05_resume50_20260523.log` | Low beat score is primarily a music-to-motion coverage/recall issue, not random timing: V2 has comparable motion-to-music RoboPerform BAS but fewer FK motion beat minima per clip than older feature baselines. The current run uses GaussianBeat as condition only (`use_beats=False`, `lambda_beat=0.0`), and the motion-energy loss matches global mean energy while the BeatEnergyEnvelope target stores speed maxima around beats. This is a definition error for the current BAS/F1 objective, which detects speed minima/holds near audio beats. |
| 2026-05-24 | condition-ablation diagnosis | passed | eval hook `eval/run_g1_dataset_eval.py --motion_energy_condition_variant`; 512-clip summary `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/condition_ablation_512/condition_ablation_512_summary.md`; predictor audit `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/energy_predictor_diagnostic_full.json`; logs `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/condition_ablation_512_*_20260524.log` | Current full eval path is oracle-GT-energy, because cached test `beat_energy_envelope` is passed directly unless the new variant flag overrides it. On the same 512 clips, `flat_energy` improved `G1FKBAS/F1` over `oracle_gt` (`0.2377/0.1767` vs `0.1890/0.1519`) while preserving diversity, proving the local speed-max envelope is hurting min-based rhythm alignment. `zero_energy` and `zero_control` raised beat scores but collapsed distribution/diversity, so deleting the control signal is not a valid fix. |
| 2026-05-24 | terminology correction | decided | this spec; user correction | The old `motion_energy` implementation is not a valid motion-beat representation. It is better named `motion_intensity` or `motion_activity`, because it encodes speed magnitude/maxima. For this project, "motion energy" should mean motion-beat salience/beatness: a high value when the motion forms a beat event, i.e. a speed local-minimum/hold/turnaround with enough surrounding motion contrast. Do not reuse the old `motion_energy_v1` cache as a rhythm target. |
| 2026-05-24 | r05 1000 pred-energy full eval | passed | log `setup_logs/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/eval_r05_ckpt1000_pred_energy_full_20260524.log`; metrics `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_pred_energy_full/metrics.json`; phase audit `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/beat_phase_diagnostic_pred_energy_full.json`; comparison `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/comparison_r05_ckpt1000_pred_energy.md` | Inference-style predicted intensity slightly improves beat alignment over oracle cached intensity (`G1BAS=0.2110`, `G1FKBAS=0.2170`, `G1BeatF1=0.1866`) and improves `G1Dist=5.3518`, but lowers `G1Div=18.0232` from oracle `19.4489`. It still trails Wav2CLIP/STFT, GaussianBeat, and Librosa35 on `G1FKBAS/F1`. Phase audit remains more max-like than min-like, so the old target should remain only an intensity/activity control; a separate true motion-beatness control is needed. |

## Results

- Active W&B run: `https://wandb.ai/realroboticslab_tianhu/EDGE/runs/wvsdt0ly`.
- First checkpoint expected at epoch 50:
  `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse/weights/train-50.pt`.
  Continued checkpoints after resume are expected under
  `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2/weights/`.
  Current verified anchor: `train-1000.pt`.
- Full eval complete for `train-1000.pt`:
  - Metrics: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_full/metrics.json`.
  - Report: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_full/paper_report.md`.
  - Benchmark comparison: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/comparison_r05_ckpt1000.md`.
  - Diagnostic renders: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_full/renders/`.
- Inference-style predicted-energy full eval complete for `train-1000.pt`:
  - Metrics: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_pred_energy_full/metrics.json`.
  - Report: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_pred_energy_full/paper_report.md`.
  - Benchmark comparison: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/comparison_r05_ckpt1000_pred_energy.md`.
  - Phase audit: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/beat_phase_diagnostic_pred_energy_full.json`.
  - Diagnostic renders: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_pred_energy_full/renders/`.
- GT/reference beat audit: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/reference_gt_beat_audit.json`. The raw test motions score `G1BAS=0.2106`, `G1FKBAS=0.2150`, `G1BeatF1=0.1791`, `G1BeatPrecision=0.2875`, and `G1BeatRecall=0.1423`, so the absolute beat numbers are strict even for dataset/reference motion under the current G1/FK detector.
- Beat phase diagnostic: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/beat_phase_diagnostic_full.json`. For V2 mean FK speed, nearest local-min soft score is `0.2105`, nearest local-max soft score is `0.2288`; for the GT motion-energy cache weighted speed, nearest local-min soft score is `0.2057`, nearest local-max soft score is `0.2324`. The current target is therefore closer to high-speed accents than to beat-min holds.
- Energy predictor diagnostic: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/energy_predictor_diagnostic_full.json`. The predictor is not the primary cause of low min-based beat score: predicted energy has higher peak-to-audio soft score than GT cached energy (`0.3864` vs `0.3538`), but the full eval previously used GT energy by default rather than the predictor.
- 512-clip condition ablation: `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/condition_ablation_512/condition_ablation_512_summary.md`.
- `train-1000.pt` headline metrics: `G1BAS=0.2055`, `G1FKBAS=0.2120`, `G1RoboPerformBAS=0.4318`, `G1FKRoboPerformBAS=0.4301`, `G1BeatF1=0.1850`, `G1Dist=5.7779`, `G1Div=19.4489`, `JointPositionStdMean=0.3389`, `JointPositionRangeMean=1.2607`, `RootFlatRangeMean=0.1759`.
- `train-1000.pt` predicted-intensity full-eval headline metrics: `G1BAS=0.2110`, `G1FKBAS=0.2170`, `G1RoboPerformBAS=0.4326`, `G1FKRoboPerformBAS=0.4353`, `G1BeatF1=0.1866`, `G1Dist=5.3518`, `G1Div=18.0232`, `JointPositionStdMean=0.3504`, `JointPositionRangeMean=1.2998`, `RootFlatRangeMean=0.1711`.
- Full eval pending: `train-500.pt`.

## Current Conclusion

Training was intentionally stopped after passing epoch 1000 so the checkpoint could be evaluated. The 1000 checkpoint is healthy and no longer looks collapsed by `G1Dist`/diversity: it has much lower `G1Dist` than Wav2CLIP/STFT r02 1000/2000, GaussianBeat 1000, and Librosa35 2000, and its `G1Div=19.4489` is close to the pure GaussianBeat amplitude/diversity regime.

The beat score should be interpreted carefully. Absolute G1/FK beat numbers are low even for GT/reference motion under the current detector. Against model baselines, V2 is weak mainly because it under-covers music beats: FK motion beats are `3.99` per clip versus `4.33-4.52` for Wav2CLIP/STFT r02 and `4.47` for Librosa35, with lower recall (`0.1447`) but competitive precision (`0.3007`). Its motion-to-music scores (`G1RoboPerformBAS=0.4318`, `G1FKRoboPerformBAS=0.4301`) are comparable to r02, so produced accents are often near some audio beat; there are just too few detected motion beat minima to cover the audio beats.

The design issue is now confirmed as a target-definition error, not just a weak objective. This run uses GaussianBeat plus BeatEnergyEnvelope as condition, but no explicit beat-distance/lbeat loss (`use_beats=False`, `lambda_beat=0.0`). The BeatEnergyEnvelope target is built from local speed maxima around audio beats, while the reported BAS/F1 metrics detect smoothed local speed minima/holds. More importantly, the name `motion_energy` was wrong for the intended concept: the implemented signal represents motion intensity/activity, not motion-beat energy. For the next branch, "motion energy" should be defined as motion beatness: a high value at FK-speed local minima/holds/turnarounds with enough surrounding motion contrast. The old `motion_energy_v1` cache should be treated as an amplitude diagnostic only and must not be reused as the rhythm target.

There is also an eval/inference contract issue: the full `r05_ckpt1000_full` eval used cached GT BeatEnergyEnvelope, not predicted energy. Keep oracle-GT-energy only as a diagnostic variant. Default future inference-style eval for this feature type should use `--motion_energy_condition_variant pred_energy` or the successor predictor path.

The predicted-energy full eval confirms that using the predictor does not solve the definition error. It slightly improves beat metrics and `G1Dist` over oracle cached intensity, but still trails older feature baselines on `G1FKBAS/F1`, and the phase audit remains more aligned to speed maxima than speed minima. The useful signal should therefore be split into two typed motion-derived controls: an intensity/activity envelope for amplitude and distribution, and a true motion-beatness envelope for beat minima/holds/turnarounds.

## Next Action

Do not resume this exact run as the mainline. Branch a v3 follow-up that uses two clearly named motion-derived controls: `motion_intensity` for amplitude/activity preservation and `motion_beatness` for speed local-minimum/hold/turnaround salience. Rebuild the affected feature and tensor caches under a new cache version, keep Wav2CLIP plus GaussianBeat as the minimal external music condition, and evaluate predicted-control inference by default. Split encoders only if the first typed-channel ablation shows interference.
