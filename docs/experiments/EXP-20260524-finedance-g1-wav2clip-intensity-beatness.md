# EXP-20260524-finedance-g1-wav2clip-intensity-beatness

Status: needs_decision
Owner: yukun
Created: 2026-05-24
Last Updated: 2026-05-26

## Research Question

Can separating motion-derived amplitude control from motion-beat control improve G1 music-to-dance rhythm without returning to averaged low-amplitude motion?

## Hypothesis

The previous `wav2clip_motion_energy_beat` run mixed two concepts in one signal: speed-max motion intensity and beat/hold salience. V3 should keep the useful amplitude signal as `motion_intensity` while adding a separate `motion_beatness` target built from FK-speed local minima/holds/turnarounds near audio beats. Predicted controls should be the main inference path; oracle controls are diagnostic only.

## Baseline Or Control

- Prior failed mixed-signal run: `EXP-20260522-finedance-g1-wav2clip-motion-energy-beat` r05 `train-1000.pt`.
- Strong beat-score baseline: Librosa35 2000.
- Distribution/diversity reference: GaussianBeat 1000.
- Rich feature reference: Wav2CLIP+STFT+GaussianBeat r02 1000/2000.

## Intervention

Add feature type `wav2clip_motion_intensity_beatness` with condition schema:

```python
cond = {
    "semantic": {"wav2clip": Tensor[B, 150, 512]},
    "control": {
        "gaussian_beat": Tensor[B, 150, 1],
        "motion_intensity": Tensor[B, 150, 1],
        "motion_beatness": Tensor[B, 150, 1],
    },
}
```

Model policy:

- `SemanticEncoder`: Wav2CLIP only.
- `ControlEncoder`: one compact typed encoder over GaussianBeat, predicted/GT motion intensity, and predicted/GT motion beatness.
- `MotionControlPredictor`: shared Wav2CLIP+GaussianBeat stem with two heads: `motion_intensity` and `motion_beatness`.
- No STFT, Librosa35, Jukebox, or old beat-distance loss in the main run.

## Data And Cache Contract

New cache:

- Metadata: `data/finedance_g1_fkbeats/motion_control_v2_metadata.json`.
- Train/test features: `data/finedance_g1_fkbeats/{train,test}/motion_control_v2_feats/*.npz`.
- Processed tensor cache: `data/finedance_g1_wav2clip_motion_intensity_beatness_dataset_backups`.

Required `.npz` fields:

- `motion_intensity_envelope`: `float32 [150, 1]`.
- `motion_beatness_envelope`: `float32 [150, 1]`.
- `weighted_fk_speed`: `float32 [150]`.
- `smoothed_weighted_fk_speed`: `float32 [150]`.
- `audio_beat_frames`: `int64 [N]`.
- `intensity_peaks`: `float32 [N]`.
- `beatness_peaks`: `float32 [N]`.

Generation command:

```bash
.venv311/bin/python -m data.audio_extraction.motion_control_v2_features \
  --data_path data/finedance_g1_fkbeats \
  --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml \
  --g1_root_quat_order xyzw \
  --batch_size 512 \
  --device cuda
```

Any change to feature semantics, normalization, keypoints, windows, or sigma requires deleting `motion_control_v2_feats` and `data/finedance_g1_wav2clip_motion_intensity_beatness_dataset_backups` before retraining.

## Training Plan

Train from scratch:

```bash
accelerate launch train.py \
  --feature_type wav2clip_motion_intensity_beatness \
  --feature_fusion linear \
  --motion_format g1 \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_wav2clip_motion_intensity_beatness_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r01 \
  --render_dir renders/EXP-20260524-finedance-g1-wav2clip-intensity-beatness \
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
  --lambda_motion_intensity 0.05 \
  --motion_intensity_norm_p05 0.16682735085487366 \
  --motion_intensity_norm_p95 2.747065782546997 \
  --lambda_motion_beatness 0.02 \
  --motion_beatness_warmup_start_epoch 100 \
  --motion_beatness_warmup_epochs 400 \
  --motion_beatness_max_fraction 0.1 \
  --lambda_energy_pred 1.0 \
  --energy_teacher_forcing_epochs 100 \
  --energy_pred_mix_prob 0.5 \
  --energy_smoothness_weight 0.1
```

## Evaluation Plan

Full eval every 500 epochs: `500`, `1000`, `1500`, `2000`.

Default/main variant: `--motion_energy_condition_variant pred_controls`.

Diagnostic variants per checkpoint:

- `oracle_controls`
- `flat_intensity`
- `zero_beatness`
- `zero_all_controls`

Track rhythm, distribution, and robot quality together: `G1BAS`, `G1FKBAS`, `G1BeatF1`, precision/recall, phase min/max diagnostics, `G1Dist`, `G1Div`, `JointPositionStdMean`, `JointPositionRangeMean`, `RootFlatRangeMean`, foot sliding, ground penetration, root drift, and jerk.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-24 | v3 implementation | ready | code paths for `wav2clip_motion_intensity_beatness`; tests pending in current implementation turn | Implements new cache schema, typed condition loader, two-head predictor, intensity/beatness losses, eval variants, and W&B metric names. Use `motion_control_v2_metadata.json` normalization values for `--motion_intensity_norm_p05/p95`. |
| 2026-05-24 | motion-control v2 cache | passed | `data/finedance_g1_fkbeats/motion_control_v2_metadata.json`; `data/finedance_g1_fkbeats/{train,test}/motion_control_v2_feats`; log `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_motion_control_v2_cache_20260524.log` | Generated `47817` train and `3265` test `.npz` files. Normalization: intensity p05 `0.1668273509`, p95 `2.7470657825`; beatness p05 `0.0022457482`, p95 `0.1129700840`; peak count `435660`. |
| 2026-05-24 | real-batch smoke | passed | command output in implementation turn; processed cache `data/finedance_g1_wav2clip_motion_intensity_beatness_dataset_backups` | Loaded train dataset with v3 structured condition and ran one `p_losses + backward` batch with normalization. Losses finite: intensity pred `0.0808`, beatness pred `0.1397`, intensity global `0.5307`, beatness valley `0.2075`. |
| 2026-05-24 | r01 launch | stopped | log `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_train_r01_20260524.log`; run dir `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r01`; W&B run `blpgtbwu` | Manually stopped at epoch 7 because stdout/stderr were redirected away from the tmux pane. Stable epochs were about `90-92s`; first epoch was `120.76s`. |
| 2026-05-24 | r02 attach/logging test | stopped | log `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_train_r02_20260524.log`; run dir `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r02`; W&B run `0i3ea6tq` | Relaunched with `tee` so tmux attach shows live logs. Per-batch metric accumulation moved to GPU and tqdm postfix lowered to every 10 batches. Epoch 2 was `89.92s`. Manually stopped to apply FK synchronization optimization. |
| 2026-05-24 | r03 optimized local run | stopped for eval | tmux `m2d_train_wav2clip_intensity_beatness`; log `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_train_r03_20260524.log`; run dir `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r03`; W&B run `05q2jasj`; checkpoint `weights/train-550.pt` | Uses live tmux logs through `tee`. `G1TorchKinematics` no longer calls CUDA `.item()` for fixed topology during forward. Epoch 1 `114.18s`; epoch 2 `88.32s`, `541.42 samples/s`, peak CUDA memory `16636.82 MB`. Training was stopped at epoch 564 to run the required 500 full eval; latest durable checkpoint is `train-550.pt`. |
| 2026-05-25 | r03 checkpoint 500 full eval | passed | main metrics `eval/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/r03_ckpt500_pred_controls/metrics.json`; phase diagnostic `eval/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/r03_ckpt500_pred_controls/phase_diagnostic.json`; variant metrics under `r03_ckpt500_{oracle_controls,flat_intensity,zero_beatness,zero_all_controls}`; logs `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/eval_r03_ckpt500_*_20260525.log` | Main `pred_controls`: `G1FKBAS=0.2189`, `G1BeatF1=0.1982`, precision `0.3264`, recall `0.1533`, `G1Dist=8.0263`, `G1Div=22.6877`, `JointPositionRangeMean=1.4667`, `RootFlatRangeMean=0.1722`, foot sliding `0.8444`, ground penetration `0.1583`. `zero_beatness` drops rhythm to `G1FKBAS=0.1931`/`F1=0.1733`, so beatness is being used. `zero_all_controls` collapses diversity/range (`G1Div=9.1583`, range `0.1062`), so controls are carrying anti-average motion. `oracle_controls` is close to `pred_controls`, so predictor quality is not the main bottleneck at 500. `flat_intensity` improves rhythm (`G1FKBAS=0.2508`, `F1=0.2241`) but worsens `G1Dist/contact`, suggesting intensity prediction/scale may be overdriving noisy amplitude and needs monitoring rather than an immediate architecture patch. |
| 2026-05-25 | r03 resume from 550 | running | tmux `m2d_train_wav2clip_intensity_beatness`; log `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_train_r03_resume550_20260525.log`; run dir `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r03_resume550`; checkpoint input `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r03/weights/train-550.pt`; W&B run `irbcya1y` | Relaunched with `--checkpoint train-550.pt`, `--epoch_offset 550`, and `--epochs 1450`, so the global counter resumed at epoch `551/2000`. Live tmux attach shows logs through `tee`. |
| 2026-05-26 | r03 checkpoint 1000 full eval | passed | checkpoint `runs/train/EXP-20260524-finedance-g1-wav2clip-intensity-beatness_r03_resume550/weights/train-1000.pt`; main metrics `eval/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/r03_ckpt1000_pred_controls/metrics.json`; phase diagnostic `eval/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/r03_ckpt1000_pred_controls/phase_diagnostic.json`; variant metrics under `r03_ckpt1000_{oracle_controls,flat_intensity,zero_beatness,zero_all_controls}`; logs `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/eval_r03_ckpt1000_*_20260526.log` | Training reached durable checkpoint 1000, then continued to about epoch 1030 before being stopped for full eval. Epoch 1000 took `87.64s`, `545.63 samples/s`, and reported global progress `1000/2000`. Main `pred_controls`: `G1BAS=0.2337`, `G1FKBAS=0.2286`, `G1BeatF1=0.2050`, precision `0.3285`, recall `0.1615`, `G1Dist=6.0560`, `G1Div=18.4838`, `JointPositionRangeMean=1.2945`, `RootFlatRangeMean=0.2282`, foot sliding `0.7462`, ground penetration `0.0483`, root drift `0.0656`. Compared with checkpoint 500, rhythm and robot/contact quality improved while diversity/range fell. `zero_beatness` drops rhythm to `G1FKBAS=0.2025`/`F1=0.1764`, confirming the beatness control is active. `zero_all_controls` is a near-static collapse (`G1Div=8.4875`, range `0.1070`, root range `0.0118`) despite deceptively high BAS, so beat metrics alone are not reliable. `flat_intensity` remains the rhythm upper diagnostic (`G1FKBAS=0.2506`, `F1=0.2237`) but has worse `G1Dist=9.2809` and contact/root metrics. |
| 2026-05-26 | 90s v3/r05/librosa comparison render | passed | comparison video `renders/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/checkpoint_comparison_012_90s_seed1234_extract_v3_r05_librosa/comparison.mp4`; manifest `renders/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/checkpoint_comparison_012_90s_seed1234_extract_v3_r05_librosa/manifest.json`; log `setup_logs/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/render_012_90s_v3_r05_librosa_20260526.log` | Rendered `012.wav` for 90s using `--feature_source extract` so custom-audio inference uses freshly extracted features, not cached dataset features. Compared `v3_1000` with `pred_controls`, `r05_1000` with `pred_energy`, and `librosa35_2000` with checkpoint-inferred beat conditioning. Verified output duration `90.006s` and audio stream AAC stereo 48 kHz. |
| 2026-05-26 | beat-score bottleneck review | analyzed | local metrics above; GaussianBeat ablation `docs/experiments/EXP-20260522-gaussian-beat-condition-ablation.md`; lbeat failure notes `docs/BEAT_ONLY_DEBUGGING_PLAYBOOK.md`; code paths `model/model.py`, `model/diffusion.py`, `eval/run_g1_dataset_eval.py` | The beatness control is not inert: zeroing it drops `G1FKBAS` by `0.0261` and F1 by `0.0286`. The remaining gap to Librosa35 is mostly recall/event density, not timing jitter: v3 pred has higher precision (`0.3285` vs Librosa35 `0.3157`) but lower recall (`0.1615` vs `0.1757`) and fewer FK beat events (`12992` vs `14598`). Stronger beat pressure alone is risky because prior normalized `Lbeat` achieved very high BAS by exploiting root/foot motion and destroying quality. The likely next design is stronger condition use plus robot-safe event pressure, not a simple larger beat-loss weight. |

## Current Conclusion

The v3 1000 checkpoint passes the preliminary acceptance gates against r05 and Wav2CLIP/STFT: `pred_controls` improves over r05 pred on rhythm (`G1FKBAS=0.2286` vs `0.2170`, `F1=0.2050` vs `0.1866`), keeps much higher diversity than STFT/Librosa35 (`G1Div=18.4838` vs `12.8445`/`11.3661`), and keeps distribution quality far better than GaussianBeat/Librosa35/STFT (`G1Dist=6.0560` vs about `8.9-9.3`). It is not a finished win: `G1FKBAS` is still below Librosa35 2000 (`0.2544`) and STFT 2000 (`0.2377`), foot sliding remains r05-like and worse than STFT/Gaussian/Librosa35, and diversity has fallen from checkpoint 500 (`22.6877` to `18.4838`). The strongest design signal is that zeroing beatness now hurts rhythm substantially, while zeroing all controls collapses motion amplitude; the two controls are doing real work. Phase diagnostics also moved in the intended local-min direction from 500 to 1000: weighted-speed `window_min_coverage_t2` rose from `0.1620` to `0.1727` and `window_min_soft_score` from `0.3083` to `0.3175`, while max alignment softened.

The beat-score bottleneck is probably a condition-strength and training-objective problem, but not in the naive sense of "turn beat loss up." Current v3 injects beatness through a compact three-channel control encoder plus mean summary, with predicted controls at inference. That is strong enough to move metrics, but still soft relative to the denoising prior, CFG dropout, and robot-quality losses. The comparison to Librosa35 suggests v3 is more selective and quality-biased: it gets higher beat precision but lower recall and fewer FK beat events. Prior `Lbeat` evidence shows that forcing more beat events without robot guards can inflate BAS through root/foot artifacts, so the next improvement should add beat-event recall pressure with explicit contact/root/quality gates.

## Next Action

Review the 90s comparison render, then decide whether to resume from `train-1000.pt` to checkpoint 1500. Recommendation: continue to 1500 only if the render does not show visually obvious averaging or contact artifacts, then run the same full eval variants plus phase diagnostic. If 1500 does not improve beat recall without shrinking diversity/range, start a focused v3b ablation: stronger/separate beatness condition path, reduced beatness/control dropout, event-recall loss on safe FK keypoints, and Pareto candidate selection against contact/root-quality metrics.
