# EXP-20260617-finedance-g1-beat8d-motion-beatness

Status: finished
Owner: yukun
Created: 2026-06-17
Last Updated: 2026-06-22

## Research Question

Can the failed old-`g1` `beat_features_8d` beat-only condition become useful when paired with a predicted root-local `motion_beatness` control, without reintroducing Wav2CLIP, STFT, `motion_intensity`, `--use_beats`, or the v5 yaw-delta representation?

## Hypothesis

The 8D beat-only run improved some paper-style beat precision but did not materially improve FK beat alignment and hurt diversity/contact. Adding a motion-derived `motion_beatness` control may give the model a usable speed-valley target around beats while preserving the clean beat-only input surface. Success requires better `G1FKBAS`, `G1BeatF1`, and recall than 8D-only without worsening the already weak `G1Div`, `G1Dist`, contact, or root behavior.

## Baseline Or Control

- Primary baseline: `EXP-20260601-finedance-g1-beat-features-8d`, checkpoint `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_resume800_to1000_b128/weights/train-1000.pt`.
- Strong Wav2CLIP reference: v3b local predicted controls checkpoint 1500 from `EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness`.
- This experiment is expected to beat 8D-only first; matching v3b is a secondary stretch goal.

## Intervention

Add feature type:

```text
feature_type=beat_features_8d_motion_beatness
feature_fusion=linear
motion_format=g1
```

Condition schema:

```text
semantic["beat_features_8d"]: Tensor[B, 150, 8]
control["motion_beatness"]: Tensor[B, 150, 1]
```

The decoder predicts `motion_beatness` from the 8D beat features. Training uses GT teacher forcing at the start, then mixes predicted beatness; deployable/custom-music inference uses `pred_controls`, not oracle GT.

## Invariant Controls

- Branch/worktree: `codex/wav2clip-stage-20260526` on `/home/tianhup/Desktop/Musics2Dance`.
- Dataset: `data/finedance_g1_fkbeats`, train `47817`, test `3265`.
- Motion format: legacy `g1`.
- Horizon/FPS: 150 frames, 30 FPS.
- No Wav2CLIP, no STFT, no `motion_intensity`, no `--use_beats`, no beat-estimator loss.
- W&B remains enabled.

## Data And Cache Contract

- 8D music cache:
  - `data/finedance_g1_fkbeats/train/beat_features_8d_feats`
  - `data/finedance_g1_fkbeats/test/beat_features_8d_feats`
- Motion beatness target source:
  - `data/finedance_g1_fkbeats/train/motion_control_v3_local_feats`
  - `data/finedance_g1_fkbeats/test/motion_control_v3_local_feats`
- Processed/tensor cache:
  - `data/finedance_g1_beat_features_8d_motion_beatness_dataset_backups`

Rebuild the processed/tensor cache if the 8D channel order, motion-control target formula, coordinate frame, or feature type schema changes.

## Implementation Scope

- Add `beat_features_8d_motion_beatness` to feature config and fusion validation.
- Add dataset support for the nested condition using `beat_features_8d_feats` plus `motion_control_v3_local_feats`.
- Add `BeatFeatures8DMotionBeatnessDecoder` and predictor.
- Add eval/render/full-song/test.py support so `auto` uses predicted beatness, with `oracle_controls`, `zero_beatness`, and `zero_all_controls` diagnostics.
- Add focused unit coverage and validator support.

## Training Plan

Validation:

```bash
.venv311/bin/python data/validate_preprocessed_data.py \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_beat_features_8d_motion_beatness_dataset_backups \
  --feature_type beat_features_8d_motion_beatness \
  --motion_format g1 \
  --sample_count 64
```

Training:

```bash
source .venv311/bin/activate
export MUJOCO_GL=egl
PYTHONUNBUFFERED=1 .venv311/bin/python -m accelerate.commands.launch train.py \
  --feature_type beat_features_8d_motion_beatness \
  --feature_fusion linear \
  --motion_format g1 \
  --lambda_beat 0.0 \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_beat_features_8d_motion_beatness_dataset_backups \
  --project runs/train \
  --exp_name EXP-20260617-finedance-g1-beat8d-motion-beatness_r01 \
  --render_dir renders/EXP-20260617-finedance-g1-beat8d-motion-beatness \
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
  --lambda_g1_kin 1.0 \
  --g1_kin_loss_warmup_epochs 0 \
  --g1_kin_loss_max_fraction 0.0 \
  --motion_energy_frame root_local \
  --motion_intensity_norm_p05 0.16980750858783722 \
  --motion_intensity_norm_p95 2.8171334266662598 \
  --lambda_motion_intensity 0.0 \
  --lambda_motion_beatness 0.02 \
  --motion_beatness_warmup_start_epoch 100 \
  --motion_beatness_warmup_epochs 400 \
  --motion_beatness_max_fraction 0.1 \
  --lambda_energy_pred 1.0 \
  --energy_teacher_forcing_epochs 100 \
  --energy_pred_mix_prob 0.5 \
  --energy_smoothness_weight 0.1 \
  --skip_train_sample_render \
  --g1_mujoco_gl egl \
  2>&1 | tee -a setup_logs/EXP-20260617-finedance-g1-beat8d-motion-beatness_train_r01_20260617.log
```

## Evaluation Plan

Run full eval at checkpoints 500 and 1000 with variants:

- `pred_controls`
- `oracle_controls`
- `zero_beatness`
- `zero_all_controls`

Primary gates versus 8D-only ckpt1000:

- `G1FKBAS >= 0.240`
- `G1BeatF1 >= 0.203`
- `G1BeatRecall >= 0.163`
- `G1Div >= 14.0`
- `G1Dist <= 8.5`
- `G1GroundPenetration <= 0.12`
- `zero_beatness` should be meaningfully worse than `pred_controls` on F1/FKBAS.

Do not call oracle GT a deployable win. If ckpt1000 fails the gates, stop and record a negative ablation. Extend to 1500 only if ckpt1000 passes.

## Rhythm Eval Suite

Implemented on 2026-06-22 before choosing the next rhythm/control training run. The suite keeps the existing G1/FK beat metrics and adds:

- Beat-density diagnostics: `G1MotionBeatDensity`, `G1AudioBeatDensity`, `G1BeatDensityRatio`, and `G1UnmatchedMotionBeatRate`.
- Body response diagnostics: wrist, foot, torso, and full-body beat precision/recall/F1, plus `G1WristDominanceRatio`.
- Robot support diagnostics: `G1FootContactOnBeatRate`, `G1NearSupportOnBeatRate`, `G1NoNearSupportRate`, `G1FootHighLiftRate`, `G1WristJerkMean`, and `G1FootJerkMean`.
- Per-file failure panels written to `failure_panel.json` with top examples for low beat recall, high unmatched motion beats, high penetration, no support, high lift, wrist jerk, and foot jerk.

Validation:

```bash
.venv311/bin/python -m py_compile \
  eval/g1_metrics.py \
  eval/run_g1_dataset_eval.py \
  eval/write_g1_metric_comparison.py \
  tests/test_g1_eval_metrics.py

.venv311/bin/python -m unittest \
  tests.test_g1_eval_metrics \
  tests.test_full_song_eval
```

Both commands passed on 2026-06-22.

ckpt1000 comparison artifacts:

- Summary JSON: `eval/EXP-20260617-finedance-g1-beat8d-motion-beatness_r01_b64_acc8/ckpt1000_rhythm_suite_summary.json`
- Comparison JSON: `eval/EXP-20260617-finedance-g1-beat8d-motion-beatness_r01_b64_acc8/ckpt1000_rhythm_suite_comparison.json`
- Comparison Markdown: `eval/EXP-20260617-finedance-g1-beat8d-motion-beatness_r01_b64_acc8/ckpt1000_rhythm_suite_comparison.md`

Key ckpt1000 results:

| Variant | G1FKBAS | G1BeatF1 | Recall | Density Ratio | Unmatched Motion | G1Dist | G1Div | Foot Sliding | Ground Pen. | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pred_controls` | 0.2629 | 0.2269 | 0.1882 | 0.5488 | 0.6633 | 10.2321 | 14.3124 | 0.6865 | 0.2168 | Rhythm/diversity gates pass; distribution and penetration fail. |
| `oracle_controls` | 0.2619 | 0.2281 | 0.1882 | 0.5408 | 0.6602 | 10.6069 | 14.3660 | 0.7087 | 0.1007 | Oracle does not fix distribution; not deployable and not a clean upper bound. |
| `zero_beatness` | 0.2305 | 0.1908 | 0.1523 | 0.5251 | 0.7089 | 10.1296 | 15.1875 | 0.6018 | 0.0376 | Beatness control is active; zeroing it clearly hurts rhythm. |
| `zero_all_controls` | 0.2547 | 0.2059 | 0.1690 | 0.5729 | 0.7042 | 14.3808 | 4.9273 | 0.8142 | 0.0216 | Control removal collapses diversity/range and worsens distribution. |

The suite supports the original gate logic: r01 is a useful rhythm-control proof but not a mainline checkpoint. Do not extend to 1500.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-06-17 | implementation | ready | source changes in feature config, dataset, model, eval/render/full-song/test.py, validator, and tests | Implements `beat_features_8d_motion_beatness` as a beat-only structured condition with predicted motion beatness. |
| 2026-06-17 | focused tests and data validation | passed | `.venv311/bin/python -m py_compile feature_config.py dataset/dance_dataset.py model/model.py EDGE.py eval/run_g1_dataset_eval.py eval/run_full_song_eval.py eval/render_g1_checkpoint_comparison.py test.py data/validate_preprocessed_data.py submit_training_pipeline.py`; `.venv311/bin/python -m unittest tests.test_feature_config_and_fusion tests.test_phase2_dataset_and_estimator tests.test_motion_energy_condition_variants tests.test_validate_preprocessed_data tests.test_phase0_cli_and_preprocess tests.test_full_song_eval tests.test_render_g1_checkpoint_comparison tests.test_submit_training_pipeline`; `.venv311/bin/python data/validate_preprocessed_data.py --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_beat_features_8d_motion_beatness_dataset_backups --feature_type beat_features_8d_motion_beatness --motion_format g1 --sample_count 64` | Py compile passed; focused unit suite ran `153` tests OK; real data validation passed with train `47817`, test `3265`, feature dirs `beat_features_8d_feats+motion_control_v3_local_feats`, and `beat_count=0`. Tiny train smoke was not run because the train CLI has no safe subset flag and `data_len` only truncates motion tensors, not filenames. |
| 2026-06-17 | r01_b64_acc8 launch | running | tmux `m2d_train_beat8d_motion_beatness_r01`; log `setup_logs/EXP-20260617-finedance-g1-beat8d-motion-beatness_train_r01_b64_acc8_20260617.log`; run dir `runs/train/EXP-20260617-finedance-g1-beat8d-motion-beatness_r01_b64_acc8`; W&B run `8cqxqz4i`; launched with direct `.venv311/bin/python -m accelerate.commands.launch train.py` | Current GPU had only about `10.5 GiB` free because a separate NaVILA VLM service was resident, so r01 uses `batch_size=64`, `gradient_accumulation_steps=8`, and `full_eval_batch_size=16` for stability while preserving effective batch `512`. Launch confirmed live tmux output, W&B sync, real train/test loading, motion-control store build, and active training through epoch 5 with no `Traceback`, CUDA OOM, `RuntimeError`, or MuJoCo `FatalError` in the log. `MUJOCO_GL=egl`, `--g1_mujoco_gl egl`, and `--skip_train_sample_render` are set. |
| 2026-06-22 | r01 ckpt1000 rhythm eval suite implementation | passed | `.venv311/bin/python -m py_compile eval/g1_metrics.py eval/run_g1_dataset_eval.py eval/write_g1_metric_comparison.py tests/test_g1_eval_metrics.py`; `.venv311/bin/python -m unittest tests.test_g1_eval_metrics tests.test_full_song_eval` | Added beat-density, body-response, support/contact, jerk, and `failure_panel.json` diagnostics to the G1 FK suite; `eval/run_g1_dataset_eval.py` now accepts `--failure_panel_path`; comparison writer includes the new fields. |
| 2026-06-22 | r01 ckpt1000 full saved-motion suite | finished | full eval directories under `eval/EXP-20260617-finedance-g1-beat8d-motion-beatness_r01_b64_acc8/ckpt1000_{pred_controls,oracle_controls,zero_beatness,zero_all_controls}/`; summary `ckpt1000_rhythm_suite_summary.json`; comparison `ckpt1000_rhythm_suite_comparison.md` | `pred_controls` passes rhythm gates but fails `G1Dist <= 8.5` and `G1GroundPenetration <= 0.12`; `zero_beatness` is much worse on FKBAS/F1, proving beatness control is active; `oracle_controls` does not solve distribution; `zero_all_controls` collapses diversity. Stop r01 at ckpt1000 and treat as a mixed/negative ablation. |
| 2026-06-22 | qualitative render comparison | regenerated | test-set comparison `renders/EXP-20260617-finedance-g1-beat8d-motion-beatness/checkpoint_comparison_012_90s_seed1234_extract_gt_8d_8dbeatness_v3b1500/comparison.mp4`; in-the-wild comparison `renders/in_the_wild/beatit_chorus_0058_40s_8d_8dbeatness_v3b1500/comparison.mp4` | Initial render showed repeated long-render motion; the same output directories were regenerated after the long-stitch fix. Root cause was hard overlap copying inside `long_ddim_sample`, which repeatedly forced each next window's first half to equal the previous window's second half during denoising. |
| 2026-06-22 | long-render repeat bug fix | passed | `.venv311/bin/python -m py_compile model/diffusion.py tests/test_phase4_to_6_beat_integration.py tests/test_g1_motion_format.py`; `.venv311/bin/python -m unittest tests.test_phase4_to_6_beat_integration tests.test_g1_motion_format`; overwritten test-set comparison `renders/EXP-20260617-finedance-g1-beat8d-motion-beatness/checkpoint_comparison_012_90s_seed1234_extract_gt_8d_8dbeatness_v3b1500/comparison.mp4`; overwritten in-the-wild comparison `renders/in_the_wild/beatit_chorus_0058_40s_8d_8dbeatness_v3b1500/comparison.mp4` | Removed denoising-time hard overlap from both `long_ddim_sample` and `long_inpaint_loop`, then aligned G1 root position during final stitch before fade blending. Regenerated renders use `feature_source=extract` and `audio_source=extract`. Test-set video is H.264/AAC stereo `1280x960`, `90.006s`; in-the-wild video is H.264/AAC stereo `1920x536`, `40.000s`. Saved-motion QA found no root reset at stitch boundaries: test-set model root-step maxima are `0.0464`, `0.0715`, `0.0562`; in-the-wild maxima are `0.0443`, `0.0451`, `0.0366`. |

## Qualitative Render Artifacts

- Test-set comparison with GT: `renders/EXP-20260617-finedance-g1-beat8d-motion-beatness/checkpoint_comparison_012_90s_seed1234_extract_gt_8d_8dbeatness_v3b1500/comparison.mp4`
- In-the-wild comparison: `renders/in_the_wild/beatit_chorus_0058_40s_8d_8dbeatness_v3b1500/comparison.mp4`
- Test-set manifest: `renders/EXP-20260617-finedance-g1-beat8d-motion-beatness/checkpoint_comparison_012_90s_seed1234_extract_gt_8d_8dbeatness_v3b1500/manifest.json`
- In-the-wild manifest: `renders/in_the_wild/beatit_chorus_0058_40s_8d_8dbeatness_v3b1500/manifest.json`
- Fixed-longstitch duplicate test-set comparison: `renders/EXP-20260617-finedance-g1-beat8d-motion-beatness/checkpoint_comparison_012_90s_seed1234_extract_gt_8d_8dbeatness_v3b1500_fixed_longstitch/comparison.mp4`
- Fixed-longstitch duplicate in-the-wild comparison: `renders/in_the_wild/beatit_chorus_0058_40s_8d_8dbeatness_v3b1500_fixed_longstitch/comparison.mp4`

## Intensity Follow-Up Decision

Do not add `motion_intensity` to r01. This run is meant to isolate whether 8D beat-structure features plus predicted `motion_beatness` can rescue the failed 8D-only beat ablation. Adding intensity now would change the research question into a broader v3-style motion-control experiment and make it harder to know whether any gain came from beat timing/control or from amplitude/range conditioning.

Prior branch evidence says `motion_beatness` is a real rhythm control: zeroing beatness in v3 lowered FK beat alignment and beat F1, while zeroing all controls collapsed diversity/range. That suggests intensity mainly belongs to motion amplitude, range, and anti-average behavior, not the first rhythm bottleneck. Add a separate second-stage `beat_features_8d_motion_intensity_beatness` only if r01 shows `pred_controls`/`oracle_controls` improve rhythm over 8D-only but `G1Div`, joint range, or qualitative energy remain too low. Do not add intensity if predicted beatness itself fails, if oracle is much better than pred, or if contact/root quality regresses.

Concrete prior metrics support keeping intensity out of this first run. In v3 r03 ckpt1000, `zero_beatness` dropped `G1FKBAS` from `0.2286` to `0.2025` and `G1BeatF1` from `0.2050` to `0.1764`. In v3b ckpt1500, `zero_beatness` dropped `G1FKBAS` from `0.2429` to `0.2171` and `G1BeatF1` from `0.2106` to `0.1869`. By contrast, v3b `flat_intensity` barely changed rhythm (`G1FKBAS 0.2429 -> 0.2441`, `G1BeatF1 0.2106 -> 0.2097`) and worsened ground penetration (`0.0517 -> 0.0993`). v3 r03 `flat_intensity` did raise beat metrics, but also worsened `G1Dist` (`6.0560 -> 9.2809`) and foot sliding (`0.7462 -> 0.8208`), so it is not a clean fix.

## Current Conclusion

r01 completed to `train-1000.pt` and is not worth extending. The intended `motion_beatness` signal is real: zeroing beatness drops `G1FKBAS` from `0.2629` to `0.2305` and `G1BeatF1` from `0.2269` to `0.1908`. However, the deployable `pred_controls` checkpoint fails the quality gates with `G1Dist=10.2321` and `G1GroundPenetration=0.2168`, and oracle controls do not produce a clean distribution fix.

The useful lesson is not "turn up beat loss" or "add intensity immediately." It is that beat-only 8D + predicted beatness can steer rhythm, but it still needs contact/support-aware pressure and a better robot-native quality gate before becoming a mainline direction.

## Next Action

Do not extend r01 to 1500. Use the new rhythm eval suite and failure panels as the default ckpt500/ckpt1000 gate for the next contact/support-aware V6a experiment. If revisiting intensity, run it as a separate second-stage ablation only after a support/contact mechanism is in place, because r01's main failure is distribution/contact quality rather than inert rhythm conditioning.
