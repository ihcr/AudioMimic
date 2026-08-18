# Experiment Index

Use this ledger as the source of truth for nontrivial research, ablations, training runs, evaluations, and paper-to-method trials in this repo.

## Active Experiments

| ID | Status | Branch/Worktree | Core Change | Latest Artifact | Next Action |
|---|---|---|---|---|---|
| [EXP-20260513-finedance-g1-wav2clip-stft-beat](EXP-20260513-finedance-g1-wav2clip-stft-beat.md) | finished | `wav2clip-stft-beat` branch on local 4090 clone | FineDance+G1 Wav2CLIP/STFT/GaussianBeat feature replacement, concat_norm vs stream_adapter; r02 continuation to 2000 | r02 `train-2000.pt`; checkpoint sweep `eval/EXP-20260513-finedance-g1-wav2clip-stft-beat/comparison_r02_resume600_epoch_sweep.md`; cache-faithful render/audit `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_012_40s_seed1234_cache/`; five extract/MuJoCo comparisons `renders/EXP-20260513-finedance-g1-wav2clip-stft-beat/checkpoint_comparison_{001,003,010,014,022}_40s_seed1234_extract_stick/`; W&B run `firfyuhd` | Start a targeted amplitude/energy-preservation follow-up; do not rely on richer music conditioning alone to fix low-amplitude averaged motion |
| [EXP-20260524-finedance-g1-wav2clip-intensity-beatness](EXP-20260524-finedance-g1-wav2clip-intensity-beatness.md) | needs_decision | `wav2clip-stft-beat` branch on local 4090 clone | V3 typed Wav2CLIP+GaussianBeat controls: separate motion intensity/activity from motion beatness/local-min salience | cache `data/finedance_g1_fkbeats/motion_control_v2_metadata.json`; r03 checkpoint-1000 full eval `eval/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/r03_ckpt1000_pred_controls/metrics.json`; phase diagnostic `eval/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/r03_ckpt1000_pred_controls/phase_diagnostic.json`; 90s v3/r05/librosa render `renders/EXP-20260524-finedance-g1-wav2clip-intensity-beatness/checkpoint_comparison_012_90s_seed1234_extract_v3_r05_librosa/comparison.mp4`; `pred_controls` `G1FKBAS=0.2286`, `G1BeatF1=0.2050`, `G1Dist=6.0560`, `G1Div=18.4838`; W&B run `irbcya1y` | Review the 90s render; if it looks acceptable, resume from `train-1000.pt` to checkpoint 1500, then repeat full eval variants plus phase diagnostic; if beat recall stalls, start a focused stronger-beatness v3b ablation with robot-quality gates |
| [EXP-20260522-finedance-g1-wav2clip-motion-energy-beat](EXP-20260522-finedance-g1-wav2clip-motion-energy-beat.md) | needs_decision | `wav2clip-stft-beat` branch on local 4090 clone | Structured Wav2CLIP semantic condition plus GaussianBeat and learned BeatEnergyEnvelope control to counter low-amplitude averaged motion | Training intentionally stopped after epoch 1000; checkpoint `runs/train/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat_r05_same_model_fk_reuse2/weights/train-1000.pt`; oracle full eval `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_full/metrics.json`; pred-energy full eval `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/r05_ckpt1000_pred_energy_full/metrics.json`; comparison `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/comparison_r05_ckpt1000_pred_energy.md`; phase audit `eval/EXP-20260522-finedance-g1-wav2clip-motion-energy-beat/beat_phase_diagnostic_pred_energy_full.json`; W&B run `wvsdt0ly` | Do not resume this exact run as mainline; branch v3 with two typed motion-derived controls: intensity/activity for amplitude and beatness for speed-min/hold/turnaround rhythm, then rebuild feature/tensor caches and make predicted-control inference the default eval path |
| [EXP-20260520-finedance-g1-gaussian-beat](EXP-20260520-finedance-g1-gaussian-beat.md) | finished | `wav2clip-stft-beat` branch on local 4090 clone | FineDance+G1 pure GaussianBeat conditioning, all other r01 controls held fixed, 1000-epoch schedule | `runs/train/EXP-20260520-finedance-g1-gaussian-beat_r01_linear/weights/train-1000.pt`; full eval `eval/EXP-20260520-finedance-g1-gaussian-beat/r01_linear_1000/metrics.json`; benchmark `eval/EXP-20260520-finedance-g1-gaussian-beat/comparison_g1_metrics.md` | Use as a lower-bound beat-only baseline; do not promote over Librosa35 without a hybrid feature follow-up |
| [EXP-20260522-gaussian-beat-condition-ablation](EXP-20260522-gaussian-beat-condition-ablation.md) | finished | `wav2clip-stft-beat` branch on local 4090 clone | Inference-only ablation of pure GaussianBeat condition: real, shifted, random, constant, unconditional | `eval/EXP-20260522-gaussian-beat-condition-ablation/comparison_g1_metrics.md`; `condition_sensitivity_vs_no_beat_uncond.md`; `condition_sensitivity_vs_real.md` | Use GaussianBeat only as a weak lower-bound/auxiliary rhythm probe; design richer conditioning with condition-sensitivity checks |

## Stage Reports

| ID | Status | Scope | Latest Artifact | Next Action |
|---|---|---|---|---|
| [EXP-20260617-music2dance-progress-report](EXP-20260617-music2dance-progress-report.md) | active_summary | Music2Dance FineDance/G1/Wav2CLIP/beat-control/yaw-delta progress through V5, 8D beat-only, and DiscoForcing-inspired compound rhythm planning | Consolidated stage report dated 2026-06-17 | First build rhythm eval suite and `V6a_compound_rhythm_only_yaw_delta`; then add contact-aware beatness; add Wav2CLIP semantics only after rhythm controllability is verified |
| [NEXT-20260617-rhythm-eval-plan](NEXT-20260617-rhythm-eval-plan.md) | execution_plan | Student-facing next-step plan for rhythm/action evaluation before new training | Metric definitions, literature/project-origin notes, implementation steps, and pass criteria | Implement rhythm eval suite first; do not start new V6 training until the current models are re-evaluated with these metrics |

## Archived Experiments

| ID | Final Status | Main Conclusion | Key Artifacts |
|---|---|---|---|
| [EXP-20260512-wav2clip-stft-beat](EXP-20260512-wav2clip-stft-beat.md) | archived | AIST/SMPL launch was superseded by the corrected FineDance+G1 target; jobs were cancelled. | Slurm jobs `4575424`, `4575425`, `4575426` |

## Status Vocabulary

`idea`, `spec`, `ready`, `running`, `blocked`, `failed`, `needs_eval`, `needs_decision`, `finished`, `archived`

## Rules

- Read this file before starting, resuming, evaluating, or comparing experiment work.
- Keep one spec per experiment at `docs/experiments/EXP-YYYYMMDD-short-slug.md`.
- Include the experiment ID in run, Slurm, render, metric, and checkpoint paths when practical.
- Update this index whenever an experiment status, latest artifact, or next action changes.
