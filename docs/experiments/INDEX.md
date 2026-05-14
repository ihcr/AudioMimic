# Experiment Index

Use this ledger as the source of truth for nontrivial research, ablations, training runs, evaluations, and paper-to-method trials in this repo.

## Active Experiments

| ID | Status | Branch/Worktree | Core Change | Latest Artifact | Next Action |
|---|---|---|---|---|---|
| [EXP-20260513-finedance-g1-wav2clip-stft-beat](EXP-20260513-finedance-g1-wav2clip-stft-beat.md) | blocked | `wav2clip-stft-beat` branch | FineDance+G1 Wav2CLIP/STFT/GaussianBeat feature replacement, concat_norm vs stream_adapter | feature cache complete; r02 `train-500.pt`; r01 train `4576168` cancelled on `AssocGrpCPUMinutesLimit`; evals `4576166/4576169` cancelled | Move code/runtime artifacts to another server, then run r01 train and resubmit evals |

## Archived Experiments

| ID | Final Status | Main Conclusion | Key Artifacts |
|---|---|---|---|
| [EXP-20260512-wav2clip-stft-beat](EXP-20260512-wav2clip-stft-beat.md) | archived | AIST/SMPL launch was superseded by the corrected FineDance+G1 target; jobs were cancelled. | Slurm jobs `4575424`, `4575425`, `4575426` |

## Status Vocabulary

`idea`, `spec`, `ready`, `running`, `blocked`, `failed`, `needs_eval`, `finished`, `archived`

## Rules

- Read this file before starting, resuming, evaluating, or comparing experiment work.
- Keep one spec per experiment at `docs/experiments/EXP-YYYYMMDD-short-slug.md`.
- Include the experiment ID in run, Slurm, render, metric, and checkpoint paths when practical.
- Update this index whenever an experiment status, latest artifact, or next action changes.
