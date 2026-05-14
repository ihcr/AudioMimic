# Experiment Index

Synced on 2026-05-12 UTC from local diffusion-worktree evidence only. This is not a full historical reconstruction. Rows below are included only when there is a local spec, metric file, Slurm log, sbatch script, or scheduler evidence.

## Active Experiments

| ID | Status | Branch/Worktree | Core Change | Latest Verified Artifact | Next Action |
|---|---|---|---|---|---|
| [EXP-20260511-finedance-g1-librosa-fullctx](EXP-20260511-finedance-g1-librosa-fullctx.md) | finished | `diffusion` / `.worktrees/diffusion` | FineDance-G1 Librosa35/Librosa34 full-song-context feature ablation plus matched 30s G1 robot renders for Jukebox/Librosa lbeat comparison | evals complete for jobs `4556032`, `4556035`, `4556038`, `4556029`, and `4545750`; 30s render-only job `4570136` completed with 9 MP4s in `slurm/pipelines/20260512-finedance-g1-30s-renders/` | Inspect matched 30s videos before selecting by rhythm scores alone |
| [EXP-20260512-finedance-g1-fk-jerk-audit](EXP-20260512-finedance-g1-fk-jerk-audit.md) | finished | `diffusion` / `.worktrees/diffusion` | Audit whether FK caused jerky/dashing FineDance-G1 inference videos | local motion-trace audit found lbeat+robotloss 5s eval clips already have root-step/root-acceleration spikes; final long-generation behavior restored to original EDGE overlap-copy timing after seam-dash render check | Keep clean anchors; only try lbeat as a low-weight finetune with root-step/root-height gates |
| [EXP-20260512-music-conditioning-redesign](EXP-20260512-music-conditioning-redesign.md) | blocked | `wav2clip-stft-beat` / `.worktrees/wav2clip` for Stage A | New music-conditioning direction: Wav2CLIP+STFT Stage A, two-stream conditioner Stage B, optional primitive/planner Stage C | Stage A moved to `.worktrees/wav2clip`; feature extraction complete and `stream_adapter` trained to 500 epochs, but `concat_norm`/eval are blocked by Slurm quota | Continue Stage A on another server/account using `.worktrees/wav2clip/HANDOFF.md` |

## Needs Triage Before Promotion

These runs are real local jobs, but this sync did not reconstruct enough intent/history to create authoritative specs. Do not fold them into an experiment conclusion until their comparison target and acceptance criteria are clear.

| Observed Run | Status Observed | Evidence | Why Not Promoted |
|---|---|---|---|
| none currently | n/a | n/a | n/a |

## Verified Historical Anchors

These are anchors for comparison, not full reconstructed specs.

| Run | Role | Key Metrics | Evidence |
|---|---|---|---|
| `20260504-g1-aist-beatdistance-fkbeats` | stable AIST G1 FKBeat anchor | `G1BAS=0.3320`, `G1RoboPerformBAS=0.5894`, `G1FKBAS=0.3497`, `G1FKRoboPerformBAS=0.5502`, `G1BeatF1=0.3333`, `G1FootSliding=0.6112`, `G1Dist=6.8652` | `slurm/pipelines/20260504-g1-aist-beatdistance-fkbeats/eval/metrics.json` |
| `20260506-g1-aist-fkbeats-lbeat-relative-scratch-lam020-cap1c` | rejected AIST lbeat scratch example | high beat scores but bad motion: `G1BeatF1=0.8716`, `G1FootSliding=3.0399`, `RootSmoothnessJerkMean=3454.9197`, `G1Dist=56.2587` | `slurm/pipelines/20260506-g1-aist-fkbeats-lbeat-relative-scratch-lam020-cap1c/eval/metrics.json` |
| `20260507-g1-aist-fkbeats-lbeat-relative-finetune-lam020-cap1` | best AIST lbeat direction so far, not default | `G1BAS=0.4484`, `G1RoboPerformBAS=0.6211`, `G1BeatF1=0.4372`, but `G1FootSliding=0.7102`, `G1GroundPenetration=0.0979` | `slurm/pipelines/20260507-g1-aist-fkbeats-lbeat-relative-finetune-lam020-cap1/eval/metrics.json` |
| `20260509-finedance-g1-fkbeatdistance-1000-r2` | cleaner FineDance G1 Jukebox FKBeat anchor | `G1BAS=0.3046`, `G1RoboPerformBAS=0.5416`, `G1FKBAS=0.3192`, `G1FKRoboPerformBAS=0.5142`, `G1BeatF1=0.2910`, `G1Dist=8.6295` | `slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2/eval/metrics.json` |
| `20260509-finedance-g1-relative-lbeat-scratch-lam010-1000-r2` | rejected FineDance G1 relative-lbeat scratch run | `G1BAS=0.3523`, `G1RoboPerformBAS=0.5023`, `G1FKRoboPerformBAS=0.4940`, `G1FootSliding=1.3512`, `G1Dist=39.5323` | `slurm/pipelines/20260509-finedance-g1-relative-lbeat-scratch-lam010-1000-r2/eval/metrics.json` |

## Status Vocabulary

`idea`, `spec`, `ready`, `running`, `blocked`, `failed`, `needs_eval`, `finished`, `archived`

## Rules

- Read this file before starting, resuming, evaluating, or comparing experiment work.
- Keep one spec per clear experiment at `docs/experiments/EXP-YYYYMMDD-short-slug.md`.
- Include the experiment ID in new run, Slurm, render, metric, and checkpoint paths when practical.
- Include a GT/reference row in metric comparison tables whenever the matching reference split can be evaluated.
- If the history is unclear, list the run under `Needs Triage Before Promotion` instead of inventing missing rationale.
- Update this index whenever an experiment status, latest artifact, or next action changes.
