# EXP-20260512-finedance-g1-fk-jerk-audit

Status: finished
Owner: Yukun
Created: 2026-05-12
Last Updated: 2026-05-12 UTC

## Research Question

Are the bad FineDance-G1 inference videos caused by FK introducing wrong labels/losses, or by another pipeline issue?

## Evidence Scope

This audit used local diffusion-worktree code, saved 30-second render payloads, eval motion pickles, training scripts/logs, normalizers, dataset metadata, and the existing `EXP-20260511-finedance-g1-librosa-fullctx` metrics. No new model training was launched.

## Checked Artifacts

- Matched 30-second render motions under `slurm/pipelines/20260512-finedance-g1-30s-renders/`
- Five-second eval motions under:
  - `slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2/eval/motions/`
  - `slurm/pipelines/20260511-finedance-g1-jukebox-lbeat-robotloss-b020-kin040-cap1-scratch-1000/eval/motions/`
  - `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/motions/`
  - `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-lbeat-robotloss-b020-kin040-cap1-scratch-1000/eval/motions/`
- FineDance FK beat metadata: `data/finedance_g1_fkbeats/metadata.json`
- G1 FK implementation paths: `data/audio_extraction/beat_features.py`, `eval/g1_kinematics.py`, `model/g1_torch_kinematics.py`, `model/diffusion.py`

## Findings

FK root quaternion convention is probably not the main bug. FineDance-G1 source/root rotations look like `xyzw`, and FK with `xyzw` gives plausible feet/keypoint ranges on reference motions. Re-running the same reference motion as `wxyz` produces much larger keypoint speeds and unrealistic foot height ranges.

The lbeat+robot-loss models are genuinely unstable before long-song stitching. A 200-file sample of 5-second eval clips showed much larger root-step and acceleration tails than the no-lbeat/motiondist anchors:

| Sample | Root Step p99 | Root Step Max | Root Acc p99 | Root Acc Max | Root-Z Saturated |
|---|---:|---:|---:|---:|---:|
| GT slices | 0.0381 | 0.1185 | 0.0219 | 0.1122 | 0.0% |
| Jukebox FKBeat no-lbeat | 0.0417 | 0.1564 | 0.0484 | 0.0963 | 0.0% |
| Jukebox lbeat+robotloss | 0.1339 | 0.2111 | 0.2508 | 0.3692 | 0.6% |
| Librosa35 motiondist | 0.0394 | 0.1048 | 0.0486 | 0.1150 | 0.0% |
| Librosa35 lbeat+robotloss | 0.0850 | 0.2616 | 0.1388 | 0.3393 | 3.18% |

The 30-second motion payloads show the same pattern. The no-lbeat Jukebox and Librosa motiondist runs stay close to reference root-step scale. The lbeat+robotloss runs introduce large root-z jumps, with the Librosa35 lbeat run repeatedly hitting the normalizer upper bound for root height (`0.9517394`).

The lbeat+robotloss training scripts were scratch runs, not fine-tunes from the clean FKBeat anchor. They used `lambda_beat=0.2`, `beat_loss_max_fraction=1.0`, and `lambda_g1_kin=0.4`, while `lambda_acc=0.0`. The robot-loss contribution visible in logs is small compared with the base loss and does not prevent root-height saturation.

There was also a long-generation stitching hypothesis in `model/diffusion.py`: forcing overlap on the final predicted sample looked plausible from the code, but the follow-up 30-second render check showed that this can create visible seam dashing. The current code therefore matches original EDGE behavior again: overlap is copied during denoising steps, but the final `x_start` is not overwritten by another hard overlap copy.

## Current Conclusion

Do not blame FK globally. FK-derived beat labels and FK evaluation look useful and conventionally sane in this checkout. The failure mode is more specific:

1. The lbeat objective is producing beat-seeking motion artifacts.
2. The robot-loss settings are too weak or incomplete to stop root-z/root-acceleration spikes.
3. Scratch lbeat+robotloss training is worse than starting from the stable FKBeat anchor.
4. Long-song generation should keep original EDGE overlap-copy timing. The later render check rejected final-sample overlap enforcement because it can create seam dashing.

The cleaner current anchors remain:

- Jukebox deployment anchor: `runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt`
- Low-dimensional Librosa anchor: `runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt`

## Recommended Next Action

Keep the existing clean checkpoints as anchors. For any new lbeat experiment, use a finetune from the clean FKBeat anchor, lower `lambda_beat` and `beat_loss_max_fraction`, re-enable an acceleration/smoothness guard, and add root-height saturation/root-step audits as acceptance gates before rendering long videos.

## Verification

```bash
source /projects/u6ed/yukun/EDGE/.venv311/bin/activate
python -m unittest \
  tests.test_phase4_to_6_beat_integration.InferenceBeatUtilityTests.test_long_ddim_sample_falls_back_to_ddim_for_single_clip \
  tests.test_phase4_to_6_beat_integration.InferenceBeatUtilityTests.test_long_overlap_constraint_matches_original_edge_window_copy
git diff --check -- model/diffusion.py tests/test_phase4_to_6_beat_integration.py docs/experiments/INDEX.md docs/experiments/EXP-20260512-finedance-g1-fk-jerk-audit.md
```

Both passed. The unittest command ran 2 tests in 68.336 seconds.
