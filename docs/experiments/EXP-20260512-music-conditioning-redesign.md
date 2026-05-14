# EXP-20260512-music-conditioning-redesign

Status: spec
Owner: Yukun
Created: 2026-05-12
Last Updated: 2026-05-14 UTC

## Evidence Scope

This is an original proposal grounded in the current FineDance-G1 results, the
dirty `diffusion` worktree, local paper notes under `docs/papers/markdown/`, and
a fresh online check of EDGE, DGFM, Beat-It, Lodge, AIST++/FACT, and Wav2CLIP
sources.

## Research Question

Can a richer and more structured music-conditioning interface improve FineDance-G1 dance quality better than the current beat-distance condition, beat-alignment loss, or direct Librosa35 replacement?

## Hypothesis

The current bottleneck is not only the music feature itself. It is the combination of:

- low-level framewise music tokens being treated as the whole music representation,
- beat-distance tokens being fused with music by simple early concatenation,
- lbeat-style supervision giving the model an exploitable target that can be satisfied by root/height/jerk artifacts,
- missing global phrase/style information that would tell the generator what kind of movement should happen, not just when a beat occurs.

A better first direction is a two-stream music conditioner:

- local rhythm stream: STFT or mel/chroma/onset features aligned at 30 fps,
- high-level music stream: Wav2CLIP embeddings, optionally later MERT or CLAP,
- separate fusion: local rhythm tokens for cross-attention, high-level stream for global FiLM/style bias, and beat-distance only as an optional gated cue, not as the primary training objective.

## Baseline Or Control

- Clean FineDance-G1 Jukebox FKBeat anchor: `20260509-finedance-g1-fkbeatdistance-1000-r2`, `G1BAS=0.3046`, `G1FKBAS=0.3192`, `G1BeatF1=0.2910`, `G1Dist=8.6295`.
- Best balanced Librosa anchor: `20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000`, `G1BAS=0.2498`, `G1FKBAS=0.2653`, `G1BeatF1=0.2255`, `G1Dist=8.9239`.
- Rejected lbeat examples should remain negative controls, not default baselines.

## Intervention

Stage A: add `wav2clip_stft` features without auxiliary beat loss.

- Extract frame-level or short-window Wav2CLIP features and align/interpolate to 30 fps.
- Extract STFT features from full-song context and align to the same clip frames.
- Start with DGFM-style projection/addition to produce a 512-dimensional per-frame music condition.
- Train the existing `DanceDecoder` first, with `use_beats=False`, to isolate the music representation.

Stage B: replace simple early fusion with a two-stream conditioner.

- Add a `MusicFusionDanceDecoder` or equivalent behind a flag such as `--music_conditioner dual_stream`.
- Encode low-level rhythm and high-level music separately.
- Feed rhythm tokens to decoder cross-attention.
- Feed pooled high-level tokens through FiLM/global bias.
- If beat-distance is included, use a learned gate and stream dropout so it cannot dominate the whole condition.
- Keep `lambda_beat=0` for the first comparison.

Stage C: optional planner/primitive layer after Stage A/B is stable.

- Inspired by Lodge, predict sparse music cue tokens or expressive-energy primitives from music.
- Use them as soft guidance for motion energy and phrase changes, not as hard beat matching.
- Only add explicit beat loss later as a small fine-tune with root-step, root-height, acceleration, and foot-contact gates.

## Invariant Controls

- Dataset/split: `data/finedance_g1_fkbeats`, same 3265-file FineDance-G1 test split.
- Motion representation: G1 FKBeat-compatible robot-native motion format.
- Model family: EDGE diffusion first; no architecture replacement until feature/fusion ablation is measured.
- Training budget: compare against the 1000/2000 epoch anchors with matched or clearly documented epoch budgets.
- Evaluation protocol: existing full-song eval plus matched 30-second qualitative videos.

## Data And Cache Contract

- Raw data: reuse FineDance audio and G1 motion data already prepared under the diffusion worktree.
- Processed data: create new processed roots for `wav2clip_stft`; do not reuse Librosa/Jukebox tensor caches.
- Feature cache: use separate feature directories, for example `wav2clip_feats`, `stft_feats`, or a combined `wav2clip_stft_feats`.
- Tensor/cache layer: bump feature-type cache keys by new `feature_type` and conditioner setting.
- Cache invalidation needed: yes, for any run that changes feature dimension, stream layout, or conditioner dict format.

## Implementation Scope

- Original planning branch: `diffusion`.
- Stage A implementation branch: `wav2clip-stft-beat`.
- Likely files:
  - `EDGE.py`
  - `args.py`
  - `data/audio_extraction/`
  - `data/create_dataset.py`
  - `data/prepare_finedance_g1_dataset.py`
  - `model/model.py`
  - `submit_training_pipeline.py`
  - focused tests under `tests/`
- Files/modules intentionally unchanged in Stage A: lbeat losses, FK beat estimator, G1 kinematics, renderer.

## Training Or Execution Plan

- Environment: `source .venv311/bin/activate`
- First implementation target: feature extraction and Stage A training with no beat condition and no beat loss.
- Slurm/run naming: include `EXP-20260512-music-conditioning-redesign` or a short derivative such as `20260512-finedance-g1-wav2clip-stft-stagea`.
- Logs/run/checkpoints: place under repo-local `slurm/pipelines/` and `runs/train/` with the run slug.

## Evaluation Plan

- Metrics:
  - Rhythm: `G1BAS`, `G1FKBAS`, `G1BeatF1`, `G1RoboPerformBAS`, `G1FKRoboPerformBAS`.
  - Quality/safety: `G1Dist`, `G1FootSliding`, `G1GroundPenetration`, `RootSmoothnessJerkMean`, root-step tails, root-height violations.
- Qualitative artifacts: matched 30-second videos for the same FineDance examples used in the latest render checks.
- Comparison target: Jukebox FKBeat no-lbeat and Librosa35 full-context motiondist.
- Success criteria:
  - improve over Librosa35 full-context motiondist on rhythm without worsening `G1Dist` and root/contact gates,
  - approach or beat Jukebox FKBeat rhythm while staying near its `G1Dist=8.6295`,
  - no lbeat-style root-z dashing or root-step spikes.
- Failure signals:
  - rhythm improves but `G1Dist`, root-height, root-step, or penetration collapse,
  - Wav2CLIP-only loses local beat timing,
  - STFT-only behaves like another low-level Librosa swap.

## Paper-Inspired Design Notes

- EDGE supports the baseline model shape: transformer diffusion with music as cross-attention context, originally relying on Jukebox.
- DGFM is the strongest direct feature clue: Wav2CLIP + STFT beats the feature-only alternatives in its FineDance study, and it uses high-level music features plus low-level temporal detail.
- AIST++/FACT is the warning that Librosa35 can work only when fusion is strong enough; a raw feature swap is not enough.
- Beat-It is the warning against direct condition concatenation. Its contribution is hierarchical multi-condition fusion, not merely nearest-beat distance plus a loss.
- Lodge points to the missing long-horizon phrase/planning layer: sparse primitives can carry global choreography better than only dense framewise music features.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-12 | spec creation | spec | `docs/experiments/EXP-20260512-music-conditioning-redesign.md` | Proposed Wav2CLIP+STFT Stage A, two-stream conditioner Stage B, primitive/planner Stage C. |
| 2026-05-14 | Stage A handoff | blocked | `wav2clip-stft-beat` branch `docs/experiments/EXP-20260513-finedance-g1-wav2clip-stft-beat.md` | Stage A moved to the dedicated `wav2clip-stft-beat` branch. Feature extraction completed and `stream_adapter` trained to 500 epochs; `concat_norm` and eval are blocked by Slurm quota and should continue on another server/account. |

## Results

- Metric files: none yet for Stage A; eval jobs were cancelled before running on
  the previous server.
- Render/report paths: none yet for Stage A.
- Checkpoint path in the `wav2clip-stft-beat` branch:
  `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt`.
- Key observations: current evidence argues against spending more effort on high-weight scratch lbeat or Librosa35-only swaps.

## Current Conclusion

Stage A has started on the separate `wav2clip-stft-beat` branch using Wav2CLIP+STFT+GaussianBeat and no auxiliary beat loss. It is not evaluated yet; do not draw a quality conclusion from training loss alone.

## Next Action

Continue Stage A from a direct clone of the `wav2clip-stft-beat` branch: copy
the runtime artifacts listed in its `HANDOFF.md`, run `concat_norm`, evaluate
both fusion variants, then compare against the clean Jukebox FKBeat and
Librosa35 full-context motiondist anchors.
