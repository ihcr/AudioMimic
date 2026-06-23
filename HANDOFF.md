# Handoff

## Goal

Preserve the 2026-06-18 side conversation about the long-term
FineDance-G1 / Musics2Dance research direction so a future Codex thread can
continue without relying on ephemeral sidechat history.

This is a documentation handoff only. It does not claim to describe the latest
training job state. For concrete runs, checkpoints, evals, and renders, read
`docs/experiments/INDEX.md` and the active experiment spec.

## Current Progress

The sidechat research decisions have been written into durable repo docs:

- `docs/research/SIDECHAT-20260618-robot-native-dance-prior-transcript.md`
- `docs/research/README.md`
- `docs/research/ROBOT_NATIVE_DANCE_PRIOR_BLUEPRINT.md`
- `docs/research/modules/G1_MOTION_LATENT_PRIOR_SPEC.md`
- `docs/research/modules/MUSIC_TO_LATENT_GENERATOR_SPEC.md`
- `docs/research/modules/FEASIBILITY_RL_SPEC.md`
- `docs/research/modules/STREAMING_POLICY_ROADMAP_SPEC.md`
- `docs/papers/INDEX.md`
- `AGENTS.md`

The main preserved conclusion is that the next structural direction should be a
music-conditioned, G1-feasible dance latent prior with rhythm-to-support
controls and robot-feasibility gates. Streaming and real-time policy remain
long-term endpoints, not the first novelty claim.

## What Worked

- The sidechat separated related-work roles clearly:
  - SONIC is a strong template for robot motion prior/control interfaces.
  - BeyondMimic is best treated as feasibility teacher, critic, tracker, or
    diagnostic.
  - RoboPerform is the closest collision for broad audio-to-humanoid policy.
  - DiscoForcing is the closest collision for streaming music-to-motion.
- The docs now keep both distillation and non-distillation streaming/policy
  routes open.
- The preferred non-distillation route is recorded as a
  `Latency-Conditioned Rhythm-to-Support G1 Dance Prior`.
- The research knowledge base now has a new-agent reading path in
  `docs/research/README.md`.

## What Didn't Work

- Do not rely on the sidechat itself as the only durable artifact. In Codex CLI
  documentation, `/side` is described as an ephemeral side conversation.
- Do not continue the old root `HANDOFF.md` operational setup as if it were
  current training status. It described an older Wav2CLIP/STFT migration flow
  and has been replaced by this research handoff.
- Do not frame the project as generic streaming, generic audio-to-policy, or a
  tracker-only wrapper. Those claims collide too directly with recent work.

## Next Steps

1. In a fresh conversation, start by reading this file, `AGENTS.md`, and
   `docs/research/README.md`.
2. Read `docs/experiments/INDEX.md` before touching runs or interpreting model
   quality.
3. For architecture work, choose one narrow ablation from the module specs:
   G1 latent prior, music-to-latent generation, feasibility critic/reranking,
   streaming interface, or policy bridge.
4. Before claiming novelty, check `docs/papers/INDEX.md` against RoboPerform,
   DiscoForcing, SONIC, BeyondMimic, and the token/prior music-to-dance papers.
5. Keep future run IDs, checkpoint paths, metric files, and render paths in
   `docs/experiments/`, not in the research blueprint.
