# Research Knowledge Base

This directory keeps durable method research for the FineDance-G1 / robot-native
music-to-dance direction. It is intentionally separate from
`docs/experiments/`, which remains the ledger for concrete runs, checkpoints,
evals, renders, and ablations.

## Start Here

- [Robot-Native Dance Prior Blueprint](ROBOT_NATIVE_DANCE_PRIOR_BLUEPRINT.md) -
  current long-term plan for moving from raw G1 pose diffusion toward a
  music-conditioned, G1-feasible dance prior.
- [Module Specs](modules/README.md) - detailed option matrices and ablation
  plans for the G1 motion latent prior, music-to-latent generator, and
  feasibility/RL module, plus the future streaming/policy roadmap.
- [Paper Library](../papers/INDEX.md) - categorized paper database with links,
  local PDF/markdown status, relevance, reusable ideas, and known limitations.
- [Sidechat Transcript: Robot-Native Dance Prior](SIDECHAT-20260618-robot-native-dance-prior-transcript.md) -
  curated recovery note for the research discussion that led to this knowledge
  base and blueprint.

## New Agent Reading Path

For a fresh agent joining this project, read in this order:

1. `AGENTS.md` at the repository root for repo rules, experiment tracking, and
   research knowledge-base routing.
2. `docs/experiments/INDEX.md` for current run status and which conclusions are
   tied to concrete checkpoints.
3. This file, then
   [Sidechat Transcript: Robot-Native Dance Prior](SIDECHAT-20260618-robot-native-dance-prior-transcript.md)
   for the reasoning chain behind the current blueprint.
4. Then read
   [Robot-Native Dance Prior Blueprint](ROBOT_NATIVE_DANCE_PRIOR_BLUEPRINT.md)
   for the durable method direction.
5. [Paper Library](../papers/INDEX.md) for the related-work map and novelty
   boundaries against RoboPerform, DiscoForcing, SONIC, and related systems.
6. Module specs only as needed:
   - [G1 Motion Latent Prior](modules/G1_MOTION_LATENT_PRIOR_SPEC.md)
   - [Music-to-Latent Generator](modules/MUSIC_TO_LATENT_GENERATOR_SPEC.md)
   - [Feasibility And RL](modules/FEASIBILITY_RL_SPEC.md)
   - [Streaming And Policy Roadmap](modules/STREAMING_POLICY_ROADMAP_SPEC.md)

The key side-chat conclusion now preserved in docs is: the project should center
on a rhythm-to-support, G1-feasible dance latent prior; streaming and policy are
long-term endpoints, with both distillation and non-distillation routes kept
available.

## Scope

Use these docs for:

- deciding whether a new paper is a direct collision, weak related work, or a
  usable method component;
- planning architecture-level changes such as motion priors, tokenizers,
  control encoders, feasibility critics, and robot policy layers;
- keeping literature conclusions out of transient experiment specs unless they
  are tied to a concrete run.

Do not put one-off run IDs, failed commands, checkpoint paths, or temporary
metric conclusions here. Those belong in `docs/experiments/`.
