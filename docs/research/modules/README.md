# Robot-Native Dance Module Specs

Last updated: 2026-06-18.

These module specs expand the top-level
[Robot-Native Dance Prior Blueprint](../ROBOT_NATIVE_DANCE_PRIOR_BLUEPRINT.md)
without turning the blueprint into a catch-all document. Each spec should keep
the alternatives, tradeoffs, failure modes, ablations, and acceptance gates for
one part of the long-term pipeline.

## Modules

| Module | Role | Current recommendation |
|---|---|---|
| [G1 Motion Latent Prior](G1_MOTION_LATENT_PRIOR_SPEC.md) | Learn the G1-feasible dance motion space before adding music | Start with continuous AE/VAE; use RVQ/HRVQ as a second ablation after contact/detail reconstruction is proven |
| [Music-to-Latent Generator](MUSIC_TO_LATENT_GENERATOR_SPEC.md) | Map compact music/rhythm controls into the learned G1 prior | Start with latent diffusion over continuous latents; compare masked latent generation and RVQ token generation later |
| [Feasibility And RL](FEASIBILITY_RL_SPEC.md) | Bring robot execution signals into selection, guidance, or training | Start with tracker/MuJoCo diagnostics and a feasibility critic; do not train end-to-end RL music-to-action as the first route |
| [Streaming And Policy Roadmap](STREAMING_POLICY_ROADMAP_SPEC.md) | Preserve a path from offline generation to causal streaming and real-time policy | Keep chunk/history/action-bridge interfaces now; track both distillation and non-distillation routes; treat streaming and policy as endpoints, not the first novelty claim |

## Maintenance Rules

- Keep the blueprint short and directional.
- Put design option tables and route comparisons in these module specs.
- Put concrete run IDs, checkpoints, eval paths, and conclusions in
  `docs/experiments/`.
- When a module graduates into a real experiment, create or update an
  experiment spec under `docs/experiments/`.
