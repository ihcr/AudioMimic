# G1 Motion Latent Prior Spec

Last updated: 2026-06-18.

## Purpose

Learn a G1-feasible dance motion space before asking music to generate motion.
This module replaces the current habit of denoising raw G1 joint trajectories as
the only representation. It should preserve contact, support, root/yaw
stability, endpoint naturalness, and dance amplitude.

## Decision Summary

Start with a continuous autoencoder or VAE. Do not make RVQ/HRVQ the first
mainline. RVQ is attractive and common in recent motion work, but our immediate
failure modes are contact/support, hand jitter, and G1 feasibility. Those can be
damaged by coarse or unstable quantization.

Recommended first route:

```text
g1_yaw_delta motion
    -> continuous G1 motion AE/VAE
    -> decoded g1_yaw_delta + contact/support head
```

Recommended ablation order:

1. Deterministic AE.
2. AE with light KL or VAE.
3. Part-wise continuous latent.
4. RVQ/HRVQ bottleneck.
5. Hierarchical coarse/detail latent.
6. Action or PD-target latent only after pose/contact prior is stable.

## Latent Options

| Option | Definition | Advantages | Risks | Recommendation |
|---|---|---|---|---|
| Deterministic autoencoder | Encode motion into continuous latent and decode motion | Easiest to debug; FK/contact losses are direct; no KL blur | Latent may be poorly shaped for sampling | Best first smoke test |
| VAE | Encode motion into Gaussian latent distribution | Better-shaped latent for diffusion; common in latent diffusion motion work | KL can over-smooth motion and wash out hands/contact | Main first route with light KL or KL warmup |
| VQ-VAE | Encode motion into discrete code indices | Reduces average-pose behavior; learns motion vocabulary | Codebook collapse; contact and endpoint detail can quantize poorly | Second-stage ablation |
| RVQ / HRVQ | Multi-level residual discrete codebooks | Strong recent trend; better reconstruction than single VQ; supports hierarchy | More moving pieces; residual levels can become noisy; harder debug | Valuable only after continuous prior baselines pass |
| Hierarchical latent | Separate coarse choreography latent from fine detail latent | Separates phrase/rhythm structure from local endpoint/contact detail | More model and ablation complexity | Long-term best fit if data is enough |
| Part-wise latent | Separate root/lower/upper/hands or body/support streams | Directly targets hand jitter and foot support | Can create incoherent body coordination if not coupled | Good V6/V7 ablation |
| Action/PD latent | Decode to action or PD target in addition to pose | Most robot-native | Requires sim/control/RL and new evaluation stack | Long-term only |

## Why Continuous First

Continuous latents let us preserve the things that currently break:

- foot height and support contact;
- endpoint velocity/jerk;
- root yaw continuity;
- FK reconstruction;
- body/hand coordination;
- dance amplitude.

RVQ/HRVQ should be judged against these. It should not be accepted just because
it improves distribution metrics or looks modern.

## Candidate Architecture

Input:

```text
motion: Tensor[B, T, D] using g1_yaw_delta
optional_fk: Tensor[B, T, J, 3]
optional_contact: Tensor[B, T, C]
```

Encoder:

- temporal Conv1D or small transformer encoder;
- downsample time by 1x, 2x, or 4x as separate ablations;
- latent dim start points: `64`, `96`, `128`;
- optional part streams for root/lower/upper/hands after the base model works.

Decoder:

- reconstruct `g1_yaw_delta`;
- predict support/contact logits;
- optionally predict FK endpoint velocities as auxiliary heads.

Losses:

- motion reconstruction;
- FK joint reconstruction;
- velocity and acceleration reconstruction;
- root/yaw continuity;
- support contact height consistency;
- sliding loss only under support/contact;
- wrist/foot endpoint jerk distribution regularization;
- VAE KL with warmup if using VAE.

## Evaluation

Do not evaluate the prior only by reconstruction MSE. Report:

- motion reconstruction MSE/MAE;
- FK reconstruction error;
- foot contact precision/recall if labels exist;
- foot sliding under contact;
- ground penetration;
- wrist FK jerk p95;
- foot FK jerk p95;
- root yaw angular velocity p99/max;
- decoded `G1Dist` and `G1Div` against training/eval distribution;
- visual render on fixed music-independent held-out motion clips.

## Ablation Matrix

| Ablation | Question | Accept signal |
|---|---|---|
| AE vs VAE | Does sampling-friendly latent hurt contact/detail? | VAE keeps contact and jerk within AE tolerance |
| latent dim `64/96/128` | How much capacity is needed? | Lower dim preserves quality without hand/contact loss |
| time stride `1/2/4` | Can phrase-level compression survive? | Stride does not add foot hover or hand jitter |
| part-wise latent | Does separating body parts reduce jitter? | Hand jerk improves without lower-body/root regression |
| RVQ/HRVQ | Does tokenization reduce average motion without damaging contact? | Equal/better contact and endpoint metrics vs continuous |
| coarse/detail hierarchy | Can high-level rhythm separate from local detail? | Better music generator quality without decoder detail loss |

## Failure Modes

- Posterior collapse or KL over-smoothing.
- Codebook collapse in VQ/RVQ.
- Good reconstruction MSE but bad FK/contact behavior.
- Token prior improves diversity but creates foot hover.
- Part-wise latent breaks whole-body coordination.
- Latent decoder learns render-plausible but tracker-unfriendly motion.

## Source Anchors

- MLD / motion latent diffusion: https://chenxin.tech/mld/
- MoMask / hierarchical residual motion tokens: https://arxiv.org/abs/2312.00063
- DuetGen / hierarchical VQ-VAE for dance: https://arxiv.org/abs/2506.18680
- SoulDance / HRVQ holistic dance: https://arxiv.org/html/2507.14915v1
- SONIC / scalable humanoid motion prior context: https://arxiv.org/abs/2511.07820

