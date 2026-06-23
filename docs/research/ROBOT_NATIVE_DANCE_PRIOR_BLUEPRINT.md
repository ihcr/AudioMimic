# Robot-Native Dance Prior Blueprint

Last updated: 2026-06-18.

## Purpose

The current FineDance-G1 line can generate robot-format motion, but recent
experiments show a persistent gap between kinematic music-to-dance metrics and
robot-native motion quality. V5 fixed severe root spinning with `g1_yaw_delta`,
but exposed hand jitter, hovering/high-lift feet, and weak support/contact
naturalness. This document turns the side-chat research discussion into a
durable project blueprint.

## Core Claim

Robot-native music-to-dance should not directly denoise raw robot joint
trajectories as the only representation. It should condition music on a learned
G1-feasible dance prior, then decode into robot-native motion, and eventually
into action or PD-target space if real execution becomes the target.

The proposed direction is:

```text
Wav2CLIP + GaussianBeat + rhythm controls
        -> music-to-G1 latent or token generator
        -> G1-feasible dance prior / decoder
        -> robot-native pose sequence
        -> optional feasibility critic or action decoder
```

This is not a direct copy of SONIC, BeyondMimic, or RoboPerform. It combines a
music-conditioned dance generator with a G1-native prior and robot feasibility
signals. The paper-level novelty must be stated narrowly: music-conditioned
robot-native dance generation with an explicit G1-feasible dance prior and
rhythm-aware controls, not generic audio-to-humanoid control.

## Why A Prior Is Better Than Direct Raw Motion Generation

Direct music-to-joint generation is underconstrained. The same music can support
many dances, while the set of physically plausible G1 dances is narrow. A single
denoiser trained on raw joint vectors must simultaneously learn:

- music semantics and rhythm;
- choreography and phrase-level motion structure;
- body and hand coordination;
- foot support, ground contact, and root dynamics;
- robot morphology and feasibility.

This tends to produce averaged motion, endpoint jitter, contact hacks, or
metric-specific behavior. A prior changes the task:

1. Learn the feasible motion space first.
2. Learn music-to-prior selection second.
3. Use feasibility supervision or tracking diagnostics to keep samples inside
   the robot-executable region.

This also gives a cleaner research story than adding more hand-tuned losses to a
raw pose denoiser.

## Related Work Map

| Area | Representative work | Prior or latent form | Transferable idea | Limitation for this project |
|---|---|---|---|---|
| Music-to-dance diffusion | EDGE, FineDance | Raw pose diffusion with audio conditioning and contact/genre modules | Strong baseline and local repo foundation | Human/SMPL oriented; robot feasibility is not the learning target |
| Long dance structure | LODGE | Coarse-to-fine diffusion with characteristic dance primitives | Separate global choreography from local detail | Still human dance; foot refinement is post/local |
| Streaming audio motion | DiscoForcing | Causal music encoder plus diffusion-forcing sequence model | Real-time, bounded-latency audio-driven rollout | Robot side is deployment/retargeting, not robot-native learning |
| Discrete motion prior | T2M-GPT, MotionGPT, MoMask | VQ/RVQ motion tokens plus GPT or masked transformers | Reduces average-pose collapse; learns reusable motion vocabulary | Tokenization can lose contact and endpoint detail |
| Hierarchical dance prior | DuetGen, SoulDance | Coarse/fine VQ or hierarchical RVQ for dance structure and details | Separates phrase/semantic motion from fine body detail | Human body, hands, face; no G1 feasibility |
| Multi-condition masked prior | DanceMosaic | Pretrained motion prior with music/pose/text towers | Mitigates guidance/condition interference | Prior is not G1-specific |
| Generalist motion prior | GENMO/GEM | Multi-task human motion estimation/generation model | Larger motion universe improves plausibility and diversity | Human representation mismatch |
| Robot motion tracking prior | SONIC | Scaled humanoid tracking foundation controller and token interface | Strong template for a universal G1-compatible motion/control prior | Not a music-to-dance model by itself |
| Guided robot control | BeyondMimic | Motion tracking plus state-action diffusion and test-time cost guidance | Feasibility diagnostics and guided execution | Best used as critic/teacher, not as the main music generator |
| Dynamic humanoid control | KungfuBot | Motion processing plus adaptive tracking policy | Shows highly dynamic G1 motion needs filtering/correction/control | Does not solve music-conditioned choreography |
| Audio-to-humanoid policy | RoboPerform | Content/style latent decomposition with audio-conditioned diffusion student policy | Closest collision and strongest robot-native audio reference | Broad audio-to-locomotion system; our novelty must be more specific |

## Proposed Architecture

The blueprint records the high-level route. Detailed option tables, failure
modes, and ablation plans live in the module specs:

- [G1 Motion Latent Prior](modules/G1_MOTION_LATENT_PRIOR_SPEC.md)
- [Music-to-Latent Generator](modules/MUSIC_TO_LATENT_GENERATOR_SPEC.md)
- [Feasibility And RL](modules/FEASIBILITY_RL_SPEC.md)
- [Streaming And Policy Roadmap](modules/STREAMING_POLICY_ROADMAP_SPEC.md)

### Stage A: G1 Native Motion Prior

Train a G1 motion autoencoder or tokenizer before adding music.

Inputs:

- current stable `g1_yaw_delta` root representation;
- G1 joint sequence;
- FK-derived support/contact labels or auxiliary targets;
- optional endpoint velocities and root/yaw dynamics for diagnostics.

Preferred first version:

- continuous latent autoencoder or VAE;
- contact/support prediction head;
- reconstruction losses in motion space and FK space;
- endpoint jerk/contact/root metrics as validation, not hidden fallbacks.

RVQ/tokenization should be a second ablation. It may reduce average motion, but
it can also damage fine contact and wrist/foot detail if the codebook is too
coarse.

### Stage B: Music-to-G1 Latent Generator

Train the music-conditioned model to generate latent sequences instead of raw
joint vectors.

Condition schema:

```text
semantic: Wav2CLIP
control: GaussianBeat, body_intensity, support_beatness
```

Design notes:

- keep external music conditions compact;
- keep beat/control streams typed instead of one undifferentiated feature blob;
- use latent diffusion or masked latent modeling as the main ablation;
- preserve predicted-control inference as the default evaluation style.

### Stage C: Feasibility Critic

Use MuJoCo, FK/contact diagnostics, and optionally BeyondMimic/SONIC-style
tracking to label generated motions.

Candidate labels:

- tracker success or failure;
- tracking error;
- fall/termination;
- joint-limit pressure;
- action or PD-target jerk;
- contact support failure;
- foot sliding and penetration;
- root drift and angular velocity tails.

Train a light surrogate:

```text
decoded motion or latent -> feasibility score
```

Use it first for checkpoint selection and reranking. Use it for guidance or
fine-tuning only after it is calibrated against real eval failures.

### Stage D: Optional Action Decoder

If pose-level latent generation becomes stable, move toward:

```text
latent -> pose + contact + PD target/action
```

This is the point where the project becomes closer to RoboPerform or
BeyondMimic-style robot policy learning. It should not be the first ablation
because it adds sim/RL/control complexity.

### Stage E: Streaming And Policy Compatibility

Streaming and real-time policy are valid long-term endpoints, but they should
not be the first novelty claim because DiscoForcing already occupies streaming
audio-to-motion and RoboPerform already occupies retargeting-free
audio-to-humanoid policy. Preserve the route by designing chunk-first APIs,
causal/bounded-lookahead condition variants, history-conditioned generation,
standalone latent decoding, and action-bridge metadata from the beginning.

## What Is Novel Here

The defensible novelty is not "we use a prior" and not "audio drives a
humanoid"; both are already represented in recent work.

The stronger claim is:

- a G1-native dance prior, not a generic human motion prior;
- music-conditioned latent generation into that G1 prior;
- explicit rhythm controls grounded in prior project evidence:
  Wav2CLIP for semantic style, GaussianBeat for audio beat, body/support
  controls for robot-relevant motion rhythm;
- feasibility supervision internal to generation through critic/rerank/guidance,
  rather than only post-hoc retargeting;
- evaluation that gates rhythm, diversity, endpoint naturalness, root behavior,
  contact, and robot tracking feasibility together.

## What This Should Not Become

- A tracker-only wrapper around the current generator.
- A raw pose diffusion model with more hand-written losses.
- A direct copy of SONIC's general humanoid tracking objective.
- A direct copy of RoboPerform's audio-to-locomotion teacher/student system.
- A direct copy of DiscoForcing's streaming endpoint without robot-native G1
  prior novelty.
- A beat-score optimizer that regresses `G1Dist`, contact, or root behavior.

## Phased Experiments

### Phase 0: Baseline Audit

Use existing v3b/v5/v6 diagnostics to define target thresholds:

- `G1FKBAS`, `G1BeatF1`, precision, recall;
- `G1Dist`, `G1Div`, joint range;
- root angular velocity and tilt;
- wrist and foot FK jerk;
- support/contact proxy rates;
- foot sliding and ground penetration;
- optional SONIC/BeyondMimic tracker success when available.

### Phase 1: Motion Prior Only

Train a G1 autoencoder or VAE on G1 motion without music. Accept only if decoded
motion improves or preserves:

- reconstruction quality;
- contact/support consistency;
- wrist/foot jerk distribution;
- root/yaw stability.

### Phase 2: Music-to-Latent

Replace raw motion denoising with latent generation. Compare:

- raw pose diffusion;
- continuous latent diffusion;
- masked latent model;
- RVQ token model, if Phase 1 shows contact detail survives tokenization.

### Phase 3: Feasibility Critic

Collect labels from eval/tracking diagnostics and train a critic. Compare:

- no critic;
- critic reranking;
- critic-guided sampling;
- critic-regularized training.

### Phase 4: Action-Aware Generation

Only after Phase 1-3 are stable, add action/PD-target outputs or a policy layer.

### Phase 5: Causal Generator And Receding-Horizon Policy

After the offline G1 dance prior is strong, there are two valid routes:

- distill the offline generator into a causal student over the same
  latent/control space;
- train a direct causal or latency-conditioned generator on real G1 dance
  latents, where `latency_budget` controls how much future audio is visible.

Then add a receding-horizon motion policy, and only later a closed-loop action
or PD-target policy. The non-distillation route should remain available because
it gives a cleaner novelty boundary than simply copying RoboPerform-style
teacher/student or DiscoForcing-style streaming compression.

## Acceptance Gates

Do not accept a checkpoint on beat metrics alone. A candidate must be reported
with:

- rhythm: `G1FKBAS`, `G1BeatF1`, precision, recall, phase diagnostic;
- distribution: `G1Dist`, `G1Div`, joint range;
- root: drift, yaw angular velocity, root-up/tilt metrics;
- contact: foot sliding, penetration, support/contact proxy;
- endpoint quality: wrist/foot FK jerk;
- sensitivity: zero beatness/support controls, flat intensity, zero controls;
- feasibility: tracker/critic score if available.

## Open Questions

- Should the first prior be continuous latent or RVQ?
- Can G1 contact/support labels be generated reliably enough without a full
  controller?
- Does feasibility critic guidance preserve dance amplitude and diversity?
- Is SONIC-style tracking available locally enough to be a diagnostic, or should
  the first critic use MuJoCo/FK metrics only?
- How much FineDance-G1 data is enough for a G1 dance prior, and should
  retargeted AMASS/LAFAN/AIST be included for non-dance feasibility coverage?
- Should streaming be reached by distillation from a full-context teacher, or by
  direct latency-conditioned causal training from real G1 dance latents?

## Source Anchors

- EDGE: https://edge-dance.github.io/
- FineDance: https://li-ronghui.github.io/finedance
- LODGE: https://li-ronghui.github.io/lodge
- DiscoForcing: https://arxiv.org/abs/2605.28491
- T2M-GPT: https://arxiv.org/abs/2301.06052
- MotionGPT: https://arxiv.org/abs/2306.14795
- MoMask: https://arxiv.org/abs/2312.00063
- DanceMosaic: https://ojs.aaai.org/index.php/AAAI/article/view/37833
- GENMO/GEM: https://arxiv.org/html/2505.01425v1
- DuetGen: https://arxiv.org/abs/2506.18680
- SoulDance/SoulNet: https://arxiv.org/html/2507.14915v1
- SONIC: https://arxiv.org/abs/2511.07820
- BeyondMimic: https://arxiv.org/abs/2508.08241
- KungfuBot: https://arxiv.org/abs/2506.12851
- RoboPerform: https://arxiv.org/abs/2512.23650
