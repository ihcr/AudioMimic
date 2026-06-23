# Sidechat Transcript: Robot-Native Dance Prior

Date: 2026-06-18

This is a curated transcript and handoff from a side conversation about the
long-term direction of the FineDance-G1 / Musics2Dance project. It is not a
verbatim chat dump. It preserves the useful decisions, objections, alternatives,
and paper-positioning logic so a future agent can recover the thread even if the
ephemeral side chat is not reopened.

## How To Use This Document

Read this document when deciding how to move beyond the current raw-pose G1
diffusion line. Then read:

- [Robot-Native Dance Prior Blueprint](ROBOT_NATIVE_DANCE_PRIOR_BLUEPRINT.md)
- [G1 Motion Latent Prior Spec](modules/G1_MOTION_LATENT_PRIOR_SPEC.md)
- [Music-to-Latent Generator Spec](modules/MUSIC_TO_LATENT_GENERATOR_SPEC.md)
- [Feasibility And RL Spec](modules/FEASIBILITY_RL_SPEC.md)
- [Streaming And Policy Roadmap Spec](modules/STREAMING_POLICY_ROADMAP_SPEC.md)
- [Paper Library](../papers/INDEX.md)

This document records why those files exist and what they are meant to
preserve.

## Starting Problem

The recent G1 experiments showed that better benchmark numbers do not
automatically mean better robot dance. The important symptoms were:

- V3/V3b improved motion amplitude with Wav2CLIP plus rhythm controls, but still
  had sudden turns or fast spinning.
- V5/yaw-delta reduced the severe rotation failure, but the render looked more
  jittery than V3b, especially in the hands, and feet often hovered or lifted in
  unnatural ways.
- Adding more hand-written losses can fix a visible artifact, but risks turning
  the project into a stack of local patches rather than a clean research method.

The sidechat therefore asked: should the next project step be another loss or
representation tweak, or a more structural method change?

## Core Conclusion

The strongest long-term direction is:

```text
music-conditioned G1-feasible dance prior
with rhythm-to-support controls and robot-feasibility gates
```

The project should not claim novelty just for using a prior, making a streaming
generator, or driving a humanoid from audio. The defensible claim is narrower:

- learn a G1-native dance prior instead of relying only on raw robot joint
  diffusion;
- generate inside that prior from compact music/rhythm conditions;
- model rhythm as support/contact/body timing, not just beat-score alignment;
- evaluate and select motions using robot feasibility, contact, endpoint
  naturalness, root behavior, diversity, and rhythm together.

## BeyondMimic, SONIC, RoboPerform, And DiscoForcing

### BeyondMimic

BeyondMimic is relevant, but not as the main music-to-dance generator. Its best
role for this project is feasibility supervision:

- tracker or execution teacher;
- feasibility critic target;
- reranking or guidance signal;
- diagnostics for whether generated G1 motion is trackable.

It should not become an external patch that merely tries to rescue bad
generated motion after the fact.

### SONIC

SONIC is the stronger template for a scalable robot motion prior or controller
interface. It is useful for thinking about:

- universal humanoid tracking/control abstractions;
- token or latent interfaces;
- G1-compatible deployment structure;
- separating high-level intent from executable robot motion.

But SONIC is not itself a music-to-dance generator. Copying it directly would
not answer the choreography or rhythm-to-support problem.

### RoboPerform

RoboPerform is the closest collision if this project claims only
"audio-to-humanoid dance" or "retargeting-free audio policy." It uses an
audio-conditioned humanoid policy framing, with teacher/student and latent
content/style decomposition. That means this project should not make its paper
claim merely:

```text
audio -> humanoid dance policy
```

The open space is more specific:

```text
music -> G1 dance latent prior -> support/contact-aware robot dance
```

with robot dance metrics and editable rhythm/support controls.

### DiscoForcing

DiscoForcing is a strong streaming music-to-dance related work. It occupies the
claim space around causal/streaming music-to-motion generation. For this
project, it is a reference for:

- causal audio windows;
- history-conditioned generation;
- fixed-latency evaluation;
- chunked generation and boundary diagnostics.

But it should not be copied as the main novelty. The project should keep
streaming as a future endpoint enabled by the G1 prior, not as the first paper
claim.

## Why A Prior Beats Direct Raw Motion Generation

Direct raw motion generation makes one model learn too many things at once:

- music semantics and rhythm;
- phrase-level choreography;
- robot joint coordination;
- support/contact timing;
- root/yaw stability;
- endpoint smoothness;
- feasibility under G1 morphology.

This is underconstrained because many dances fit the same music, while the set
of feasible G1 dances is narrow. A learned prior decomposes the problem:

1. Learn what G1-feasible dance motion looks like.
2. Learn how music selects or schedules that motion.
3. Use feasibility diagnostics or critics to keep generation inside the robot
   motion manifold.

This is cleaner than adding many artifact-specific losses to the raw pose
denoiser.

## Latent Choices

The sidechat kept multiple routes open instead of pretending there is one
obvious answer.

| Option | Why use it | Risks | Best first use |
|---|---|---|---|
| Deterministic autoencoder | Simple, fast, preserves detail if capacity is enough | Latent may be unstructured and hard to sample | First reconstruction baseline |
| VAE | Common continuous prior, smoother sampling space | Posterior collapse or over-smoothing | First generative latent prior if AE is stable |
| VQ-VAE | Discrete motion units, less average-pose collapse | Codebook collapse and contact/detail loss | Ablation after contact survives AE/VAE |
| RVQ / HRVQ | Stronger discrete hierarchy, common in newer motion work | More stages and harder debugging | Second-stage ablation for phrase/body detail |
| FSQ-style tokens | Simpler scalar quantization than codebook VQ | Unknown for G1 contact fidelity | Later token ablation |

The preferred first route is continuous AE or VAE, because this project first
needs to prove a G1 latent can reconstruct support/contact, root, hands, and
feet without adding tokenization artifacts. RVQ/HRVQ should be an important
ablation, not the first forced choice.

## RL And Control Routes

RL should not be the first mainline. It can help once the motion prior is good,
but starting with RL would mix choreography, control, sim reward design, and
data issues too early.

Possible routes:

| Route | Description | Advantage | Risk |
|---|---|---|---|
| No RL, FK/MuJoCo diagnostics only | Train generator and select by offline robot metrics | Fastest and cleanest first step | May miss closed-loop execution failures |
| Feasibility critic | Train a surrogate from MuJoCo/tracker/failure labels | Adds robot awareness without full policy learning | Critic can be miscalibrated |
| Reranking | Sample multiple motions and keep feasible ones | Simple and robust for offline generation | Does not fix generator distribution |
| Guided sampling | Use critic or cost during sampling | Can improve feasibility directly | Can hurt diversity or rhythm if over-weighted |
| Tracker/policy distillation | Distill generated or reference motion into closed-loop control | Path to real robot policy | High collision with RoboPerform if framed broadly |
| RL fine-tuning | Optimize support, tracking, stability, energy, rhythm rewards | Real execution objective | Reward hacking and heavy sim complexity |

Recommended order:

1. motion prior only;
2. music-to-latent generator;
3. feasibility critic and reranking;
4. guided sampling or critic-regularized training;
5. tracker/action/PD-target policy only after pose-level generation is strong.

## Streaming And Policy Discussion

The sidechat separated three endpoints:

| Endpoint | Output | What it means |
|---|---|---|
| Offline generator | full G1 motion sequence | best choreography, full context |
| Streaming generator | next latent or motion chunk | causal or bounded-lookahead response |
| Real-time policy | PD target or action chunk | closed-loop robot-state-conditioned control |

The concern was that RoboPerform already occupies policy and DiscoForcing
already occupies streaming. The answer was: do not make streaming or policy the
core novelty. Make them endpoints of a robot-native dance prior.

### Distillation Route

```text
offline high-quality generator
    -> causal student
    -> receding-horizon motion policy
    -> tracker/action distillation
```

This is practical, but close to existing teacher/student narratives.

### Non-Distillation Route

```text
G1 dance latent prior
    -> direct causal or latency-conditioned latent generator
    -> receding-horizon motion generator
    -> tracker-conditioned motion policy
```

This trains causal behavior directly from real G1 dance latents and history
windows, without making the offline generator the teacher.

The preferred non-distillation candidate is:

```text
Latency-Conditioned Rhythm-to-Support G1 Dance Prior
```

Training input:

```text
past audio
bounded-lookahead audio based on latency_budget
previous latent or decoded motion history
previous support/contact state
latency_budget token
```

Training target:

```text
next real G1 dance latent chunk
```

This leaves a clean paper path that is not simply "we distilled a full-context
teacher into a streaming student."

## Condition Design Direction

The sidechat preserved the earlier decision to keep music conditions compact:

```text
semantic: Wav2CLIP
control: GaussianBeat, body_intensity, support_beatness
```

The important correction is conceptual:

- body intensity is not "motion beat";
- support beatness should represent rhythm/support events, not just high speed;
- controls should be typed and testable by condition ablations;
- future models should avoid blindly accumulating handcrafted features.

The next architecture should make these controls interact with a learned G1
motion prior rather than trying to force raw joint diffusion with more losses.

## What Not To Do

Avoid these directions unless a later experiment explicitly justifies them:

- Do not add more artifact-specific losses as the main contribution.
- Do not claim a policy unless robot state is in the loop.
- Do not claim streaming unless causal or bounded-lookahead eval is performed.
- Do not treat higher BAS as success if contact, root, endpoint quality, or
  diversity regresses.
- Do not make BeyondMimic an external cleanup layer for a broken generator.
- Do not copy RoboPerform's broad audio-to-humanoid teacher/student claim.
- Do not copy DiscoForcing's streaming claim without the G1 prior and support
  novelty.

## Concrete Documents Created From The Sidechat

- `docs/research/README.md`: entry point and new-agent reading path.
- `docs/research/ROBOT_NATIVE_DANCE_PRIOR_BLUEPRINT.md`: high-level method
  blueprint.
- `docs/research/modules/G1_MOTION_LATENT_PRIOR_SPEC.md`: continuous, VAE,
  VQ/RVQ, support/contact reconstruction options.
- `docs/research/modules/MUSIC_TO_LATENT_GENERATOR_SPEC.md`: music conditions
  and generator design.
- `docs/research/modules/FEASIBILITY_RL_SPEC.md`: feasibility critic, reranking,
  guided sampling, and RL/control routes.
- `docs/research/modules/STREAMING_POLICY_ROADMAP_SPEC.md`: distillation and
  non-distillation streaming/policy endpoints.
- `docs/papers/INDEX.md`: categorized paper library and novelty boundaries.
- `AGENTS.md`: research knowledge-base routing added for future agents.

## Next-Agent Checklist

Before proposing another architecture change, a new agent should:

1. Read `AGENTS.md`.
2. Read `docs/experiments/INDEX.md` to separate measured results from research
   ideas.
3. Read `docs/research/README.md` and this sidechat transcript.
4. Read the blueprint and the module spec for the component being changed.
5. Check `docs/papers/INDEX.md` before claiming novelty against RoboPerform,
   DiscoForcing, SONIC, or BeyondMimic.
6. Keep the next ablation narrow enough to prove one structural claim:
   G1 latent prior, music-to-latent generation, feasibility critic, streaming
   interface, or policy bridge.

## Current Best Blueprint In One Paragraph

Build a G1-native motion prior first, ideally continuous AE/VAE before RVQ.
Then train a compact Wav2CLIP + GaussianBeat + typed rhythm-control
music-to-latent generator. Add robot feasibility through calibrated diagnostics,
critic, and reranking before using heavier guidance or RL. Preserve chunk,
history, latency-budget, and decoder interfaces so the same system can later
become streaming or policy-like without rewriting the architecture. The
paper-level claim should be the G1-feasible rhythm-to-support dance prior, not
generic streaming or generic audio-to-humanoid control.

