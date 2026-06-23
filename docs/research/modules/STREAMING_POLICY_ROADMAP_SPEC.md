# Streaming And Policy Roadmap Spec

Last updated: 2026-06-18.

## Purpose

Keep the offline robot-native dance generator compatible with later streaming
and real-time policy upgrades. This spec records what must be designed now so
the project does not become a full-song-only generator that later requires a
rewrite.

## Boundary

Do not conflate three different endpoints:

| Endpoint | Output | Timing | What it solves | What it does not solve |
|---|---|---|---|---|
| Offline generator | Full G1 motion sequence | Can use full-song context and slower inference | Best choreography and motion quality | Not causal or closed-loop |
| Streaming generator | Next latent/motion chunk | Causal or bounded-lookahead, interactive frame rate | Live music response and continuity | Not necessarily robot action control |
| Real-time policy | PD target or action chunk | Closed-loop, robot-state conditioned | Execution, tracking, disturbance correction | Choreography quality can still depend on upstream intent |

The current project should first build the offline G1 dance prior well, but its
interfaces should be chunkable, causal-testable, and policy-ready.

## Collision Map

| Work | Occupied endpoint | Why it matters | Open space for this project |
|---|---|---|---|
| RoboPerform | Retargeting-free audio-to-humanoid policy with teacher-student audio style injection | Strongest collision if the claim is simply "audio drives humanoid dance" | Dance-specific G1 prior, rhythm-to-support controls, editable robot dance, robot dance benchmark |
| DiscoForcing | Real-time streaming audio-to-motion with causal music encoder and diffusion forcing | Strongest collision if the claim is simply "streaming music-to-dance" | Robot-native generation and support/contact-aware dance prior |
| SONIC | Scaled universal humanoid tracking controller and token/control interface | Strongest template for policy/control layer | Music-conditioned choreography and G1 dance latent prior |
| DanceMosaic/DuetGen/SoulDance | Human dance priors, tokens, hierarchy, editability | Strong human dance references | Robot feasibility, support/contact rhythm, G1-specific evaluation |

Therefore the first paper claim should not be "streaming" or "audio-to-policy".
The stronger claim is:

```text
music-conditioned G1-feasible dance prior
with rhythm-to-support controls and robot-feasibility gates
```

Streaming and policy should be framed as natural extensions enabled by the
chosen architecture.

## Core Routes

There are two viable long-term upgrade routes. Distillation is the safest
engineering route, but it should not be the only route because RoboPerform and
DiscoForcing already make teacher/student and streaming endpoints visible.

### Route A: Distillation Path

```text
offline high-quality G1 dance generator
    -> causal student over the same latent/control space
    -> receding-horizon motion policy
    -> tracker/action distillation
    -> real-time audio-conditioned robot policy
```

This makes the G1 dance prior the reusable center of the project, rather than a
throwaway offline generator.

### Route B: Non-Distillation Path

```text
G1 dance latent prior
    -> direct causal or latency-conditioned latent generator
    -> receding-horizon motion generator
    -> tracker-conditioned motion policy
    -> action or PD-target policy
```

This path trains causal/streaming behavior directly from real G1 dance data and
history windows instead of imitating an offline teacher. It gives a cleaner
novelty route if the project wants to avoid looking like a RoboPerform-style
teacher/student policy or a DiscoForcing-style streaming compression pipeline.

The preferred non-distillation variant is:

```text
past audio + latency_budget + latent/motion history + support/contact history
    -> next G1 support-aware dance latent chunk
```

where `latency_budget` is sampled during training from values such as `0ms`,
`250ms`, `500ms`, `1000ms`, and `full_context`.

## Design Requirements To Preserve Now

### 1. Chunk-First API

Even if training starts offline, model code should support fixed windows:

```text
audio_window, control_window, optional_history
    -> latent_chunk
    -> motion_chunk
```

Full-song generation should be a wrapper around chunk generation, not a separate
hard-coded path.

Required future hooks:

- chunk length: `0.5s`, `1s`, `2s`, `4s` as ablations;
- overlap and crossfade or latent inpainting;
- explicit sequence start/reset markers;
- fixed-latency evaluation with the same chunk boundaries.

### 2. Causal Audio And Control Variants

Every non-causal condition should have a causal or bounded-lookahead variant:

- Wav2CLIP semantic window with past-only or limited lookahead;
- GaussianBeat updates from online beat tracking or bounded-lookahead beat
  estimation;
- body/support controls predicted from past audio and history, not future motion;
- explicit lookahead settings: `0ms`, `250ms`, `500ms`, `1000ms`.

Offline teacher quality can use full context, but causal student evaluation must
not.

### 3. History-Conditioned Generator

A streaming student must receive history:

```text
past audio features
past generated latent
past decoded motion
past support/contact state
optional robot state
    -> next latent or motion chunk
```

This is where DiscoForcing and MotionStreamer are useful references. The key is
not to copy their whole architecture, but to preserve history buffers and
causal/latency-aware evaluation.

### 4. Standalone Latent Decoder

The G1 latent decoder must remain callable independently:

```text
latent_chunk -> g1_yaw_delta motion + support/contact
```

This enables:

- offline generator;
- causal student;
- policy/action decoder;
- teacher-student datasets;
- tracker or MuJoCo diagnostics.

If the decoder is entangled with full-song audio conditioning, future streaming
or policy distillation becomes much harder.

### 5. Action Bridge

Pose generation should preserve fields needed by a future policy layer:

- root-local velocity;
- yaw delta;
- support/contact channels;
- foot phase or support transition;
- body intensity/support beatness controls;
- optional PD-target/action placeholders;
- timestamps and frame-rate metadata.

The generator does not need to output actions first, but its representation
should not discard the information an action decoder or tracker needs.

### 6. Teacher And Training Data Logging

Offline generation should save enough data to train a causal student later, but
the same logs should also support direct causal training diagnostics:

```text
audio features
control features
offline teacher latent
decoded motion
support/contact predictions
random seed and sampler settings
quality/feasibility metrics
latency budget if sampled
history window metadata
```

This avoids rerunning expensive full-song generation when training a streaming
student. It also allows direct causal models to be evaluated against the same
fixed histories and quality gates without treating the offline model as a
teacher.

### 7. Streaming Evaluation

Add evaluation modes before claiming streaming:

- offline teacher full context;
- direct causal generator or causal student with `0ms`, `250ms`, `500ms`,
  `1000ms` lookahead;
- chunk boundary jerk and discontinuity;
- response to beat/tempo edits;
- long-horizon drift;
- rhythm/quality/contact metrics under identical audio clips.

The goal is not only throughput. It must preserve rhythm, contact, endpoint
naturalness, and diversity.

### 8. Policy Evaluation

Do not call the method a policy unless robot state is part of the loop.

Policy-level evaluation should include:

- proprioceptive state input;
- action or PD-target output;
- closed-loop rollout;
- tracking error;
- fall or termination;
- contact/support success;
- action jerk and joint limit pressure.

Before that stage, call the system a streaming generator or receding-horizon
motion generator.

## Route Options

| Route | Description | Advantages | Risks | When to use |
|---|---|---|---|---|
| Offline-only prior | Full-context music-to-latent generator | Highest quality first; easiest paper baseline | Can become non-causal dead end | Phase 1, but keep chunk API |
| Causal student distillation | Offline teacher generates targets, student sees only causal windows/history | Converts quality model into streaming model without retraining from scratch | Student may lag or smooth beats | After offline prior works |
| Direct causal latent training | Train `history + causal audio -> next latent chunk` directly on real data latents | Native streaming ability; no teacher/student claim | Harder than offline; exposure bias and long-horizon drift | Best non-distillation mainline |
| Latency-conditioned generator | Sample `latency_budget` during training and condition the model on available lookahead | One model covers offline, bounded-lookahead, and strict-causal modes | May trade off peak offline quality for flexibility | Strongest non-distillation novelty candidate |
| Diffusion-forcing or horizon-noise training | Train different temporal horizons with different noise/history conditions | Learns short-term responsiveness and long-term consistency jointly | Close to DiscoForcing if copied directly | Use only with G1 support/rhythm latent novelty |
| Receding-horizon motion generator | Replan short latent/motion chunks repeatedly | Bridges streaming and robot execution | Needs continuity handling | After causal student works |
| Tracker-conditioned motion policy | Include current tracked state and generate next reference chunk | More robust under execution mismatch | Requires tracker/MuJoCo loop | Before action policy |
| Action/PD-target decoder | Decode latent intent plus robot state into actions | Real robot-native policy endpoint | Heavy control/sim engineering | Final stage |
| End-to-end audio-to-action | Direct audio/proprioception to action | Clean endpoint in principle | High collision with RoboPerform; high reward/data risk | Avoid as first claim |
| Latent MPC or online optimization | Optimize the next latent/action chunk online using rhythm and feasibility costs | No teacher, strong constraint control, useful upper bound | May be too slow for real-time and hard to tune | Diagnostic or upper-bound route |

## Preferred Non-Distillation Design

The most promising non-distillation direction is:

```text
Latency-Conditioned Rhythm-to-Support G1 Dance Prior
```

Training target:

```text
real G1 dance latent at t:t+h
```

Inputs:

```text
past audio window
bounded-lookahead audio window depending on latency_budget
previous latent or decoded motion history
previous support/contact state
latency_budget token
optional robot/tracker state in later policy stages
```

Training policy:

- randomly sample `latency_budget`;
- mask future audio according to that budget;
- condition on ground-truth history early;
- add scheduled sampling or two-forward/self-history training later;
- predict the next latent chunk directly;
- decode and apply FK/contact/rhythm losses;
- evaluate strict-causal, bounded-lookahead, and full-context modes separately.

Why this is different from distillation:

- no offline teacher motion is required as the target;
- streaming behavior is native to the training objective;
- the claim can focus on latency-conditioned G1 support/rhythm dynamics rather
  than teacher/student compression.

## Novelty Boundary

The project should not claim novelty for:

- teacher-student distillation by itself;
- avoiding teacher-student distillation by itself;
- streaming by itself;
- audio-to-humanoid policy by itself;
- universal humanoid tracking by itself.

The stronger novelty boundary is:

- rhythm-to-support modeling for robot dance;
- G1-feasible dance latent prior;
- music-conditioned generation in that robot-native prior;
- support/contact-aware condition sensitivity;
- feasibility-aware evaluation and critic/reranking;
- future-compatible chunk/causal/policy interfaces;
- optional latency-conditioned causal generation without relying on an offline
  teacher.

This leaves streaming and policy as credible endpoints without making them the
core novelty that RoboPerform or DiscoForcing already occupy.

## Acceptance Gates

Before moving from offline to streaming:

- offline latent generator beats or matches raw-pose baseline on rhythm,
  `G1Dist`, `G1Div`, contact, root, and endpoint gates;
- causal generator or student with `500ms` or less lookahead preserves most
  offline quality;
- chunk boundaries do not introduce visible motion pops;
- support/contact predictions remain coherent across chunks;
- beat response does not lag more than the chosen latency budget.

Before moving from streaming to policy:

- tracker/MuJoCo closed-loop diagnostics exist;
- generated motion can be tracked without frequent fall/termination;
- action/PD-target data can be collected or synthesized;
- policy evaluation includes robot state, not only reference playback.

## Source Anchors

- DiscoForcing / causal streaming audio-to-motion: https://arxiv.org/abs/2605.28491
- MotionStreamer / continuous causal latent streaming motion: https://arxiv.org/abs/2503.15451
- SONIC / universal humanoid tracking and token/control interface: https://arxiv.org/abs/2511.07820
- RoboPerform / retargeting-free audio-to-humanoid policy: https://arxiv.org/abs/2512.23650
- Diffusion Policy / receding-horizon action diffusion: https://diffusion-policy.cs.columbia.edu/
- RTC / real-time action chunking: https://arxiv.org/abs/2506.07339
