# Feasibility And RL Spec

Last updated: 2026-06-18.

## Purpose

Bring robot execution information into the robot-native dance generator without
turning the first architecture ablation into a full RL/control project. This
module defines how to use MuJoCo, trackers, RL policies, and feasibility critics
as staged components.

## Decision Summary

Use RL signals, but do not start with end-to-end RL music-to-action. The first
use of robot execution should be diagnostics and labels. Then train a
feasibility critic. Only after the pose/latent generator is stable should the
project consider an action or PD-target decoder.

Recommended route:

```text
generated motion
    -> FK/MuJoCo/tracker diagnostics
    -> feasibility labels
    -> feasibility critic
    -> rerank or guide latent generation
```

## RL And Feasibility Routes

| Route | Definition | Advantages | Risks | Recommendation |
|---|---|---|---|---|
| No RL, kinematic prior only | AE/VAE plus FK/contact/root losses | Fast, local, debuggable | Physical feasibility is still proxy-based | Required first baseline |
| Tracker as evaluator | Run generated motion through SONIC/BeyondMimic-style tracker or MuJoCo tracking proxy | Minimal intrusion; directly tests executability | External stack may be slow or brittle | First robot-control integration |
| RL-derived feasibility critic | Use tracker/MuJoCo success, error, action smoothness, contact failure as labels | Internalizes robot feedback into generation without full RL | Critic can learn biased shortcuts | Best medium-term route |
| Critic reranking | Sample multiple motions and choose high feasibility/rhythm Pareto candidates | Easy and low-risk | More inference cost; no training-time improvement | First critic use |
| Critic-guided sampling | Use critic gradients or score guidance during denoising | More direct generation improvement | Can cause adversarial critic hacking | Use after critic calibration |
| Critic-regularized training | Add critic loss during generator training | Internalizes feasibility | Can suppress dance amplitude/diversity | Later ablation with caps |
| BC/RL action decoder | Decode latent to PD targets/actions using behavior cloning and RL | Truly robot-native execution layer | Heavy sim/control engineering | Long-term only |
| End-to-end RL music-to-action | Reward combines beat, style, stability, contact | Direct in principle | Reward hacking, unstable training, weak music semantics | Not recommended as mainline |

## Why Not Direct RL First

Direct RL music-to-action would mix several unsolved problems:

- learning choreography from audio;
- preserving beat and semantic alignment;
- maintaining dance diversity;
- enforcing contact/support;
- learning stable humanoid control;
- avoiding reward hacking.

Recent control papers support using RL for tracking, correction, and execution
robustness, but they do not imply that our first music generator should be an
RL policy. PDP uses RL policies for corrective actions in sub-optimal states.
BeyondMimic uses tracking and state-action diffusion for versatile control.
KungfuBot uses motion processing and adaptive tracking for highly dynamic G1
skills. SONIC scales motion tracking as a foundation controller. These are
powerful feasibility/control references, not direct replacements for a
music-to-dance generator.

## Candidate Labels

From FK and existing eval:

- foot sliding;
- ground penetration;
- no-near-support rate;
- high-lift frame rate;
- wrist/foot FK jerk p95;
- root drift;
- root yaw angular velocity p99/max;
- root tilt/root-up metrics.

From MuJoCo/tracker when available:

- tracker success/failure;
- fall/termination;
- tracking error;
- action magnitude;
- action jerk;
- joint limit pressure;
- contact mismatch;
- center-of-mass or support polygon failure if exposed.

## Feasibility Critic Design

Inputs:

```text
motion sequence
FK endpoints
contact/support channels
root/yaw features
optional latent sequence
```

Outputs:

```text
score_feasible: scalar or Tensor[B]
score_contact: scalar
score_smoothness: scalar
score_tracking: scalar if tracker labels exist
```

Training:

- start as supervised regression/classification from diagnostic labels;
- normalize labels by train/eval percentiles;
- train on real generated failures, not only clean dataset clips;
- keep a held-out diagnostic set from fixed checkpoints and music clips.

Usage order:

1. Report-only diagnostic.
2. Reranking among multiple generated samples.
3. Pareto selection with rhythm/diversity/contact metrics.
4. Guidance during denoising.
5. Training regularizer with cap.

## Acceptance Gates

A critic is useful only if:

- high critic score correlates with lower foot sliding/penetration;
- high critic score correlates with lower wrist/foot jerk;
- high critic score does not simply prefer low-amplitude still motion;
- reranking improves contact/root without collapsing `G1Div`;
- zero/flat control sensitivity remains visible after reranking;
- tracker success improves when tracker labels are available.

## Ablation Matrix

| Ablation | Question | Accept signal |
|---|---|---|
| FK-only labels vs MuJoCo labels | Are cheap labels enough? | FK labels predict render/tracker failures |
| critic report-only vs rerank | Does critic ranking help without training risk? | Rerank improves naturalness at fixed rhythm |
| rerank sample count | How many candidates are needed? | Small N improves Pareto without large cost |
| critic guidance scale | Can guidance improve samples directly? | Smooth Pareto improvement, no stillness collapse |
| critic loss cap | Can training use critic safely? | No `G1Div` or amplitude collapse |
| tracker evaluator | Does generated motion execute under SONIC/BeyondMimic-style tracking? | Higher success and lower correction effort |

## Failure Modes

- Critic prefers still or low-amplitude motion.
- Critic learns metric artifacts rather than true feasibility.
- Guidance exploits critic blind spots.
- Tracker stack corrections hide generator failures.
- RL/control work consumes effort before the generator prior is validated.
- Feasibility improvements regress beat alignment and diversity.

## Source Anchors

- SONIC / motion tracking as scalable humanoid prior: https://arxiv.org/abs/2511.07820
- BeyondMimic / tracking plus guided state-action diffusion: https://arxiv.org/abs/2508.08241
- KungfuBot / adaptive tracking for highly dynamic G1 skills: https://arxiv.org/abs/2506.12851
- PDP / BC plus RL corrective policies for physics-based animation: https://arxiv.org/abs/2406.00960
- RoboPerform / audio-to-humanoid teacher/student reference: https://arxiv.org/abs/2512.23650

