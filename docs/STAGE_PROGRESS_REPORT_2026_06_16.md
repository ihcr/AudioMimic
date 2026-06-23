# Musics2Dance Stage Progress Report

Date: 2026-06-16

Scope: This document summarizes the current research progress in the
FineDance-G1 music-to-dance branch, with emphasis on the recent Wav2CLIP-based
direction and the beat-feature ablations. The goal is to explain the method
evolution and the main findings, not to document implementation details.

## Executive Summary

The project has moved from the original EDGE/Jukebox setup toward a lighter and
more controllable FineDance-to-G1 generation pipeline. The main lesson so far is
that better beat scores alone are not enough. A model can improve beat metrics
by collapsing motion, exploiting root rotation, creating endpoint jitter, or
breaking foot support. Therefore, every promising result is now judged by a
multi-objective gate: rhythm, diversity, distribution quality, root behavior,
contact/grounding, and qualitative long renders.

The most promising current direction is:

```text
Wav2CLIP semantic audio
+ GaussianBeat
+ root-local motion intensity and motion beatness controls
+ yaw-delta G1 root representation
```

This direction has produced the strongest recent quantitative rhythm and
distribution results, especially in the V5 yaw-delta experiment. However, V5 is
not yet a final model: it fixes the earlier root-spin and root-tilt problems,
but introduces visible endpoint/contact failures such as wrist jitter and
hovering or high-lift feet. The next stage should keep the yaw-delta
representation and redesign the motion-control target around contact-aware
support and endpoint smoothness.

The beat-only 8D feature experiment is complete and should be treated as a
negative ablation. It shows that adding richer beat timing channels alone does
not replace Wav2CLIP-family conditioning.

## Unified Model Comparison

The table below uses one BAS convention throughout: `G1FKRoboPerformBAS`.
This is the FK-based RoboPerform-style motion-to-music BAS, so it is aligned
with the RoboPerform interpretation rather than mixing paper-style
music-to-motion BAS variants. The selected checkpoint is the best balanced or
most representative checkpoint from each model family's experiment notes, not
the maximum of a single metric in isolation.

This table compares the FineDance-G1 full-test runs. Older AIST/Jukebox
beat-loss runs are useful historical motivation, but they are not mixed into
this table because the dataset and task setting are different.

| Model family | Selected checkpoint | Condition and representation | BAS (FK RoboPerform) | Beat F1 | G1Dist | G1Div | Foot slide | Ground pen. | Root/render status | Main takeaway |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| Librosa35 baseline | 2000 | Librosa35, old `g1` | 0.4504 | 0.2139 | 9.2544 | 11.3661 | 0.5349 | 0.0352 | Calm reference; low diversity | Strong contact/rhythm reference, but conservative motion |
| 1D GaussianBeat | 1000 | GaussianBeat only, old `g1` | 0.4199 | 0.1913 | 9.2000 | 20.5369 | 0.6015 | 0.0803 | Root drift 0.2709 | Useful beat-only lower bound with high diversity |
| Wav2CLIP/STFT r01 | 500 | Wav2CLIP + STFT + GaussianBeat concat, old `g1` | 0.4329 | 0.1924 | 11.8843 | 17.4055 | 0.8473 | 0.0751 | Root drift 0.3429 | Naive fusion keeps diversity but hurts quality |
| Wav2CLIP/STFT r02 | 2000 | Wav2CLIP + STFT + GaussianBeat, old `g1` | 0.4245 | 0.1979 | 8.9113 | 12.8445 | 0.5572 | 0.0408 | Root drift 0.2726 | Lightweight audio stack works, but remains averaged |
| R05 intensity control | 1000 | Wav2CLIP + GaussianBeat + predicted intensity, old `g1` | 0.4353 | 0.1866 | 5.3518 | 18.0232 | 0.7267 | 0.0861 | Root drift 0.0625 | Intensity improves distribution, but is not true beatness |
| V3 world-frame controls | 1000 | Wav2CLIP + GaussianBeat + intensity/beatness, old `g1` | 0.4568 | 0.2050 | 6.0560 | 18.4838 | 0.7462 | 0.0483 | Root-yaw exploit in long render | Controls are active, but world-frame target is unsafe |
| V3b root-local controls | 1500 | Wav2CLIP + GaussianBeat + root-local controls, old `g1` | 0.4517 | 0.2106 | 5.7822 | 14.0929 | 0.7639 | 0.0517 | RootAngP99 5.91, max 93.91 | Best pre-V5 Wav2CLIP anchor, but root tail remains |
| V4 root-delta | 1500 | V3b controls, `g1_root_delta` | 0.4418 | 0.2025 | 3.3720 | 14.4285 | 0.8544 | 0.0614 | Root max 16.72, but long-render tilt failure | Solves yaw spikes, fails through roll/pitch drift |
| V5 yaw-delta | 1000 | V3b controls, `g1_yaw_delta` | 0.4693 | 0.2340 | 3.9908 | 16.5609 | 0.8384 | 0.0757 | RootUpZP01 1.0, RootAngP99 2.95; contact/jitter fail | Strongest metrics so far, but not deployable |
| 8D beat-only | 1000 | 8D beat structure only, old `g1` | 0.4296 | 0.1934 | 9.1177 | 14.4888 | 0.6561 | 0.1580 | RootUpZP01 0.902, RootAngMax 72.14 | Negative ablation; beat channels alone are insufficient |

The table makes the main tradeoff clear. V5 is the strongest quantitative
checkpoint under the unified BAS convention, while Librosa35 still has the
cleanest contact. V4 and V5 show that representation changes can dramatically
improve root behavior and distribution quality, but the next failure mode moves
into endpoint/contact quality. The 8D beat-only model is not competitive with
the Wav2CLIP-family models once rhythm, distribution, and grounding are viewed
together.

## Starting Point: EDGE, Jukebox, and G1 Beat Loss

The original EDGE method uses a Transformer diffusion model conditioned on
Jukebox music features. In this branch, the goal changed from human dance only
to Unitree G1 robot motion, using FineDance-derived music and retargeted G1
motion. This introduced a harder acceptance criterion: the generated motion must
not only align with music, but also remain physically plausible for the robot.

Early G1 beat-loss work showed that explicit beat supervision can raise rhythm
metrics substantially. This was useful because it proved that the model can be
steered toward stronger musical alignment. But it also exposed a central
failure mode: a higher BAS or F1 score can come with worse foot sliding, worse
ground penetration, and lower physical quality.

This changed the research methodology. Since then, beat metrics are never
interpreted alone. A result is only considered useful if rhythm improves while
motion range, diversity, contact, root stability, and long-render quality remain
healthy.

## Methodology Evolution

### 1. Replace heavy Jukebox features with lighter music features

The first Wav2CLIP experiment tested whether the model could train with a
lighter music stack:

```text
Wav2CLIP + STFT + GaussianBeat
```

Two fusion schemes were tested: direct normalized concatenation and a stream
adapter. The stream-adapter version was continued to 2000 epochs and became the
first usable Wav2CLIP/STFT/GaussianBeat checkpoint.

Main finding: Wav2CLIP-based conditioning is trainable and useful, but richer
audio features alone do not solve motion quality. The model still tended toward
low-amplitude, averaged motion. This suggested that the problem was not only
audio representation, but also the lack of explicit motion-control structure.

### 2. Add explicit motion-derived control signals

The next direction introduced a structured condition:

```text
semantic condition: Wav2CLIP
control condition: GaussianBeat + motion-derived control
```

The first version used a "motion energy" target built from speed maxima around
beats. Evaluation showed that this signal was misnamed. It represented motion
intensity or activity, not true motion beatness. The reported FK beat metrics
reward local speed minima, holds, and turnarounds near audio beats, so a
speed-max target was not aligned with the evaluation objective.

The corrected V3 formulation separated the control into two typed signals:

- `motion_intensity`: when the motion should be large or active.
- `motion_beatness`: when the motion should form a beat event through a local
  minimum, hold, or turnaround.

Main finding: this split worked. Zeroing the beatness control lowered rhythm
metrics, and zeroing all controls collapsed diversity and motion range. The
controls were therefore active, not decorative.

### 3. Remove root-yaw exploits from motion controls

V3 still had a serious problem: because motion intensity and beatness were
computed in the world frame, the model could satisfy the control signal by
spinning the root. Fast yaw changes moved wrists, ankles, and torso through
world space even when the local dance motion was not healthy.

V3b rebuilt the motion-control features in a root-local frame and added root
angular diagnostics. This improved the best Wav2CLIP-family reference point.
The best V3b checkpoint is currently the checkpoint-1500 predicted-control
model:

| Model | BAS (FK RoboPerform) | Beat F1 | G1Dist | G1Div | Notes |
|---|---:|---:|---:|---:|---|
| V3b local controls, ckpt1500 | 0.4517 | 0.2106 | 5.7822 | 14.0929 | Strongest stable Wav2CLIP-family comparison anchor before V5 |

Main finding: root-local controls improved the signal, but did not eliminate
the high-speed root-rotation tail. This shifted the focus from feature design
to motion representation.

### 4. Redesign the G1 root representation

The root representation experiments tested whether the root trajectory itself
was causing the failure.

V4 used local root deltas and relative full SO(3) root rotation. It solved the
rare extreme yaw spikes: root angular velocity maxima dropped dramatically.
However, long 90-second renders revealed a worse failure: roll/pitch drift
accumulated over time, and the robot could lie sideways or float. This showed
that full SO(3) delta integration is not safe for long robot dance generation.

V5 kept the successful local root translation idea but represented root heading
as yaw-only delta:

```text
root local xy delta
+ root height
+ delta-yaw sin/cos
+ G1 joint positions
```

This removed roll/pitch integration entirely. At checkpoint 1000, V5 passed the
main root-stability gates and produced the strongest recent quantitative
result:

| Model | BAS (FK RoboPerform) | Beat F1 | G1Dist | G1Div | RootUpZP01 | RootAngP99 | RootAngMax |
|---|---:|---:|---:|---:|---:|---:|---:|
| V5 yaw-delta, ckpt1000 | 0.4693 | 0.2340 | 3.9908 | 16.5609 | 1.0000 | 2.9524 | 14.9566 |

Main finding: yaw-delta is the right root-representation direction so far. It
fixes the root-spin and root-tilt issues better than the earlier absolute-root
or full-SO(3)-delta representations.

However, V5 failed the naturalness gate in qualitative render and endpoint
diagnostics. Compared with V3b, V5 showed substantially more wrist FK jerk, more
frames without near support, more high-lift foot frames, and lower contact-proxy
rate. This means the model found a new way to satisfy rhythm and intensity:
high-frequency endpoint motion and poor support contact.

The next step should not be "train V5 longer." It should be V6: keep yaw-delta,
but redesign the control target and objective around support/contact.

## Latest Wav2CLIP Direction: What We Have Learned

The Wav2CLIP direction has produced a clear methodological path:

1. Wav2CLIP can replace Jukebox as a lighter semantic audio stream.
2. Wav2CLIP alone is not enough because the model can average motion.
3. Motion-derived controls help, especially when intensity and beatness are
   separated.
4. Control signals must be computed in a robot-meaningful frame; world-frame
   speed can be exploited by root yaw.
5. Root representation matters as much as conditioning. The current best root
   direction is yaw-only local root delta.
6. The next bottleneck is contact and endpoint behavior, not root spin.

In short, the project has moved from feature replacement to structured
robot-aware control.

## Beat-Feature Exploration

The beat-feature line was designed to answer a narrower scientific question:
can explicit beat timing features alone replace richer music features?

### 1D GaussianBeat baseline

The 1D GaussianBeat model removed Wav2CLIP, STFT, Librosa, and Jukebox. It used
only a single Gaussian beat curve as the music condition. This worked as a
beat-only lower-bound model and preserved high diversity:

| Model | BAS (FK RoboPerform) | Beat F1 | G1Dist | G1Div |
|---|---:|---:|---:|---:|
| 1D GaussianBeat, ckpt1000 | 0.4199 | 0.1913 | 9.2000 | 20.5369 |

But condition sensitivity showed a weakness: real beats, shifted beats, and
random beats produced very similar benchmark scores. The model reacted to
having a beat-like condition, but it did not strongly use the exact timing.

### 8D beat-only feature

The 8D beat-feature experiment extended GaussianBeat with denser beat structure:

```text
beat pulse
GaussianBeat
distance to previous beat
distance to next beat
beat phase sin/cos
beat interval
onset strength
```

This was intentionally beat-only: no Wav2CLIP, no STFT, no Librosa, and no
structured motion-control predictor. The run completed to 1000 epochs and full
evaluation.

Final result:

| Model | BAS (FK RoboPerform) | Beat F1 | Precision | Recall | G1Dist | G1Div |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8D beat-only, ckpt1000 | 0.4296 | 0.1934 | 0.2979 | 0.1551 | 9.1177 | 14.4888 |

Compared with 1D GaussianBeat, the 8D feature slightly improved precision, but
did not improve the main robot-aligned rhythm picture. It also caused large
regressions in diversity, ground penetration, and root drift. Compared with the
strongest pre-V5 Wav2CLIP-family reference, V3b checkpoint 1500, the 8D model is
clearly behind on rhythm and distribution quality:

| Model | BAS (FK RoboPerform) | Beat F1 | Precision | Recall | G1Dist | G1Div |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8D beat-only, ckpt1000 | 0.4296 | 0.1934 | 0.2979 | 0.1551 | 9.1177 | 14.4888 |
| V3b Wav2CLIP-family, ckpt1500 | 0.4517 | 0.2106 | 0.3225 | 0.1687 | 5.7822 | 14.0929 |

Main finding: richer beat timing features alone are not enough. They are useful
as a lower-bound ablation and diagnostic, but they are not a practical
replacement for richer audio semantics and robot-aware motion controls.

## Current Interpretation

The best current explanation of the results is:

- Music semantics and beat timing solve different parts of the problem.
- Beat-only features can provide a rhythm prior, but do not reliably control
  exact motion beat placement.
- Wav2CLIP provides useful high-level music context, but needs explicit
  motion-control channels to avoid averaged motion.
- Motion controls must be aligned with the metric and the robot body. A target
  based on wrist-heavy speed can create endpoint jitter; a target based on
  world-frame speed can create root-spin exploits.
- Robot root representation is a first-class modeling choice. Yaw-delta is the
  best current compromise for long-horizon G1 dance.

The biggest open issue is no longer whether the model can improve rhythm. It
can. The issue is how to improve rhythm without letting the model trade it for
bad contact, jitter, high-lift feet, or unnatural endpoint behavior.

## Metric Hacking and Model Quality

One important lesson from the Wav2CLIP V3 and later experiments is that higher
scores do not automatically mean better generated dance. Several designs raised
one or more rhythm metrics while producing visibly worse behavior in render:

- V3 improved structured control use, but world-frame motion controls allowed a
  root-yaw exploit. The model could create large world-space wrist, ankle, and
  torso movement by spinning the root, which helped satisfy motion-control
  signals without producing natural local dance.
- V4 root-delta fixed the rare extreme yaw spikes, but full SO(3) integration
  introduced long-horizon roll/pitch drift. The short metrics looked promising,
  while the 90-second render showed the robot lying sideways or floating.
- V5 yaw-delta fixed the root-up and root-spin problems and produced the best
  unified BAS/F1 numbers so far, but qualitative render and diagnostics exposed
  wrist jitter, hovering feet, high-lift support failures, and weak contact.
- The 8D beat-only ablation modestly improved some beat-timing quantities, but
  diversity, grounding, and root behavior regressed enough that it is not a
  practical model.

This means the evaluation protocol should be treated as a Pareto filter rather
than a leaderboard. A checkpoint is only better if the score improvement also
survives contact, root, diversity, endpoint-smoothness, and long-render checks.

The same lesson also suggests that the next major gains may not come from
adding another small conditioning channel or increasing the beat loss. The
overall architecture may need to change. Promising longer-term directions
include:

- **Streaming or real-time policy-style generation.** Instead of generating
  fixed offline clips and stitching windows, the model could move toward a
  causal policy that continuously reacts to music. DiscoForcing is relevant as
  an example of treating music-to-dance as a streaming generation problem.
- **Robot-native physical plausibility.** The model should include stronger
  physical constraints or learned physical priors, especially support contact,
  foot height, center-of-mass behavior, endpoint smoothness, and balance-related
  root motion. This is the path toward motions that are not only visually
  rhythmic but also plausible for deployment on a humanoid robot.
- **Retargeting-free generation.** The long-term goal should be explicitly
  robot-native: generate motions that do not rely on human-to-robot retargeting
  as a required postprocess, and make deployment constraints part of the
  generation problem itself.

In short, the recent experiments show that feature engineering and loss design
are useful, but the bigger research direction is architectural: move from
"generate a motion sequence that scores well" toward "generate robot-native
motion that remains stable, physically plausible, and musically responsive
under deployment constraints."

## Recommended Next Stage

The next stage should be a V6 experiment with these design choices:

1. Keep the V5 `g1_yaw_delta` root representation.
2. Keep Wav2CLIP as the semantic audio stream.
3. Keep GaussianBeat as an auxiliary rhythm signal.
4. Rebuild motion beatness to be support-aware and contact-aware.
5. Reduce or cap wrist contribution in the motion-control target.
6. Add explicit support/contact supervision or representation.
7. Add endpoint smoothness and contact-quality gates to every 500-epoch full
   evaluation.

Suggested acceptance gates for V6 should include:

- Rhythm: `G1FKRoboPerformBAS`, `G1BeatF1`, precision, and recall.
- Distribution: `G1Dist`, `G1Div`, joint range, and root range.
- Root stability: root-up metrics, root angular p99/max, drift.
- Contact: foot sliding, ground penetration, support/contact proxy rate.
- Endpoint quality: wrist FK jerk, high-lift rate, and no-near-support rate.
- Qualitative validation: matched long MuJoCo render with verified feature
  extraction route.

## Bottom Line

The current research has produced a strong direction, but not yet a final
deployable model. The most important progress is conceptual: the project now
has a clearer decomposition of the problem.

```text
Wav2CLIP gives music semantics.
GaussianBeat gives an explicit rhythm prior.
Motion intensity prevents averaged low-amplitude dance.
Motion beatness encourages beat events.
Yaw-delta root representation stabilizes long-horizon G1 root motion.
Contact-aware support is the next missing piece.
```

The 8D beat-only experiment is valuable because it rules out a tempting simple
solution. More beat channels alone do not solve G1 music-to-dance generation.
The practical path forward is richer audio semantics plus robot-aware,
support-aware motion controls.
