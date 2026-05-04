# Beat-Only Debugging Playbook

## Scope

This branch is **not** a full reproduction of Beat-It. It is better understood as:

- **EDGE baseline**: the original Jukebox-conditioned diffusion model
- **Beat-only extension**: add Beat-It-inspired beat conditioning and beat supervision
- **No keyframe path**: deliberately excluded for now

So the research question for this branch is:

> Can Beat-It-style beat conditioning improve EDGE, without adding keyframes?

That means the correct comparison is:

1. local EDGE baseline in this branch
2. beat-conditioned EDGE without `Lbeat`
3. beat-conditioned EDGE with `Lbeat`

and **not** a direct claim that we have already reproduced the full Beat-It model.

## Why We Still Train A Local Baseline

There is already an official EDGE checkpoint at `checkpoint.pt`, and it is very useful as an anchor.
However, we still need a **fresh local baseline run** in this branch for fair ablations.

Reasons:

1. This branch changed training, preprocessing, evaluation, checkpoint loading, and condition handling.
2. A fresh local baseline tells us whether the branch still trains a healthy non-beat model.
3. If the local baseline is already bad, the problem is branch-wide rather than beat-specific.
4. If the local baseline is healthy but beat-conditioned runs are worse, the failure is in the beat path.
5. Comparing new beat runs only against the official checkpoint mixes together architecture changes, infra changes, and beat changes.

In short:

- `checkpoint.pt` answers: "What does official EDGE look like under our evaluator?"
- a new local baseline run answers: "What does **this branch** do before we add beats?"

We need both.

## Current Evidence

### 2026-04-30 safe `Lbeat` update

The old high-weight `Lbeat` run collapsed, so `Lbeat` is now treated as a cautious fine-tune on top of the working BeatDistance model.

Safe defaults:

1. start from a BeatDistance checkpoint
2. use `lambda_beat=0.02`
3. use `lambda_acc=0.1`
4. train for `500` epochs
5. save every `50` epochs
6. start beat loss at epoch `25`
7. warm beat loss over `200` epochs
8. cap beat-loss contribution at `25%` of the normal motion loss
9. require a beat-estimator checkpoint with finite validation loss no higher than `8.0`

The safe Slurm preset screens checkpoints at `50,100,200,300,400,500`, then runs the full 186-clip eval only on the selected checkpoint. The selection report is written to `eval/lbeat_selection.json`.

Important interpretation rule:

- If `accepted=false`, the model was evaluated successfully but did not pass the quality gate.
- A rejected model should be reported as rejected, not as an evaluator crash and not as a better model.
- The helper only exits non-zero on rejection when `--fail_on_rejection` is explicitly passed.

### Previous evaluator anchor

Earlier evidence showed the old high-weight `Lbeat` path was genuinely worse than the stable BeatDistance path. The practical lesson remains: do not accept a beat-alignment gain without checking PFC and the full quality gate.

### Structural context

This branch intentionally omits major Beat-It components:

1. keyframe conditioning
2. hierarchical multi-condition fusion
3. the full Beat-It evaluation protocol

So we should describe the current work as a **beat-only ablation inspired by Beat-It**, not as a full Beat-It reproduction.

### Strongest quantitative warnings found so far

1. The old `Lbeat` recipe used too much beat-loss weight too early.
2. The beat estimator must pass validation before it is trusted for `Lbeat`.
3. Generated motions from bad beat-loss runs showed physical problems.
   - Foot skating and motion scale looked much worse than GT.
   - This is consistent with PFC inflation.
4. Stored `motion_dist` and `audio_dist` labels are only weakly aligned.
   - That makes beat-only control harder, especially without keyframes.

## What `Lbeat` Is

`Lbeat` is the auxiliary beat-alignment loss used during diffusion training.

In this branch:

1. the model generates motion
2. FK converts that motion to joints
3. the frozen beat estimator predicts a beat-distance curve from generated joints
4. that predicted curve is compared with the target beat-distance curve

So `Lbeat` is only as trustworthy as the beat estimator itself.

If the estimator is weak, `Lbeat` can actively push training in the wrong direction.

That is why the most important ablation is:

- **BeatDistance without `Lbeat`**

If that helps over baseline, but adding `Lbeat` hurts, then the beat representation may be fine and the estimator-supervised loss is the real problem.

## Most Likely Failure Causes

### 1. Weak beat estimator harms `Lbeat`

Confidence: **High**

Why this matters:

1. `Lbeat` depends on a frozen estimator.
2. A bad estimator gives misleading gradients.
3. The older `10`-epoch checkpoint looked much too weak to trust.

What to test:

1. Train a fresh estimator for `50` epochs with validation.
2. Compare `BeatDistance` with `lambda_beat=0.0` against `BeatDistance` with `lambda_beat=0.5`.
3. Accept `Lbeat` only if the fresh-estimator run improves BAS without wrecking PFC.

### 2. Beat conditioning itself may currently hurt EDGE

Confidence: **High**

Why this matters:

1. Even without `Lbeat`, the beat-conditioned decoder changes the conditioning pathway.
2. If fusion is poor, the model can lose motion quality before the auxiliary beat loss even matters.

What to test:

1. Compare local baseline vs `BeatDistance` with `lambda_beat=0.0`.
2. If no-`Lbeat` already hurts PFC badly, the problem is likely the conditioning path or representation, not the estimator.

### 3. Training recipe drift from the original EDGE baseline

Confidence: **Medium-high**

Why this matters:

1. Official EDGE was trained under a validated recipe and environment.
2. This branch changed defaults, launcher logic, checkpoint behavior, and evaluation automation.
3. Even small training-recipe drift can move the baseline.

What to test:

1. Compare the fresh local baseline against `checkpoint.pt`.
2. If the local baseline is much worse than the official checkpoint, fix branch-wide training drift before blaming beats.

### 4. Undertraining may still matter

Confidence: **Medium**

Why this matters:

1. Earlier runs often stopped around `600` epochs because of walltime.
2. AIST++ diffusion training can continue improving after that.
3. The beat estimator also looked undertrained at only `10` epochs.

What to test:

1. Use the `600`-epoch jobs as screening runs.
2. Only resume promising variants.
3. Do not spend more GPU time extending clearly bad runs.

Decision rule:

- If a run is already worse than baseline by epoch `600`, do not blindly extend it to `2000`.

### 5. Beat labels may be too noisy for beat-only control

Confidence: **Medium-high**

Why this matters:

1. The training path uses GT motion beats.
2. The test path uses audio beats.
3. Weak alignment between the two makes the task harder.
4. Without keyframes, the beat signal may be too ambiguous on its own.

What to test:

1. Visually inspect random `motion_dist` and `audio_dist` pairs.
2. Compare stored `motion_beats` against the evaluation-time motion-beat detector on GT clips.
3. Measure beat count and spacing distributions.

Important:

- Do this **after** the no-`Lbeat` ablation, not before.
- Otherwise we risk rewriting labels before we know whether labels are actually the bottleneck.

### 6. Physical scale or skating instability in generated motion

Confidence: **High**

Why this matters:

1. The biggest gap so far is physical plausibility, not just beat alignment.
2. Huge PFC suggests skating, unstable root acceleration, or motion scale inflation.
3. If beat conditioning pushes motion amplitude too hard, BAS can still stay mediocre while PFC explodes.

What to test:

1. Track foot-min velocity and root-acceleration components for each variant.
2. Compare mean joint-coordinate magnitude against GT.
3. Inspect qualitative outputs from baseline and beat runs side by side.

### 7. Evaluation mismatch with the papers

Confidence: **Medium**

Why this matters:

1. Our current evaluator is useful for relative comparison.
2. Its absolute PFC scale is not the same as the numbers printed in the papers.
3. Beat-It also reports metrics we do not yet use as the main headline in this branch.

What to do:

1. Use the current evaluator for **branch-local comparisons**.
2. Do not claim paper-level reproduction from these numbers alone.
3. Treat official `checkpoint.pt` as the sanity anchor for the evaluator.

### 8. Missing keyframes limit ceiling performance

Confidence: **Conceptually true, but not the first blocker**

Why this matters:

1. Beat-It uses both beats and keyframes.
2. Removing keyframes makes the control problem harder.
3. Even a perfect beat-only extension may not match full Beat-It numbers.

Important:

- This explains a possible ceiling.
- It does **not** explain why the current beat run is much worse than official EDGE.

So this is not the first thing to fix.

## Debugging Order

Follow this order and do not skip steps.

### Step 1. Establish the local baseline

Goal:

- Determine whether this branch still trains a healthy EDGE model.

Success condition:

- local baseline is reasonably close to official `checkpoint.pt`
- PFC is not catastrophically inflated

If this fails:

- stop beat debugging
- fix branch-wide training/eval drift first

### Step 2. Test BeatDistance without `Lbeat`

Goal:

- isolate the effect of beat conditioning alone

Success condition:

- BAS improves or stays competitive with baseline
- PFC does not collapse

Interpretation:

1. If this helps, the beat representation is promising.
2. If this hurts badly, the conditioning path itself needs work.

### Step 3. Test safe BeatDistance fine-tune with `Lbeat`

Goal:

- determine whether `Lbeat` helps when it starts from a working BeatDistance checkpoint and cannot dominate training

Success condition:

- full eval passes the safe `Lbeat` quality gate
- PFC, Distg, and Distk stay within the gate thresholds
- BAP or BAS improves enough to justify the extra loss

Interpretation:

1. If this passes, estimator-supervised beat loss is worth keeping as a fine-tune tool.
2. If this is rejected, keep using BeatDistance as the stable model and report the best rejected checkpoint separately.

### Step 4. Only resume the promising run

Goal:

- avoid wasting GPU on dead ends

Resume only if the `600`-epoch checkpoint is already promising on:

1. BAS
2. PFC
3. qualitative motion quality

## Current Recommended Experiment Set

This is the correct minimum set for the beat-only question:

1. current trusted local baseline
2. current trusted BeatDistance run
3. safe `edge_beatdistance_lbeat` fine-tune from the trusted BeatDistance checkpoint

Interpret them in this order:

1. local baseline vs official checkpoint
2. beat-distance no-`Lbeat` vs local baseline
3. safe `Lbeat` fine-tune vs beat-distance no-`Lbeat`

## Decision Table

### Case A: local baseline is already bad

Meaning:

- the branch has a core training, data, or evaluation problem

Next action:

- debug the baseline path before changing anything beat-specific

### Case B: local baseline is good, no-`Lbeat` is bad

Meaning:

- beat conditioning or beat fusion is the main problem

Next action:

- inspect beat representation, encoder fusion, and condition strength

### Case C: local baseline is good, no-`Lbeat` is okay, safe `Lbeat` is rejected

Meaning:

- the beat loss still does not add enough value under the quality gate

Next action:

- keep BeatDistance as the default model
- inspect the best rejected checkpoint before spending more GPU time
- change only one safe-control knob at a time if another `Lbeat` attempt is needed

### Case D: no-`Lbeat` and `Lbeat` both help

Meaning:

- the beat-only idea is working

Next action:

- extend the best run to a longer schedule
- then optimize details such as estimator quality and label quality

## What Not To Do Yet

Avoid these until the ablation matrix is resolved:

1. do not add keyframes just to rescue the current beat path
2. do not rewrite beat labels before checking no-`Lbeat`
3. do not claim Beat-It reproduction
4. do not report a rejected safe `Lbeat` run as better just because eval completed

## Working Conclusion

The current evidence points to this priority order:

1. verify the local baseline in this branch
2. isolate beat conditioning without `Lbeat`
3. validate whether a stronger estimator makes `Lbeat` helpful

The biggest practical insight is:

> the current branch should be debugged as an **EDGE + beat-only ablation**, not as a full Beat-It reproduction

That framing makes the next decisions much clearer and avoids chasing the wrong target too early.
