# AudioMimic ICRA: GT-Calibrated Evaluation and Experiment Plan

Update date: 2026-08-23

Status: proposed paper protocol; freeze before final-test evaluation

The current experiment-by-experiment evidence and claim boundary are tracked in
[`EXPERIMENT_CONCLUSIONS_20260823.md`](../experiments/EXPERIMENT_CONCLUSIONS_20260823.md).
This plan defines the protocol; that ledger records what the present results do and do
not support.

This document turns the existing metric map into a paper-level experimental
design. It is deliberately independent of M0/M2/M3/M4 naming and of any one
generator architecture. A future model enters the benchmark by exporting the
same audio, motion, timing, and provenance contract; evaluation code and test
data remain unchanged.

The metric definitions remain in
[`EVALUATION_MAP_MUSIC_TO_G1.md`](EVALUATION_MAP_MUSIC_TO_G1.md). Human-study
details remain in
[`HUMAN_EVALUATION_PROTOCOL.md`](HUMAN_EVALUATION_PROTOCOL.md). This document
defines which scientific claims are made, which experiments support them, and
how paired FineDance/AIST++ ground truth is used.

## 1. Recommended Paper Story

The paper should answer one central question:

> Can a humanoid generate a coherent dance online from arrived music, under a
> strict causal deadline, and preserve the resulting musical and expressive
> motion after physical tracking?

The recommended method story is:

```text
arrived audio prefix
  -> causal/predicted-future music condition
  -> H8/C4 robot-native motion planning
  -> committed generated history and boundary state
  -> continuous G1 reference stream
  -> SONIC tracking
  -> executed robot dance
```

The diffusion generator is trained on GMR/retargeted G1 motion and directly outputs
G1 reference motion. GMR is a GT preparation step, not an additional inference stage.
For evaluation, O-Human (SMPL/SMPLH), O-G1 (GMR), M-ref (generator G1) and M-exec
(SONIC G1) are parallel objects under the same music identity.

Commit Forcing, coarse-to-fine motion representation, and predicted-future
music are mechanisms supporting this system claim. They should not become
three unrelated papers inside one paper. The evidence should follow the same
causal chain as the deployed system.

The strongest defensible claim hierarchy is:

| Claim | Required conclusion | Main comparison |
|---|---|---|
| C1 Long-horizon generation | Generated-history training reduces drift, freezing, repetition, or quality decay during free rollout | Commit Forcing vs matched Teacher Forcing |
| C2 Correct music use | The correct causal music condition produces motion that fits that music better than null, wrong, shifted, and past-only controls | predicted-future audio vs counterfactual controls |
| C3 Real-time causality | No future waveform is read and every stage meets the streaming deadline | timestamp and latency audit |
| C4 Executable expression | SONIC executes the references stably while retaining meaningful motion and music-response information | reference vs execution, calibrated by GT-to-SONIC |
| C5 Perceptual result | People judge the final motion as natural, dance-like, and better matched to music | blinded human studies |

Open-loop online generation is sufficient for C1--C5: the generator may use
its committed decoded history while SONIC tracks the stream independently.
Tracker feedback is not required for the term `online`. A tracker-aware
closed-loop generator is a useful later extension only if it produces a
measured advantage; it should not be made a submission dependency now.

## 2. Current Evidence and Claim Boundary

As of 2026-08-23, the evidence supports the following narrow statements:

1. The coarse/fine representation has completed intervention and
   reconstruction evidence in the current paper draft.
2. M3 changes its generated trajectory when the music sidecar changes. Across
   the current two-song, three-seed ablation, wrong/shifted/null conditions
   change motion by about 0.16--0.17 rad RMSE, roughly half the paired motion
   scale.
3. That ablation does **not** show that the paired condition is more suitable
   for the target music. BAS and onset-impact correlation do not reliably rank
   paired above wrong, shifted, or null.
4. The two available M3 trajectories survive 60 s SONIC execution, with about
   100 ms measured lag and aligned joint RMSE of 0.219/0.201 rad.
5. SONIC preserves most joint amplitude but only about 42--46% of total motion
   energy in those two runs; 1--3 Hz content is especially attenuated.
6. The AIST++ and FineDance test manifests and the first multi-dataset
   metric-calibration audit are available locally. The current stratified pilot uses
   38 declared paired sequences (AIST++ 20, FineDance 18) and 494 controlled
   clean/jitter/low-pass/freeze/repeat samples. Seven of eight dataset-level direction
   checks pass; FineDance freeze Beat F1 is a warning, so that detector is not accepted
   as a standalone corruption test. BAS remains a core beat-alignment metric and is
   always reported with coverage, onset response, lag, tempo and phase. This validates
   metric sensitivity only; it does not validate a model or replace the sealed test.
   Results are in `eval/benchmark_v1/gt/stratified_audit_v1/`.
7. The full paired reference distribution is also available for calibration only:
   1,408 AIST++ sequences plus 203 FineDance sequences, summarized over 32
   dataset/tempo/style strata in `eval/benchmark_v1/gt/stratified_audit_all_v1/`.
   These full-data statistics must not be used to tune the sealed test or to claim
   that one pooled scalar is an ideal dance score. Small style groups remain
   descriptive until more paired songs are added.

The formal M3 generator-level ablation is recorded in
`eval/m3_music_ablation/formal_30s/aggregate_v2/`. It contains 24 trajectories
from two songs, three seeds, and four music conditions. The paired sidecar
changes the trajectory, but its mean BAS and impact-correlation advantage over
the counterfactual controls is not reliable. This result is a gate for the
next model iteration, not evidence to promote M3 directly to the main SONIC
matrix.

Therefore the current project can demonstrate a working music-conditioned
streaming mechanism and a working tracker interface. It cannot yet claim that
the robot dances *better to the correct music* or that the two available songs
represent the test distribution.

## 3. What Ground Truth Is and Is Not

FineDance and AIST++ paired music-dance data should be the reference foundation,
but `GT` must be separated into three levels.

| Oracle level | Object | What it calibrates |
|---|---|---|
| O-Human | original paired human music and motion | natural dance distribution and positive music-motion pairing |
| O-G1 | the same motion retargeted to the fixed G1 representation | robot-morphology motion distribution and generator target domain |
| O-Exec | SONIC execution of O-G1 under the frozen protocol | tracker capability and unavoidable execution loss |

`Oracle future audio` is a fourth, different concept. It is a diagnostic input
condition with real future music access, not ground-truth dance and not a
deployable method. Paper tables must call it `oracle future audio`, never just
`oracle`, to avoid confusing it with O-Human/O-G1/O-Exec.

### 3.1 GT is a distribution, not one exact answer

Music-to-dance is one-to-many. A generated choreography can be valid without
matching the recorded GT pose frame by frame. Consequently:

- do not use generated-motion-to-GT joint RMSE as a dance-quality metric;
- use GT feature distributions, GT percentile envelopes, paired-audio
  discrimination, and human preference for generator quality;
- use direct RMSE only where the target is uniquely defined, such as codec
  reconstruction, retargeting validation, or SONIC following a fixed reference.

GT also need not achieve the numerical maximum on every metric. Diversity is
not better without bound; low jerk can mean over-smoothed motion; BAS can fail
when either beat detector misses valid accents. The target is the GT range and
expected ordering, not a single universal score of 1.0.

### 3.2 Operational evaluation sequence

The project will use a staged benchmark rather than jumping directly to model
comparison:

1. **38-sequence calibration set.** Evaluate the paired AIST++ and FineDance
   held-out sequences with the complete frozen scorecard. O-Human establishes
   the positive reference distribution; O-G1 measures the change introduced by
   GMR/retargeting; O-Exec measures the additional SONIC execution loss.
2. **Dataset expansion.** After the metric directions, thresholds and
   counterfactual checks are validated on the 38 sequences, expand to the
   larger audited AIST++ and FineDance paired sets. The sealed test split is
   never selected by metric outcome.
3. **Generator comparison.** Evaluate M2 and M3 using the same songs, clips,
   seeds, duration and input counterfactuals at the G1 reference level. Their
   results are interpreted relative to the O-Human/O-G1 distributions, not
   against a frame-matched choreography target.
4. **Execution comparison.** Send the fixed M2/M3 references through SONIC and
   recompute the same dance-quality and music-adaptation metrics on M-Exec.
   The difference `M_ref -> M_exec` is the tracking/retention loss, while
   `O-G1 -> O-Exec` is the tracker capability calibration.

The 38-sequence stage is therefore a calibration and sanity check, not the
final statistical claim. A valid oracle should occupy the expected GT range
and correctly rank clean versus corrupted or counterfactual data; it does not
need to maximize every individual metric.

## 4. Dataset and Split Protocol

### 4.1 FineDance: primary in-domain benchmark

The authoritative local audit now joins 203 raw FineDance motions, 203
retargeted G1 motions, 207 WAV files, and label metadata by the same numeric
sequence ID. The official cross-genre test contains 18 usable paired
sequences. The manifest is
`eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json`.
Before final evaluation, freeze one manifest that joins, for every sequence:

- source sequence ID and source dataset split;
- audio path, motion path, duration, FPS, and SHA256;
- song identity, style/genre labels when available, tempo, and dancer/motion ID;
- retargeting version, G1 joint order, quaternion convention, and FK model hash;
- all legal 20 s and 60 s evaluation windows.

Audit that no source song, choreography, or duplicate audio crosses the
train/test boundary. If the existing split is not song-disjoint, the final
paper must either retrain with a strict split or explicitly narrow the claim
from unseen-music generalization to held-out-window generation.

Use all eligible sealed test sequences in generator tables. If only a subset
supports 60 s, report:

- a 20 s table over every test sequence;
- a 60 s long-horizon table over every eligible sequence.

Do not select songs based on model scores.

The current pairing audit compares each same-ID WAV/G1 pair against all other
test songs. It finds a positive mean best-lag correlation margin (`+0.0771`)
and paired rank percentile (`0.8366`), but only `22.2%` paired top-1 retrieval.
Therefore the result is benchmark-integrity evidence, not a claim that a simple
energy correlation is a sufficient musicality metric. The final music-motion
evaluation must add beat/onset timing, tempo/phase agreement, lag-aware
retrieval, counterfactual wrong-song controls, and human preference.

### 4.2 AIST++: external benchmark

Use the official FACT split as an external test: ten held-out music pieces and
40 unique test choreographies, with training/test music and choreography
separated. The local 1,408 retargeted sequences and matching audio are enough
to reconstruct this split, but the 40 official test IDs must be frozen in a
manifest before running models.

AIST++ and FineDance scores are reported separately. They differ in skeleton,
style, duration, frame rate, and motion distribution, so pooling them into one
mean is not meaningful. AIST++ is an external-generalization result, not a
replacement for the FineDance test.

### 4.3 Development, calibration, and sealed test

Maintain three disjoint uses of data:

| Partition | Allowed use |
|---|---|
| Train | model and auxiliary encoder training |
| Calibration/validation | checkpoint choice, metric thresholds, corruption severity, and one global inference policy |
| Sealed test | one final model comparison after all choices are frozen |

The same test manifest, clip windows, seeds, and metric configuration must be
used by every method. A checkpoint may be selected on validation once; an
individual metric may not choose its own favorable checkpoint, seed, or song
subset.

## 5. Model-Independent Input and Output Contract

Every model adapter must write a trajectory package containing at least:

```text
root_pos[T,3]          metres
root_rot[T,4]          declared xyzw or wxyz
dof_pos[T,29]          radians in frozen G1 order
fps                    source motion rate
audio_id/path/hash     exact conditioning audio
audio_start_seconds    motion frame zero on the audio clock
source_sequence_id     benchmark item identity
generation_seed        model sampling seed
initial_history_id     K64/cold-start identity
model/checkpoint_sha   immutable model identity
condition_type         none/past/predicted_future/oracle_future
audio_lookahead_s      real waveform lookahead, normally zero
history_s, horizon_s, commit_s
per-cycle timestamps   audio, feature, inference, publish, execution
```

The evaluator must not import model internals. It reads this package, validates
the contract, computes metrics, optionally streams the reference to SONIC, and
writes a versioned result. This is what makes the protocol reusable for a new
diffusion model, autoregressive model, flow model, or policy.

Each result must bind the following hashes:

```text
dataset manifest
model/checkpoint
metric configuration
FK/robot model
audio and motion files
SONIC policy/configuration
evaluation code commit
```

## 6. Metric Calibration Before Model Ranking

The first experiment is not a model comparison. It tests whether each proposed
metric measures the property assigned to it.

### 6.1 Motion-quality corruptions

Apply deterministic corruptions to O-G1 at three frozen severity levels:

| Corruption | Intended failure | Metrics expected to respond |
|---|---|---|
| joint jitter | high-frequency noise | jerk, FIDk, PFC, human naturalness |
| temporal low-pass | over-smoothing/average motion | energy, diversity, expressiveness |
| freeze segments | static collapse | static ratio, FID, long-horizon human rating |
| repeated segments | mode repetition | non-local repetition, diversity, coherence |
| C4 boundary offsets | streaming discontinuity | C4 position/velocity jumps, jerk |
| synthetic foot slide | contact inconsistency | FSR, PFC, foot-contact retention |
| root drift/tilt | unstable global behavior | root envelope, penetration, executability |
| time scaling | excessive or weak dynamics | velocity/acceleration/jerk envelopes |

A useful quality metric should rank clean GT above corrupted GT and generally
worsen monotonically with severity. Failure to do so demotes the metric to
supplementary or removes it from the corresponding claim.

### 6.2 Music-pairing counterfactuals

For every clean GT pair, construct negatives that isolate different musical
properties:

| Condition | What is preserved | What is broken |
|---|---|---|
| paired | everything | nothing |
| circular shift by 1/2/4 s | song identity and global style | local phase and event timing |
| segment shuffle | local timbre and event statistics | phrase order and long structure |
| wrong song, matched tempo/style | approximate rhythm/style | song-specific phrasing |
| wrong song, matched tempo but different style | tempo | style/semantics |
| null/silence | motion history and seed | music information |

A music metric is admitted to the main paper only if it ranks paired GT above
the relevant negative for the property it claims to measure. BAS is tested in
this calibration but remains supplementary because it is one-sided and local.

### 6.3 Provisional metric-admission rules

Freeze final thresholds using calibration data before opening sealed test.
Reasonable provisional gates are:

1. at least 70% directional accuracy for clean-vs-corrupted or
   paired-vs-counterfactual comparisons, with the trajectory-level bootstrap
   interval above chance;
2. Spearman correlation of at least 0.6 with corruption severity for metrics
   that claim monotonic behavior;
3. test-retest reliability of at least ICC 0.75 for stochastic extraction or
   repeated SONIC measurements;
4. no dependence on model identity, test-song-specific tuning, or per-method
   audio offset fitting.

These are project admission gates, not values claimed by FACT, EDGE, Lodge,
Beat-It, RoboPerform, or DiscoForcing.

## 7. Frozen Metric Scorecard

No aggregate `dance score` is used. Each claim has a small set of primary
endpoints and a larger diagnostic set.

### 7.1 Generator quality

Primary:

- FIDk/FIDg with one frozen canonical extractor and set-level bootstrap;
- long-horizon degradation from the first to last non-overlapping 20 s window;
- PFC/FSR and failure-free validity;
- blinded H-NATURAL and H-DANCE preference.

Diagnostics:

- Divk/Divg and same-audio seed diversity, targeted to the GT interval;
- velocity, acceleration, jerk, energy, static ratio, repetition;
- C4 boundary continuity, root behavior, penetration, and joint limits.

For Commit Forcing, the most relevant result is not only average FID. Report
whether quality, dynamics, and repetition degrade with rollout time. A method
addressing exposure bias should improve the last-window result or the
first-to-last degradation slope.

### 7.2 Music-motion correspondence

Primary:

- R@1/R@2/R@3 and MMDist from an independently trained, frozen audio-motion
  retrieval encoder that passes the GT pairing calibration;
- paired counterfactual margin for one calibrated temporal-response metric and
  one calibrated structure/semantic metric;
- blinded H-RHYTHM and H-STYLE preference.

Recommended temporal/structural candidates:

- onset-to-motion-energy response and reliable response lag;
- tempo consistency and circular phase error;
- phrase-boundary response;
- motion-energy-envelope correspondence;
- retrieval similarity for global style/semantic pairing.

Supplementary:

- BAS/BeatAlign, motion-beat count, and bidirectional beat coverage;
- Beat F1/BAP/KPD only for models with explicit beat or key-pose targets.

Define a counterfactual pairing margin for a validated similarity score `s`:

```text
PairMargin_i = s(audio_i, motion_i_paired)
             - mean_k s(audio_i, motion_i_counterfactual_k)
```

The important result is a positive paired margin relative to controls, not
merely a large difference between two generated trajectories. The current M3
ablation establishes condition sensitivity, not a positive PairMargin.

### 7.3 Reference executability and SONIC tracking

Primary:

- Success Rate and time to fall;
- raw and lag-compensated joint error plus EMPKPE;
- motion-energy and 1--3 Hz band-power retention;
- degradation of the validated music-motion endpoints from reference to
  execution.

Diagnostics:

- amplitude retention, 0--1/3--8 Hz retention, per-body-group lag;
- contact-state F1, foot-slip change, root height/orientation;
- reference dynamic-envelope violations.

Use O-G1 -> O-Exec to establish the tracker ceiling. For an error metric,
report the model-reference result both absolutely and relative to the O-G1
distribution. For a bounded quality metric, report GT percentiles rather than
claiming that the single GT mean is a universal optimum.

### 7.4 Real-time system

Primary hard evidence:

- zero future-waveform access for the deployable route;
- audio-to-feature, feature-to-motion, motion-to-reference, and
  reference-to-execution latency at P50/P95/P99;
- deadline-miss, stale-condition, packet-drop, and fallback rates;
- real-time factor over uninterrupted 60 s runs.

Mean inference time alone is insufficient. Offline PKL replay evaluates
tracking capability but cannot support an online music-generation claim.

### 7.5 Metric provenance and literature comparison

The frozen scorecard intentionally combines established dance-generation
metrics with robot-system metrics:

| Source | Imported evidence | How AudioMimic uses it |
|---|---|---|
| FACT/AIST++ | FIDk/FIDg, Distk/Distg, BeatAlign, pairwise user study | conventional human-dance quality and diversity; GT pairing calibration |
| EDGE | PFC, BeatAlign, diversity, large-scale pairwise preference | physical plausibility and human evidence; retain its warning that small-test FID can disagree with perception |
| Lodge | long-sequence FID/Div, FSR, BAS, efficiency | long-horizon quality, foot skating, and generation efficiency |
| Beat-It | PFC, Div, BAS, KPD, BAP, user study | physical/rhythm metrics; KPD/BAP only when explicit key-pose/beat targets exist |
| RoboPerform | R@K, MMDist, Success, EMPJPE, EMPKPE, latency | global audio-motion correspondence and robot execution |
| DiscoForcing | strict causality, FID/FSR/Div/BAS, ms/frame/FPS | causal streaming contract and runtime comparison |
| AudioMimic | PairMargin, long-horizon degradation, expression/frequency retention, complete latency chain | system-specific evidence for causal condition use and generator-to-tracker loss |

Published values are directly comparable only when dataset split, sequence
length, skeleton/canonical mapping, feature extractor, beat detector, and
aggregation are identical. Otherwise, include them in a separate
`reported in original paper` table and do not use boldface/SOTA ranking against
locally computed G1 values. The preferred comparison is to run released
baselines through the same frozen adapter and evaluator.

### 7.6 Required baseline policy

The minimum direct baselines have different proof roles:

| Baseline | Proof role |
|---|---|
| matched Teacher Forcing | isolates Commit Forcing |
| no-audio parent | tests whether music provides value beyond the motion prior |
| past/current-audio causal model | tests whether predicted future context is useful |
| oracle-future audio | estimates the remaining music-forecast gap |
| O-G1 GT | calibrates target motion and pairing distributions |
| O-G1 -> O-Exec | calibrates the SONIC ceiling |

If resources permit, add one released causal streaming method such as
DiscoForcing and one offline full-song method such as EDGE or Lodge. Their
outputs must first be converted through the same documented G1 adapter. A
baseline that uses future audio remains useful as a non-causal quality
reference but cannot be placed in the real-time column as if its input contract
were matched. RoboPerform is a direct audio-to-policy system and is best treated
as a system-level reference unless its complete training/evaluation contract is
available for a matched reproduction.

## 8. Required Experiment Matrix

### E0. Dataset and metric validation

Run O-Human/O-G1 paired and corrupted controls before evaluating a model.

Outputs:

- frozen FineDance and AIST++ manifests;
- metric calibration curves, directional accuracy, reliability, and accepted
  primary metrics;
- GT P5/P50/P95 envelopes by dataset and, where useful, by body group or
  dynamic tier.

Gate: no final model ranking until E0 is complete.

### E1. Representation mechanism

Keep the existing matched codec/representation study:

- discrete only;
- continuous only or matched continuous baseline;
- D+C without kinematics-guided objective;
- proposed D+C representation.

Use reconstruction/intervention metrics for the mechanism and standard motion
metrics for output quality. Do not use representation-specific diagnostic
scores as substitutes for final dance quality.

### E2. Commit Forcing

Minimum matched routes:

| Route | Difference |
|---|---|
| Teacher Forcing | recorded history and boundary during training |
| generated token/history control | generated history but incomplete transition closure |
| full Commit Forcing | generated D+C commit, decoded motion, reconstructed boundary |

Hold representation, parents, data, optimizer budget, sampling policy, K64,
song, and generation seed fixed. Run three training seeds and three sampling
seeds on the complete sealed manifest.

Report 20 s all-test quality, 60 s long-horizon quality, and three 20 s
position windows. The primary contrast is paired by source and seed. There is
no metric-specific source or seed selection.

### E3. Causal music conditioning

Use capability names in the paper even if code uses M0/M2/M3/M4:

| Paper route | Music access | Purpose |
|---|---|---|
| No-audio parent | none | negative control |
| Past/current audio | arrived audio only, no forecast | reactive causal baseline |
| Predicted-future audio | forecast from arrived prefix | deployable proposed route |
| Oracle-future audio | real future features | non-deployable diagnostic ceiling |
| Offline full-song model | full waveform, if available | non-causal quality reference |

For the causal counterfactual test, fix model noise, initial state, K64 history,
and all non-music inputs while changing only the music condition. Use paired,
wrong, shifted, shuffled, and null conditions.

Two initialization protocols must be separated:

1. **shared neutral K64**, independent of the target song, is the primary
   test of whether music drives subsequent choreography;
2. **matched GT K64**, taken from the same source sequence, is a secondary
   continuation test because the motion prefix already contains style and
   song-related information.

Without this separation, the model can appear music-aware by continuing the
initial choreography or by reacting arbitrarily to audio identity.

Minimum paper-ready target:

- every eligible FineDance test song;
- three fixed generation seeds;
- 60 s where source duration allows, otherwise the all-test 20 s protocol;
- the same condition matrix for every proposed and baseline route.

### E4. SONIC capability ceiling

Track a preregistered O-G1 set covering low, medium, and high GT dynamics.
Select the set from GT metadata only, before viewing model or SONIC results.
Run three SONIC repeats per reference.

This experiment answers how much of a valid retargeted dance SONIC can retain.
It also sets the reference dynamic envelope used by executability diagnostics.

### E5. End-to-end generator-to-SONIC evaluation

Run only frozen finalist routes and required controls through SONIC; there is
no need to execute every exploratory training ablation.

Recommended minimum:

- 8--12 FineDance songs selected by a frozen style/tempo/dynamic stratification;
- three generation seeds per route;
- three SONIC repeats per fixed generated reference;
- 60 s, 1.0x, full packet, 3 s measured-state alignment plus 1 s hold;
- identical SONIC checkpoint, simulator, initialization, and CPU-load gate.

Report reference metrics, execution metrics, and their degradation in one
paired table. Failed executions remain in Success/TTF and contribute only their
pre-fall interval to tracking-error summaries.

### E6. Streaming and stress tests

For at least 60 s per run, test:

- nominal load;
- inference-time jitter;
- delayed audio features;
- packet drop and feedback drop;
- cold start and K64 warm start;
- deadline miss with fallback;
- CPU-load gate and simulator real-time factor.

The deployable route passes only when the complete audio-to-reference loop, not
just the diffusion call, meets the C4 deadline.

### E7. Human evaluation

Use the existing three-study design:

1. muted generator naturalness/dance quality;
2. audible rhythm/style match;
3. muted reference-vs-execution expressiveness retention, with audible result
   as supplementary.

Add GT-based attention/calibration trials:

- O-G1 paired vs jitter/freeze/repeat corruption for naturalness;
- the same O-G1 motion with paired vs shifted/wrong audio for music match.

These trials verify that participants understand the intended distinction and
provide a perceptual reference for automatic metrics. Use participant- and
song-aware confidence intervals; do not treat individual video votes as
independent trajectories.

### E8. Real G1 validation

If the paper claims a physical robot system, simulation video alone is not
enough. Freeze one safe controller and runtime configuration, then run a
predeclared set rather than only an edited hero sequence.

Recommended target:

- at least six held-out songs across tempo/style tiers;
- three independent 30 s trials per song;
- fall/intervention/safety-stop counts, achieved duration, latency, and video;
- no post-hoc playback slowing in the main result.

An edited hero video may accompany these results, but it is not statistical
evidence. If this matrix cannot be completed, scope the quantitative claim to
MuJoCo sim2sim and describe hardware footage only as a qualitative demo.

## 9. Statistical Protocol

1. The independent unit is a source trajectory/song, not a frame.
2. Compare methods with the same song, start, K64, and generation seed using
   paired differences.
3. Use hierarchical bootstrap: sample songs first, then generation seeds; for
   execution metrics, sample SONIC repeats inside the fixed reference.
4. For set-level FID, bootstrap complete source blocks and recompute FID; do
   not assign one pseudo-FID value to each frame.
5. Report mean/median, SD/IQR, effect size, and 95% confidence interval.
6. Use Holm correction within each predeclared claim family. Primary endpoints
   are tested; diagnostics are interpreted descriptively.
7. Use right-censored survival analysis for time to fall.
8. For human studies, use a Davidson/Bradley-Terry or mixed logistic model with
   participant and song effects.
9. Report every failed run and every eligible test sequence.

The current paper TODO that proposes choosing a favorable seed per metric and
then retaining the best 25% of tracks must not be used for formal results. It
creates different test sets for different claims and makes the reported result
conditional on observed test performance. Model selection belongs on
validation data; the sealed test reports all preregistered eligible samples.

## 10. Paper Tables and Figures

Keep the main paper compact and move diagnostic detail to the supplement.

### Main figures

1. **System figure:** causal audio, forecast, H8/C4 planner, committed state,
   SONIC execution.
2. **GT-calibrated evaluation figure:** O-Human and O-G1 as GT preparation/reference
   pairs, beside M-ref and M-exec model results; the arrows indicate provenance and
   comparison, not an extra GMR stage inside generator inference.
   reference -> model execution.
3. **Long-horizon figure:** first/middle/last window degradation for Teacher
   Forcing and Commit Forcing.
4. **Music counterfactual figure:** paired/wrong/shifted/null PairMargin across
   songs.
5. **Expression-retention figure:** reference/execution spectral power and
   energy by body group.

### Main tables

| Table | Rows | Primary columns |
|---|---|---|
| T1 Representation | D-only, C-only, legacy D+C, proposed D+C | structure/detail reconstruction and intervention, compact quality summary |
| T2 Streaming generation | Teacher Forcing, partial closure, Commit Forcing | FIDk/FIDg, long-horizon degradation, repetition, PFC/FSR |
| T3 Music use | no audio, past audio, predicted future, oracle future | retrieval, PairMargin-temporal, PairMargin-structure, human rhythm/style |
| T4 End-to-end | O-G1 and finalist references before/after SONIC | success, EMPKPE, lag, energy/1--3 Hz retention, music-score degradation |
| T5 Runtime | deployable routes | lookahead, P50/P95 latency, deadline miss, RTF |

The supplement contains per-song values, GT calibration curves, all dynamics,
all frequency bands, failures, stress tests, and complete human-study counts.

## 11. Acceptance Gates and Paper Reframing

### Gate G0: evaluation validity

- split and pairing audit passes;
- metric calibration passes;
- manifests and hashes are frozen;
- no metric-specific test selection.

### Gate G1: Commit Forcing

- full Commit Forcing improves the predeclared long-horizon endpoint over
  matched Teacher Forcing;
- it does not materially degrade GT-calibrated quality or diversity;
- the result holds across source windows rather than only an aggregate.

### Gate G2: music conditioning

- predicted-future audio beats no-audio and past/current-audio controls on at
  least one validated temporal endpoint and one validated global pairing
  endpoint;
- paired counterfactual margins have confidence intervals above zero;
- H-RHYTHM or H-STYLE supports the same direction;
- motion quality and executability do not regress materially.

If G2 fails, do not describe the model as learning appropriate choreography
from music. Reframe it as causal streaming motion generation with an active but
unvalidated music interface, or improve the music objective before submission.

### Gate G3: real-time causality

- future waveform lookahead is zero for the proposed route;
- 60 s real-time factor is at least 1.0;
- deadline miss is at most 1% under the frozen deployment machine;
- all buffering and predicted-audio horizon are disclosed.

### Gate G4: SONIC execution

- the preregistered simulation Success Rate target is met;
- tracker error and expression loss are compared with O-G1 -> O-Exec;
- the music advantage established at the reference level remains detectable
  after execution;
- failure and post-fall handling follow the frozen protocol.

If G4 fails, scope the paper to reference generation and present SONIC only as
an interface diagnosis. Do not let tracker failure obscure a generator claim,
and do not claim end-to-end robot dance from reference-only videos.

### Gate G5: hardware

- the real-G1 protocol completes with recorded successes and failures;
- safety intervention and latency are reported;
- qualitative video agrees with, but does not replace, the trial table.

## 12. Implementation Work Packages

### WP0: freeze the paper protocol

- remove metric-specific seed/song selection from the paper TODO;
- choose primary endpoints and correction families;
- freeze dataset, metric, model, and SONIC configuration manifests.

Deliverable: signed protocol version and sealed-test checksum.

### WP1: build the GT benchmark

- audit FineDance pairs and split;
- reconstruct the official AIST++ test split;
- export O-Human/O-G1 standardized packages;
- implement deterministic motion corruptions and music counterfactuals.

Deliverable: `gt_benchmark_manifest.json` plus corruption manifests.

### WP2: validate and freeze metrics

- unify G1 FK, contact, FID/Div extractors, timing, and body groups;
- run GT calibration and retain only validated primary metrics;
- train the retrieval encoder on train/validation only and freeze it;
- compare automatic metrics with the human calibration trials.

Deliverable: metric card for every admitted metric, including source,
implementation hash, expected direction, calibration result, and limitation.

### WP3: evaluate motion generation

- run matched Commit Forcing and Teacher Forcing models;
- run no-audio, past-audio, predicted-future, and oracle-future routes;
- execute causal counterfactuals with shared neutral K64 and fixed noise;
- produce complete 20 s and 60 s reference-level tables.

Deliverable: frozen generator result package and failure gallery.

### WP4: evaluate SONIC and the full runtime

- run O-G1 capability calibration;
- execute the frozen finalist matrix;
- compute tracking, expression, and music-response degradation;
- run deadline/drop/jitter stress tests.

Deliverable: end-to-end table, runtime trace, and synchronized videos.

### WP5: human study and hardware

- render model-independent blinded clips;
- obtain ethics/participant approval as required;
- run the frozen questionnaire and statistical model;
- perform the preregistered real-G1 matrix when safe.

Deliverable: anonymized responses, analysis script, hardware trial manifest,
and unedited source videos.

## 13. Immediate Execution Order

The next work should proceed in this order:

1. freeze the one-sentence paper claim and remove favorable-subset evaluation;
2. audit the FineDance split and generate the complete O-G1 paired manifest;
3. build GT corruptions and music negatives;
4. validate existing metrics before adding new ones;
5. implement/freeze the independent audio-motion retrieval metric;
6. rerun M3 and later models on the same full generator benchmark;
7. promote only models that pass the music PairMargin gate to SONIC;
8. run O-G1 and finalist SONIC matrices under the fixed CPU/RTF gate;
9. perform blinded human evaluation;
10. complete real-G1 trials or explicitly narrow the paper scope.

Further ad hoc two-song M3 playback is lower priority than WP1--WP2. The two
current M3/SONIC runs remain valuable pilot evidence, but the reusable GT
benchmark must be established before they can become a paper-level result.

## 14. Recommended Claim Wording by Outcome

If all gates pass:

> AudioMimic generates G1 dance online from arrived music under a strict causal
> commit schedule, improves correct-music correspondence over causal controls,
> and preserves that advantage through SONIC execution.

If music correspondence fails but Commit Forcing passes:

> AudioMimic improves long-horizon causal G1 motion continuation under a
> deployment-matched commit schedule; its music interface remains a system
> component without a demonstrated choreography-quality gain.

If SONIC retention fails but generator and music pass:

> AudioMimic provides causal music-conditioned robot-native references; the
> current tracker interface remains an execution bottleneck.

This decision tree keeps the paper accurate while allowing the same benchmark
to evaluate every later model release.
