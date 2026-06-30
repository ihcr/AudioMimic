# Music-to-Latent Generator Spec

Last updated: 2026-06-30.

## Purpose

Generate G1-feasible dance latents from music and rhythm controls. This module
should replace raw pose diffusion only after the G1 motion latent prior can
reconstruct stable, natural robot motion.

## Decision Summary

Start with continuous latent diffusion because it is closest to the current
diffusion code and easiest to compare against the raw-motion baseline. Use
masked latent generation or RVQ token generation only after the latent prior
ablation proves that tokenization preserves contact/support and endpoint detail.

Post V6b-C update: the first continuous music-to-latent diffusion route is
rejected as a mainline. V6b-B and V6b-C renders show globally unnatural,
chaotic, non-feasible motion, and the Wav2CLIP route amplifies endpoint jerk.
This means the immediate blocker is no longer only weak condition sensitivity;
it is off-manifold latent generation. The next route must constrain generated
latents by decoded motion feasibility, not just denoising MSE in latent space.

Original first route, now treated as a negative baseline:

```text
Wav2CLIP + GaussianBeat + body/support controls
    -> latent diffusion
    -> G1 motion decoder
```

## Condition Schema

Keep conditions compact and typed:

```text
semantic:
  wav2clip: Tensor[B, T, 512]

control:
  gaussian_beat: Tensor[B, T, 1]
  body_intensity: Tensor[B, T, 1]
  support_beatness: Tensor[B, T, 1]
```

Do not reintroduce STFT/Librosa/Jukebox into the mainline unless an ablation has
a specific reason. The current long-term direction is fewer, better controls.

## Generator Options

| Option | Definition | Advantages | Risks | Recommendation |
|---|---|---|---|---|
| Continuous latent diffusion | Diffuse in AE/VAE latent space | Close to current model; good for smooth detail; lower dimensional than raw pose | Can still average if latent prior is weak | First mainline |
| Masked latent model | Predict masked latent frames/tokens with bidirectional context | Good for editing/inpainting and global context | Needs careful masking schedule; less aligned with current code | Second ablation |
| Autoregressive latent model | Generate latent sequence step by step | Simple causal/streaming path | Error accumulation; may freeze | Use only for streaming-specific follow-up |
| RVQ token transformer | Generate discrete codebook indices | Strong recent precedent; can improve diversity and avoid mean pose | Quantization/contact risks; token sampling can be brittle | Only if RVQ prior passes reconstruction/contact gates |
| Hierarchical generator | Generate coarse dance latent then detail latent | Best long-term fit for phrase/detail separation | More modules and many ablations | Long-term after continuous baseline |
| Part-wise generator | Separate root/lower/upper/hands latent generation | Directly attacks hand/foot mismatch | Condition streams may conflict | Useful if whole-body latent still jitters |

## Injection Choices

| Injection | Use case | Notes |
|---|---|---|
| Cross-attention from Wav2CLIP | Music semantics and phrase-level style | Keep as semantic path |
| FiLM or adaptive layer norm from controls | Beat/intensity/support scalars | Good for low-dimensional controls |
| Separate control encoder | Avoid semantic/control interference | Start compact, split only if ablation shows conflict |
| Classifier-free guidance | Condition sensitivity and strength tuning | Needs quality gates to avoid beat metric hacking |
| Control dropout | Test whether model uses controls | Required for zero/flat condition variants |

## Training Policy

Start with:

- frozen or separately pretrained G1 latent decoder;
- latent reconstruction target from the encoder;
- music-conditioned diffusion loss in latent space;
- optional decoded FK/contact auxiliary loss with small cap;
- predicted-control inference as default eval, not oracle controls.

Do not jointly train everything from scratch until the frozen-prior baseline is
understood. Otherwise decoder failure and generator failure become entangled.

After V6b-C, do not rely on latent denoising loss alone. Add a feasibility path
before another long run:

- monitor latent norm, latent variance, and train-latent nearest-neighbor
  distance during sampling;
- decode generated latents during training or scheduled validation and penalize
  extreme FK velocity/acceleration/jerk, ground penetration, foot high-lift, and
  no-support frames;
- evaluate true learned-null conditioning separately from raw zero controls,
  because raw zeros are out-of-distribution inputs to the control tower;
- gate checkpoints on render/quality Pareto, not on beat metrics alone.

## V6b-B Condition-Sensitivity Follow-Up

If `EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion` still shows weak
separation between `real_beat8d`, `shifted_beat8d`, and `random_beat8d` at
epochs 1000 or 1500, treat this as a condition-use failure rather than simply
an undertraining problem. Do not keep extending beat8d-only latent diffusion as
the mainline unless the full eval shows real-condition gains on `G1BeatF1`,
`G1FKBAS`, and `G1FKRoboPerformBAS` without `zero_beat8d` winning through
quality collapse.

Recent music-to-dance work points to the same pattern:

- Beat-It separates beat conditions from dense music features with nearest-beat
  distance, hierarchical multi-condition fusion, and an explicit beat alignment
  loss.
- MambaDance replaces sparse one-hot beat signals with Gaussian beat
  representations and uses a temporal fusion backbone designed for long
  rhythmic sequences.
- TokenDance tokenizes both music and dance, decomposing music into semantic
  and acoustic codebooks instead of relying on a small continuous control
  vector alone.
- LODGE uses a coarse-to-fine hierarchy with characteristic dance primitives and
  a foot refinement block, so low-level motion quality is not delegated only to
  the music condition.
- DanceMosaic uses a pretrained generative masked motion prior plus
  synchronized/progressive multimodal guidance to avoid multimodal gradient
  interference and improve editability/control response.

Recommended next ablation if V6b-B fails at 1000 or 1500:

```text
semantic music encoder + GaussianBeat / beat-distance controls
    -> separate semantic and control encoders
    -> cross-attention for semantic music
    -> AdaLN or FiLM for beat/support controls
    -> latent diffusion over frozen V6b-A latents
    -> decoded FK/contact auxiliary losses with small weights
```

Add one condition-sensitivity objective instead of relying only on denoising
loss: either a decoded beat-alignment auxiliary loss, or an in-batch ranking
loss that scores real audio/beat controls above shifted and random controls.
Keep the same acceptance test: real, shifted, random, and zero variants at every
500-epoch checkpoint, plus contact/support/root/endpoint gates.

## Cross-Domain Weak-Conditioning Lessons

The V6b-B failure should be treated as a weak conditional-control problem. The
accepted V6b-A motion prior can reconstruct stable G1 motion, and longer V6b-B
training improves robot quality, but the generated latent distribution remains
nearly invariant to real, shifted, and random beat conditions. Recent conditional
diffusion and motion-generation work usually strengthens conditioning through
one or more of these mechanisms:

- **Trainable condition branch over a frozen generator.** ControlNet and
  T2I-Adapter add trainable control adapters while freezing the strong generative
  backbone. For this repo, the equivalent is not to finetune V6b-A first; add a
  dedicated music/control adapter or tower that maps audio controls into the
  latent denoiser.
- **Condition densification.** GMD turns sparse spatial constraints into dense
  guidance because sparse signals can be ignored during reverse diffusion.
  Beat-only impulses should similarly become dense Gaussian/beat-distance/phase
  fields, not only low-dimensional pulse features.
- **Explicit condition-use objective.** Attend-and-Excite fixes text concepts
  being ignored by optimizing cross-attention during inference. For dance, use
  training-time ranking or auxiliary losses so real audio/beat controls predict
  the denoising target better than shifted or random controls; monitor
  condition-token attention/gradient response.
- **Separate guidance and realism objectives.** OmniControl combines control
  accuracy guidance with realism guidance, rather than letting control strength
  destroy motion quality. For G1, pair beat-alignment guidance with
  support/contact/root/jerk gates.
- **Progressive multi-branch training.** DanceMosaic trains modality-specific
  music/pose towers with synchronized/progressive masking to reduce gradient
  interference. For V6b-C, keep semantic music and beat/support controls in
  separate encoders and ablate each branch before merging.
- **Preference or reward alignment after supervised training.** Diffusion-DPO,
  DDPO, and reward-gradient methods align diffusion models to objectives that
  the likelihood/denoising loss does not capture. For later V6d work, use
  pairwise real-vs-shift/random preference data or differentiable rhythm/contact
  rewards after the supervised V6b-C baseline shows some condition sensitivity.

Recommended escalation order:

```text
V6b-C: semantic/control split + dense beat controls + ranking loss (rejected)
V6b-D: manifold-checked latent generator with decoded feasibility losses/gates
V6b-E: ControlNet/T2I-Adapter-style control branch over a stable denoiser
V6b-F: inference-time beat/contact guidance or attention excitation
V6d: reward or preference alignment using real-vs-corrupted condition pairs
```

The priority changed after qualitative renders: conditioning improvements are
useful only if the generator remains on the decoded G1-feasible manifold.

## Final V6b-C Paired Design

Run V6b-C as a paired comparison, not a single Wav2CLIP-only bet:

```text
r01_control_only:
  dense rhythm/control tower
  no Wav2CLIP semantic tower

r02_wav2clip_control:
  same dense rhythm/control tower
  + Wav2CLIP semantic tower
```

The two routes should share the same frozen V6b-A prior, latent cache, model
depth, diffusion schedule, batch size, loss weights, and full-eval cadence. The
only intended difference is whether Wav2CLIP semantic tokens are present.

Use `beat_features_8d` as dense control rather than as a semantic condition:

```text
beat_pulse
gaussian_beat
dist_to_prev_beat_norm
dist_to_next_beat_norm
beat_phase_sin
beat_phase_cos
beat_interval_norm
onset_strength_norm
```

Use Wav2CLIP only from the existing `wav2clip_stft_beat_feats` cache:

```text
wav2clip = wav2clip_stft_beat_feats[:, :512]
```

Do not include the STFT channels in V6b-C. The paired comparison is testing
dense rhythm/control versus dense rhythm/control plus compact semantic music.

Both routes must include a real-vs-corrupted condition-use loss. The core
diagnostic is whether real conditions predict the denoising target better than
shifted/random conditions before full-eval metrics are even considered.

At ckpt500, interpret the pair as follows:

- if both fail condition sensitivity, the loss/architecture still does not
  solve weak conditioning;
- if control-only succeeds and Wav2CLIP fails, Wav2CLIP is interfering and the
  mainline should stay rhythm/control-only;
- if Wav2CLIP succeeds and control-only fails, semantic music is necessary;
- if both succeed, choose by the rhythm-quality Pareto and keep the other as a
  clean ablation.

## Evaluation

Every candidate must decode to motion and run the same G1 gates:

- rhythm: `G1FKBAS`, `G1BeatF1`, precision, recall;
- distribution: `G1Dist`, `G1Div`, joint range;
- root: drift, yaw angular velocity, tilt/root-up;
- contact: sliding, penetration, support proxy;
- endpoint quality: wrist/foot FK jerk;
- control sensitivity: zero support beatness, flat intensity, zero all controls;
- latent diagnostics: latent norm, latent variance, decoder reconstruction gap.

Use stable raw-diffusion anchors in every comparison, even when a new latent
route is being tested:

```text
v3b_1500_pred:
  balanced raw-diffusion reference for render naturalness and overall motion

beat8d_1000_auto:
  single-8D raw-diffusion reference for rhythm/control behavior
```

These anchors are not promoted because they win every metric. They are retained
because qualitative render inspection makes them the most useful stable
references for detecting whether a new route is genuinely better or only
exploits the metric suite. Treat metrics as diagnostic evidence, not as the
sole acceptance rule. If a new model improves beat scores but looks less natural
or regresses on support/contact, ground behavior, endpoint jerk, root stability,
or motion feasibility, call that out as a failure mode rather than accepting the
checkpoint.

Do not include other 8D variants as stable anchors. Variants such as
`beat8d_beatness_1000_pred` can stay in diagnostic tables, but they are not the
8D raw-diffusion baseline for future acceptance comparisons.

## Ablation Matrix

| Ablation | Question | Accept signal |
|---|---|---|
| raw pose diffusion vs latent diffusion | Does prior help beyond model capacity? | Better contact/naturalness without rhythm collapse |
| frozen decoder vs finetuned decoder | Does music training need decoder adaptation? | Finetune improves metrics without corrupting prior |
| Wav2CLIP only vs +GaussianBeat vs +support controls | Which controls matter? | Sensitivity variants show real, not cosmetic, condition use |
| continuous latent vs masked latent | Does bidirectional masked modeling help structure? | Better long-window coherence and no endpoint regression |
| continuous latent vs RVQ token | Does tokenization reduce average motion? | Higher amplitude/diversity without contact loss |
| CFG scale sweep | How strong should condition guidance be? | Better rhythm Pareto, not beat-only hacking |

## Failure Modes

- Latent generator ignores beat/support controls.
- Strong guidance improves beat metrics while hurting contact/root.
- Decoder hides generator errors by smoothing, causing low-amplitude dance.
- RVQ tokens increase diversity but damage foot support.
- Joint training corrupts the learned G1 prior.

## Source Anchors

- EDGE / music-conditioned diffusion baseline: https://edge-dance.github.io/
- MLD / latent diffusion for motion: https://chenxin.tech/mld/
- Classifier-free guidance / conditional score scaling: https://arxiv.org/abs/2207.12598
- ControlNet / trainable control branch over frozen diffusion backbone: https://arxiv.org/abs/2302.05543
- T2I-Adapter / lightweight condition adapters: https://arxiv.org/abs/2302.08453
- Attend-and-Excite / attention-based condition-use guidance: https://arxiv.org/abs/2301.13826
- GMD / dense guidance for ignored sparse motion constraints: https://arxiv.org/abs/2305.12577
- OmniControl / balancing control accuracy and realism guidance: https://arxiv.org/abs/2310.08580
- Beat-It / beat distance, multi-condition fusion, alignment loss: https://arxiv.org/abs/2407.07554
- MambaDance / Gaussian beat representation: https://arxiv.org/abs/2603.08023
- TokenDance / dual music-dance tokenization: https://arxiv.org/abs/2603.27314
- LODGE / coarse-to-fine primitives and foot refinement: https://arxiv.org/abs/2403.10518
- DanceMosaic / multi-tower masked dance prior: https://ojs.aaai.org/index.php/AAAI/article/view/37833
- Diffusion-DPO / preference alignment: https://arxiv.org/abs/2311.12908
- DDPO / reinforcement learning for diffusion objectives: https://arxiv.org/abs/2305.13301
- VADER / video diffusion reward-gradient alignment: https://arxiv.org/abs/2407.08737
- DuetGen / music-to-hierarchical dance tokens: https://arxiv.org/abs/2506.18680
- RoboPerform / audio style latents for humanoid policy: https://arxiv.org/abs/2512.23650
