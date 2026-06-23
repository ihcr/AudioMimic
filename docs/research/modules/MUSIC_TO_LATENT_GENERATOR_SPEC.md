# Music-to-Latent Generator Spec

Last updated: 2026-06-18.

## Purpose

Generate G1-feasible dance latents from music and rhythm controls. This module
should replace raw pose diffusion only after the G1 motion latent prior can
reconstruct stable, natural robot motion.

## Decision Summary

Start with continuous latent diffusion because it is closest to the current
diffusion code and easiest to compare against the raw-motion baseline. Use
masked latent generation or RVQ token generation only after the latent prior
ablation proves that tokenization preserves contact/support and endpoint detail.

Recommended first route:

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

## Evaluation

Every candidate must decode to motion and run the same G1 gates:

- rhythm: `G1FKBAS`, `G1BeatF1`, precision, recall;
- distribution: `G1Dist`, `G1Div`, joint range;
- root: drift, yaw angular velocity, tilt/root-up;
- contact: sliding, penetration, support proxy;
- endpoint quality: wrist/foot FK jerk;
- control sensitivity: zero support beatness, flat intensity, zero all controls;
- latent diagnostics: latent norm, latent variance, decoder reconstruction gap.

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
- DanceMosaic / multi-tower masked dance prior: https://ojs.aaai.org/index.php/AAAI/article/view/37833
- DuetGen / music-to-hierarchical dance tokens: https://arxiv.org/abs/2506.18680
- RoboPerform / audio style latents for humanoid policy: https://arxiv.org/abs/2512.23650

