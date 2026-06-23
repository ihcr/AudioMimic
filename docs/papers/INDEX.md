# Paper Library

Last updated: 2026-06-18.

This is the project paper database for FineDance-G1, robot-native
music-to-dance, motion priors, rhythm metrics, and humanoid feasibility. Local
PDFs and converted markdown live in this directory when available. Missing PDFs
are not a blocker; keep the canonical link and add notes first.

## How To Use This Library

- Read the "Core Reading Order" before proposing a new architecture.
- Use "Categories" to find related work by method role.
- Use "Local Assets" to see what is already downloaded or converted.
- Put transient conclusions from specific runs in `docs/experiments/`, not here.
- When adding a paper, include: year, venue/status, category, link, local asset
  status, relevance, reusable idea, limitation.

## Core Reading Order

1. EDGE - local baseline and original music-to-dance diffusion stack.
2. FineDance and LODGE - dance data, hand/body detail, long choreography, foot
   refinement.
3. MoMask, T2M-GPT, MotionGPT - discrete motion priors and tokenized motion.
4. DanceMosaic, DuetGen, SoulDance, GENMO/GEM - 2025/2026 direction for masked,
   hierarchical, multi-condition, and generalist motion priors.
5. DiscoForcing, MotionStreamer, SONIC, BeyondMimic, KungfuBot, RoboPerform -
   streaming, robot-native control, tracking,
   audio-to-humanoid, and feasibility prior references.
6. Diffusion Policy, RTC, MLD, and PDP - practical latent diffusion,
   receding-horizon policy, and RL/BC routes for motion/control.
7. PhysDiff, RobotMDM, PHC/HOVER/ExBody2 - physics/contact/control references
   for feasibility and execution.

## Categories

### A. Local Baselines And Dance Generation

| Paper | Year | Venue/status | Local asset | Relevance | Reusable idea | Limitation |
|---|---:|---|---|---|---|---|
| [EDGE: Editable Dance Generation From Music](https://arxiv.org/abs/2211.10658) | 2023 | CVPR | `edge.pdf`, `markdown/edge/edge.md` | Repository foundation | Transformer diffusion, Jukebox conditioning, contact consistency, in-betweening | Human/SMPL target; robot feasibility is indirect |
| [FineDance](https://arxiv.org/abs/2212.03741) | 2023 | ICCV | no local PDF listed by exact title | FineDance-G1 data source and hand-detail reference | Fine-grained dance genres, hand motion, expert modules | Human dance; retargeting does not guarantee G1 feasibility |
| [LODGE](https://arxiv.org/abs/2403.10518) | 2024 | CVPR | `lodge.pdf`, `markdown/lodge/lodge.md` | Long dance and contact reference | Coarse-to-fine diffusion, characteristic primitives, Foot Refine Block | Human representation; local foot fix rather than learned robot prior |
| [DiscoForcing](https://arxiv.org/abs/2605.28491) | 2026 | arXiv | not local | Streaming/causal related work | Causal music encoder, diffusion-forcing, history-guided streaming sampler | Robot path is retargeting/deployment, not robot-native generator |
| [MotionStreamer](https://arxiv.org/abs/2503.15451) | 2025 | ICCV | not local | Streaming motion latent reference | Continuous causal latent space and history-aware streaming motion generation | General motion streaming; not G1 dance or policy |
| [DanceMosaic / Walk Before You Dance](https://ojs.aaai.org/index.php/AAAI/article/view/37833) | 2026 | AAAI | not local | Multi-condition dance prior | Pretrained motion prior, multi-tower guidance, progressive masked training | Prior is not G1-specific |
| [DuetGen](https://arxiv.org/abs/2506.18680) | 2025 | SIGGRAPH | not local | Hierarchical dance token prior | Coarse/fine VQ-VAE and masked transformers for music-to-token generation | Two-person human dance; no robot feasibility |
| [SoulDance / SoulNet](https://arxiv.org/html/2507.14915v1) | 2025 | ICCV | not local | Holistic body/hand/facial dance | HRVQ for body, hands, face; music-motion retrieval prior | Human holistic data; face is irrelevant to G1 |
| [GENMO / GEM](https://arxiv.org/html/2505.01425v1) | 2025 | ICCV Highlight | not local | Generalist motion prior | Joint estimation/generation training across modalities, including music-to-dance | Human representation mismatch |

### B. Motion Priors, Tokens, And Latent Spaces

| Paper | Year | Venue/status | Local asset | Relevance | Reusable idea | Limitation |
|---|---:|---|---|---|---|---|
| [MDM](https://arxiv.org/abs/2209.14916) | 2023 | ICLR | not local | Diffusion motion baseline | Predict clean motion so geometric losses can be applied directly | Human motion; no music or robot-specific feasibility |
| [T2M-GPT](https://arxiv.org/abs/2301.06052) | 2023 | CVPR | not local | Motion tokenizer baseline | VQ-VAE plus GPT for motion tokens | Text-to-motion; contact detail can be quantized away |
| [MotionGPT](https://arxiv.org/abs/2306.14795) | 2023 | NeurIPS | not local | Motion-language/token prior | Motion vocabulary and language-style modeling | Human motion tasks; not robot execution |
| [MLD / Motion Latent Diffusion](https://chenxin.tech/mld/) | 2023 | CVPR | not local | Continuous latent diffusion baseline | Train an autoencoder first, then diffuse in lower-dimensional motion latent space | Human motion; G1 contact/support must be revalidated |
| [MoMask](https://arxiv.org/abs/2312.00063) | 2024 | CVPR | not local | Strong discrete motion prior | Hierarchical residual quantization and masked token generation | Human text-to-motion; not G1/contact-specific |
| [Human Motion Diffusion as a Generative Prior](https://openreview.net/forum?id=dTpbEdN9kr) | 2024 | ICLR | not local | Prior-as-component reference | Use off-the-shelf diffusion priors for composition/control | Still human motion; prior not robot-native |
| [Motion Generalist](https://openreview.net/forum?id=mx9jLnzQGr) | 2026 | ICLR withdrawn | not local | Weak signal only | Condition-aware masking and Text-Music-Dance framing | Withdrawn; do not use as main evidence |

### C. Robot-Native Control, Tracking, And Feasibility

| Paper | Year | Venue/status | Local asset | Relevance | Reusable idea | Limitation |
|---|---:|---|---|---|---|---|
| [SONIC](https://arxiv.org/abs/2511.07820) | 2025 | arXiv | external manual in repo: `操作手册_AudioMimic_SONIC.md` | Best template for scalable humanoid motion prior | Motion tracking at scale, universal token/control interface, G1-compatible deployment stack | Not a music-to-dance generator by itself |
| [BeyondMimic](https://arxiv.org/abs/2508.08241) | 2025 | arXiv | not local | Feasibility tracker/critic template | High-quality motion tracking, state-action diffusion, test-time cost guidance | Best as teacher/critic/execution layer; not choreography/audio model |
| [RoboPerform](https://arxiv.org/abs/2512.23650) | 2026 | CVPR | `robopeform.pdf`, `markdown/robopeform/robopeform.md` | Closest direct collision | Retargeting-free audio-to-locomotion, content/style latents, diffusion student policy | Broad audio-to-humanoid system; our novelty must be more specific |
| [KungfuBot](https://arxiv.org/abs/2506.12851) | 2025 | NeurIPS | not local | Dynamic G1 control reference | Motion processing, filtering/correction, adaptive motion tracking | Does not solve music-conditioned choreography |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 2023/2024 | RSS / IJRR | not local | Receding-horizon action policy reference | Denoise action chunks and execute in receding horizon | Manipulation-centric; does not solve choreography |
| [RTC / Real-Time Chunking](https://arxiv.org/abs/2506.07339) | 2025 | arXiv | not local | Real-time action chunk execution reference | Generate action chunks while executing committed actions | Robot action execution framework; not dance generation |
| [PDP / Physics-Based Character Animation via Diffusion Policy](https://arxiv.org/abs/2406.00960) | 2024 | SIGGRAPH Asia | not local | RL/BC route for robust control | Use RL policies for corrective actions, then train diffusion policy with behavior cloning | Character/control setting; too heavy as first music-to-dance step |
| [RobotMDM](https://la.disneyresearch.com/publication/robot-motion-diffusion-model-motion-generation-for-robotic-characters/) | 2024 | SIGGRAPH Asia | not local | Differentiable feasibility surrogate | Couple kinematic generator with physics-based controller via reward surrogate | Robotic character setting; not music-to-dance |
| [PHC](https://www.zhengyiluo.com/PHC-Site/) | 2023 | ICCV | not local | Humanoid tracking/control background | Robust physical humanoid control for human motion imitation | Human/humanoid control layer, not music generator |
| [HOVER](https://hover-versatile-humanoid.github.io/) | 2024 | project/arXiv | not local | Versatile humanoid control background | Commanded whole-body control and tracking | Not a dance/music generator |
| [ExBody2](https://exbody2.github.io/) | 2025 | project/arXiv | not local | Expressive humanoid control background | Expressive whole-body tracking and stability | Not music-conditioned generation |

### D. Physics, Contact, And Guidance

| Paper | Year | Venue/status | Local asset | Relevance | Reusable idea | Limitation |
|---|---:|---|---|---|---|---|
| [PhysDiff](https://research.nvidia.com/labs/lpr/publication/yuan2023physdiff/) | 2023 | ICCV Oral | not local | Physical plausibility reference | Inject physics/projection into diffusion rather than post-processing only | Human motion; not robot-specific |
| [GMD](https://korrawe.github.io/gmd-project/) | 2023 | ICCV | not local | Dense guidance reference | Convert sparse controls to dense guided motion constraints | Human path/control setting |
| [OmniControl](https://neu-vi.github.io/omnicontrol/) | 2024 | ICLR | not local | Controllability reference | Spatial control with realism/coherence tradeoff | Human motion; no G1 feasibility |
| [InterDance](https://arxiv.org/html/2412.16982v1) | 2024/2025 | arXiv | not local | Contact/interaction realism | Contact/surface-aware realism evaluation and progressive guidance | Human interaction, not robot dance |

### E. Audio And Music Representations

| Paper/model | Year | Venue/status | Local asset | Relevance | Reusable idea | Limitation |
|---|---:|---|---|---|---|---|
| [Jukebox](https://arxiv.org/abs/2005.00341) | 2020 | arXiv | not local | Original EDGE audio feature | Strong high-level music embeddings | Slow feature extraction; heavy cache |
| [Wav2CLIP](https://arxiv.org/abs/2110.11499) | 2021 | arXiv | not local | Current compact semantic audio feature | Audio embedding aligned to CLIP-like semantic space | Does not explicitly encode beat/phase |
| [MERT](https://arxiv.org/abs/2306.00107) | 2023 | ICLR? / arXiv | `mert.pdf`, `markdown/mert/mert.md` | Candidate music encoder | Music understanding representation | May be overkill and not rhythm-specific for this repo |

## Local PDF And Markdown Inventory

These files already exist under `docs/papers/` as of 2026-06-18. The names are
kept as-is, including typos, to avoid breaking paths.

| Local PDF | Converted markdown | Current classification |
|---|---|---|
| `edge.pdf` | `markdown/edge/edge.md` | Core baseline |
| `lodge.pdf` | `markdown/lodge/lodge.md` | Long dance and coarse-to-fine prior |
| `robopeform.pdf` | `markdown/robopeform/robopeform.md` | Robot-native audio reference |
| `aist-fact.pdf` | `markdown/aist-fact/aist-fact.md` | Dataset / music-dance baseline |
| `mert.pdf` | `markdown/mert/mert.md` | Audio/music representation |
| `DGFM.pdf` | `markdown/DGFM/DGFM.md` | Needs triage |
| `BidirectionalDiffusion .pdf` | `markdown/BidirectionalDiffusion/BidirectionalDiffusion.md` | Needs triage |
| `beat-it.pdf` | `markdown/beat-it/beat-it.md` | Needs triage for beat/rhythm |
| `danceba.pdf` | `markdown/danceba/danceba.md` | Needs triage for dance/beat alignment |
| `infinitedance.pdf` | `markdown/infinitedance/infinitedance.md` | Needs triage for long dance |
| `lrcm.pdf` | `markdown/lrcm/lrcm.md` | Needs triage |
| `mambadance.pdf` | `markdown/mambadance/mambadance.md` | Needs triage for sequence backbone |
| `mathdance.pdf` | `markdown/mathdance/mathdance.md` | Needs triage |
| `pamd.pdf` | `markdown/pamd/pamd.md` | Needs triage |
| `skeleton2stage.pdf` | `markdown/skeleton2stage/skeleton2stage.md` | Needs triage for representation/staging |
| `tokendance.pdf` | `markdown/tokendance/tokendance.md` | Needs triage for token dance prior |

## Project-Specific Takeaways

1. Direct raw-motion denoising is no longer the best long-term design for G1
   dance. Recent motion generation and humanoid control work increasingly uses
   priors, latent spaces, tokens, or policy layers.
2. SONIC is the strongest template for a robot motion prior, but it should not
   be copied as a generic tracker. The project needs a G1 dance prior.
3. BeyondMimic is best used as a feasibility teacher, critic, or diagnostic
   route. Using it only as a post-hoc tracker would not make the generator
   robot-native.
4. RoboPerform is the strongest collision on audio-to-humanoid performance.
   Any paper claim here must distinguish itself through dance-specific G1 prior,
   explicit rhythm controls, benchmark/eval rigor, and not just audio-to-action.
5. Token priors are attractive for reducing average motion, but contact and
   endpoint fidelity must be validated before making RVQ the mainline.
6. Streaming and policy should be kept as future-compatible endpoints. The
   current novelty should stay centered on rhythm-to-support G1 dance prior,
   not on copying DiscoForcing's streaming claim or RoboPerform's policy claim.
7. Non-distillation streaming should remain an open route: train direct causal
   or latency-conditioned G1 latent generation from real data, not only an
   offline-teacher to causal-student compression pipeline.

## Maintenance Backlog

- Triage each "Needs triage" local PDF and assign it to a category.
- Add BibTeX blocks for all core papers.
- Add short one-page notes for SONIC, BeyondMimic, RoboPerform, and KungfuBot
  after full reading.
- Add local PDF/markdown assets for SONIC, BeyondMimic, KungfuBot, DanceMosaic,
  DuetGen, SoulDance, GENMO/GEM, MoMask, T2M-GPT, MotionGPT, PhysDiff,
  RobotMDM, PHC, HOVER, and ExBody2 if licensing permits.
