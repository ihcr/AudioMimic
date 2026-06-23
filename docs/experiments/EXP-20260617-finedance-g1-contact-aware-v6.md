# EXP-20260617-finedance-g1-contact-aware-v6

## Question

How should the FineDance-G1 music-to-robot pipeline change after v5 fixed root spinning but exposed hand jitter and hovering/high-lift feet?

## Local Evidence

V5 `g1_yaw_delta` is a useful representation change but not a final model. It removes the v4 roll/pitch drift and sharply reduces the v3b root-spin tail, but the matched render and endpoint/contact diagnostic show a new failure mode:

- `v5 / v3b1500` wrist FK jerk p95: `1.67x`.
- `v5 / v3b1500` no-near-support rate: `2.86x`.
- `v5 / v3b1500` high-lift frame rate: `2.48x`.
- `v5 / v3b1500` contact-proxy rate: `0.78x`.
- Diagnostic: `eval/EXP-20260601-finedance-g1-yaw-delta-repr_r02_resume250_controlstore_b128_acc4/v5_jitter_foot_hacking_diagnostic_20260614.json`.

Diagnosis: this is objective/metric mismatch rather than cache corruption. The current motion-control target puts `70%` of weighted speed mass on wrists, the FK beat metric can be improved by endpoint speed valleys, and the current foot loss does not enforce predicted support height/contact.

## Literature Read

- MDM, ICLR 2023: predicts clean motion samples so geometric losses on joint positions, velocities, and foot contact can be applied directly. This supports keeping `predict_epsilon=False` and using FK/contact losses as first-class training objectives rather than downstream render fixes. Source: https://guytevet.github.io/mdm-page/
- EDGE, CVPR 2023: strong music-to-dance baseline with diffusion, Jukebox conditioning, joint-wise conditioning, and temporal constraints. The transferable lesson is not only the audio feature, but the ability to constrain subsets of the body and stitch long windows with explicit temporal consistency. Source: https://edge-dance.github.io/
- FineDance, ICCV 2023: explicitly targets fine-grained hand motion and uses expert networks to reduce unrealistic full-body output. This supports separating hand expressivity from whole-body/support rhythm rather than letting a single wrist-heavy scalar drive both. Source: https://li-ronghui.github.io/finedance
- Lodge, CVPR 2024: uses a two-stage coarse-to-fine dance architecture and a Foot Refine Block for feet-ground contact. This supports a global rhythm/choreography planner plus local contact-aware refinement instead of one monolithic denoiser/loss. Source: https://li-ronghui.github.io/lodge
- MoMask, CVPR 2024: uses hierarchical residual motion tokens and masked generation for high-fidelity motion details. This supports a longer-term robot-aware motion prior/tokenizer to prevent per-frame diffusion from inventing high-frequency endpoint artifacts. Source: https://ericguo5513.github.io/momask/
- OmniControl, ICLR 2024 and GMD, ICCV 2023: both show that control accuracy needs a realism/coherence counterpart; sparse controls are easy to ignore or satisfy unnaturally unless converted into dense or hybrid guidance. Sources: https://neu-vi.github.io/omnicontrol/ and https://korrawe.github.io/gmd-project/
- PhysDiff, ICCV 2023 oral: physically implausible diffusion artifacts such as floating, foot sliding, and penetration are not reliably fixed by post-processing; physics or physical projection should be injected into generation. Source: https://research.nvidia.com/labs/lpr/publication/yuan2023physdiff/
- InterDance, 2024/2025: contact and physical realism improve when representation includes richer contact/surface information and guidance optimizes realism progressively; evaluation includes contact frequency and penetration, not only BAS/FID. Source: https://arxiv.org/html/2412.16982v1
- DanceMosaic / Walk Before You Dance, AAAI 2026: uses a pretrained motion prior, multi-tower music/pose guidance, progressive training, and inference-time guidance to reduce gradient interference between guidance modalities. Source: https://ojs.aaai.org/index.php/AAAI/article/view/37833
- RobotMDM, SIGGRAPH Asia 2024: integrates a kinematic generative model with a downstream physics-based controller through a differentiable reward surrogate. This is the closest long-term template for robot-native generation: train the generator against robot executability, not only visual FK metrics. Source: https://la.disneyresearch.com/publication/robot-motion-diffusion-model-motion-generation-for-robotic-characters/
- PHC, ICCV 2023; HOVER/ExBody2, 2024-2025: robust humanoid motion depends on a tracking/control layer that handles noisy or generated references, full-body command modes, and stability. Sources: https://www.zhengyiluo.com/PHC-Site/ , https://hover-versatile-humanoid.github.io/ , https://exbody2.github.io/
- GENMO/GEM, ICCV 2025 highlight: a generalist human-motion model trained across estimation and generation tasks, including audio/music-to-dance. It reports music-to-dance evaluation with `PFC` and `BAS`, and shows that multi-task generative priors can improve physical plausibility and motion/music correlation. Source: https://arxiv.org/html/2505.01425v1
- SoulDance/SoulNet, ICCV 2025: a holistic music-dance dataset and hierarchical motion model for body, hands, and face. It reinforces that hand/fine-body behavior should be modeled as coordinated hierarchical components rather than a single wrist-heavy scalar. Source: https://arxiv.org/html/2507.14915v1
- DuetGen, SIGGRAPH 2025: hierarchical VQ-VAE plus two-stage masked transformers for music-driven dance; high-level semantic tokens and low-level detail tokens are generated separately. This supports a coarse-to-fine/token-prior route for separating choreography rhythm from endpoint detail. Source: https://anindita127.github.io/DuetGen/
- RoboPerform, CVPR 2026: audio-conditioned humanoid performance model that separates high-level content latents from temporally aligned audio/style latents, denoising executable humanoid actions. This is the closest post-2024 reference for our goal: retargeting-free, audio-driven, physically plausible humanoid dance/control. Source: https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Do_You_Have_Freestyle_Expressive_Humanoid_Locomotion_via_Audio_Control_CVPR_2026_paper.pdf
- KungfuBot, NeurIPS 2025: physics-based humanoid whole-body control for highly dynamic skills including dancing; it uses motion processing, physical filtering/correction/retargeting, and adaptive motion imitation, and reports deployment on Unitree G1. This supports adding a feasibility/control layer and not trusting raw generated kinematics. Source: https://openreview.net/forum?id=LCPoXt0pzm
- Motion Generalist, ICLR 2026 withdrawn submission: proposes condition-aware masking for text/music motion generation and a Text-Music-Dance dataset, but because it is withdrawn it should be treated as a weak signal, not a main reference. Source: https://openreview.net/forum?id=mx9jLnzQGr

## Design Decision

V6 should not be another small loss on top of v5. Keep the successful yaw-only root representation, but redesign the architecture around three explicit layers:

1. Music-to-rhythm planner: produce sparse, low-dimensional musical intent and support/rhythm targets.
2. Robot-aware motion prior/decoder: generate motion under a learned or structured prior that preserves natural endpoint dynamics.
3. Contact/physics feasibility layer: enforce or predict support/contact and reject or fine-tune physically poor samples.

## Proposed V6 Mainline

### Representation

- Keep `g1_yaw_delta` for root trajectory.
- Add predicted/support contact channels to make a new representation or auxiliary head:
  - `left_foot_contact`, `right_foot_contact`.
  - Optional later: toe/heel split if MuJoCo geoms make it reliable.
- Do not restore full root roll/pitch integration.

### Controls

Use fewer, better controls:

- `semantic`: `Wav2CLIP`.
- `control`: `GaussianBeat`, `body_intensity`, `support_beatness`.
- Remove wrist-dominated `motion_beatness` as the primary rhythm control.
- `body_intensity`: robust group-level motion amplitude with capped per-group contribution.
- `support_beatness`: rhythm salience from torso/root-local body motion plus foot support transitions/holds, not wrist local minima.
- Optional second-stage ablation only: `upper_body_accent`, if v6 becomes under-expressive.

### Architecture

- Keep separate `SemanticEncoder(Wav2CLIP)`.
- Replace the compact scalar `ControlEncoder` with a typed rhythm/support encoder:
  - beat pulse stream for musical timing,
  - body/support control stream for motion feasibility,
  - cross-attention or FiLM injection into the denoiser at multiple layers.
- Add a contact/support prediction head sharing the Wav2CLIP+GaussianBeat stem.
- Decode with contact-aware FK losses and endpoint-quality gates.

### Training Objective

Use losses that correspond to durable modeling assumptions:

- Core diffusion reconstruction and velocity/acceleration losses stay.
- FK contact loss becomes two-part:
  - support height/contact consistency when target or predicted contact is active,
  - horizontal velocity/sliding only under support.
- Endpoint smoothness is measured in FK space and data-normalized by train-set percentiles for wrists/feet. This is a robot morphology prior, not a one-off visual patch.
- Beatness loss uses support/body beat valleys and should be capped like other aux losses.
- Long-term upgrade: train a RobotMDM-style differentiable feasibility surrogate from MuJoCo/contact/jerk/root metrics and use it for fine-tuning or sample selection.

## Evaluation Gates

Full eval every 500 epochs must include:

- Rhythm: `G1FKBAS`, `G1BeatF1`, precision/recall, phase diagnostics.
- Distribution: `G1Dist`, `G1Div`, `JointPositionRangeMean`.
- Root: `RootUpZP01`, `RootTiltGt60DegRate`, `RootAngularVelocityP99/Max`, root drift.
- Contact: `G1FootSliding`, `G1GroundPenetration`, contact proxy rate, no-near-support rate, high-lift frame rate.
- Endpoint naturalness: wrist FK jerk p95, foot FK jerk p95, joint smoothness jerk.
- Sensitivity: zero `support_beatness`, flat `body_intensity`, zero all controls.

Acceptance against v5/v3b:

- Keep v5's root-spin fix: `RootAngularVelocityMax < 30`, `RootTiltGt60DegRate == 0`.
- Keep rhythm near v5: `G1FKBAS >= 0.25`, `BeatF1 >= 0.225` by 1000 if possible.
- Restore naturalness near v3b: wrist FK jerk p95 no more than `1.2x` v3b1500, no-near-support rate no more than `1.3x` v3b1500, high-lift rate no more than `1.3x` v3b1500.
- Do not regress distribution: `G1Dist <= 4.5`, `G1Div >= 14.0`.

## Next Action

Implement V6 in two phases:

1. V6a: contact/support-aware controls plus eval gates. Rebuild feature and processed caches; train from scratch for a clean ablation.
2. V6b: robot-aware prior/surrogate. Use v6a outputs plus MuJoCo feasibility metrics to train a differentiable quality surrogate or reranker, following the RobotMDM direction.

Do not resume v5 to 1500/2000 as the mainline until v6a is tested.
