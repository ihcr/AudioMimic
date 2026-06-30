# EXP-20260629-v6bb-comparison-renders

## Question

Produce fresh PPT-ready comparison renders that include the rejected V6b-B
latent-diffusion model beside the stronger raw-diffusion anchors, without
reusing prior videos or saved generated motions.

## Status

`finished`

Initial Slurm render job `5417344` failed quickly because the raw helper
comparison was asked to compose a `grid2x2` layout with only three inputs. The
wrapper was fixed to use a horizontal layout for that intermediate raw helper
comparison while keeping the final V6b-B comparison layout unchanged.

Current Slurm render job `5417348` was submitted on 2026-06-29 and started on
`nid010757`; it completed successfully (`COMPLETED`, exit `0:0`, elapsed
`00:01:35`). The job log is:

```text
slurm/EXP-20260629-v6bb-comparison-renders/render_5417348.out
```

The user requested two fresh render packs:

- FineDance test-set clip: V6b-B, ground truth, v3b, and Librosa35.
- In-the-wild Beat It clip: V6b-B, v3b, and Librosa35. No ground truth is
  available for this audio.

The render contract is generate-first: every model output in this pack must be
newly sampled/rendered by the job. Prior generated MP4s and saved predicted
motions are not inputs. The only reused inputs are checkpoints, audio files,
dataset ground-truth motion for the GT tile, and deterministic feature caches
where the selected FineDance slice already has canonical cached features.

## Models

- V6b-B beat8d latent diffusion:
  `runs/train/EXP-20260626-finedance-g1-v6b-beat8d-latent-diffusion_r01_beat8d_only/weights/train-1500.pt`.
- Frozen V6b-A motion prior:
  `runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/weights/train-500.pt`.
- v3b:
  `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt`.
- Librosa35:
  `runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt`.

## Render Plan

Test-set comparison:

```bash
.venv311/bin/python -m eval.render_g1_checkpoint_comparison \
  --music data/finedance/music_wav/012.wav \
  --out_length 5 \
  --seed 1234 \
  --slice_start 26 \
  --feature_source cache \
  --output_dir renders/EXP-20260629-v6bb-comparison-renders/012_slice26_v6bb_gt_v3b_librosa35/raw_gt_v3b_librosa35 \
  --overwrite \
  --g1_render_backend mujoco \
  --g1_mujoco_gl egl \
  --comparison_layout horizontal \
  --model v3b_1500:wav2clip_local_motion_intensity_beatness:linear:runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt:pred_controls \
  --model librosa35_2000:baseline:linear:runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt

.venv311/bin/python -m eval.render_v6b_latent_audio \
  --music data/finedance/music_wav/012.wav \
  --out_length 5 \
  --seed 1234 \
  --slice_start 26 \
  --feature_source cache \
  --output_dir renders/EXP-20260629-v6bb-comparison-renders/012_slice26_v6bb_gt_v3b_librosa35 \
  --overwrite \
  --comparison_layout grid2x2 \
  --compose_video gt=renders/EXP-20260629-v6bb-comparison-renders/012_slice26_v6bb_gt_v3b_librosa35/raw_gt_v3b_librosa35/videos/gt/gt_0_012_g1.mp4 \
  --compose_video v3b_1500=renders/EXP-20260629-v6bb-comparison-renders/012_slice26_v6bb_gt_v3b_librosa35/raw_gt_v3b_librosa35/videos/v3b_1500/v3b_1500_0_012_g1.mp4 \
  --compose_video librosa35_2000=renders/EXP-20260629-v6bb-comparison-renders/012_slice26_v6bb_gt_v3b_librosa35/raw_gt_v3b_librosa35/videos/librosa35_2000/librosa35_2000_0_012_g1.mp4
```

Beat It in-the-wild comparison:

```bash
.venv311/bin/python -m eval.render_g1_checkpoint_comparison \
  --music custom_music/in_the_wild/beatit_chorus_0058_40s.wav \
  --out_length 40 \
  --seed 1234 \
  --feature_source extract \
  --output_dir renders/EXP-20260629-v6bb-comparison-renders/beatit_chorus_0058_40s_v6bb_v3b_librosa35/raw_v3b_librosa35 \
  --overwrite \
  --no_gt \
  --g1_render_backend mujoco \
  --g1_mujoco_gl egl \
  --comparison_layout horizontal \
  --model v3b_1500:wav2clip_local_motion_intensity_beatness:linear:runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt:pred_controls \
  --model librosa35_2000:baseline:linear:runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt

.venv311/bin/python -m eval.render_v6b_latent_audio \
  --music custom_music/in_the_wild/beatit_chorus_0058_40s.wav \
  --out_length 40 \
  --seed 1234 \
  --feature_source extract \
  --output_dir renders/EXP-20260629-v6bb-comparison-renders/beatit_chorus_0058_40s_v6bb_v3b_librosa35 \
  --overwrite \
  --comparison_layout horizontal \
  --compose_video v3b_1500=renders/EXP-20260629-v6bb-comparison-renders/beatit_chorus_0058_40s_v6bb_v3b_librosa35/raw_v3b_librosa35/videos/v3b_1500/v3b_1500_0_beatit_chorus_0058_40s_g1.mp4 \
  --compose_video librosa35_2000=renders/EXP-20260629-v6bb-comparison-renders/beatit_chorus_0058_40s_v6bb_v3b_librosa35/raw_v3b_librosa35/videos/librosa35_2000/librosa35_2000_0_beatit_chorus_0058_40s_g1.mp4
```

## Outputs

Expected comparison MP4s after the Slurm job completes:

```text
renders/EXP-20260629-v6bb-comparison-renders/012_slice26_v6bb_gt_v3b_librosa35/comparison.mp4
renders/EXP-20260629-v6bb-comparison-renders/beatit_chorus_0058_40s_v6bb_v3b_librosa35/comparison.mp4
```

Final verification:

- Test-set comparison:
  `renders/EXP-20260629-v6bb-comparison-renders/012_slice26_v6bb_gt_v3b_librosa35/comparison.mp4`,
  duration `5.01s`, `30.0 fps`, frame size `1280x960`, nonblank frame means
  `57.57` and `58.26`.
- Beat It comparison:
  `renders/EXP-20260629-v6bb-comparison-renders/beatit_chorus_0058_40s_v6bb_v3b_librosa35/comparison.mp4`,
  duration `40.0s`, `25.0 fps`, frame size `1920x536`, nonblank frame means
  `54.20` and `54.54`.

## Notes

- `礼包5` is interpreted as Librosa35 from the feature-selection slide context.
- V6b-B is a rejected beat8d-only latent-diffusion ablation; this render is for
  visual comparison, not acceptance.
