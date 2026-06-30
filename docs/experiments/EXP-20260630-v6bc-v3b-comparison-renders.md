# EXP-20260630-v6bc-v3b-comparison-renders

## Question

Produce fresh comparison renders for the latest two trained V6b-C models beside
the v3b raw-diffusion anchor.

## Status

`finished`

Correction on 2026-06-30: the first test-set four-grid was incorrectly rendered
as a 5-second single-slice comparison. The correct PPT comparison should be a
long video. A fresh 90-second four-grid rerender was launched as Slurm job
`5432733` on `nid010994` with output root:

```text
renders/EXP-20260630-v6bc-v3b-comparison-renders/012_90s_v6bc_r01_r02_v3b/
```

Follow it with:

```bash
tail -f slurm/EXP-20260630-v6bc-v3b-comparison-renders/render_90s_5432733.out
```

Job `5432733` failed before rendering because the old single-slice
`slice_start=26` is outside the valid 90-second window range for song `012`.
The wrapper was corrected to match the existing 90-second feature-video
segment: `feature_source=extract`, `slice_start=3`.

Corrected Slurm job `5432746` was submitted on 2026-06-30 and started on
`nid010557`; it completed successfully (`COMPLETED`, exit `0:0`, elapsed
`00:02:14`). The job log is:

```text
slurm/EXP-20260630-v6bc-v3b-comparison-renders/render_90s_5432746.out
```

Slurm render job `5432379` was submitted on 2026-06-30 and started on
`nid010556`. Follow the live log with:

```bash
tail -f slurm/EXP-20260630-v6bc-v3b-comparison-renders/render_5432379.out
```

Job `5432379` failed after the test-set four-grid completed because the Beat It
raw helper had a single v3b tile and the shared comparison composer tried to
run `hstack=inputs=1`, which ffmpeg rejects. The composer was patched to copy a
single video input directly before adding the label banner. The render wrapper
will be relaunched with `--overwrite`, so final artifacts should come from the
fixed rerun rather than the partial failed run.

Fixed Slurm render job `5432388` was submitted on 2026-06-30 and started on
`nid010556`; it completed successfully (`COMPLETED`, exit `0:0`, elapsed
`00:01:33`). The job log is:

```text
slurm/EXP-20260630-v6bc-v3b-comparison-renders/render_5432388.out
```

Requested render packs:

- FineDance test-set four-grid: ground truth, V6b-C r01 control-only,
  V6b-C r02 Wav2CLIP-control, and v3b.
- In-the-wild Beat It three-tile comparison: V6b-C r01 control-only,
  V6b-C r02 Wav2CLIP-control, and v3b. No ground truth exists for this audio.

Render contract: generate fresh model outputs in this pack. Reuse checkpoints,
input audio, deterministic cached dataset features for the cached FineDance
slice, and dataset ground truth for the GT tile only.

## Models

- V6b-C r01 control-only:
  `runs/train/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r01_control_only/weights/train-1500.pt`.
- V6b-C r02 Wav2CLIP-control:
  `runs/train/EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion_r02_wav2clip_control/weights/train-1500.pt`.
- Frozen V6b-A motion prior:
  `runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/weights/train-500.pt`.
- v3b:
  `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt`.

## Render Plan

Test-set source:

```text
music: data/finedance/music_wav/012.wav
slice_start: 26
feature_source: cache
out_length: 5
```

Beat It source:

```text
music: custom_music/in_the_wild/beatit_chorus_0058_40s.wav
feature_source: extract
out_length: 40
```

The render wrapper generates v3b and GT through
`eval.render_g1_checkpoint_comparison`, generates V6b-C r01/r02 through
`eval.render_v6b_latent_audio`, then composes:

```text
GT | v6bc_r01_control_1500
v6bc_r02_wav2clip_1500 | v3b_1500
```

for the test set, and:

```text
v6bc_r01_control_1500 | v6bc_r02_wav2clip_1500 | v3b_1500
```

for Beat It.

## Outputs

Expected final comparison MP4s:

```text
renders/EXP-20260630-v6bc-v3b-comparison-renders/012_slice26_v6bc_r01_r02_v3b/comparison.mp4
renders/EXP-20260630-v6bc-v3b-comparison-renders/beatit_chorus_0058_40s_v6bc_r01_r02_v3b/comparison.mp4
```

Final verification:

- Test-set four-grid:
  `renders/EXP-20260630-v6bc-v3b-comparison-renders/012_90s_v6bc_r01_r02_v3b/comparison.mp4`,
  duration `90.01s`, `30.0 fps`, frame size `1280x960`, nonblank frame means
  `58.11` and `58.14`. This corrected the earlier incorrect 5-second
  single-slice four-grid.
- Beat It three-tile comparison:
  `renders/EXP-20260630-v6bc-v3b-comparison-renders/beatit_chorus_0058_40s_v6bc_r01_r02_v3b/comparison.mp4`,
  duration `40.0s`, `25.0 fps`, frame size `1920x536`, nonblank frame means
  `54.36` and `54.23`.

## Notes

- The experiment ledger marks V6b-C as rejected overall; these videos are
  diagnostic/slide comparison artifacts, not an acceptance claim.
- Although r02 checkpoint 1000 is noted as a useful diagnostic anchor in the
  V6b-C spec, this render uses the latest trained checkpoints, both
  `train-1500.pt`, to match the user request.
