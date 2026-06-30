# EXP-20260629-music-feature-selection-videos

Status: running
Owner: yukun
Created: 2026-06-29
Last Updated: 2026-06-29

## Goal

Generate five separate, presentation-ready long G1 render videos for the "Music
Feature Selection" slide:

1. JukeBox
2. Librosa35
3. Wav2CLIP/STFT
4. Gaussian-8D
5. Hybrid architecture, using v3b as the best hybrid checkpoint

## Feasibility Check

This is feasible as a same-song, same-seed, same-duration render set. The
JukeBox candidate is the early G1 beat-distance checkpoint:

```text
runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt
feature_type=jukebox
use_beats=True
beat_rep=distance
motion_format=g1
```

The other four checkpoints already have existing evaluation or render evidence
in this checkout.

## Model Set

| Slide label | Render label | Checkpoint | Feature type | Fusion | Condition variant |
|---|---|---|---|---|---|
| JukeBox | `jukebox_beatdistance_1000` | `runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt` | `jukebox` | `linear` | `auto` |
| Librosa35 | `librosa35_2000` | `runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt` | `baseline` | `linear` | `auto` |
| Wav2CLIP/STFT | `wav2clip_stft_r02_2000` | `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter_resume600_to2000/weights/train-2000.pt` | `wav2clip_stft_beat` | `stream_adapter` | `auto` |
| Gaussian-8D | `gaussian8d_1000` | `runs/train/EXP-20260601-finedance-g1-beat-features-8d_r01_resume800_to1000_b128/weights/train-1000.pt` | `beat_features_8d` | `linear` | `auto` |
| Hybrid architecture - v3b | `hybrid_v3b_1500` | `runs/train/EXP-20260526-finedance-g1-wav2clip-local-intensity-beatness_r01_resume550/weights/train-1500.pt` | `wav2clip_local_motion_intensity_beatness` | `linear` | `pred_controls` |

## Render Protocol

- Music: `data/finedance/music_wav/012.wav`
- Duration: `90s`
- Seed: `1234`
- Feature source: `extract`, so the videos reflect inference-time extraction
  rather than cached feature replay
- Ground truth tile: omitted; the deliverable is five single-model videos
- Render backend: MuJoCo G1 with EGL on Isambard
- Output root:

```text
renders/EXP-20260629-music-feature-selection-videos/012_90s_seed1234_extract_ppt_models/
```

PPT-friendly copies should be placed under:

```text
renders/EXP-20260629-music-feature-selection-videos/012_90s_seed1234_extract_ppt_models/single_model_videos/
```

## Slurm Plan

Submit one GPU render job from the repo root:

```bash
sbatch slurm/EXP-20260629-music-feature-selection-videos/render_012_90s_ppt_models.sbatch
```

Log-follow command after submission:

```bash
tail -f slurm/EXP-20260629-music-feature-selection-videos/render_<jobid>.out
```

## Verification Plan

After the job completes:

- Confirm all five `single_model_videos/*.mp4` files exist.
- Inspect `manifest.json` and confirm `feature_source=extract` and
  `audio_source=extract`.
- Verify video/audio metadata with the repo-local ffmpeg helper or available
  ffprobe equivalent.
- Record final paths and any failed render/extraction issue here.

## Progress Log

| Date | Step | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-06-29 | feasibility | passed | checkpoint configs checked for the five selected models; uploaded slide inspected | JukeBox is represented by `finedance_g1_fkbeatdistance_1000`, which uses `feature_type=jukebox`, `use_beats=True`, and `beat_rep=distance`. |
| 2026-06-29 | Slurm render launch | running | job `5417007`; Slurm log `slurm/EXP-20260629-music-feature-selection-videos/render_5417007.out`; curated log `setup_logs/EXP-20260629-music-feature-selection-videos/render_012_90s_ppt_models.log` | Job started on `nid011137`; baseline, beat_features_8d, and gaussian_beat extraction completed; JukeBox extraction is downloading the large legacy model cache before rendering. Follow with `tail -f slurm/EXP-20260629-music-feature-selection-videos/render_5417007.out`. |
