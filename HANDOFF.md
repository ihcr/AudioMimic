# Handoff

## Goal

Continue the FineDance G1 diffusion ablation work in the `diffusion` worktree:

- compare Jukebox vs Librosa35/34 music features fairly on retargeted FineDance-G1;
- keep GT/reference rows in every future metric comparison;
- use matched robot renders to sanity-check whether high beat scores correspond to usable G1 motion;
- avoid treating paper claims about Librosa35/DGFM as directly comparable without matching architecture, representation, and metrics.

New-server checkout:

- Clone/fetch this branch directly as a normal repo root; it does not require the
  old EDGE `main` checkout for code or environment.
- Suggested path: `/path/to/EDGE-diffusion`
- Branch: `diffusion`
- Repo-local env: `source .venv311/bin/activate`
- Long/heavy jobs must use Slurm (`srun`/`sbatch`), not the login node.

## Current Progress

Pack-up update, 2026-05-14:

- This diffusion branch is being pushed together with the new `wav2clip-stft-beat` branch for migration to another server.
- Stage A of the music-conditioning redesign moved to the separate
  `wav2clip-stft-beat` branch.
- The wav2clip branch completed Wav2CLIP/STFT/GaussianBeat feature extraction
  and trained the `stream_adapter` variant to 500 epochs, but `concat_norm` and
  eval are blocked by Slurm quota. Clone/fetch `wav2clip-stft-beat` directly and
  read its `HANDOFF.md` before continuing Stage A.
- Runtime videos remain local under `videos/` and are ignored; do not commit them.

The main experiment spec is finished and should be read first:

- `docs/experiments/EXP-20260511-finedance-g1-librosa-fullctx.md`
- Index: `docs/experiments/INDEX.md`
- Dataset/run summary: `docs/DATASET_PREPARATION.md`

Completed FineDance-G1 full-context Librosa/Jukebox comparison:

| Run | Evidence | Key result |
|---|---|---|
| GT/reference | `slurm/pipelines/20260511-finedance-g1-librosa-fullctx-gt/eval_with_audio/metrics.json` | `G1BAS=0.2130`, `G1FKBAS=0.2160`, `G1BeatF1=0.1770`, `G1Dist=0.0000`, `G1FootSliding=0.2176` |
| Jukebox FKBeat no-lbeat | `slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2/eval/metrics.json` | `G1BAS=0.3046`, `G1FKBAS=0.3192`, `G1BeatF1=0.2910`, `G1Dist=8.6295` |
| Jukebox lbeat + robot loss | `slurm/pipelines/20260511-finedance-g1-jukebox-lbeat-robotloss-b020-kin040-cap1-scratch-1000/eval/metrics.json` | strong beat metrics, weak quality: `G1FKBAS=0.7218`, `G1BeatF1=0.7140`, `G1Dist=28.8001` |
| Librosa35 full-context baseline | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-baseline-2000/eval/metrics.json` | `G1BAS=0.2246`, `G1FKBAS=0.2421`, `G1Dist=9.5547` |
| Librosa35 full-context motiondist | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/metrics.json` | best balanced Librosa run: `G1BAS=0.2498`, `G1FKBAS=0.2653`, `G1Dist=8.9239` |
| Librosa34 full-context motiondist | `slurm/pipelines/20260511-finedance-g1-librosa34-fullctx-motiondist-cond-2000/eval/metrics.json` | slightly worse than Librosa35 on rhythm and much worse on `G1Dist=12.0596` |
| Librosa35 full-context lbeat + robot loss | `slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-lbeat-robotloss-b020-kin040-cap1-scratch-1000/eval/metrics.json` | high FK beat, quality regressions: `G1FKBAS=0.6298`, `G1Dist=14.8613`, `G1GroundPenetration=0.1715` |

Completed training/eval Slurm IDs recorded in the spec:

- Librosa35 full-context baseline: validation `4556030`, train `4556031`, eval `4556032`
- Librosa35 full-context motiondist: validation `4556033`, train `4556034`, eval `4556035`
- Librosa34 full-context motiondist: validation `4556036`, train `4556037`, eval `4556038`
- Librosa35 full-context lbeat+robotloss: validation `4556026`, train `4556028`, eval `4556029`
- Jukebox lbeat+robotloss: validation `4545745`, train `4545749`, eval `4545750`
- GT/reference eval with audio paths: Slurm `4571181`, completed `0:0`

Qualitative render artifacts:

- Model render root: `slurm/pipelines/20260512-finedance-g1-30s-renders/`
- Matched 30s songs: FineDance `012`, `063`, `143`
- Jukebox no-loss renders: `slurm/pipelines/20260512-finedance-g1-30s-renders/jukebox_fkbeat_no_lbeat/renders/`
- Jukebox lbeat+robotloss renders: `slurm/pipelines/20260512-finedance-g1-30s-renders/jukebox_lbeat_robotloss/renders/`
- Librosa35 lbeat+robotloss renders: `slurm/pipelines/20260512-finedance-g1-30s-renders/librosa35_lbeat_robotloss/renders/`
- GT/source reference renders: `slurm/pipelines/20260512-finedance-g1-30s-renders/data_reference/renders/`
- Convenience source copies: `videos/finedance_g1_30s/source/`

The 30s model render Slurm job was `4570136`, completed `0:0`, and produced 9 non-empty MP4s. A later reference-render job `4571201` produced the GT/source videos.

The source/doc changes for this handoff have been committed and pushed. Runtime
artifacts under `data/`, `runs/`, `slurm/`, and `videos/` remain local and should
be copied explicitly when needed.

## What Worked

- Full-song-context rebuilding for Librosa features improved the Librosa35 motiondist run versus sliced-context feature extraction: `G1Dist` improved from `11.2301` to `8.9239` and foot sliding improved from `0.5798` to `0.5246`.
- Keeping the 35th binary music-beat channel was better than dropping it. `baseline34 + motiondist` did not beat `baseline35 + motiondist`.
- Best balanced Librosa checkpoint is currently:
  - `runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt`
- FineDance-G1 Jukebox FKBeat no-lbeat remains the cleaner Jukebox anchor:
  - `runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt`
- Jukebox lbeat + robot loss improved beat scores over old relative-lbeat scratch, but still has poor distribution/contact quality.
- Added `--skip_metrics` to `eval/run_full_song_eval.py`; this is useful for render-only jobs where the prepared tree does not expose `test/motions`.
- GT/reference rows now work by augmenting reference pickle copies with matching `test/wavs_sliced/*.wav` audio paths, then running G1 eval against the original reference motions.
- Experiment documentation now has a standing rule: include a GT/reference row whenever the matching reference split can be evaluated.

## What Didn't Work

- Do not compare FineDance full songs directly for quick qualitative video checks; they are too long. Use 30s clips unless the user explicitly wants full-length output.
- Cancelled full-song render attempt:
  - Slurm `4570014`
  - script `slurm/pipelines/20260512-finedance-g1-full-song-renders/render_fullsong_three_models.sbatch`
- First 30s render attempt failed after producing the first model's videos because post-render metrics looked for `data/finedance_g1_fkbeats/test/motions`, but this tree uses `test/motions_sliced`.
  - Failed Slurm job: `4570052`
  - fixed by adding `--skip_metrics` and resubmitting as `4570136`.
- Raw GT/reference pickles do not include `audio_path`, so direct GT eval reports no beat scores. Use the augmented runtime reference copy under:
  - `slurm/pipelines/20260511-finedance-g1-librosa-fullctx-gt/reference_with_audio/`
- GT beat scores are not an upper bound. Generated motions can over-align motion minima to music beats and score higher than real choreography, so always read beat metrics with `G1Dist`, foot sliding, ground penetration, and smoothness.
- The DGFM/FineDance paper comparison is not a direct contradiction. DGFM's best setup is Wav2CLIP + STFT, not plain 35D Librosa alone, and its architecture/representation/metrics differ from this G1-retargeted EDGE branch.

## Next Steps

1. Inspect the matched 30s videos before choosing any checkpoint by rhythm metrics alone:
   - model renders under `slurm/pipelines/20260512-finedance-g1-30s-renders/*/renders/`
   - GT/source renders under `slurm/pipelines/20260512-finedance-g1-30s-renders/data_reference/renders/`
2. If reporting comparisons again, include GT/reference first. Use the row from:
   - `slurm/pipelines/20260511-finedance-g1-librosa-fullctx-gt/eval_with_audio/metrics.json`
3. Treat `Librosa35 full-context motiondist` as the current low-dimensional Librosa baseline, not the lbeat+robotloss run.
4. Treat `Jukebox FKBeat no-lbeat` as the cleaner deployment anchor unless a qualitative/render review says otherwise.
5. If continuing toward paper-aligned feature work, the next fair direction is probably Wav2CLIP + STFT or a rhythm-aware STFT/PRE-style feature module, not plain 35D Librosa alone.
6. For new-server continuation of Wav2CLIP/STFT/GaussianBeat Stage A, clone or
   switch to the pushed `wav2clip-stft-beat` branch and follow its `HANDOFF.md`.
7. Before any future push, review dirty source changes and keep runtime artifacts under `slurm/`, `data/`, `videos/`, and generated caches untracked. Do not commit checkpoints, dataset files, caches, renders, or Slurm logs.
8. If preparing another PR/push, rerun focused tests that cover the changed code paths:
   - `python -m unittest tests.test_beat_features tests.test_validate_preprocessed_data`
   - use Slurm for any heavier validation.
