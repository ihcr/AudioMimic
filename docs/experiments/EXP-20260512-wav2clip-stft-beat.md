# EXP-20260512-wav2clip-stft-beat

Status: archived
Owner: yukun
Created: 2026-05-12
Last Updated: 2026-05-13

## Research Question

Can EDGE replace Jukebox conditioning with a lighter music feature stack while preserving the current Transformer diffusion backbone and offline full-song generation workflow?

## Hypothesis

`Wav2CLIP + STFT + GaussianBeat` should provide a better first non-Jukebox baseline than either handcrafted audio features alone or a single foundation embedding alone. Wav2CLIP supplies high-level music semantics, STFT preserves local acoustic texture, and GaussianBeat provides an explicit frame-level rhythm prior.

## Baseline Or Control

- Current EDGE Transformer diffusion with Jukebox features.
- Optional control: current EDGE Transformer diffusion with 35-D Librosa baseline features.

## Intervention

Train two first-stage fusion variants from the same raw feature cache:

1. `r01_concat_norm`: normalize each raw stream and concatenate raw dimensions directly. Expected condition width is `512 + 193 + 1 = 706`.
2. `r02_stream_adapter`: split the same raw feature tensor into Wav2CLIP, STFT, and GaussianBeat streams, pass each stream through its own learned adapter and normalization, then concatenate the adapted streams. Use a 512-D fused condition to match the current EDGE latent dimension:
   - Wav2CLIP `512 -> 256`
   - STFT `193 -> 192`
   - GaussianBeat `1 -> 64`
   - concatenated fused condition: `256 + 192 + 64 = 512`

The adapter dimensions follow common multimodal fusion practice: keep the fused condition close to the model hidden size, use simple multiples of 32/64 for GPU-friendly dense layers, compress high-dimensional foundation features, preserve low-level time-frequency detail with little compression, and expand scalar control signals into a small embedding.

The first experiment intentionally does not add Mamba, tokenization, beat gating, or feature-wise rhythm control inside the denoiser.

## Invariant Controls

- Dataset/split: use the existing EDGE/AIST++ train-test split and record the exact split files before launch.
- Motion or input representation: current EDGE SMPL body representation with 4 contact channels, root translation, and 24-joint 6D rotations.
- Model family: keep the current Transformer `DanceDecoder` unchanged for the first feature-only baseline.
- Training budget: match the chosen Jukebox baseline as closely as queue and compute allow; record any budget mismatch.
- Evaluation protocol: use the same eval split, metrics, render settings, and matched music clips when comparing to Jukebox.

## Data And Cache Contract

- Raw data: existing EDGE AIST++ raw/sliced data tree.
- Processed data: keep current motion preprocessing semantics unchanged.
- Feature cache: create `wav2clip_stft_beat_feats`; do not overwrite Jukebox or baseline feature folders.
- Tensor/cache layer: use feature-specific processed and tensor dataset caches.
- Cache invalidation needed: yes, because the feature tensor shape and feature semantics differ from both Jukebox and baseline features.

## Implementation Scope

- Branch: `wav2clip-stft-beat`; this archived AIST/SMPL attempt predated the
  FineDance+G1 correction and should be read as historical context.
- Files/modules expected to change:
  - `args.py`
  - `EDGE.py`
  - `dataset/dance_dataset.py`
  - `test.py`
  - `train.py`
  - `data/create_dataset.py`
  - `data/audio_extraction/`
  - `model/model.py`
  - focused tests under `tests/`
- Files/modules intentionally unchanged for r01/r02:
  - Diffusion process and losses in `model/diffusion.py`.
  - Motion preprocessing and evaluator semantics.
  - Backbone attention blocks beyond the minimal conditioning fusion module.

## Training Or Execution Plan

- Environment: `source .venv311/bin/activate` from a direct branch clone.
- Command or script:
  - `sbatch slurm/EXP-20260512-wav2clip-stft-beat/preprocess.sbatch`
  - `sbatch --dependency=afterok:4575424 slurm/EXP-20260512-wav2clip-stft-beat/train_concat.sbatch`
  - `sbatch --dependency=afterok:4575424 slurm/EXP-20260512-wav2clip-stft-beat/train_adapter.sbatch`
- Slurm jobs:
  - failed preprocess attempt: `4575392`
  - preprocess: `4575424`
  - `r01_concat_norm`: `4575425`
  - `r02_stream_adapter`: `4575426`
- Logs: use `slurm/EXP-20260512-wav2clip-stft-beat/`.
- Run directory: use `runs/train/EXP-20260512-wav2clip-stft-beat_r01_concat_norm/` and `runs/train/EXP-20260512-wav2clip-stft-beat_r02_stream_adapter/`.
- Checkpoints: save under the run directory with `--save_interval 50`.
- Resume point: none yet.

## Evaluation Plan

- Command or script: to be filled after training and feature cache are available.
- Metrics: EDGE-compatible FID/PFC/BAS/diversity metrics, plus any existing full benchmark tables used for Jukebox comparisons.
- Qualitative artifacts: skeleton renders and saved motions under `renders/EXP-20260512-wav2clip-stft-beat/` and `eval/motions/EXP-20260512-wav2clip-stft-beat/`.
- Comparison target: current Jukebox Transformer EDGE checkpoint and any baseline-feature control produced with the same split.
- Success criteria:
  - feature extraction completes without Jukebox dependency;
  - feature tensors are finite and aligned to 30 FPS motion windows;
  - fresh forward loss is finite;
  - short training does not produce NaN weights;
  - generated motions render and evaluation metrics are comparable to the control.
- Failure signals:
  - feature/motion length mismatch;
  - feature scale dominates or collapses one stream;
  - NaN loss or NaN checkpoint weights;
  - BAS or visual rhythm clearly degrades despite good motion realism.

## Music-To-Dance Notes

- Music feature type/extractor: Wav2CLIP + STFT + GaussianBeat, with `r01_concat_norm` and `r02_stream_adapter` trained as parallel feature-fusion variants.
- Motion representation/body target: EDGE/AIST++ SMPL body, no hand/finger expansion in r01/r02.
- FPS, horizon, stride: keep 30 FPS and 5-second model horizon. Keep current offline inference with 5-second windows and 2.5-second overlap.
- Beat/alignment conditioning: GaussianBeat is a conditioning feature only.
- Beat/alignment supervision: unchanged; do not add a new beat loss in r01/r02.
- Beat/alignment metrics: BAS and qualitative matched-music render inspection.
- Render/video paths: `renders/EXP-20260512-wav2clip-stft-beat/`.

## Future Architecture Experiments

These are explicitly deferred until the feature-only baseline is stable:

- Beat-gated FiLM instead of simple conditioning concat.
- Transformer + Mamba hybrid decoder.
- Full Mamba or BiMamba replacement for the denoiser blocks.
- Longer horizon such as 10 seconds.
- FSQ/tokenized motion or music pipeline.
- MuQ or MERT feature variants.

## Run Log

| Date | Run | Status | Evidence | Notes |
|---|---|---|---|---|
| 2026-05-12 | design | spec | `docs/experiments/EXP-20260512-wav2clip-stft-beat.md` | Parallel fusion variants selected: raw stream-normalized concat and learned stream adapter. |
| 2026-05-12 | implementation | passed | `python -m unittest tests.test_feature_config_and_fusion tests.test_wav2clip_stft_beat_features tests.test_edge_checkpoint_load` | 9 focused tests pass, including beat stream not being normalized away. |
| 2026-05-12 | feature smoke | passed | `/tmp/edge_wav2clip_stft_beat_smoke/gHO_sBM_cAll_d19_mHO1_ch03.npy` | Real Wav2CLIP/STFT/GaussianBeat output shape `(150, 706)`, dtype `float32`, finite. |
| 2026-05-12 | full loss smoke | passed | synthetic `EDGE('wav2clip_stft_beat', feature_fusion=...)` loss | `concat_norm` total loss `9.6763`; `stream_adapter` total loss `9.7790`; both finite. |
| 2026-05-12 | preprocessing attempt 1 | failed | Slurm job `4575392` | Running `create_dataset.py` from `data/` could not import repo-root `feature_config`; fixed extractor repo-root path handling. |
| 2026-05-12 | preprocessing attempt 2 | queued | Slurm job `4575424` | Builds `wav2clip_stft_beat_feats` and prebuilds feature-specific dataset/tensor caches. |
| 2026-05-12 | r01_concat_norm train | queued | Slurm job `4575425`, dependency `afterok:4575424` | 500-epoch first-stage run, batch size 64. |
| 2026-05-12 | r02_stream_adapter train | queued | Slurm job `4575426`, dependency `afterok:4575424` | 500-epoch first-stage run, batch size 64. |
| 2026-05-13 | scope correction | archived | Slurm jobs `4575424`, `4575425`, `4575426` were cancelled | Superseded by FineDance+G1 run `EXP-20260513-finedance-g1-wav2clip-stft-beat`; do not use the AIST/SMPL queue as the active experiment. |

## Results

- Metric files: none yet.
- Render/report paths: none yet.
- Checkpoint paths: none yet.
- Key observations:
  - PyPI `wav2clip` needs `torchaudio==2.11.0+cu126` in this environment; the generic `torchaudio==2.11.0` wheel looked for CUDA 13.
  - `wav2clip.embed_audio` returns channel-first embeddings for this package version, so the extractor transposes `(512, T)` to `(T, 512)`.
  - Do not apply `LayerNorm(1)` to the scalar GaussianBeat stream; it collapses the signal to zero.

## Current Conclusion

The AIST/SMPL launch was intentionally stopped after the target dataset/body was
corrected. The implementation work carries forward, but the active run is now
FineDance+G1 in the `wav2clip-stft-beat` branch.

## Next Action

Use `EXP-20260513-finedance-g1-wav2clip-stft-beat` as the active experiment.
