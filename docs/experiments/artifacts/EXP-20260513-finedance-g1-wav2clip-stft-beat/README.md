# EXP-20260513 Wav2CLIP/STFT/Beat Artifacts

Curated Slurm evidence for the `wav2clip-stft-beat` branch. These files are
small enough for GitHub and are enough to reconstruct what ran without
committing runtime data, feature caches, checkpoints, or full `slurm/`.

## Kept

- `preprocess/`: feature-extraction sbatch plus logs proving the first
  single-process attempt was slow and the 8-way array completed all shards.
- `r01_concat_norm/`: generated pipeline scripts and validation log. There is no
  train/eval output because the job was cancelled by quota before training ran.
- `r02_stream_adapter/`: generated pipeline scripts, validation log, and the
  compressed full `train.out` proving the 500-epoch training run completed.

## Deliberately Not Kept

- `data/finedance_g1_fkbeats/`: copy with `rsync -aL` when migrating.
- `data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups/`: copy
  only when you want to avoid rebuilding the tensor cache.
- `runs/train/.../train-500.pt`: copy as a runtime checkpoint, not through Git.
- cancelled eval output: no metrics were produced before the quota block.
