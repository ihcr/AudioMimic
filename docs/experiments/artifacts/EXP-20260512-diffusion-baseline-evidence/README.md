# EXP-20260512 Diffusion Baseline Evidence

Curated GitHub-safe evidence from the `diffusion` branch. This is meant to keep
the important train/eval provenance available after moving servers without
committing large runtime folders, generated motions, videos, or checkpoint
weights.

## Kept

- `jukebox_fkbeat_no_lbeat/`: train/eval scripts, compressed train/eval logs,
  metrics, G1 table, report, and compressed motion audit for the clean Jukebox
  FKBeat anchor.
- `librosa35_fullctx_motiondist/`: train/eval scripts, compressed train/eval
  logs, metrics, G1 table, report, and compressed motion audit for the best
  balanced Librosa35 full-context anchor.
- `gt_reference/`: GT/reference metrics with audio paths and compressed motion
  audit.
- `render_30s/`: 30-second render job scripts/logs and small metric summaries
  only.

## Deliberately Not Kept

- Checkpoints under `runs/train/`: copy final anchor `.pt` files as runtime
  artifacts when needed.
- Generated motions, reference pickle copies, input wavs, and MP4s.
- Full `slurm/` trees for non-anchor or superseded jobs.
