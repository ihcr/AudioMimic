# New Server Setup

This branch can be restored on a new server with one script. The script prepares
`.venv311`, fetches the runtime artifacts, and checks that the Wav2CLIP/STFT/Beat
feature cache has the expected train/test counts.

## Option A: Copy From Another Server

From a fresh clone of `wav2clip-stft-beat`:

```bash
scripts/setup_new_server.sh \
  --artifact-source OLD:/projects/u6ed/yukun/EDGE/.worktrees/wav2clip \
  --include-cache \
  --include-checkpoint
```

`data/finedance_g1_fkbeats/` is copied with `rsync -aL`, so the old symlinked
motion/audio/baseline/beat directories become real files in the new clone.

## Option B: Download From Hugging Face

All EDGE/G1 Hugging Face artifacts should live in this repo:

```text
wyksdsg/edge-g1-beatdistance
```

The current URL is a model repo (`https://huggingface.co/wyksdsg/edge-g1-beatdistance`),
so the scripts default to `--repo-type model` / `--hf-repo-type model`.

First upload the runtime artifacts from the old server. This combines the real
prepared FineDance+G1 data from the `diffusion` worktree with the Wav2CLIP/STFT
feature directories from this `wav2clip-stft-beat` branch:

```bash
# On the current old server, the shared EDGE env is here; in a fresh clone,
# use the .venv311 created by scripts/setup_new_server.sh.
source /projects/u6ed/yukun/EDGE/.venv311/bin/activate
export HF_TOKEN=...
python scripts/upload_wav2clip_artifacts_to_hf.py \
  --repo-id wyksdsg/edge-g1-beatdistance \
  --repo-type model \
  --diffusion-root /projects/u6ed/yukun/EDGE/.worktrees/diffusion \
  --include-cache \
  --include-checkpoint \
  --include-diffusion-caches \
  --include-diffusion-checkpoints \
  --progress-seconds 30
```

Add `--private` only when creating a new private HF repo.
The uploader uses Hugging Face's resumable large-folder path and prints a
progress report every `--progress-seconds`.

Then on the new server:

```bash
export HF_TOKEN=...
scripts/setup_new_server.sh \
  --hf-repo wyksdsg/edge-g1-beatdistance \
  --hf-repo-type model \
  --include-cache \
  --include-checkpoint \
  --include-diffusion-caches \
  --include-diffusion-checkpoints \
  --include-evidence
```

The HF repo should preserve repo-relative paths, for example
`data/finedance_g1_fkbeats/...` and
`runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt`.
The diffusion anchor data/checkpoints use their original repo-relative paths,
for example `runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt`.

## Torch/CUDA Wheels

If the server needs a specific PyTorch CUDA wheel, pass the wheel index:

```bash
scripts/setup_new_server.sh \
  --hf-repo wyksdsg/edge-g1-beatdistance \
  --torch-index-url https://download.pytorch.org/whl/cu126 \
  --include-cache \
  --include-checkpoint
```

If the cluster already provides torch in the active environment, use
`--skip-torch`.

## Validation And Continuation

The default setup performs a lightweight artifact count check. For a deeper
sample validation, add `--run-validation`.

After setup, launch the unfinished concat run:

```bash
source .venv311/bin/activate
python submit_training_pipeline.py \
  --preset g1_finedance_wav2clip_stft_beat_concat_norm \
  --train_name EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm \
  --run_id EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm \
  --skip_preprocess \
  --train_time 04:00:00 \
  --eval_time 02:00:00
```

Then evaluate the existing stream-adapter checkpoint:

```bash
sbatch --time=02:00:00 \
  slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/evaluate.sbatch
```
