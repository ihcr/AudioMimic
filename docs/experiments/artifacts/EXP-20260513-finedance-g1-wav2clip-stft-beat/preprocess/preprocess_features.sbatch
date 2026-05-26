#!/usr/bin/env bash
#SBATCH --job-name=edge_fdg1_w2s_prep
#SBATCH --output=slurm/EXP-20260513-finedance-g1-wav2clip-stft-beat/preprocess_features_%A_%a.out
#SBATCH --error=slurm/EXP-20260513-finedance-g1-wav2clip-stft-beat/preprocess_features_%A_%a.err
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

set -euo pipefail

cd /projects/u6ed/yukun/EDGE/.worktrees/wav2clip
source ../../.venv311/bin/activate

export PYTHONUNBUFFERED=1
export TERM=xterm-256color
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_DISABLED=true

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting FineDance G1 Wav2CLIP/STFT/GaussianBeat extraction shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

python - <<'PY'
from pathlib import Path
import os

from data.audio_extraction.wav2clip_stft_beat_features import extract_folder

root = Path("data/finedance_g1_fkbeats")
shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
for split in ("train", "test"):
    src = root / split / "wavs_sliced"
    dest = root / split / "wav2clip_stft_beat_feats"
    dest.mkdir(parents=True, exist_ok=True)
    print(
        f"extracting {split} shard {shard_index}/{num_shards}: {src} -> {dest}",
        flush=True,
    )
    extract_folder(
        src,
        dest,
        skip_completed=True,
        shard_index=shard_index,
        num_shards=num_shards,
    )
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Feature extraction shard complete"
