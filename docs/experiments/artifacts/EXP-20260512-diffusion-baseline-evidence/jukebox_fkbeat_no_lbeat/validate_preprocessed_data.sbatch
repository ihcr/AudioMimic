#!/usr/bin/env bash
#SBATCH --job-name=edge_validate_preprocessed_data
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2/validate_preprocessed_data.out
#SBATCH --partition=workq
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion
source /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.venv311/bin/activate
export PYTHONUNBUFFERED=1
export TERM=xterm-256color
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export WANDB_DISABLED=true
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting stage"
stdbuf -oL -eL python data/validate_preprocessed_data.py --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_fkbeats_dataset_backups_fkbeat1000 --feature_type jukebox --motion_format g1 --feature_cache_mode memmap --feature_cache_dtype float16 --sample_count 64 --use_beats --beat_rep distance
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete"
