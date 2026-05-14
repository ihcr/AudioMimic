#!/usr/bin/env bash
#SBATCH --job-name=edge_train
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2/train.out
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

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
stdbuf -oL -eL python train.py --feature_type jukebox --motion_format g1 --use_beats --beat_rep distance --lambda_beat 0.0 --lambda_acc 0.0 --beat_loss_start_epoch 0 --beat_loss_warmup_epochs 0 --beat_loss_max_fraction 0.0 --beat_loss_cap_mode hard --beat_estimator_max_val_loss 8.0 --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_fkbeats_dataset_backups_fkbeat1000 --project runs/train --exp_name finedance_g1_fkbeatdistance_1000 --render_dir renders --batch_size 512 --gradient_accumulation_steps 1 --epochs 1000 --save_interval 100 --ema_interval 1 --learning_rate 0.0002 --weight_decay 0.02 --train_num_workers 2 --test_num_workers 2 --mixed_precision bf16 --feature_cache_mode memmap --feature_cache_dtype float16
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete"
