#!/usr/bin/env bash
#SBATCH --job-name=edge_train
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm/train.out
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail
cd /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip
source /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.venv311/bin/activate
export PYTHONUNBUFFERED=1
export TERM=xterm-256color
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_DISABLED=true
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting stage"
stdbuf -oL -eL python train.py --feature_type wav2clip_stft_beat --motion_format g1 --lambda_beat 0.0 --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_wav2clip_stft_beat_concat_norm_dataset_backups --project runs/train --exp_name EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm --render_dir renders --batch_size 512 --gradient_accumulation_steps 1 --epochs 500 --epoch_offset 0 --save_interval 50 --ema_interval 1 --learning_rate 0.0002 --weight_decay 0.02 --train_num_workers 2 --test_num_workers 2 --mixed_precision bf16 --feature_cache_mode memmap --feature_cache_dtype float16 --feature_fusion concat_norm --lambda_g1_fk 0.0 --lambda_g1_fk_vel 0.0 --lambda_g1_fk_acc 0.0 --lambda_g1_foot 0.0 --lambda_g1_kin 1.0 --g1_kin_loss_warmup_epochs 0 --g1_kin_loss_max_fraction 0.0 --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml --g1_root_quat_order xyzw
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete"
