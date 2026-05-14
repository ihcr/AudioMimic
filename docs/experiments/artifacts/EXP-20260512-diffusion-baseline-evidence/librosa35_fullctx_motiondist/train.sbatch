#!/usr/bin/env bash
#SBATCH --job-name=edge_train
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/train.out
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
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_DISABLED=true
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting stage"
stdbuf -oL -eL python train.py --feature_type baseline --motion_format g1 --use_beats --beat_rep distance --lambda_beat 0.0 --lambda_acc 0.0 --beat_loss_start_epoch 0 --beat_loss_warmup_epochs 0 --beat_loss_max_fraction 0.0 --beat_loss_cap_mode hard --beat_estimator_max_val_loss 8.0 --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_librosa35_fullctx_motiondist_cond_dataset_backups --project runs/train --exp_name finedance_g1_librosa35_fullctx_motiondist_cond_2000 --render_dir renders --batch_size 512 --gradient_accumulation_steps 1 --epochs 2000 --epoch_offset 0 --save_interval 100 --ema_interval 1 --learning_rate 0.0002 --weight_decay 0.02 --train_num_workers 2 --test_num_workers 2 --mixed_precision bf16 --feature_cache_mode memmap --feature_cache_dtype float16 --lambda_g1_fk 0.0 --lambda_g1_fk_vel 0.0 --lambda_g1_fk_acc 0.0 --lambda_g1_foot 0.0 --lambda_g1_kin 1.0 --g1_kin_loss_warmup_epochs 0 --g1_kin_loss_max_fraction 0.0 --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml --g1_root_quat_order xyzw
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete"
