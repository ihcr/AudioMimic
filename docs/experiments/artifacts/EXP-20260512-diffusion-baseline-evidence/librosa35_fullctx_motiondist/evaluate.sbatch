#!/usr/bin/env bash
#SBATCH --job-name=edge_evaluate
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/evaluate.out
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
stdbuf -oL -eL python eval/run_g1_dataset_eval.py --checkpoint runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt --feature_type baseline --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_librosa35_fullctx_motiondist_cond_dataset_backups --render_dir /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/renders --motion_save_dir /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/motions --metrics_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/metrics.json --g1_table_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/g1_table.json --motion_audit_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/motion_audit.json --paper_report_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260511-finedance-g1-librosa35-fullctx-motiondist-cond-2000/eval/paper_report.md --seed 1234 --use_beats --beat_rep distance --enable_fk_metrics --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml --g1_root_quat_order xyzw
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete"
