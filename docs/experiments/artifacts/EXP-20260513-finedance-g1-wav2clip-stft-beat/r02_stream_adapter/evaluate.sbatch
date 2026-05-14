#!/usr/bin/env bash
#SBATCH --job-name=edge_evaluate
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/evaluate.out
#SBATCH --partition=workq
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
stdbuf -oL -eL python eval/run_g1_dataset_eval.py --checkpoint runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt --feature_type wav2clip_stft_beat --feature_fusion stream_adapter --data_path data/finedance_g1_fkbeats --processed_data_dir data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups --render_dir /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/eval/renders --motion_save_dir /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/eval/motions --metrics_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/eval/metrics.json --g1_table_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/eval/g1_table.json --motion_audit_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/eval/motion_audit.json --paper_report_path /lus/lfs1aip2/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/eval/paper_report.md --seed 1234 --enable_fk_metrics --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml --g1_root_quat_order xyzw
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete"
