#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm_train_g1_motion_prior.sh [--dry-run]

Submit EXP-20260623 V6b Stage A G1 motion-prior training as a Slurm job.

Environment overrides:
  PARTITION              Slurm partition. Default: workq
  TIME_LIMIT             Job time limit. Default: 24:00:00
  CPUS_PER_TASK          CPU cores. Default: 8
  MEMORY                 Slurm memory. Default: 64G
  GPUS                   GPU count. Default: 1
  ACCOUNT                Optional Slurm account.
  WANDB_MODE             W&B mode. Default: online
  BATCH_SIZE             Training batch size. Default: 256
  NUM_WORKERS            DataLoader workers. Default: 0
  CACHE_BATCH_SIZE       Cache build batch size. Default: 256
  MIXED_PRECISION        AMP precision: no, fp16, or bf16. Default: fp16
  LEARNING_RATE          Training learning rate. Default: 2e-4
  EPOCHS                 Training epochs. Default: 500
  SAVE_INTERVAL          Checkpoint interval. Default: 100
  EVAL_INTERVAL          Lightweight eval interval. Default: 50
  FULL_EVAL_INTERVAL     Full eval interval. Default: 500
  RUN_SUFFIX             Run suffix. Default: r01_ae_s2_latent128
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

EXP_ID="EXP-20260623-finedance-g1-v6b-motion-prior"
RUN_SUFFIX="${RUN_SUFFIX:-r01_ae_s2_latent128}"
EXP_NAME="${EXP_ID}_${RUN_SUFFIX}"
PARTITION="${PARTITION:-workq}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
GPUS="${GPUS:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-256}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EPOCHS="${EPOCHS:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
FULL_EVAL_INTERVAL="${FULL_EVAL_INTERVAL:-500}"

SLURM_DIR="$REPO_ROOT/slurm/$EXP_ID/$RUN_SUFFIX"
LOG_DIR="$REPO_ROOT/setup_logs/$EXP_ID"
SBATCH_PATH="$SLURM_DIR/train_${RUN_SUFFIX}.sbatch"
SLURM_LOG="$SLURM_DIR/train_%j.out"
TEE_LOG="$LOG_DIR/train_${RUN_SUFFIX}.log"
PYTHON="$REPO_ROOT/.venv311/bin/python"

mkdir -p "$SLURM_DIR" "$LOG_DIR"

ACCOUNT_LINE=""
if [[ -n "${ACCOUNT:-}" ]]; then
  ACCOUNT_LINE="#SBATCH --account=$ACCOUNT"
fi

cat > "$SBATCH_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=m2d_v6b_prior
#SBATCH --output=$SLURM_LOG
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME_LIMIT
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --gres=gpu:$GPUS
$ACCOUNT_LINE

set -euo pipefail
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export TERM=xterm-256color
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=\${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
export WANDB_MODE=\${WANDB_MODE:-$WANDB_MODE}

echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting $EXP_NAME"
"$PYTHON" -m train_g1_motion_prior \\
  --data_path data/finedance_g1_fkbeats \\
  --processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \\
  --exp_name "$EXP_NAME" \\
  --motion_format g1_yaw_delta \\
  --prior_type ae \\
  --latent_dim 128 \\
  --temporal_downsample 2 \\
  --batch_size "$BATCH_SIZE" \\
  --num_workers "$NUM_WORKERS" \\
  --cache_batch_size "$CACHE_BATCH_SIZE" \\
  --epochs "$EPOCHS" \\
  --learning_rate "$LEARNING_RATE" \\
  --weight_decay 0.02 \\
  --mixed_precision "$MIXED_PRECISION" \\
  --save_interval "$SAVE_INTERVAL" \\
  --eval_interval "$EVAL_INTERVAL" \\
  --full_eval_interval "$FULL_EVAL_INTERVAL" \\
  --wandb_pj_name Musics2Dance \\
  --wandb_mode "$WANDB_MODE" \\
  2>&1 | tee -a "$TEE_LOG"
echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished $EXP_NAME"
EOF

echo "Wrote $SBATCH_PATH"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. Submit with:"
  echo "  sbatch $SBATCH_PATH"
else
  sbatch "$SBATCH_PATH"
fi
