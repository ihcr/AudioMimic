#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm_train_g1_latent_diffusion.sh [--dry-run]

Submit EXP-20260629 V6b-C dual-route latent diffusion training as a Slurm job.

Environment overrides:
  PARTITION              Slurm partition. Default: workq
  TIME_LIMIT             Job time limit. Default: 08:00:00
  CPUS_PER_TASK          CPU cores. Default: 16
  MEMORY                 Slurm memory. Default: 128G
  GPUS                   GPU count. Default: 1
  ACCOUNT                Optional Slurm account.
  WANDB_MODE             W&B mode. Default: offline
  BATCH_SIZE             Training batch size. Default: 512
  NUM_WORKERS            DataLoader workers. Default: 8
  CACHE_BATCH_SIZE       Latent cache build batch size. Default: 1024
  MIXED_PRECISION        AMP precision: no, fp16, or bf16. Default: bf16
  LEARNING_RATE          Training learning rate. Default: 2e-4
  EPOCHS                 Training epochs. Default: 500
  CHECKPOINT             Optional checkpoint to resume from.
  RESUME_OPTIMIZER       Resume optimizer/scaler state when CHECKPOINT is set. Default: 0
  SAVE_INTERVAL          Checkpoint interval. Default: 100
  EVAL_INTERVAL          Lightweight eval interval. Default: 25
  FULL_EVAL_INTERVAL     Full eval interval. Default: 500
  FULL_EVAL_METRIC_WORKERS
                         Parallel CPU workers for full-eval metrics. Default: 8
  USE_WAV2CLIP_SEMANTIC  Enable Wav2CLIP semantic tower: 0 or 1. Default: 0
  RUN_SUFFIX             Run suffix. Defaults to r01_control_only or r02_wav2clip_control
  CONTROL_RANK_WEIGHT    Control ranking loss weight. Default: 0.10
  CONTROL_RANK_MARGIN    Control ranking margin. Default: 0.02
  SEMANTIC_RANK_WEIGHT   Semantic ranking loss weight. Default: 0.03
  SEMANTIC_RANK_MARGIN   Semantic ranking margin. Default: 0.01
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

EXP_ID="EXP-20260629-finedance-g1-v6bc-dual-route-latent-diffusion"
USE_WAV2CLIP_SEMANTIC="${USE_WAV2CLIP_SEMANTIC:-0}"
if [[ -z "${RUN_SUFFIX:-}" ]]; then
  if [[ "$USE_WAV2CLIP_SEMANTIC" == "1" || "$USE_WAV2CLIP_SEMANTIC" == "true" ]]; then
    RUN_SUFFIX="r02_wav2clip_control"
  else
    RUN_SUFFIX="r01_control_only"
  fi
fi
EXP_NAME="${EXP_ID}_${RUN_SUFFIX}"
PARTITION="${PARTITION:-workq}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-128G}"
GPUS="${GPUS:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-1024}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EPOCHS="${EPOCHS:-500}"
CHECKPOINT="${CHECKPOINT:-}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
FULL_EVAL_INTERVAL="${FULL_EVAL_INTERVAL:-500}"
FULL_EVAL_METRIC_WORKERS="${FULL_EVAL_METRIC_WORKERS:-8}"
CONTROL_RANK_WEIGHT="${CONTROL_RANK_WEIGHT:-0.10}"
CONTROL_RANK_MARGIN="${CONTROL_RANK_MARGIN:-0.02}"
SEMANTIC_RANK_WEIGHT="${SEMANTIC_RANK_WEIGHT:-0.03}"
SEMANTIC_RANK_MARGIN="${SEMANTIC_RANK_MARGIN:-0.01}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r02_gh200_b1024_w8_bf16/weights/train-500.pt}"

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
#SBATCH --job-name=m2d_v6bc_latent
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
RESUME_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
  RESUME_ARGS+=(--checkpoint "$CHECKPOINT")
  if [[ "$RESUME_OPTIMIZER" == "1" || "$RESUME_OPTIMIZER" == "true" ]]; then
    RESUME_ARGS+=(--resume_optimizer)
  fi
fi
SEMANTIC_ARGS=()
if [[ "$USE_WAV2CLIP_SEMANTIC" == "1" || "$USE_WAV2CLIP_SEMANTIC" == "true" ]]; then
  SEMANTIC_ARGS+=(--use_wav2clip_semantic)
fi
"$PYTHON" -m train_g1_latent_diffusion \\
  --data_path data/finedance_g1_fkbeats \\
  --motion_prior_processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \\
  --latent_processed_data_dir data/finedance_g1_v6bc_music_control_latent_dataset_backups \\
  --prior_checkpoint "$PRIOR_CHECKPOINT" \\
  --exp_name "$EXP_NAME" \\
  --motion_format g1_yaw_delta \\
  --batch_size "$BATCH_SIZE" \\
  --num_workers "$NUM_WORKERS" \\
  --cache_batch_size "$CACHE_BATCH_SIZE" \\
  --epochs "$EPOCHS" \\
  --learning_rate "$LEARNING_RATE" \\
  --weight_decay 0.02 \\
  --mixed_precision "$MIXED_PRECISION" \\
  --control_rank_weight "$CONTROL_RANK_WEIGHT" \\
  --control_rank_margin "$CONTROL_RANK_MARGIN" \\
  --semantic_rank_weight "$SEMANTIC_RANK_WEIGHT" \\
  --semantic_rank_margin "$SEMANTIC_RANK_MARGIN" \\
  --save_interval "$SAVE_INTERVAL" \\
  --eval_interval "$EVAL_INTERVAL" \\
  --full_eval_interval "$FULL_EVAL_INTERVAL" \\
  --full_eval_metric_workers "$FULL_EVAL_METRIC_WORKERS" \\
  --wandb_pj_name Musics2Dance \\
  --wandb_mode "$WANDB_MODE" \\
  "\${SEMANTIC_ARGS[@]}" \\
  "\${RESUME_ARGS[@]}" \\
  2>&1 | tee -a "$TEE_LOG"
echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished $EXP_NAME"
EOF

echo "Wrote $SBATCH_PATH"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. Submit with:"
  echo "  sbatch $SBATCH_PATH"
else
  SUBMIT_OUTPUT="$(sbatch "$SBATCH_PATH")"
  echo "$SUBMIT_OUTPUT"
  JOB_ID="$(awk '{print $4}' <<<"$SUBMIT_OUTPUT")"
  echo "Follow live Slurm log with:"
  echo "  tail -f $SLURM_DIR/train_${JOB_ID}.out"
  echo "Curated tee log fallback:"
  echo "  tail -f $TEE_LOG"
fi
