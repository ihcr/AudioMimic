#!/usr/bin/env bash
set -euo pipefail

resolve_shared_root() {
  local current
  current="$(cd "$1" && pwd)"
  while [[ "$current" != "/" ]]; do
    if [[ "$(basename "$current")" == ".worktrees" ]]; then
      dirname "$current"
      return
    fi
    current="$(dirname "$current")"
  done
  cd "$1" && pwd
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHARED_ROOT="$(resolve_shared_root "$REPO_ROOT")"

RUN_NAME="beat_estimator"
if [[ $# -gt 0 && "$1" != -* ]]; then
  RUN_NAME="$1"
  shift
fi

PARTITION="${PARTITION:-workq}"
ESTIMATOR_TIME="${ESTIMATOR_TIME:-08:00:00}"
ESTIMATOR_CPUS="${ESTIMATOR_CPUS:-8}"
ESTIMATOR_MEM="${ESTIMATOR_MEM:-32G}"
ESTIMATOR_GPUS="${ESTIMATOR_GPUS:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-$RUN_NAME}"
RUN_DIR="$REPO_ROOT/slurm/beat_estimator/$RUN_ID"
SCRIPT_PATH="$RUN_DIR/train_beat_estimator.sbatch"
LOG_PATH="$RUN_DIR/beat_estimator.out"
OUTPUT_PATH="$RUN_DIR/beat_estimator.pt"
mkdir -p "$RUN_DIR"

EXTRA_ARGS=""
if [[ $# -gt 0 ]]; then
  printf -v EXTRA_ARGS ' %q' "$@"
fi

cat > "$SCRIPT_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=edge_beat_estimator
#SBATCH --output=$LOG_PATH
#SBATCH --partition=$PARTITION
#SBATCH --time=$ESTIMATOR_TIME
#SBATCH --cpus-per-task=$ESTIMATOR_CPUS
#SBATCH --mem=$ESTIMATOR_MEM
#SBATCH --gres=gpu:$ESTIMATOR_GPUS

set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
source $(printf '%q' "$SHARED_ROOT/.venv311/bin/activate")
export PYTHONUNBUFFERED=1
export TERM=xterm-256color
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting stage"
stdbuf -oL -eL python train_beat_estimator.py --motion_dir data/train/motions_sliced --beat_dir data/train/beat_feats --output_path $(printf '%q' "$OUTPUT_PATH")$EXTRA_ARGS
echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage complete"
EOF

sbatch "$SCRIPT_PATH"
