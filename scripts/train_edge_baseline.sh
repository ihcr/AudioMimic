#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_NAME="edge_baseline"
if [[ $# -gt 0 && "$1" != -* ]]; then
  TRAIN_NAME="$1"
  shift
fi

"$REPO_ROOT/submit_training_pipeline.sh" \
  --preset edge_baseline \
  --train_name "$TRAIN_NAME" \
  "$@"
