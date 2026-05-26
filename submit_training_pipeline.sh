#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_VENV_PYTHON="$(cd "$SCRIPT_DIR/../.." && pwd)/.venv311/bin/python"

if [[ -x "$REPO_VENV_PYTHON" ]]; then
  "$REPO_VENV_PYTHON" "$SCRIPT_DIR/submit_training_pipeline.py" "$@"
else
  python3 "$SCRIPT_DIR/submit_training_pipeline.py" "$@"
fi
