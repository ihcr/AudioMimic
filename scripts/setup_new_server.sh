#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_new_server.sh [options]

Sets up the wav2clip-stft-beat branch on a new server.

Data sources, choose at most one:
  --artifact-source SRC      rsync source root. Examples:
                             OLD:/projects/u6ed/yukun/EDGE/.worktrees/wav2clip
                             /mnt/edge_wav2clip_artifacts
  --hf-repo REPO_ID          Hugging Face dataset repo containing repo-relative
                             paths such as data/finedance_g1_fkbeats/.
                             Default: wyksdsg/edge-g1-beatdistance

Options:
  --hf-revision REV          HF revision, branch, or commit. Default: main.
  --include-cache            Also fetch stream-adapter tensor/cache backup.
  --include-checkpoint       Also fetch r02 stream-adapter train-500.pt.
  --include-diffusion-caches Also fetch selected diffusion baseline caches.
  --include-diffusion-checkpoints
                             Also fetch selected diffusion anchor checkpoints.
  --include-evidence         Also fetch curated Slurm/metric evidence.
  --skip-env                 Do not create .venv311 or install Python deps.
  --skip-data                Do not fetch/copy runtime artifacts.
  --skip-torch               Do not install torch/torchaudio automatically.
  --torch-index-url URL      PyTorch wheel index, e.g.
                             https://download.pytorch.org/whl/cu126
  --run-validation           Run data/validate_preprocessed_data.py after setup.
  --no-fetch-g1-model        Do not fetch Unitree G1 MJCF/meshes if missing.
  --help                     Show this help.

Environment:
  PYTHON_BIN                 Python used to create .venv311. Default: python3.11.
  HF_TOKEN                   Hugging Face token for private dataset repos.

Examples:
  scripts/setup_new_server.sh \
    --artifact-source OLD:/projects/u6ed/yukun/EDGE/.worktrees/wav2clip \
    --include-cache --include-checkpoint

  scripts/setup_new_server.sh \
    --hf-repo wyksdsg/edge-g1-beatdistance \
    --include-cache --include-checkpoint --run-validation
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

ARTIFACT_SOURCE=""
HF_REPO=""
DEFAULT_HF_REPO="wyksdsg/edge-g1-beatdistance"
HF_REVISION="main"
INCLUDE_CACHE=0
INCLUDE_CHECKPOINT=0
INCLUDE_DIFFUSION_CACHES=0
INCLUDE_DIFFUSION_CHECKPOINTS=0
INCLUDE_EVIDENCE=0
SKIP_ENV=0
SKIP_DATA=0
SKIP_TORCH=0
RUN_VALIDATION=0
FETCH_G1_MODEL=1
TORCH_INDEX_URL=""

DATA_REL="data/finedance_g1_fkbeats"
CACHE_REL="data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups"
CHECKPOINT_REL="runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt"
G1_MODEL_REL="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-source)
      ARTIFACT_SOURCE="${2:?missing value for --artifact-source}"
      shift 2
      ;;
    --hf-repo)
      HF_REPO="${2:?missing value for --hf-repo}"
      shift 2
      ;;
    --hf-revision)
      HF_REVISION="${2:?missing value for --hf-revision}"
      shift 2
      ;;
    --include-cache)
      INCLUDE_CACHE=1
      shift
      ;;
    --include-checkpoint)
      INCLUDE_CHECKPOINT=1
      shift
      ;;
    --include-diffusion-caches)
      INCLUDE_DIFFUSION_CACHES=1
      shift
      ;;
    --include-diffusion-checkpoints)
      INCLUDE_DIFFUSION_CHECKPOINTS=1
      shift
      ;;
    --include-evidence)
      INCLUDE_EVIDENCE=1
      shift
      ;;
    --skip-env)
      SKIP_ENV=1
      shift
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:?missing value for --torch-index-url}"
      shift 2
      ;;
    --run-validation)
      RUN_VALIDATION=1
      shift
      ;;
    --no-fetch-g1-model)
      FETCH_G1_MODEL=0
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

if [[ -n "$ARTIFACT_SOURCE" && -n "$HF_REPO" ]]; then
  echo "Choose only one of --artifact-source or --hf-repo." >&2
  exit 2
fi

log() {
  printf '[setup_new_server] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

venv_python() {
  printf '%s/.venv311/bin/python' "$REPO_ROOT"
}

runtime_python() {
  if [[ -x "$(venv_python)" ]]; then
    venv_python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "No Python interpreter found." >&2
  exit 1
}

install_env() {
  if [[ "$SKIP_ENV" -eq 1 ]]; then
    log "Skipping Python environment setup."
    return
  fi

  if [[ ! -x "$(venv_python)" ]]; then
    require_cmd "$PYTHON_BIN"
    log "Creating .venv311 with $PYTHON_BIN."
    "$PYTHON_BIN" -m venv "$REPO_ROOT/.venv311"
  fi

  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv311/bin/activate"
  log "Upgrading pip tooling."
  python -m pip install --upgrade pip setuptools wheel

  if [[ "$SKIP_TORCH" -eq 0 ]]; then
    if python -c 'import torch, torchaudio' >/dev/null 2>&1; then
      log "torch and torchaudio are already importable."
    else
      log "Installing torch and torchaudio."
      if [[ -n "$TORCH_INDEX_URL" ]]; then
        python -m pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
      else
        python -m pip install torch torchaudio
      fi
    fi
  fi

  log "Installing repo dependencies from requirements-new-server.txt."
  python -m pip install -r "$REPO_ROOT/requirements-new-server.txt"

  if [[ "$FETCH_G1_MODEL" -eq 1 && ! -f "$REPO_ROOT/$G1_MODEL_REL" ]]; then
    log "Fetching Unitree G1 model files."
    python "$REPO_ROOT/scripts/fetch_unitree_g1_description.py" \
      --output_root "$REPO_ROOT/third_party/unitree_g1_description"
  fi
}

rsync_dir_from_source() {
  local rel="$1"
  local follow_symlinks="$2"
  local src="${ARTIFACT_SOURCE%/}/$rel/"
  local dst="$REPO_ROOT/$rel/"
  mkdir -p "$dst"
  require_cmd rsync
  if [[ "$follow_symlinks" -eq 1 ]]; then
    log "Copying $rel with rsync -aL."
    rsync -aL --info=progress2 "$src" "$dst"
  else
    log "Copying $rel with rsync -a."
    rsync -a --info=progress2 "$src" "$dst"
  fi
}

rsync_file_from_source() {
  local rel="$1"
  local src="${ARTIFACT_SOURCE%/}/$rel"
  local dst="$REPO_ROOT/$rel"
  mkdir -p "$(dirname "$dst")"
  require_cmd rsync
  log "Copying $rel."
  rsync -a --info=progress2 "$src" "$dst"
}

download_from_hf() {
  local patterns
  patterns="'$DATA_REL/**'"
  if [[ "$INCLUDE_CACHE" -eq 1 ]]; then
    patterns="$patterns, '$CACHE_REL/**'"
  fi
  if [[ "$INCLUDE_CHECKPOINT" -eq 1 ]]; then
    patterns="$patterns, '$CHECKPOINT_REL'"
  fi
  if [[ "$INCLUDE_DIFFUSION_CACHES" -eq 1 ]]; then
    patterns="$patterns, 'data/finedance_g1_fkbeats_dataset_backups_fkbeat1000/**'"
    patterns="$patterns, 'data/finedance_g1_librosa35_fullctx_motiondist_cond_dataset_backups/**'"
  fi
  if [[ "$INCLUDE_DIFFUSION_CHECKPOINTS" -eq 1 ]]; then
    patterns="$patterns, 'runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt'"
    patterns="$patterns, 'runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt'"
  fi
  if [[ "$INCLUDE_EVIDENCE" -eq 1 ]]; then
    patterns="$patterns, 'docs/experiments/artifacts/**'"
  fi

  log "Downloading artifacts from HF dataset repo $HF_REPO@$HF_REVISION."
  "$(runtime_python)" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    revision="$HF_REVISION",
    local_dir="$REPO_ROOT",
    allow_patterns=[$patterns],
)
PY
}

fetch_data() {
  if [[ "$SKIP_DATA" -eq 1 ]]; then
    log "Skipping runtime artifact fetch."
    return
  fi

  if [[ -n "$ARTIFACT_SOURCE" ]]; then
    rsync_dir_from_source "$DATA_REL" 1
    if [[ "$INCLUDE_CACHE" -eq 1 ]]; then
      rsync_dir_from_source "$CACHE_REL" 0
    fi
    if [[ "$INCLUDE_CHECKPOINT" -eq 1 ]]; then
      rsync_file_from_source "$CHECKPOINT_REL"
    fi
    return
  fi

  if [[ -n "$HF_REPO" ]]; then
    download_from_hf
    return
  fi

  HF_REPO="$DEFAULT_HF_REPO"
  download_from_hf
}

quick_check() {
  log "Running quick artifact checks."
  "$(runtime_python)" - <<'PY'
from pathlib import Path

checks = [
    ("data/finedance_g1_fkbeats/train/wav2clip_stft_beat_feats", "*.npy", 47817),
    ("data/finedance_g1_fkbeats/test/wav2clip_stft_beat_feats", "*.npy", 3265),
]
for rel, pattern, expected in checks:
    path = Path(rel)
    if not path.exists():
        raise SystemExit(f"Missing required artifact directory: {rel}")
    count = sum(1 for _ in path.glob(pattern))
    print(f"{rel}: {count}/{expected}")
    if count != expected:
        raise SystemExit(f"{rel}: expected {expected}, found {count}")

optional = [
    "data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups",
    "runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt",
]
for rel in optional:
    state = "present" if Path(rel).exists() else "missing"
    print(f"{rel}: {state}")
PY
}

run_validation() {
  log "Running sample validation."
  "$(runtime_python)" "$REPO_ROOT/data/validate_preprocessed_data.py" \
    --data_path "$REPO_ROOT/data/finedance_g1_fkbeats" \
    --processed_data_dir "$REPO_ROOT/data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups" \
    --feature_type wav2clip_stft_beat \
    --motion_format g1 \
    --feature_cache_mode memmap \
    --feature_cache_dtype float16 \
    --sample_count 64
}

main() {
  cd "$REPO_ROOT"
  mkdir -p data runs/train slurm
  install_env
  fetch_data
  if [[ "$SKIP_DATA" -eq 0 ]]; then
    quick_check
    if [[ "$RUN_VALIDATION" -eq 1 ]]; then
      run_validation
    fi
  fi
  log "Done. Activate with: source .venv311/bin/activate"
}

main
