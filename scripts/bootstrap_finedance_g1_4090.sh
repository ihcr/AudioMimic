#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/bootstrap_finedance_g1_4090.sh [options]

One-command setup for a direct-attached 4090 server. This script does not use
Slurm. It creates/uses .venv311, downloads compact artifacts from Hugging Face,
then rebuilds FineDance+G1 prepared data and audio features locally.

Options:
  --hf-repo REPO_ID          Hugging Face repo. Default: wyksdsg/edge-g1-beatdistance.
  --hf-repo-type TYPE        HF repo type: model or dataset. Default: model.
  --hf-revision REV          HF revision, branch, or commit. Default: main.
  --torch-index-url URL      PyTorch wheel index. Default:
                             https://download.pytorch.org/whl/cu126
  --features LIST            Comma-separated features to rebuild.
                             Default: baseline,jukebox,wav2clip_stft_beat
  --jukebox-batch-size N     Jukebox extraction batch size. Default: 4.
  --skip-env                 Do not create .venv311 or install Python deps.
  --skip-download            Do not download HF compact artifacts.
  --skip-prepare             Do not rebuild prepared FineDance/G1 trees.
  --skip-features            Do not extract audio features.
  --skip-model-warmup        Do not pre-download Wav2CLIP/Jukebox weights.
  --run-validation           Run focused data validation after rebuild.
  --clean                    Delete prepared output trees before rebuilding.
  --dry-run                  Print planned commands without running them.
  --help                     Show this help.

Environment:
  PYTHON_BIN                 Python used to create .venv311. Default: python3.11.
  HF_TOKEN                   Hugging Face token if the repo is private.
  CUDA_VISIBLE_DEVICES       Optional GPU selection for the 4090.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

HF_REPO="wyksdsg/edge-g1-beatdistance"
HF_REPO_TYPE="model"
HF_REVISION="main"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
FEATURES="baseline,jukebox,wav2clip_stft_beat"
JUKEBOX_BATCH_SIZE="${EDGE_JUKEBOX_BATCH_SIZE:-4}"

SKIP_ENV=0
SKIP_DOWNLOAD=0
SKIP_PREPARE=0
SKIP_FEATURES=0
SKIP_MODEL_WARMUP=0
RUN_VALIDATION=0
CLEAN=0
DRY_RUN=0

FINEDANCE_ROOT="data/finedance"
G1_MOTION_DIR="data/finedance-g1-retargeted"
SOURCE_PREPARED_ROOT="data/finedance_aistpp"
G1_OUTPUT_ROOT="data/finedance_g1_fkbeats"
G1_MODEL_REL="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"
LOG_DIR="setup_logs/finedance_g1_4090"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo)
      HF_REPO="${2:?missing value for --hf-repo}"
      shift 2
      ;;
    --hf-repo-type)
      HF_REPO_TYPE="${2:?missing value for --hf-repo-type}"
      if [[ "$HF_REPO_TYPE" != "model" && "$HF_REPO_TYPE" != "dataset" ]]; then
        echo "--hf-repo-type must be model or dataset." >&2
        exit 2
      fi
      shift 2
      ;;
    --hf-revision)
      HF_REVISION="${2:?missing value for --hf-revision}"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:?missing value for --torch-index-url}"
      shift 2
      ;;
    --features)
      FEATURES="${2:?missing value for --features}"
      shift 2
      ;;
    --jukebox-batch-size)
      JUKEBOX_BATCH_SIZE="${2:?missing value for --jukebox-batch-size}"
      shift 2
      ;;
    --skip-env)
      SKIP_ENV=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-prepare)
      SKIP_PREPARE=1
      shift
      ;;
    --skip-features)
      SKIP_FEATURES=1
      shift
      ;;
    --skip-model-warmup)
      SKIP_MODEL_WARMUP=1
      shift
      ;;
    --run-validation)
      RUN_VALIDATION=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
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

log() {
  printf '[bootstrap_4090] %s\n' "$*"
}

venv_python() {
  printf '%s/.venv311/bin/python' "$REPO_ROOT"
}

run_cmd() {
  log "$*"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi
  "$@"
}

run_logged() {
  local name="$1"
  shift
  mkdir -p "$REPO_ROOT/$LOG_DIR"
  log "$*"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi
  "$@" 2>&1 | tee -a "$REPO_ROOT/$LOG_DIR/$name.log"
}

feature_enabled() {
  local needle="$1"
  case ",$FEATURES," in
    *,"$needle",*) return 0 ;;
    *) return 1 ;;
  esac
}

install_env() {
  if [[ "$SKIP_ENV" -eq 1 ]]; then
    log "Skipping env setup."
    return
  fi
  if [[ ! -x "$(venv_python)" ]]; then
    run_cmd "$PYTHON_BIN" -m venv "$REPO_ROOT/.venv311"
  fi
  run_logged pip_upgrade "$(venv_python)" -m pip install --upgrade pip setuptools wheel
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    run_logged torch_install "$(venv_python)" -m pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
  else
    run_logged torch_install "$(venv_python)" -m pip install torch torchaudio
  fi
  run_logged deps_install "$(venv_python)" -m pip install -r "$REPO_ROOT/requirements-new-server.txt"
}

download_compact_artifacts() {
  if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
    log "Skipping HF download."
    return
  fi
  run_logged hf_download "$(venv_python)" - <<PY
from huggingface_hub import snapshot_download

repo_type = None if "$HF_REPO_TYPE" == "model" else "$HF_REPO_TYPE"
snapshot_download(
    repo_id="$HF_REPO",
    repo_type=repo_type,
    revision="$HF_REVISION",
    local_dir="$REPO_ROOT",
    allow_patterns=[
        "data/finedance/**",
        "data/finedance-g1-retargeted/**",
        "runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt",
        "runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt",
        "runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt",
        "hf_manifest.json",
    ],
)
PY
}

warmup_models() {
  if [[ "$SKIP_MODEL_WARMUP" -eq 1 || "$SKIP_FEATURES" -eq 1 ]]; then
    log "Skipping model warmup."
    return
  fi
  run_logged model_warmup "$(venv_python)" - <<'PY'
from data.audio_extraction.jukebox_features import ensure_jukebox_models
from data.audio_extraction.wav2clip_stft_beat_features import load_wav2clip_model

ensure_jukebox_models()
load_wav2clip_model()
print("Jukebox and Wav2CLIP models are available.")
PY
}

prepare_source_features() {
  local clean_arg=()
  if [[ "$CLEAN" -eq 1 ]]; then
    clean_arg=(--clean)
  fi
  if feature_enabled baseline; then
    run_logged prepare_source_baseline "$(venv_python)" data/prepare_finedance_dataset.py \
      --finedance_root "$FINEDANCE_ROOT" \
      --output_root "$SOURCE_PREPARED_ROOT" \
      --feature_type baseline \
      "${clean_arg[@]}"
  fi
  if feature_enabled jukebox; then
    export EDGE_JUKEBOX_BATCH_SIZE="$JUKEBOX_BATCH_SIZE"
    run_logged prepare_source_jukebox "$(venv_python)" data/prepare_finedance_dataset.py \
      --finedance_root "$FINEDANCE_ROOT" \
      --output_root "$SOURCE_PREPARED_ROOT" \
      --feature_type jukebox
  fi
}

prepare_g1_tree() {
  local clean_arg=()
  if [[ "$CLEAN" -eq 1 ]]; then
    clean_arg=(--clean)
  fi
  if feature_enabled baseline; then
    run_logged prepare_g1_baseline "$(venv_python)" data/prepare_finedance_g1_dataset.py \
      --g1_motion_dir "$G1_MOTION_DIR" \
      --source_prepared_root "$SOURCE_PREPARED_ROOT" \
      --output_root "$G1_OUTPUT_ROOT" \
      --feature_type baseline \
      --extract_beats \
      --g1_motion_beat_source fk \
      --g1_fk_model_path "$G1_MODEL_REL" \
      --g1_root_quat_order xyzw \
      "${clean_arg[@]}"
  fi
  if feature_enabled jukebox; then
    export EDGE_JUKEBOX_BATCH_SIZE="$JUKEBOX_BATCH_SIZE"
    run_logged prepare_g1_jukebox "$(venv_python)" data/prepare_finedance_g1_dataset.py \
      --g1_motion_dir "$G1_MOTION_DIR" \
      --source_prepared_root "$SOURCE_PREPARED_ROOT" \
      --output_root "$G1_OUTPUT_ROOT" \
      --feature_type jukebox
  fi
}

extract_wav2clip_stft_beat() {
  if ! feature_enabled wav2clip_stft_beat; then
    return
  fi
  for split in train test; do
    run_logged "extract_wav2clip_${split}" "$(venv_python)" data/audio_extraction/wav2clip_stft_beat_features.py \
      --src "$G1_OUTPUT_ROOT/$split/wavs_sliced" \
      --dest "$G1_OUTPUT_ROOT/$split/wav2clip_stft_beat_feats"
  done
}

validate_outputs() {
  if [[ "$RUN_VALIDATION" -eq 0 ]]; then
    log "Skipping validation. Pass --run-validation to enable it."
    return
  fi
  if feature_enabled baseline; then
    run_logged validate_baseline "$(venv_python)" data/validate_preprocessed_data.py \
      --data_path "$G1_OUTPUT_ROOT" \
      --processed_data_dir data/finedance_g1_baseline_dataset_backups \
      --feature_type baseline \
      --motion_format g1 \
      --feature_cache_mode memmap \
      --feature_cache_dtype float32 \
      --sample_count 64 \
      --use_beats \
      --beat_rep distance
  fi
  if feature_enabled jukebox; then
    run_logged validate_jukebox "$(venv_python)" data/validate_preprocessed_data.py \
      --data_path "$G1_OUTPUT_ROOT" \
      --processed_data_dir data/finedance_g1_fkbeats_dataset_backups_fkbeat1000 \
      --feature_type jukebox \
      --motion_format g1 \
      --feature_cache_mode memmap \
      --feature_cache_dtype float32 \
      --sample_count 64 \
      --use_beats \
      --beat_rep distance
  fi
  if feature_enabled wav2clip_stft_beat; then
    run_logged validate_wav2clip "$(venv_python)" data/validate_preprocessed_data.py \
      --data_path "$G1_OUTPUT_ROOT" \
      --processed_data_dir data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups \
      --feature_type wav2clip_stft_beat \
      --motion_format g1 \
      --feature_cache_mode memmap \
      --feature_cache_dtype float16 \
      --sample_count 64
  fi
}

main() {
  cd "$REPO_ROOT"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  mkdir -p data runs/train "$LOG_DIR"
  install_env
  download_compact_artifacts
  warmup_models
  if [[ "$SKIP_PREPARE" -eq 0 && "$SKIP_FEATURES" -eq 0 ]]; then
    prepare_source_features
    prepare_g1_tree
    extract_wav2clip_stft_beat
  elif [[ "$SKIP_PREPARE" -eq 0 ]]; then
    prepare_g1_tree
  else
    log "Skipping preparation."
  fi
  validate_outputs
  log "Done. Activate with: source .venv311/bin/activate"
  log "Logs: $LOG_DIR"
}

main
