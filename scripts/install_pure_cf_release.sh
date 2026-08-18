#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_REPO="${1:-$HOME/Musics2Dance-prior-dev}"
RELEASE_NAME="pure-cf-d16-zero-200k-v1"
MODEL_DIR="$REPO_ROOT/models/releases/$RELEASE_NAME"
SEED_DIR="$REPO_ROOT/models/seeds"
UPSTREAM_RELEASE="https://github.com/lbtwyk/Musics2Dance/releases/download/$RELEASE_NAME"
UPSTREAM_RAW="https://raw.githubusercontent.com/lbtwyk/Musics2Dance/prior-dev"

mkdir -p "$MODEL_DIR" "$SEED_DIR"

release_files=(
  README.md
  SHA256SUMS
  pipeline_manifest.json
  codec-d16-seed1234-train300.pt
  pure-cf-d16-zero-seed1234-update200000-inference.pt
  q0-d16-seed1234-update100000.pt
)

if [[ -d "$SOURCE_REPO/models/releases/$RELEASE_NAME" ]]; then
  echo "Installing model release from $SOURCE_REPO"
  cp --reflink=auto \
    "$SOURCE_REPO/models/releases/$RELEASE_NAME/"* \
    "$MODEL_DIR/"
else
  echo "Local Musics2Dance checkout not found; downloading release assets"
  for file in "${release_files[@]}"; do
    if [[ "$file" == *.pt ]]; then
      url="$UPSTREAM_RELEASE/$file"
    else
      url="$UPSTREAM_RAW/models/releases/$RELEASE_NAME/$file"
    fi
    wget -c -O "$MODEL_DIR/$file" "$url"
  done
fi

seed_name="m2_train1234_sample1234_u100000_best_song098.pkl"
seed_relative="onlinegeneratedmotion/m2_predicted_fms/$seed_name"
if [[ -f "$SOURCE_REPO/$seed_relative" ]]; then
  cp --reflink=auto "$SOURCE_REPO/$seed_relative" "$SEED_DIR/$seed_name"
else
  wget -c -O "$SEED_DIR/$seed_name" "$UPSTREAM_RAW/$seed_relative"
fi

(
  cd "$MODEL_DIR"
  sha256sum -c SHA256SUMS
)

echo "Model release installed at $MODEL_DIR"
echo "K64 seed installed at $SEED_DIR/$seed_name"
