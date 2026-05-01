#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python download_dataset.py
