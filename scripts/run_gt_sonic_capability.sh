#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_gt_sonic_capability.sh <low|medium|high> <1|2|3>" >&2
  exit 2
fi

LEVEL="$1"
REPEAT="$2"
case "$LEVEL" in
  low|medium|high) ;;
  *) echo "Invalid level: $LEVEL" >&2; exit 2 ;;
esac
case "$REPEAT" in
  1|2|3) ;;
  *) echo "Invalid repeat: $REPEAT" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_ROOT/eval/gt_sonic_capability/selection_20260819/selection_manifest.json"

readarray -t RUN_FIELDS < <(
  python -c '
import json, sys
manifest, level, repeat = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = json.load(open(manifest, encoding="utf-8"))
run = next(item for item in payload["planned_runs"] if item["level"] == level and item["repeat"] == repeat)
print(run["run_id"])
print(run["motion_path"])
' "$MANIFEST" "$LEVEL" "$REPEAT"
)

RUN_ID="${RUN_FIELDS[0]}"
GT_PKL="${RUN_FIELDS[1]}"
OUTPUT_DIR="$REPO_ROOT/eval/gt_sonic_capability/runs/$RUN_ID"

echo "GT capability run: $RUN_ID"
echo "Reference: $GT_PKL"
echo "Output: $OUTPUT_DIR"

cd "$REPO_ROOT"
python stream_to_sonic.py \
  --pkl "$GT_PKL" \
  --root_quat_order xyzw \
  --packet_mode full \
  --playback_rate 1.0 \
  --align_from_feedback_seconds 3 \
  --align_hold_seconds 1 \
  --output_dir "$OUTPUT_DIR" \
  --record_feedback \
  --feedback_port 5557 \
  --sim_state_port 5559 \
  --reference_safety none \
  --sonic_reference_fps 50 \
  --startup_wait 2
