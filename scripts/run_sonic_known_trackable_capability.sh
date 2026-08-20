#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <low|medium|high> <1|2|3>" >&2
  exit 2
fi

level=$1
repeat=$2
case "$level" in
  low) motion=walking_quip_360_R_002__A428_M ;;
  medium) motion=macarena_001__A545_M ;;
  high) motion=dance_in_da_party_001__A464_M ;;
  *) echo "Unknown level: $level" >&2; exit 2 ;;
esac
case "$repeat" in
  1|2|3) ;;
  *) echo "Repeat must be 1, 2, or 3" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
reference_root="$repo_root/eval/gt_sonic_capability/sonic_known_trackable_20260820"
run_id="sonic_native_${level}_${motion}_r0${repeat}"

echo "Prerequisite: CONTROL stable -> elastic band released -> unassisted standing -> ZMQ enabled"
echo "SONIC native capability run: $run_id"

python "$repo_root/stream_to_sonic.py" \
  --pkl "$reference_root/$motion.pkl" \
  --root_quat_order xyzw \
  --packet_mode full \
  --playback_rate 1.0 \
  --align_from_feedback_seconds 3 \
  --align_hold_seconds 1 \
  --output_dir "$repo_root/eval/gt_sonic_capability/known_trackable_runs/$run_id" \
  --record_feedback \
  --feedback_port 5557 \
  --sim_state_port 5559 \
  --reference_safety none \
  --sonic_reference_fps 50 \
  --startup_wait 2
