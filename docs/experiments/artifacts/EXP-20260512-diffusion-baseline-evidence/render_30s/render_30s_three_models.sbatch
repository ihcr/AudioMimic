#!/usr/bin/env bash
#SBATCH --job-name=g1_30s_renders
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260512-finedance-g1-30s-renders/render_30s_three_models.%j.out
#SBATCH --error=/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260512-finedance-g1-30s-renders/render_30s_three_models.%j.err

set -euo pipefail

cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion
source /projects/u6ed/yukun/EDGE/.venv311/bin/activate

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

ROOT="slurm/pipelines/20260512-finedance-g1-30s-renders"
FULL_WAV_DIR="${ROOT}/input_wavs"
DATA_PATH="data/finedance_g1_fkbeats"
G1_XML="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"

run_render() {
  local run_key="$1"
  local checkpoint="$2"
  local feature_type="$3"
  local slice_key="$4"

  local out_root="${ROOT}/${run_key}"
  mkdir -p "${out_root}/motions" "${out_root}/renders" "${out_root}/slices" "${out_root}/logs"

  echo "[$(date --iso-8601=seconds)] start ${run_key}"
  python -m eval.run_full_song_eval \
    --checkpoint "${checkpoint}" \
    --data_path "${DATA_PATH}" \
    --full_wav_dir "${FULL_WAV_DIR}" \
    --feature_type "${feature_type}" \
    --motion_format g1 \
    --use_beats \
    --beat_rep distance \
    --beat_source audio \
    --motion_save_dir "${out_root}/motions" \
    --render_dir "${out_root}/renders" \
    --slice_work_dir "${ROOT}/slice_cache/${slice_key}" \
    --metrics_path "${out_root}/metrics.json" \
    --skip_metrics \
    --render \
    --g1_render_backend mujoco \
    --g1_root_quat_order xyzw \
    --g1_fk_model_path "${G1_XML}" \
    --g1_render_width 960 \
    --g1_render_height 720
  echo "[$(date --iso-8601=seconds)] done ${run_key}"
}

run_render \
  "jukebox_fkbeat_no_lbeat" \
  "runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt" \
  "jukebox" \
  "jukebox"

run_render \
  "jukebox_lbeat_robotloss" \
  "runs/train/finedance_g1_jukebox_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt" \
  "jukebox" \
  "jukebox"

run_render \
  "librosa35_lbeat_robotloss" \
  "runs/train/finedance_g1_librosa35_fullctx_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt" \
  "baseline" \
  "librosa35"
