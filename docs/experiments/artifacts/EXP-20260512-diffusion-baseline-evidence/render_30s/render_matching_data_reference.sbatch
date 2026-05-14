#!/usr/bin/env bash
#SBATCH --job-name=g1_data_ref
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260512-finedance-g1-30s-renders/render_matching_data_reference.%j.out
#SBATCH --error=/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260512-finedance-g1-30s-renders/render_matching_data_reference.%j.err

set -euo pipefail

cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion
source /projects/u6ed/yukun/EDGE/.venv311/bin/activate

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

python - <<'PY'
import json
import pickle
from pathlib import Path

from eval.g1_visualization import render_g1_motion


ROOT = Path("slurm/pipelines/20260512-finedance-g1-30s-renders")
RAW_ROOT = Path("/projects/u6ed/yukun/EDGE/data/finedance-g1-retargeted")
OUT_ROOT = ROOT / "data_reference"
RENDER_DIR = OUT_ROOT / "renders"
MOTION_DIR = OUT_ROOT / "motions"
G1_XML = "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"
SEQUENCE_IDS = ["012", "063", "143"]
SECONDS = 30.0


def load_cropped_motion(sequence_id):
    source = RAW_ROOT / f"{sequence_id}.pkl"
    with source.open("rb") as handle:
        motion = pickle.load(handle)
    fps = float(motion.get("fps", 30.0))
    frames = int(round(fps * SECONDS))
    if len(motion["root_pos"]) < frames:
        raise ValueError(f"{source} has {len(motion['root_pos'])} frames; need {frames}")
    cropped = dict(motion)
    cropped["fps"] = fps
    for key in ("root_pos", "root_rot", "dof_pos"):
        cropped[key] = motion[key][:frames].copy()
    return source, cropped, frames


RENDER_DIR.mkdir(parents=True, exist_ok=True)
MOTION_DIR.mkdir(parents=True, exist_ok=True)

summary = []
for sequence_id in SEQUENCE_IDS:
    source_motion, motion, frames = load_cropped_motion(sequence_id)
    audio = ROOT / "input_wavs" / f"{sequence_id}_30s.wav"
    cropped_motion_path = MOTION_DIR / f"{sequence_id}_30s.pkl"
    with cropped_motion_path.open("wb") as handle:
        pickle.dump(motion, handle)

    output = render_g1_motion(
        motion,
        out=RENDER_DIR,
        epoch="fullsong",
        num=0,
        name=str(audio),
        sound=True,
        stitch=False,
        model_path=G1_XML,
        root_quat_order="xyzw",
        render_backend="mujoco",
        width=960,
        height=720,
        mujoco_gl="egl",
        output_name=f"{sequence_id}_30s",
    )
    metadata = {
        "dataset": "finedance-retargeted-data-reference",
        "sequence_id": sequence_id,
        "source_motion": str(source_motion),
        "cropped_motion": str(cropped_motion_path),
        "source_audio": str(audio),
        "output_video": output,
        "fps": float(motion["fps"]),
        "frames": frames,
        "duration_sec": frames / float(motion["fps"]),
        "root_quat_order": "xyzw",
        "render_backend": "mujoco",
        "matches_render_set": str(ROOT),
    }
    metadata_path = Path(output).with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    summary.append(metadata)
    print(json.dumps(metadata, indent=2), flush=True)

(OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Wrote {len(summary)} matching data-reference renders to {RENDER_DIR}", flush=True)
PY
