#!/usr/bin/env bash
#SBATCH --job-name=fd_more7_vids
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260512-finedance-g1-30s-renders/render_more7_finedance_videos.%j.out
#SBATCH --error=/projects/u6ed/yukun/EDGE/.worktrees/diffusion/slurm/pipelines/20260512-finedance-g1-30s-renders/render_more7_finedance_videos.%j.err

set -euo pipefail

cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion
source /projects/u6ed/yukun/EDGE/.venv311/bin/activate

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TERM=xterm-256color
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export WANDB_DISABLED=true

ROOT="slurm/pipelines/20260512-finedance-g1-30s-renders"
MORE_WAV_DIR="${ROOT}/input_wavs_more7"
DATA_PATH="data/finedance_g1_fkbeats"
G1_XML="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"
VIDEO_ROOT="videos/finedance_g1_30s"

python - <<'PY'
import json
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation as R

from eval.g1_visualization import render_g1_motion


SEQUENCE_IDS = ["036", "065", "092", "098", "109", "137", "144"]
FPS = 30.0
SECONDS = 30.0
FRAMES = int(FPS * SECONDS)
ROOT = Path("slurm/pipelines/20260512-finedance-g1-30s-renders")
WAV_DIR = ROOT / "input_wavs_more7"
VIDEO_ROOT = Path("videos/finedance_g1_30s")
SOURCE_DIR = VIDEO_ROOT / "source"
SOURCE_MOTION_ROOT = Path("/projects/u6ed/yukun/EDGE/data/finedance-g1-retargeted")
SOURCE_AUDIO_ROOT = Path("/projects/u6ed/yukun/EDGE/data/finedance/music_wav")
MODEL_PATH = "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"


def yaw_deg(quat_xyzw):
    return float(R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)[2])


def crop_30s(motion, source_path):
    fps = float(motion.get("fps", FPS))
    frames = int(round(fps * SECONDS))
    if len(motion["root_pos"]) < frames:
        raise ValueError(f"{source_path} has only {len(motion['root_pos'])} frames, need {frames}")
    cropped = dict(motion)
    cropped["fps"] = fps
    for key in ("root_pos", "root_rot", "dof_pos"):
        cropped[key] = np.asarray(motion[key][:frames]).copy()
    return cropped


def rotate_motion_yaw(motion, degrees):
    rotated = dict(motion)
    yaw = R.from_euler("z", degrees, degrees=True)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64).copy()
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64).copy()

    pivot = root_pos[0, :2].copy()
    xy = root_pos[:, :2] - pivot
    theta = np.deg2rad(degrees)
    rot2 = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    root_pos[:, :2] = xy @ rot2.T + pivot
    root_rot = (yaw * R.from_quat(root_rot)).as_quat()

    rotated["root_pos"] = root_pos.astype(np.float32)
    rotated["root_rot"] = root_rot.astype(np.float32)
    rotated["dof_pos"] = np.asarray(motion["dof_pos"], dtype=np.float32).copy()
    return rotated


def write_30s_wav(source, dest):
    audio, sr = sf.read(str(source), always_2d=True)
    samples = int(round(sr * SECONDS))
    if audio.shape[0] < samples:
        raise ValueError(f"{source} has only {audio.shape[0] / sr:.2f}s, need {SECONDS:.2f}s")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), audio[:samples], sr)


WAV_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
source_records = []
for index, sequence_id in enumerate(SEQUENCE_IDS):
    motion_path = SOURCE_MOTION_ROOT / f"{sequence_id}.pkl"
    source_audio = SOURCE_AUDIO_ROOT / f"{sequence_id}.wav"
    input_wav = WAV_DIR / f"{sequence_id}_30s.wav"
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    if not source_audio.is_file():
        raise FileNotFoundError(source_audio)

    write_30s_wav(source_audio, input_wav)
    with motion_path.open("rb") as handle:
        motion = crop_30s(pickle.load(handle), motion_path)

    before = yaw_deg(motion["root_rot"][0])
    correction = 180.0 if before > 0.0 else 0.0
    if correction:
        motion = rotate_motion_yaw(motion, correction)
    after = yaw_deg(motion["root_rot"][0])

    output = render_g1_motion(
        motion,
        out=SOURCE_DIR,
        epoch=0,
        num=index,
        name=str(input_wav),
        sound=True,
        stitch=False,
        model_path=MODEL_PATH,
        root_quat_order="xyzw",
        render_backend="mujoco",
        width=960,
        height=720,
        mujoco_gl="egl",
        output_name=f"{sequence_id}_30s",
    )
    source_video = SOURCE_DIR / f"fullsong_0_{sequence_id}_30s_g1.mp4"
    Path(output).replace(source_video)
    source_records.append(
        {
            "sequence_id": sequence_id,
            "source_motion": str(motion_path),
            "source_audio": str(source_audio),
            "input_wav": str(input_wav),
            "source_video": str(source_video),
            "yaw_before_deg": before,
            "yaw_after_deg": after,
            "applied_yaw_correction_deg": correction,
        }
    )

(ROOT / "render_more7_source_records.json").write_text(
    json.dumps(source_records, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(source_records, indent=2), flush=True)
PY

run_render() {
  local run_key="$1"
  local checkpoint="$2"
  local feature_type="$3"

  local out_root="${ROOT}/${run_key}_more7"
  mkdir -p "${out_root}/motions" "${out_root}/renders" "${out_root}/slices" "${out_root}/logs"

  echo "[$(date --iso-8601=seconds)] render ${run_key} more7"
  python -m eval.run_full_song_eval \
    --checkpoint "${checkpoint}" \
    --data_path "${DATA_PATH}" \
    --full_wav_dir "${MORE_WAV_DIR}" \
    --feature_type "${feature_type}" \
    --motion_format g1 \
    --use_beats \
    --beat_rep distance \
    --beat_source audio \
    --motion_save_dir "${out_root}/motions" \
    --render_dir "${out_root}/renders" \
    --slice_work_dir "${ROOT}/slice_cache/more7_${run_key}" \
    --metrics_path "${out_root}/metrics.json" \
    --skip_metrics \
    --render \
    --g1_render_backend mujoco \
    --g1_root_quat_order xyzw \
    --g1_fk_model_path "${G1_XML}" \
    --g1_render_width 960 \
    --g1_render_height 720
}

run_render \
  "jukebox_fkbeat_no_lbeat" \
  "runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt" \
  "jukebox"

run_render \
  "jukebox_lbeat_robotloss" \
  "runs/train/finedance_g1_jukebox_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt" \
  "jukebox"

run_render \
  "librosa35_fullctx_motiondist_cond" \
  "runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt" \
  "baseline"

run_render \
  "librosa35_lbeat_robotloss" \
  "runs/train/finedance_g1_librosa35_fullctx_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt" \
  "baseline"

python - <<'PY'
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg


SEQUENCE_IDS = ["036", "065", "092", "098", "109", "137", "144"]
ROOT = Path("slurm/pipelines/20260512-finedance-g1-30s-renders")
VIDEO_ROOT = Path("videos/finedance_g1_30s")
RUNS = {
    "jukebox_fkbeat_no_lbeat": {
        "checkpoint": "runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt",
        "feature_type": "jukebox",
    },
    "jukebox_lbeat_robotloss": {
        "checkpoint": "runs/train/finedance_g1_jukebox_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt",
        "feature_type": "jukebox",
    },
    "librosa35_fullctx_motiondist_cond": {
        "checkpoint": "runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/weights/train-2000.pt",
        "feature_type": "baseline",
    },
    "librosa35_lbeat_robotloss": {
        "checkpoint": "runs/train/finedance_g1_librosa35_fullctx_lbeat_robotloss_b020_kin040_cap1_scratch_1000/weights/train-1000.pt",
        "feature_type": "baseline",
    },
}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def write_side_by_side(source, generated, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-i",
        str(generated),
        "-filter_complex",
        "[0:v]scale=960:720,setsar=1[left];"
        "[1:v]scale=960:720,setsar=1[right];"
        "[left][right]hstack=inputs=2:shortest=1[v]",
        "-map",
        "[v]",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


manifest_path = VIDEO_ROOT / "manifest.json"
manifest = []
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest = [
    row
    for row in manifest
    if row.get("sequence_id") not in set(SEQUENCE_IDS)
]

new_records = []
for run_key, config in RUNS.items():
    generated_dir = VIDEO_ROOT / "generated" / run_key
    side_dir = VIDEO_ROOT / "side_by_side" / run_key
    generated_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    for sequence_id in SEQUENCE_IDS:
        source = VIDEO_ROOT / "source" / f"fullsong_0_{sequence_id}_30s_g1.mp4"
        generated = ROOT / f"{run_key}_more7" / "renders" / f"fullsong_0_{sequence_id}_30s_g1.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        if not generated.is_file():
            raise FileNotFoundError(generated)

        generated_copy = generated_dir / generated.name
        shutil.copy2(generated, generated_copy)
        side_output = side_dir / f"{sequence_id}_source_vs_{run_key}.mp4"
        write_side_by_side(source, generated, side_output)
        new_records.append(
            {
                "sequence_id": sequence_id,
                "run_key": run_key,
                "source_video": str(source),
                "generated_video": str(generated_copy),
                "side_by_side_video": str(side_output),
                "side_by_side_layout": "left=retargeted source data, right=generated inference render",
                "side_by_side_video_policy": "trimmed_to_shortest_input",
                "duration_sec": 30.0,
                "resolution": "1920x720",
                "checkpoint": config["checkpoint"],
                "feature_type": config["feature_type"],
                "beat_rep": "distance",
                "render_set": "more7",
            }
        )

manifest.extend(new_records)
manifest.sort(key=lambda row: (row.get("run_key", ""), row.get("sequence_id", "")))
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Added {len(new_records)} generated/side-by-side videos for {len(SEQUENCE_IDS)} new FineDance sequences", flush=True)
PY

find "${VIDEO_ROOT}" -type f -name '*.mp4' -printf '%p %s bytes\n' | sort
