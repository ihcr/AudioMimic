# G1 Forward Kinematics Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trusted Unitree G1 forward kinematics to the G1 evaluation path so robot beat and foot diagnostics use real robot geometry.

**Architecture:** Keep the change evaluation-first. Add a small Unitree model fetcher, a focused FK adapter around MuJoCo, and new FK metrics wired into the existing G1 evaluator without changing training data yet.

**Tech Stack:** Python 3.11, `mujoco`, NumPy, existing `unittest` tests, official Unitree G1 MJCF from `unitreerobotics/unitree_ros`.

## Implementation Status

Status on 2026-05-04: implemented through the optional FK-beat data rebuild and
submitted for training.

Completed outcomes:

- Unitree G1 model fetcher and local model files are present.
- `requirements-g1-fk.txt` records the MuJoCo FK dependency.
- `eval/g1_kinematics.py` provides the G1 FK adapter.
- `eval/g1_metrics.py` and `eval/run_g1_dataset_eval.py` support opt-in FK metrics.
- `submit_training_pipeline.py` can launch FK-enabled G1 eval.
- `data/audio_extraction/beat_features.py` and `data/prepare_g1_aist_dataset.py`
  support `--g1_motion_beat_source fk`.
- `data/g1_aistpp_full_fkbeats` was built and validated with `17,733` train clips
  and `186` test clips.
- `slurm/pipelines/20260504-g1-aist-beatdistance-fkbeats` was submitted with
  training job `4446773` and evaluation job `4446774`.

---

## File Structure

- Create `scripts/fetch_unitree_g1_description.py`
  - Downloads the pinned official G1 MJCF, README, LICENSE, and referenced meshes.
  - Writes them under `third_party/unitree_g1_description`.
  - Does not run during normal evaluation.

- Create `requirements-g1-fk.txt`
  - Adds the MuJoCo dependency for FK evaluation without disturbing the older
    fairmotion eval stack.

- Create `eval/g1_kinematics.py`
  - Owns model loading, joint-order validation, root quaternion handling, FK, keypoint extraction, and foot geom extraction.
  - No dependency on training code.

- Modify `eval/g1_metrics.py`
  - Adds optional FK metric computation.
  - Keeps existing G1 metrics and names for continuity.
  - Records FK metadata in the audit/report outputs.

- Modify `eval/run_g1_dataset_eval.py`
  - Adds CLI flags for FK model path, enabling FK metrics, and root quaternion order.

- Create `tests/test_g1_kinematics.py`
  - Unit tests for qpos construction, joint order validation, keypoint extraction, and missing dependency/model errors.

- Modify `tests/test_g1_eval_metrics.py`
  - Adds tests that FK metrics are included only when enabled and all FK values are finite.

- Optional later file: `data/audio_extraction/beat_features.py`
  - Only after eval validation, add FK-based G1 motion-beat extraction for future rebuilt training data.

## Task 1: Add Unitree G1 Model Fetcher

**Files:**
- Create: `scripts/fetch_unitree_g1_description.py`
- Test: `tests/test_g1_kinematics.py`

- [ ] **Step 1: Write a test for URL discovery without network**

```python
def test_extracts_mesh_urls_from_mjcf():
    from scripts.fetch_unitree_g1_description import referenced_mesh_files

    xml = '<mujoco><asset><mesh file="pelvis.STL"/><mesh file="leg.STL"/></asset></mujoco>'
    assert referenced_mesh_files(xml) == ["pelvis.STL", "leg.STL"]
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
python -m unittest tests.test_g1_kinematics.G1ModelFetchTests.test_extracts_mesh_urls_from_mjcf
```

Expected: fails because the fetcher module does not exist yet.

- [ ] **Step 3: Implement the fetcher helpers**

Create `scripts/fetch_unitree_g1_description.py` with:

```python
from pathlib import Path
from urllib.request import urlopen
import re

UPSTREAM_BASE = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master"
G1_ROOT = "robots/g1_description"
MODEL_NAME = "g1_29dof_rev_1_0.xml"


def referenced_mesh_files(xml_text):
    return sorted(set(re.findall(r'<mesh[^>]+file="([^"]+)"', xml_text)))


def download_text(url):
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def download_bytes(url):
    with urlopen(url, timeout=60) as response:
        return response.read()
```

Then add `fetch_unitree_g1_description(output_root)` to write the model files,
mesh files, and a `SOURCE.txt` with the upstream URL.

- [ ] **Step 4: Run the focused test**

Run:

```bash
python -m unittest tests.test_g1_kinematics.G1ModelFetchTests
```

Expected: passes.

- [ ] **Step 5: Fetch the model once**

Run:

```bash
python scripts/fetch_unitree_g1_description.py --output_root third_party/unitree_g1_description
```

Expected:

```text
third_party/unitree_g1_description/g1_29dof_rev_1_0.xml
third_party/unitree_g1_description/meshes/*.STL
third_party/unitree_g1_description/LICENSE
third_party/unitree_g1_description/SOURCE.txt
```

If the download takes longer than one minute, rerun it through `srun`.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_unitree_g1_description.py tests/test_g1_kinematics.py third_party/unitree_g1_description
git commit -m "feat: add Unitree G1 model fetcher"
```

## Task 2: Add MuJoCo Dependency

**Files:**
- Create: `requirements-g1-fk.txt`
- Test: `tests/test_g1_kinematics.py`

- [ ] **Step 1: Write a missing-dependency test**

Add a test that imports `eval.g1_kinematics` with `mujoco` unavailable and
expects a clear error only when model loading is attempted.

- [ ] **Step 2: Add FK requirements**

Create `requirements-g1-fk.txt`:

```text
mujoco>=3.2,<4
PyOpenGL==3.1.0
```

The separate file avoids making pip re-resolve `fairmotion==0.0.4`, which is
already installed in the repo environment but declares old transitive
dependencies that are not compatible with this Python stack.

- [ ] **Step 3: Install in the repo environment on a compute node if needed**

Run:

```bash
srun --partition=workq --time=00:10:00 --ntasks=1 --cpus-per-task=2 --mem=8G bash -lc 'cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion && source ../../.venv311/bin/activate && pip install -r requirements-g1-fk.txt'
```

Expected: `mujoco` installs successfully.

- [ ] **Step 4: Verify import**

Run:

```bash
source ../../.venv311/bin/activate
python - <<'PY'
import mujoco
print(mujoco.__version__)
PY
```

Expected: prints a MuJoCo version.

- [ ] **Step 5: Commit**

```bash
git add requirements-g1-fk.txt tests/test_g1_kinematics.py
git commit -m "chore: add MuJoCo eval dependency"
```

## Task 3: Implement G1 FK Adapter

**Files:**
- Create: `eval/g1_kinematics.py`
- Test: `tests/test_g1_kinematics.py`

- [ ] **Step 1: Write tests for joint order**

Test that the official model joint order, excluding `floating_base_joint`,
matches:

```python
EXPECTED_G1_29DOF_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
```

- [ ] **Step 2: Write tests for qpos construction**

Use a fake motion with one frame:

```python
root_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
root_rot = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
dof_pos = np.arange(29, dtype=np.float32)[None]
```

Expected MuJoCo qpos starts with:

```text
1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0
```

and then the 29 joint values.

- [ ] **Step 3: Run tests and confirm they fail**

Run:

```bash
python -m unittest tests.test_g1_kinematics.G1KinematicsTests
```

Expected: fails because `eval.g1_kinematics` does not exist yet.

- [ ] **Step 4: Implement `eval/g1_kinematics.py`**

Required public functions:

```python
def load_g1_mujoco_model(model_path):
    ...

def validate_g1_joint_order(model):
    ...

def build_g1_qpos(root_pos, root_rot, dof_pos, root_quat_order="wxyz"):
    ...

def forward_g1_kinematics(motion, model_path, root_quat_order="wxyz"):
    ...
```

Implementation notes:

- Import `mujoco` inside `load_g1_mujoco_model`, not at module import time.
- Normalize root quaternions.
- Support `root_quat_order="wxyz"` and `"xyzw"`.
- Fail on unsupported order.
- Return body positions and selected keypoints.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m unittest tests.test_g1_kinematics.G1KinematicsTests
```

Expected: passes.

- [ ] **Step 6: Commit**

```bash
git add eval/g1_kinematics.py tests/test_g1_kinematics.py
git commit -m "feat: add G1 forward kinematics adapter"
```

## Task 4: Add FK Beat and Timing Metrics

**Files:**
- Modify: `eval/g1_metrics.py`
- Test: `tests/test_g1_eval_metrics.py`

- [ ] **Step 1: Write tests for beat timing**

Add a pure function test:

```python
def test_beat_timing_reports_precision_recall_f1_and_offsets():
    generated = np.array([10, 31, 80])
    target = np.array([12, 30, 60])
    report = compute_beat_timing_report(generated, target, tolerance=2)
    assert report["precision"] == 2 / 3
    assert report["recall"] == 2 / 3
    assert report["f1"] == 2 / 3
```

- [ ] **Step 2: Write tests for FK metric inclusion**

Patch `forward_g1_kinematics` to return small fake keypoints and assert
`G1FKBAS`, `G1BeatF1`, and foot diagnostics appear when FK is enabled.

- [ ] **Step 3: Run tests and confirm they fail**

Run:

```bash
python -m unittest tests.test_g1_eval_metrics.G1MetricTests
```

Expected: fails because the new metrics do not exist.

- [ ] **Step 4: Implement metric helpers**

Add functions in `eval/g1_metrics.py`:

```python
def compute_fk_keypoint_speed_curve(keypoints, fps):
    ...

def detect_fk_motion_beat_frames(keypoints, fps, sigma=5):
    ...

def compute_beat_timing_report(generated_beats, target_beats, tolerance=2):
    ...

def compute_fk_foot_diagnostics(fk_result, fps):
    ...
```

Use existing `compute_bas_score` and `greedy_match_count` where possible.

- [ ] **Step 5: Wire FK metrics into `run_g1_motion_evaluation`**

Add optional parameters:

```python
fk_model_path=None
enable_fk_metrics=False
root_quat_order="wxyz"
```

When enabled:

- run FK for each motion
- compute FK beat metrics
- compute foot diagnostics
- record finite aggregate values
- include FK model metadata in `motion_audit.json`

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m unittest tests.test_g1_eval_metrics.G1MetricTests tests.test_g1_kinematics.G1KinematicsTests
```

Expected: passes.

- [ ] **Step 7: Commit**

```bash
git add eval/g1_metrics.py tests/test_g1_eval_metrics.py
git commit -m "feat: add FK-based G1 evaluation metrics"
```

## Task 5: Add G1 Eval CLI Flags

**Files:**
- Modify: `eval/run_g1_dataset_eval.py`
- Modify: `submit_training_pipeline.py`
- Test: `tests/test_submit_training_pipeline.py`

- [ ] **Step 1: Write CLI tests**

Add tests that:

- `eval/run_g1_dataset_eval.py` accepts `--enable_fk_metrics`
- `--g1_fk_model_path` defaults to `third_party/unitree_g1_description/g1_29dof_rev_1_0.xml`
- `submit_training_pipeline.py` forwards FK flags when requested

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
python -m unittest tests.test_submit_training_pipeline
```

Expected: fails because the flags are not wired.

- [ ] **Step 3: Add CLI flags**

In `eval/run_g1_dataset_eval.py`:

```python
parser.add_argument("--enable_fk_metrics", action="store_true")
parser.add_argument("--g1_fk_model_path", default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml")
parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="wxyz")
```

Pass those values into `run_g1_motion_evaluation`.

- [ ] **Step 4: Add Slurm pipeline support**

In `submit_training_pipeline.py`, add opt-in flags:

```text
--enable_g1_fk_metrics
--g1_fk_model_path
--g1_root_quat_order
```

Only append FK flags for G1 eval commands.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m unittest tests.test_submit_training_pipeline tests.test_g1_eval_metrics
```

Expected: passes.

- [ ] **Step 6: Commit**

```bash
git add eval/run_g1_dataset_eval.py submit_training_pipeline.py tests/test_submit_training_pipeline.py
git commit -m "feat: wire FK metrics into G1 eval CLI"
```

## Task 6: Smoke Test On Saved G1 Motions

**Files:**
- Runtime output only under `slurm/` or a scratch eval folder.

- [ ] **Step 1: Run a small local import check**

Run:

```bash
source ../../.venv311/bin/activate
python -m py_compile eval/g1_kinematics.py eval/g1_metrics.py eval/run_g1_dataset_eval.py
```

Expected: exits with code 0.

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
source ../../.venv311/bin/activate
python -m unittest tests.test_g1_kinematics tests.test_g1_eval_metrics tests.test_submit_training_pipeline
```

Expected: all tests pass.

- [ ] **Step 3: Run a 4-clip FK smoke eval through `srun`**

Use the shortest command that evaluates a tiny copied subset of saved motions.
If a helper flag does not exist yet, create a small scratch folder with 4 motion
files and use `run_g1_motion_evaluation` directly.

Run:

```bash
srun --partition=workq --time=00:20:00 --ntasks=1 --cpus-per-task=4 --mem=16G bash -lc 'cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion && source ../../.venv311/bin/activate && python -m eval.run_g1_dataset_eval --checkpoint runs/train/g1_aist_beatdistance_featurecache/weights/train-2000.pt --data_path data/g1_aistpp_full --processed_data_dir data/g1_aistpp_full_dataset_backups --motion_save_dir slurm/fk_smoke/motions --metrics_path slurm/fk_smoke/metrics.json --g1_table_path slurm/fk_smoke/g1_table.json --motion_audit_path slurm/fk_smoke/motion_audit.json --paper_report_path slurm/fk_smoke/paper_report.md --render_dir slurm/fk_smoke/renders --use_beats --beat_rep distance --enable_fk_metrics'
```

Expected:

- `metrics.json` exists
- `G1FKBAS` is finite
- `G1BeatF1` is finite
- foot diagnostics are finite
- audit records FK model path and quaternion order

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-04-g1-forward-kinematics-eval-design.md docs/superpowers/plans/2026-05-04-g1-forward-kinematics-eval.md
git commit -m "docs: plan G1 forward kinematics evaluation"
```

## Task 7: Optional FK Beat Metadata Rebuild

Do this only after Task 6 validates the FK evaluator on real saved motions.

**Files:**
- Modify: `data/audio_extraction/beat_features.py`
- Modify: `data/prepare_g1_aist_dataset.py`
- Test: `tests/test_g1_beat_features.py`

- [ ] **Step 1: Add a new beat extraction mode**

Add an explicit option such as:

```text
--g1_motion_beat_source fk
```

Do not change the default until old-vs-new eval is compared.

- [ ] **Step 2: Write a new prepared tree**

Use a new output path, for example:

```text
data/g1_aistpp_full_fkbeats
```

Do not overwrite:

```text
data/g1_aistpp_full
```

- [ ] **Step 3: Validate the rebuilt tree through `srun`**

Run the full data validator on a compute node because it reads many files:

```bash
srun --partition=workq --time=00:30:00 --ntasks=1 --cpus-per-task=4 --mem=24G bash -lc 'cd /projects/u6ed/yukun/EDGE/.worktrees/diffusion && source ../../.venv311/bin/activate && python data/validate_preprocessed_data.py --data_path data/g1_aistpp_full_fkbeats --motion_format g1 --feature_type jukebox --use_beats --beat_rep distance'
```

Expected: validation passes and reports matching motion, wav, feature, and beat
counts.

- [ ] **Step 4: Retrain only after accepting new labels**

Submit a new G1 BeatDistance run with a fresh processed-cache path. Treat this
as a new experiment, not a continuation of the current checkpoint.

## Final Verification Checklist

- [ ] Official Unitree model files are present locally.
- [ ] `mujoco` imports from `.venv311`.
- [ ] G1 joint order is validated.
- [ ] FK metrics are opt-in and do not break old G1 evaluation.
- [ ] New metrics are finite on a real smoke eval.
- [ ] No SMPL-only metric names are introduced into G1 FK reports.
- [ ] Long checks run with `srun` or `sbatch`.
