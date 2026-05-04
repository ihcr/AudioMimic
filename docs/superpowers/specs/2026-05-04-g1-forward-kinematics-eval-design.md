# G1 Forward Kinematics Evaluation Design

## Purpose

The current G1 evaluator computes motion beats from root movement and joint-angle
speed. That is useful as a temporary robot-native proxy, but it does not use the
robot's real body geometry. The next evaluation upgrade should use trusted G1
forward kinematics so beat detection and physical diagnostics are based on
actual robot link positions.

The first implementation should be evaluation-only. It should not change
training labels or retrain models until the new evaluator is validated against
the real G1 test motions.

## 2026-05-04 Implementation Status

The evaluation-first FK path has been implemented, and the optional FK-beat
training-data follow-up has started as a new experiment rather than a
continuation of older G1 checkpoints.

Current concrete state:

- local Unitree G1 MJCF: `third_party/unitree_g1_description/g1_29dof_rev_1_0.xml`
- FK-beat prepared tree: `data/g1_aistpp_full_fkbeats`
- fresh processed cache: `data/g1_aistpp_full_fkbeats_dataset_backups`
- train clips: `17,733`
- test clips: `186`
- training pipeline: `slurm/pipelines/20260504-g1-aist-beatdistance-fkbeats`
- training job: `4446773`
- chained evaluation job: `4446774`

## Trusted Robot Model Source

Use the official Unitree G1 29-DoF robot description first:

- Source: `unitreerobotics/unitree_ros`
- Model folder:
  `robots/g1_description`
- Model file:
  `g1_29dof_rev_1_0.xml`
- License: BSD-3-Clause

This model matches our current saved G1 motion dimensionality:

- 6 leg DoF per side
- 3 waist DoF
- 7 arm DoF per side
- 29 actuated joints total

The official MuJoCo joint order observed in `g1_29dof_rev_1_0.xml` is:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

Before metrics use this model, the implementation must verify that our
`dof_pos` arrays follow this order. If the order is not proven, FK metrics must
fail loudly instead of silently reporting misleading numbers.

## Dependency Choice

Use the Python `mujoco` package for the first version.

Reasons:

- Unitree publishes MJCF files directly.
- GMR uses MuJoCo for G1 robot-motion visualization.
- MuJoCo gives us both pure kinematics now and a direct path to simulator-based
  checks later.
- The official MJCF already contains body, joint, geom, and actuator structure.

Do not use Pinocchio in the first pass unless MuJoCo becomes blocked. Pinocchio
is still a good future option for fast batched FK, but it adds a second robot
model parsing path and is not needed for the first evaluation upgrade.

## Source Pull Policy

Add a small fetch script that downloads the pinned official Unitree model files
into:

```text
third_party/unitree_g1_description
```

The fetch script should download:

- `README.md`
- repository `LICENSE`
- `g1_29dof_rev_1_0.xml`
- every mesh referenced by that MJCF file

The script should record the upstream repository URL and commit or branch used.
The evaluator should never download network files at runtime. Runtime evaluation
should only read the local third-party copy.

## Data Contract

The FK module consumes the existing G1 payload format:

```text
root_pos: [T, 3]
root_rot: [T, 4]
dof_pos: [T, 29]
fps: scalar, normally 30
```

It outputs a dictionary of robot body and geom positions:

```text
bodies: [T, B, 3]
body_names: [B]
geoms: [T, G, 3]
geom_names: [G]
keypoints: [T, K, 3]
keypoint_names: [K]
```

The first required keypoints are:

- pelvis
- torso
- left ankle roll link
- right ankle roll link
- left wrist yaw link
- right wrist yaw link
- lowest left-foot geom point
- lowest right-foot geom point

If the official MJCF names differ from these expectations, the implementation
should fail during model validation and report the missing name.

## Root Pose Convention

MuJoCo free-joint `qpos` expects:

```text
x, y, z, qw, qx, qy, qz
```

Our G1 payload stores `root_pos` and `root_rot`, but the quaternion order is not
encoded in the file. The implementation must add a convention check before using
FK metrics.

The accepted first-pass policy is:

1. Assume `root_rot` is `wxyz` only after validating it on real test clips.
2. Add a command-line option to override the order if needed.
3. Record the selected convention in `metrics.json` and `motion_audit.json`.
4. Fail loudly if the convention is unknown.

## New Metrics

Add these metrics beside the existing G1 metrics. Do not replace old names until
we compare old and new reports on the same saved motions.

### FK Beat Alignment

Name:

```text
G1FKBAS
```

Definition:

- compute robot keypoint speed from FK keypoints
- smooth the speed curve
- detect local minima as robot motion beats
- compare each robot motion beat to the nearest music beat with the same
  AIST-style BAS formula already used by the repo

This keeps continuity with AIST/RoboPerform while making the motion beat source
robot-geometry based.

### Beat Timing Report

Names:

```text
G1BeatF1
G1BeatPrecision
G1BeatRecall
G1BeatTimingMeanFrames
G1BeatTimingStdFrames
```

Definition:

- greedily match generated robot beats to audio or designated beats within a
  configurable tolerance
- report precision, recall, F1, mean signed offset, and offset jitter

This is more informative than BAS alone because it shows missed beats and timing
bias.

### Foot Diagnostics

Names:

```text
G1FootSliding
G1GroundPenetration
G1FootClearanceMean
```

Definition:

- infer foot contact from low foot height and low vertical speed
- measure horizontal foot speed during inferred contact
- measure lowest foot point below the estimated ground plane
- report mean foot clearance

These are still kinematic diagnostics, not simulator success metrics.

## Metrics Not In Scope Yet

Do not add these in the first FK pass:

- simulator success rate
- controller tracking error
- torque or energy metrics
- fall rate from rollout
- real robot deployment checks

Those require a controller or simulator rollout, not only forward kinematics.

## Evaluation Flow

The updated G1 evaluation should:

1. load saved G1 motion files
2. load the local Unitree model
3. run FK for each motion
4. compute FK beat, timing, and foot diagnostics
5. keep existing root/joint summary metrics for continuity
6. write the new metrics into the existing G1 report files
7. write audit entries describing model path, joint order, and quaternion order

Expected output files remain:

```text
metrics.json
g1_table.json
motion_audit.json
paper_report.md
renders/*
```

## Training Data Follow-Up

After FK evaluation is validated, we can optionally regenerate G1 beat metadata
from FK keypoint beats. That would change training labels for G1 BeatDistance,
so it must be treated as a new data version and followed by retraining.

Do not silently overwrite the current prepared G1 beat files. Use a new prepared
tree or a new beat-cache version.

## Acceptance Criteria

- FK metrics use official Unitree G1 geometry, not hand-written link offsets.
- The implementation verifies the 29-DoF joint order before scoring.
- The selected root quaternion convention is recorded in evaluation output.
- Existing G1 eval still works when FK metrics are disabled or unavailable.
- FK-enabled eval fails with a clear message if the model files or dependency
  are missing.
- Unit tests cover model loading, qpos construction, beat matching, timing
  metrics, and foot diagnostics on small fixtures.
- A smoke eval on saved G1 motions produces finite FK metrics.
