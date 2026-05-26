from pathlib import Path

import numpy as np


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

DEFAULT_KEYPOINT_BODIES = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "G1 forward kinematics requires the mujoco package. "
            "Install it with `pip install -r requirements-eval.txt`."
        ) from exc
    return mujoco


def _as_float_array(value, name, ndim=None):
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} expected {ndim} dimensions, got {array.ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize_quat(quat):
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return quat / norm


def load_g1_mujoco_model(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"G1 MuJoCo model not found: {model_path}")
    mujoco = _require_mujoco()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    validate_g1_joint_order(model)
    return model


def _joint_names_from_model(model):
    mujoco = _require_mujoco()
    names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and name != "floating_base_joint":
            names.append(name)
    return names


def validate_g1_joint_order(model_or_joint_names):
    if isinstance(model_or_joint_names, (list, tuple)):
        joint_names = list(model_or_joint_names)
    else:
        joint_names = _joint_names_from_model(model_or_joint_names)
    expected = list(EXPECTED_G1_29DOF_JOINTS)
    if joint_names != expected:
        raise ValueError(
            "G1 joint order mismatch. "
            f"Expected {expected}, got {joint_names}."
        )
    return joint_names


def build_g1_qpos(root_pos, root_rot, dof_pos, root_quat_order="xyzw"):
    root_pos = _as_float_array(root_pos, "root_pos", ndim=2)
    root_rot = _as_float_array(root_rot, "root_rot", ndim=2)
    dof_pos = _as_float_array(dof_pos, "dof_pos", ndim=2)
    if root_pos.shape[-1] != 3:
        raise ValueError(f"root_pos expected 3 channels, got {root_pos.shape[-1]}")
    if root_rot.shape[-1] != 4:
        raise ValueError(f"root_rot expected 4 channels, got {root_rot.shape[-1]}")
    if dof_pos.shape[-1] != len(EXPECTED_G1_29DOF_JOINTS):
        raise ValueError(f"dof_pos expected 29 channels, got {dof_pos.shape[-1]}")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError("root_pos, root_rot, and dof_pos must have matching frames")

    if root_quat_order == "wxyz":
        quat = root_rot
    elif root_quat_order == "xyzw":
        quat = root_rot[:, [3, 0, 1, 2]]
    else:
        raise ValueError(f"Unsupported root quaternion order: {root_quat_order}")
    quat = _normalize_quat(quat)
    return np.concatenate([root_pos, quat, dof_pos], axis=-1).astype(np.float64)


def _names_for_objects(model, object_type, count):
    mujoco = _require_mujoco()
    names = []
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        names.append(name or f"{object_type.name.lower()}_{object_id}")
    return names


def _body_id(model, name):
    mujoco = _require_mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"G1 MuJoCo model is missing body: {name}")
    return body_id


def _lowest_geom_point_for_body(model, data, body_name):
    body_id = _body_id(model, body_name)
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    if geom_ids.size == 0:
        return data.xpos[body_id].copy()
    geom_positions = data.geom_xpos[geom_ids].copy()
    geom_radii = model.geom_size[geom_ids, 0].copy()
    lowest_z = geom_positions[:, 2] - geom_radii
    selected = int(np.argmin(lowest_z))
    point = geom_positions[selected].copy()
    point[2] = lowest_z[selected]
    return point


def _extract_keypoints(model, data, body_ids):
    keypoint_names = list(DEFAULT_KEYPOINT_BODIES)
    keypoints = [data.xpos[body_ids[name]].copy() for name in DEFAULT_KEYPOINT_BODIES]
    left_foot = _lowest_geom_point_for_body(model, data, "left_ankle_roll_link")
    right_foot = _lowest_geom_point_for_body(model, data, "right_ankle_roll_link")
    keypoint_names.extend(["left_lowest_foot_geom", "right_lowest_foot_geom"])
    keypoints.extend([left_foot, right_foot])
    return np.asarray(keypoints, dtype=np.float32), left_foot, right_foot, keypoint_names


def forward_g1_kinematics(motion, model_path, root_quat_order="xyzw"):
    model = load_g1_mujoco_model(model_path)
    mujoco = _require_mujoco()
    qpos = build_g1_qpos(
        motion["root_pos"],
        motion["root_rot"],
        motion["dof_pos"],
        root_quat_order=root_quat_order,
    )
    if qpos.shape[-1] != model.nq:
        raise ValueError(f"G1 qpos expected {model.nq} channels, got {qpos.shape[-1]}")

    data = mujoco.MjData(model)
    body_names = _names_for_objects(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    body_parent_ids = model.body_parentid.copy().astype(np.int64)
    geom_names = _names_for_objects(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    body_ids = {name: _body_id(model, name) for name in DEFAULT_KEYPOINT_BODIES}

    bodies = []
    geoms = []
    keypoints = []
    left_foot_points = []
    right_foot_points = []
    keypoint_names = None
    for frame_qpos in qpos:
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        bodies.append(data.xpos.copy())
        geoms.append(data.geom_xpos.copy())
        frame_keypoints, left_foot, right_foot, keypoint_names = _extract_keypoints(
            model,
            data,
            body_ids,
        )
        keypoints.append(frame_keypoints)
        left_foot_points.append(left_foot)
        right_foot_points.append(right_foot)

    return {
        "bodies": np.asarray(bodies, dtype=np.float32),
        "body_names": body_names,
        "body_parent_ids": body_parent_ids,
        "geoms": np.asarray(geoms, dtype=np.float32),
        "geom_names": geom_names,
        "keypoints": np.asarray(keypoints, dtype=np.float32),
        "keypoint_names": keypoint_names or [],
        "left_foot_points": np.asarray(left_foot_points, dtype=np.float32),
        "right_foot_points": np.asarray(right_foot_points, dtype=np.float32),
        "metadata": {
            "model_path": str(model_path),
            "root_quat_order": root_quat_order,
            "joint_names": list(EXPECTED_G1_29DOF_JOINTS),
        },
    }
