import argparse
import glob
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

from rotation_transforms import (axis_angle_to_quaternion, quaternion_multiply,
                                 quaternion_to_axis_angle)
from vis import SMPLSkeleton

PFC_SCALE = 10000.0
UP_DIR = 2  # z is up
FLAT_DIRS = [i for i in range(3) if i != UP_DIR]
DT = 1 / 30

SMPL_SKELETON = SMPLSkeleton()


def compute_pfc_components(full_pose):
    joint3d = np.asarray(full_pose, dtype=np.float32)
    root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT
    root_a = (root_v[1:] - root_v[:-1]) / DT
    root_a[:, UP_DIR] = np.maximum(root_a[:, UP_DIR], 0)
    root_a = np.linalg.norm(root_a, axis=-1)
    acceleration_scale = float(root_a.max()) if root_a.size else 0.0
    root_a_norm = root_a / acceleration_scale if acceleration_scale > 0 else root_a

    foot_idx = [7, 10, 8, 11]
    feet = joint3d[:, foot_idx]
    foot_v = np.linalg.norm(
        feet[2:, :, FLAT_DIRS] - feet[1:-1, :, FLAT_DIRS],
        axis=-1,
    )
    foot_mins = np.zeros((len(foot_v), 2), dtype=np.float32)
    foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
    foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

    foot_loss = foot_mins[:, 0] * foot_mins[:, 1] * root_a_norm
    return {
        "raw_pfc": float(foot_loss.mean()) if foot_loss.size else 0.0,
        "mean_root_acceleration": float(root_a.mean()) if root_a.size else 0.0,
        "mean_root_acceleration_norm": (
            float(root_a_norm.mean()) if root_a_norm.size else 0.0
        ),
        "mean_left_foot_min": float(foot_mins[:, 0].mean()) if foot_mins.size else 0.0,
        "mean_right_foot_min": float(foot_mins[:, 1].mean()) if foot_mins.size else 0.0,
        "acceleration_scale": acceleration_scale,
        "num_frames": int(joint3d.shape[0]),
    }


def motion_pickle_to_full_pose(payload):
    root_pos = np.asarray(payload["pos"], dtype=np.float32)
    local_q = np.asarray(payload["q"], dtype=np.float32).reshape(-1, 24, 3)
    scale = payload.get("scale")
    if scale is not None:
        scale_value = float(np.asarray(scale).reshape(-1)[0])
        if scale_value != 0:
            root_pos = root_pos / scale_value

    root_pos = root_pos[::2]
    local_q = local_q[::2]

    local_q = np.asarray(local_q, dtype=np.float32)
    root_pos = torch.from_numpy(root_pos)
    local_q = torch.from_numpy(local_q)
    root_q = local_q[:, :1, :]
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0], dtype=torch.float32)
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    local_q[:, :1, :] = quaternion_to_axis_angle(root_q_quat)

    root_pos = torch.stack((root_pos[:, 0], -root_pos[:, 2], root_pos[:, 1]), dim=-1)
    return SMPL_SKELETON.forward(local_q[None], root_pos[None]).detach().cpu().numpy()[0]


def smpl_motion_to_full_pose(payload):
    smpl_poses = np.asarray(payload["smpl_poses"], dtype=np.float32).reshape(-1, 24, 3)
    smpl_trans = np.asarray(payload["smpl_trans"], dtype=np.float32).reshape(-1, 3)
    local_q = torch.from_numpy(smpl_poses)
    root_pos = torch.from_numpy(smpl_trans)
    return SMPL_SKELETON.forward(local_q[None], root_pos[None]).detach().cpu().numpy()[0]


def _has_smpl_motion(payload):
    return "smpl_poses" in payload and "smpl_trans" in payload


def _has_legacy_motion(payload):
    return "pos" in payload and "q" in payload


def _reconstruct_full_pose(payload):
    if _has_smpl_motion(payload):
        return smpl_motion_to_full_pose(payload)
    if _has_legacy_motion(payload):
        return motion_pickle_to_full_pose(payload)
    raise KeyError("Unsupported motion payload. Expected smpl_poses/smpl_trans or pos/q.")


def _validate_full_pose_consistency(payload, full_pose, atol=1e-4, rtol=1e-4):
    if not (_has_smpl_motion(payload) or _has_legacy_motion(payload)):
        return full_pose
    reconstructed = _reconstruct_full_pose(payload)
    if not np.allclose(full_pose, reconstructed, atol=atol, rtol=rtol):
        diff = np.abs(full_pose - reconstructed)
        raise ValueError(
            "Saved full_pose does not match reconstructed motion data "
            f"(max diff {float(diff.max()):.6f}, mean diff {float(diff.mean()):.6f})."
        )
    return full_pose


def load_full_pose(payload, source_preference="auto", validate_consistency=True):
    if source_preference == "full_pose":
        if "full_pose" not in payload:
            raise KeyError("Expected payload to contain full_pose.")
        full_pose = np.asarray(payload["full_pose"], dtype=np.float32)
        if validate_consistency:
            return _validate_full_pose_consistency(payload, full_pose)
        return full_pose

    if source_preference == "smpl":
        if not _has_smpl_motion(payload):
            raise KeyError("Expected payload to contain smpl_poses and smpl_trans.")
        return smpl_motion_to_full_pose(payload)

    if source_preference == "legacy":
        if not _has_legacy_motion(payload):
            raise KeyError("Expected payload to contain pos and q.")
        return motion_pickle_to_full_pose(payload)

    if source_preference == "smpl_or_legacy":
        return _reconstruct_full_pose(payload)

    if source_preference != "auto":
        raise ValueError(f"Unsupported source_preference {source_preference!r}")

    if "full_pose" in payload:
        return load_full_pose(
            payload,
            source_preference="full_pose",
            validate_consistency=validate_consistency,
        )
    if _has_smpl_motion(payload) or _has_legacy_motion(payload):
        return _reconstruct_full_pose(payload)
    raise KeyError("Unsupported motion payload. Expected full_pose or pos/q fields.")


def compute_physical_score(dir, return_details=False, sample_limit=1000, seed=1234):
    scores = []
    component_records = []

    motion_files = glob.glob(os.path.join(dir, "*.pkl"))
    if len(motion_files) > sample_limit:
        rng = random.Random(seed)
        motion_files = rng.sample(motion_files, sample_limit)

    for pkl in tqdm(motion_files, desc="PFC", unit="file"):
        with open(pkl, "rb") as handle:
            info = pickle.load(handle)
        components = compute_pfc_components(load_full_pose(info))
        scores.append(components["raw_pfc"])
        component_records.append(components)

    pfc = float(np.mean(scores) * PFC_SCALE) if scores else float("nan")
    if not return_details:
        return pfc

    if not component_records:
        return {
            "PFC": float("nan"),
            "num_files": 0,
            "mean_raw_pfc": float("nan"),
            "mean_root_acceleration": float("nan"),
            "mean_root_acceleration_norm": float("nan"),
            "mean_left_foot_min": float("nan"),
            "mean_right_foot_min": float("nan"),
            "mean_acceleration_scale": float("nan"),
        }

    return {
        "PFC": pfc,
        "num_files": len(component_records),
        "mean_raw_pfc": float(np.mean([row["raw_pfc"] for row in component_records])),
        "mean_root_acceleration": float(
            np.mean([row["mean_root_acceleration"] for row in component_records])
        ),
        "mean_root_acceleration_norm": float(
            np.mean([row["mean_root_acceleration_norm"] for row in component_records])
        ),
        "mean_left_foot_min": float(
            np.mean([row["mean_left_foot_min"] for row in component_records])
        ),
        "mean_right_foot_min": float(
            np.mean([row["mean_right_foot_min"] for row in component_records])
        ),
        "mean_acceleration_scale": float(
            np.mean([row["acceleration_scale"] for row in component_records])
        ),
    }


def calc_physical_score(dir):
    out = compute_physical_score(dir)
    print(f"{dir} has a mean PFC of {out}")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_physical_score(opt.motion_path)
