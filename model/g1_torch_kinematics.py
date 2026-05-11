from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn

from eval.g1_kinematics import DEFAULT_KEYPOINT_BODIES, EXPECTED_G1_29DOF_JOINTS
from rotation_transforms import axis_angle_to_quaternion, quaternion_to_matrix


FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
LOWEST_FOOT_NAMES = ("left_lowest_foot_geom", "right_lowest_foot_geom")


@dataclass(frozen=True)
class BodySpec:
    name: str
    parent: int
    pos: tuple
    quat: tuple
    joint_name: str | None
    joint_axis: tuple


def _parse_floats(value, expected, default):
    if value is None:
        return tuple(default)
    parts = tuple(float(part) for part in value.split())
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} values, got {len(parts)} in {value!r}")
    return parts


def _joint_spec(body_elem):
    for joint in body_elem.findall("joint"):
        name = joint.attrib.get("name", "")
        if name == "floating_base_joint":
            return None, (0.0, 0.0, 0.0)
        if joint.attrib.get("type", "hinge") != "hinge":
            raise ValueError(f"Unsupported G1 joint type for {name!r}")
        axis = _parse_floats(joint.attrib.get("axis"), 3, (0.0, 0.0, 1.0))
        joint_pos = _parse_floats(joint.attrib.get("pos"), 3, (0.0, 0.0, 0.0))
        if any(abs(value) > 1e-8 for value in joint_pos):
            raise ValueError(f"Non-zero G1 joint anchor is not supported for {name!r}")
        return name, axis
    return None, (0.0, 0.0, 0.0)


def _geom_points(body_elem):
    offsets = []
    radii = []
    for geom in body_elem.findall("geom"):
        geom_type = geom.attrib.get("type", "sphere")
        if geom_type == "mesh":
            continue
        pos = _parse_floats(geom.attrib.get("pos"), 3, (0.0, 0.0, 0.0))
        size = _parse_floats(geom.attrib.get("size"), 1, (0.0,))
        offsets.append(pos)
        radii.append(size[0])
    return offsets, radii


def parse_g1_mjcf(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"G1 MuJoCo model not found: {model_path}")
    root = ET.parse(model_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"G1 MuJoCo model is missing worldbody: {model_path}")

    bodies = []
    foot_geoms = {name: ([], []) for name in FOOT_BODY_NAMES}

    def visit(body_elem, parent):
        name = body_elem.attrib["name"]
        joint_name, joint_axis = _joint_spec(body_elem)
        body_index = len(bodies)
        bodies.append(
            BodySpec(
                name=name,
                parent=parent,
                pos=_parse_floats(body_elem.attrib.get("pos"), 3, (0.0, 0.0, 0.0)),
                quat=_parse_floats(body_elem.attrib.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)),
                joint_name=joint_name,
                joint_axis=joint_axis,
            )
        )
        if name in foot_geoms:
            foot_geoms[name] = _geom_points(body_elem)
        for child in body_elem.findall("body"):
            visit(child, body_index)

    for child in worldbody.findall("body"):
        visit(child, -1)

    joint_names = [body.joint_name for body in bodies if body.joint_name is not None]
    expected = list(EXPECTED_G1_29DOF_JOINTS)
    if joint_names != expected:
        raise ValueError(f"G1 joint order mismatch. Expected {expected}, got {joint_names}.")
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(model_path))
        for foot_name in FOOT_BODY_NAMES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
            if body_id < 0:
                raise ValueError(f"G1 MuJoCo model is missing body: {foot_name}")
            geom_ids = [idx for idx, geom_body in enumerate(model.geom_bodyid) if geom_body == body_id]
            offsets = [tuple(model.geom_pos[idx].tolist()) for idx in geom_ids]
            radii = [float(model.geom_size[idx, 0]) for idx in geom_ids]
            if offsets:
                foot_geoms[foot_name] = (offsets, radii)
    except ImportError:
        pass
    return bodies, foot_geoms


class G1TorchKinematics(nn.Module):
    def __init__(self, model_path, root_quat_order="xyzw"):
        super().__init__()
        if root_quat_order not in ("wxyz", "xyzw"):
            raise ValueError("root_quat_order must be 'wxyz' or 'xyzw'")
        self.root_quat_order = root_quat_order
        bodies, foot_geoms = parse_g1_mjcf(model_path)
        self.body_names = [body.name for body in bodies]
        self.keypoint_names = list(DEFAULT_KEYPOINT_BODIES) + list(LOWEST_FOOT_NAMES)

        joint_to_index = {
            name: index for index, name in enumerate(EXPECTED_G1_29DOF_JOINTS)
        }
        joint_indices = [
            -1 if body.joint_name is None else joint_to_index[body.joint_name]
            for body in bodies
        ]
        keypoint_indices = [self.body_names.index(name) for name in DEFAULT_KEYPOINT_BODIES]
        foot_body_indices = [self.body_names.index(name) for name in FOOT_BODY_NAMES]

        self.register_buffer("parent_ids", torch.tensor([body.parent for body in bodies], dtype=torch.long))
        self.register_buffer("body_pos", torch.tensor([body.pos for body in bodies], dtype=torch.float32))
        self.register_buffer("body_quat", torch.tensor([body.quat for body in bodies], dtype=torch.float32))
        self.register_buffer("joint_indices", torch.tensor(joint_indices, dtype=torch.long))
        self.register_buffer("joint_axes", torch.tensor([body.joint_axis for body in bodies], dtype=torch.float32))
        self.register_buffer("keypoint_indices", torch.tensor(keypoint_indices, dtype=torch.long))
        self.register_buffer("foot_body_indices", torch.tensor(foot_body_indices, dtype=torch.long))

        for foot_name in FOOT_BODY_NAMES:
            offsets, radii = foot_geoms[foot_name]
            if not offsets:
                offsets = [(0.0, 0.0, 0.0)]
                radii = [0.0]
            self.register_buffer(
                f"{foot_name}_geom_offsets",
                torch.tensor(offsets, dtype=torch.float32),
            )
            self.register_buffer(
                f"{foot_name}_geom_radii",
                torch.tensor(radii, dtype=torch.float32),
            )

    def _root_quaternion_wxyz(self, root_rot):
        if self.root_quat_order == "wxyz":
            return root_rot
        return root_rot[..., [3, 0, 1, 2]]

    def _lowest_foot_points(self, body_positions, body_rotations, foot_name, body_index):
        offsets = getattr(self, f"{foot_name}_geom_offsets").to(
            device=body_positions.device,
            dtype=body_positions.dtype,
        )
        radii = getattr(self, f"{foot_name}_geom_radii").to(
            device=body_positions.device,
            dtype=body_positions.dtype,
        )
        foot_pos = body_positions[:, body_index]
        foot_rot = body_rotations[:, body_index]
        geom_pos = foot_pos[:, None, :] + torch.matmul(
            foot_rot[:, None, :, :],
            offsets[None, :, :, None],
        ).squeeze(-1)
        lowest_z = geom_pos[..., 2] - radii[None, :]
        selected = torch.argmin(lowest_z, dim=-1)
        gather_index = selected[:, None, None].expand(-1, 1, 3)
        point = torch.gather(geom_pos, dim=1, index=gather_index).squeeze(1).clone()
        point[:, 2] = torch.gather(lowest_z, dim=1, index=selected[:, None]).squeeze(1)
        return point

    def forward(self, root_pos, root_rot, dof_pos):
        if root_pos.shape[-1] != 3:
            raise ValueError(f"root_pos expected 3 channels, got {root_pos.shape[-1]}")
        if root_rot.shape[-1] != 4:
            raise ValueError(f"root_rot expected 4 channels, got {root_rot.shape[-1]}")
        if dof_pos.shape[-1] != len(EXPECTED_G1_29DOF_JOINTS):
            raise ValueError(f"dof_pos expected 29 channels, got {dof_pos.shape[-1]}")
        if root_pos.shape[:-1] != root_rot.shape[:-1] or root_pos.shape[:-1] != dof_pos.shape[:-1]:
            raise ValueError("root_pos, root_rot, and dof_pos must share leading dimensions")

        leading_shape = root_pos.shape[:-1]
        flat_root_pos = root_pos.reshape(-1, 3)
        flat_root_rot = self._root_quaternion_wxyz(root_rot).reshape(-1, 4)
        flat_dof = dof_pos.reshape(-1, len(EXPECTED_G1_29DOF_JOINTS))

        batch = flat_root_pos.shape[0]
        body_count = self.body_pos.shape[0]
        body_positions = flat_root_pos.new_zeros((batch, body_count, 3))
        body_rotations = flat_root_pos.new_zeros((batch, body_count, 3, 3))

        fixed_rotations = quaternion_to_matrix(
            self.body_quat.to(device=flat_root_pos.device, dtype=flat_root_pos.dtype)
        )
        root_rotation = quaternion_to_matrix(flat_root_rot)

        for body_index in range(body_count):
            parent = int(self.parent_ids[body_index].item())
            if parent < 0:
                body_positions[:, body_index] = flat_root_pos
                body_rotations[:, body_index] = root_rotation
                continue

            parent_pos = body_positions[:, parent]
            parent_rot = body_rotations[:, parent]
            local_pos = self.body_pos[body_index].to(
                device=flat_root_pos.device,
                dtype=flat_root_pos.dtype,
            )
            fixed_rot = fixed_rotations[body_index].expand(batch, -1, -1)
            base_pos = parent_pos + torch.matmul(
                parent_rot,
                local_pos.view(1, 3, 1).expand(batch, -1, -1),
            ).squeeze(-1)
            base_rot = torch.matmul(parent_rot, fixed_rot)

            joint_index = int(self.joint_indices[body_index].item())
            if joint_index >= 0:
                axis = self.joint_axes[body_index].to(
                    device=flat_root_pos.device,
                    dtype=flat_root_pos.dtype,
                )
                angle_axis = axis.expand(batch, 3) * flat_dof[:, joint_index : joint_index + 1]
                joint_rot = quaternion_to_matrix(axis_angle_to_quaternion(angle_axis))
                body_rot = torch.matmul(base_rot, joint_rot)
            else:
                body_rot = base_rot

            body_positions[:, body_index] = base_pos
            body_rotations[:, body_index] = body_rot

        body_positions = body_positions.reshape(*leading_shape, body_count, 3)
        body_rotations = body_rotations.reshape(*leading_shape, body_count, 3, 3)
        keypoints = body_positions.index_select(-2, self.keypoint_indices)

        flat_positions = body_positions.reshape(-1, body_count, 3)
        flat_rotations = body_rotations.reshape(-1, body_count, 3, 3)
        left_foot = self._lowest_foot_points(
            flat_positions,
            flat_rotations,
            FOOT_BODY_NAMES[0],
            int(self.foot_body_indices[0].item()),
        )
        right_foot = self._lowest_foot_points(
            flat_positions,
            flat_rotations,
            FOOT_BODY_NAMES[1],
            int(self.foot_body_indices[1].item()),
        )
        feet = torch.stack((left_foot, right_foot), dim=-2).reshape(*leading_shape, 2, 3)
        keypoints = torch.cat((keypoints, feet), dim=-2)
        return {
            "bodies": body_positions,
            "keypoints": keypoints,
            "feet": feet,
            "body_names": self.body_names,
            "keypoint_names": self.keypoint_names,
        }
