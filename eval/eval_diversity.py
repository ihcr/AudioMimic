import argparse
import glob
import pickle
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.eval_pfc import load_full_pose

JOINT_NAMES = [
    "root",
    "lhip", "rhip", "belly",
    "lknee", "rknee", "spine",
    "lankle", "rankle", "chest",
    "ltoes", "rtoes", "neck",
    "linshoulder", "rinshoulder",
    "head", "lshoulder", "rshoulder",
    "lelbow", "relbow",
    "lwrist", "rwrist",
    "lhand", "rhand",
]

FRAME_TIME = 1.0 / 60.0
UP_AXIS = 1
GENERIC_HUMERUS = np.array(
    [[1.99113488e-01, 2.36807942e-01, -1.80702247e-02],
     [4.54445392e-01, 2.21158922e-01, -4.10167128e-02]],
    dtype=np.float32,
)
GENERIC_SHOULDER = np.array(
    [[1.99113488e-01, 2.36807942e-01, -1.80702247e-02],
     [-1.91692337e-01, 2.36928746e-01, -1.23055102e-02]],
    dtype=np.float32,
)
GENERIC_HIP = np.array(
    [[5.64076714e-02, -3.23069185e-01, 1.09197125e-02],
     [-6.24834076e-02, -3.31302464e-01, 1.50412619e-02]],
    dtype=np.float32,
)


def distance_between_points(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def distance_from_plane(a, b, c, p, threshold):
    ba = np.asarray(b) - np.asarray(a)
    ca = np.asarray(c) - np.asarray(a)
    cross = np.cross(ca, ba)
    denom = np.linalg.norm(cross)
    if denom == 0:
        return False
    pa = np.asarray(p) - np.asarray(a)
    return bool(np.dot(cross, pa) / denom > threshold)


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = np.asarray(n2) - np.asarray(n1)
    denom = np.linalg.norm(normal)
    if denom == 0:
        return False
    pa = np.asarray(p) - np.asarray(a)
    return bool(np.dot(normal, pa) / denom > threshold)


def angle_within_range(j1, j2, k1, k2, angle_range):
    j = np.asarray(j2) - np.asarray(j1)
    k = np.asarray(k2) - np.asarray(k1)
    denom = np.linalg.norm(j) * np.linalg.norm(k)
    if denom == 0:
        return False
    cos_theta = np.clip(np.dot(j, k) / denom, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return bool(angle_range[0] < angle < angle_range[1])


def velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, p, p_prev, threshold):
    velocity = np.asarray(p) - np.asarray(j1) - (np.asarray(p_prev) - np.asarray(j1_prev))
    direction = np.asarray(j2) - np.asarray(j1)
    denom = np.linalg.norm(direction)
    if denom == 0:
        return False
    velocity_along_direction = np.dot(velocity, direction) / denom
    velocity_along_direction /= FRAME_TIME
    return bool(velocity_along_direction > threshold)


def velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, p, p_prev, threshold):
    velocity = np.asarray(p) - np.asarray(j1) - (np.asarray(p_prev) - np.asarray(j1_prev))
    j31 = np.asarray(j3) - np.asarray(j1)
    j21 = np.asarray(j2) - np.asarray(j1)
    direction = np.cross(j31, j21)
    denom = np.linalg.norm(direction)
    if denom == 0:
        return False
    velocity_along_direction = np.dot(velocity, direction) / denom
    velocity_along_direction /= FRAME_TIME
    return bool(velocity_along_direction > threshold)


def velocity_above_threshold(p, p_prev, threshold):
    velocity = np.linalg.norm(np.asarray(p) - np.asarray(p_prev)) / FRAME_TIME
    return bool(velocity > threshold)


def calc_average_velocity(positions, frame_idx, joint_idx, sliding_window):
    current_window = 0
    average_velocity = np.zeros(3, dtype=np.float32)
    for offset in range(-sliding_window, sliding_window + 1):
        if frame_idx + offset - 1 < 0 or frame_idx + offset >= len(positions):
            continue
        average_velocity += positions[frame_idx + offset][joint_idx] - positions[frame_idx + offset - 1][joint_idx]
        current_window += 1
    if current_window == 0:
        return 0.0
    return float(np.linalg.norm(average_velocity / (current_window * FRAME_TIME)))


def calc_average_acceleration(positions, frame_idx, joint_idx, sliding_window):
    current_window = 0
    average_acceleration = np.zeros(3, dtype=np.float32)
    for offset in range(-sliding_window, sliding_window + 1):
        if frame_idx + offset - 1 < 0 or frame_idx + offset + 1 >= len(positions):
            continue
        v2 = (positions[frame_idx + offset + 1][joint_idx] - positions[frame_idx + offset][joint_idx]) / FRAME_TIME
        v1 = (positions[frame_idx + offset][joint_idx] - positions[frame_idx + offset - 1][joint_idx]) / FRAME_TIME
        average_acceleration += (v2 - v1) / FRAME_TIME
        current_window += 1
    if current_window == 0:
        return 0.0
    return float(np.linalg.norm(average_acceleration / current_window))


def calc_average_velocity_horizontal(positions, frame_idx, joint_idx, sliding_window):
    current_window = 0
    average_velocity = np.zeros(2, dtype=np.float32)
    flat_axes = [0, 2]
    for offset in range(-sliding_window, sliding_window + 1):
        if frame_idx + offset - 1 < 0 or frame_idx + offset >= len(positions):
            continue
        average_velocity += (
            positions[frame_idx + offset][joint_idx, flat_axes]
            - positions[frame_idx + offset - 1][joint_idx, flat_axes]
        )
        current_window += 1
    if current_window == 0:
        return 0.0
    return float(np.linalg.norm(average_velocity / (current_window * FRAME_TIME)))


def calc_average_velocity_vertical(positions, frame_idx, joint_idx, sliding_window):
    current_window = 0
    average_velocity = 0.0
    for offset in range(-sliding_window, sliding_window + 1):
        if frame_idx + offset - 1 < 0 or frame_idx + offset >= len(positions):
            continue
        average_velocity += positions[frame_idx + offset][joint_idx, UP_AXIS] - positions[frame_idx + offset - 1][joint_idx, UP_AXIS]
        current_window += 1
    if current_window == 0:
        return 0.0
    return float(abs(average_velocity / (current_window * FRAME_TIME)))


def extract_kinetic_features(positions, sliding_window=2):
    positions = np.asarray(positions, dtype=np.float32)
    features = []
    for joint_idx in range(positions.shape[1]):
        horizontal = 0.0
        vertical = 0.0
        acceleration = 0.0
        for frame_idx in range(1, len(positions)):
            horizontal += calc_average_velocity_horizontal(
                positions, frame_idx, joint_idx, sliding_window
            ) ** 2
            vertical += calc_average_velocity_vertical(
                positions, frame_idx, joint_idx, sliding_window
            ) ** 2
            acceleration += calc_average_acceleration(
                positions, frame_idx, joint_idx, sliding_window
            )
        denom = max(len(positions) - 1.0, 1.0)
        features.extend([horizontal / denom, vertical / denom, acceleration / denom])
    return np.asarray(features, dtype=np.float32)


class ManualFeatureExtractor:
    def __init__(self, positions):
        self.positions = np.asarray(positions, dtype=np.float32)
        self.frame_num = 1
        self.joint_index = {name: idx for idx, name in enumerate(JOINT_NAMES)}
        self.hl = distance_between_points(*GENERIC_HUMERUS)
        self.sw = distance_between_points(*GENERIC_SHOULDER)
        self.hw = distance_between_points(*GENERIC_HIP)

    def next_frame(self):
        self.frame_num += 1

    def pos(self, name):
        if name == "y_unit":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if name == "minus_y_unit":
            return np.array([0.0, -1.0, 0.0], dtype=np.float32)
        if name == "zero":
            return np.zeros(3, dtype=np.float32)
        if name == "y_min":
            return np.array([0.0, np.min(self.positions[self.frame_num, :, UP_AXIS]), 0.0], dtype=np.float32)
        return self.positions[self.frame_num, self.joint_index[name]]

    def prev(self, name):
        return self.positions[self.frame_num - 1, self.joint_index[name]]

    def f_move(self, j1, j2, j3, j4, threshold):
        return velocity_direction_above_threshold(
            self.pos(j1), self.prev(j1), self.pos(j2), self.prev(j2), self.pos(j3), self.prev(j3), threshold
        )

    def f_nmove(self, j1, j2, j3, j4, threshold):
        return velocity_direction_above_threshold_normal(
            self.pos(j1), self.prev(j1), self.pos(j2), self.pos(j3), self.pos(j4), self.prev(j4), threshold
        )

    def f_plane(self, j1, j2, j3, j4, threshold):
        return distance_from_plane(self.pos(j1), self.pos(j2), self.pos(j3), self.pos(j4), threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        return distance_from_plane_normal(self.pos(j1), self.pos(j2), self.pos(j3), self.pos(j4), threshold)

    def f_angle(self, j1, j2, j3, j4, angle_range):
        return angle_within_range(self.pos(j1), self.pos(j2), self.pos(j3), self.pos(j4), angle_range)

    def f_fast(self, joint_name, threshold):
        return velocity_above_threshold(self.pos(joint_name), self.prev(joint_name), threshold)


def extract_manual_features(positions):
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) < 2:
        return np.zeros(32, dtype=np.float32)

    features = []
    extractor = ManualFeatureExtractor(positions)
    for _ in range(1, positions.shape[0]):
        pose_features = [
            extractor.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * extractor.hl),
            extractor.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * extractor.hl),
            extractor.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * extractor.hl),
            extractor.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * extractor.hl),
            extractor.f_move("belly", "chest", "chest", "rwrist", 1.8 * extractor.hl),
            extractor.f_move("belly", "chest", "chest", "lwrist", 1.8 * extractor.hl),
            extractor.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110]),
            extractor.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110]),
            extractor.f_nplane("lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * extractor.sw),
            extractor.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * extractor.hl),
            extractor.f_move("rwrist", "root", "lwrist", "root", 1.4 * extractor.hl),
            extractor.f_move("lwrist", "root", "rwrist", "root", 1.4 * extractor.hl),
            extractor.f_fast("rwrist", 2.5 * extractor.hl),
            extractor.f_fast("lwrist", 2.5 * extractor.hl),
            extractor.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * extractor.hl),
            extractor.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * extractor.hl),
            extractor.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * extractor.hl),
            extractor.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * extractor.hl),
            extractor.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * extractor.hw),
            extractor.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110]),
            extractor.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110]),
            extractor.f_fast("rankle", 2.5 * extractor.hl),
            extractor.f_fast("lankle", 2.5 * extractor.hl),
            extractor.f_angle("neck", "root", "rshoulder", "relbow", [25, 180]),
            extractor.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180]),
            extractor.f_angle("neck", "root", "rhip", "rknee", [50, 180]),
            extractor.f_angle("neck", "root", "lhip", "lknee", [50, 180]),
            extractor.f_plane("rankle", "neck", "lankle", "root", 0.5 * extractor.hl),
            extractor.f_angle("neck", "root", "zero", "y_unit", [70, 110]),
            extractor.f_nplane("zero", "minus_y_unit", "y_min", "rwrist", -1.2 * extractor.hl),
            extractor.f_nplane("zero", "minus_y_unit", "y_min", "lwrist", -1.2 * extractor.hl),
            extractor.f_fast("root", 2.3 * extractor.hl),
        ]
        features.append(np.asarray(pose_features, dtype=np.float32))
        extractor.next_frame()
    return np.mean(np.stack(features, axis=0), axis=0)


def to_y_up(full_pose):
    full_pose = np.asarray(full_pose, dtype=np.float32)
    return np.stack([full_pose[..., 0], full_pose[..., 2], -full_pose[..., 1]], axis=-1)


def zero_start_root(full_pose):
    return full_pose - full_pose[:1, :1, :]


def prepare_positions_for_features(full_pose):
    return zero_start_root(to_y_up(full_pose))


def extract_diversity_features(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    full_pose = prepare_positions_for_features(load_full_pose(payload))
    return extract_kinetic_features(full_pose), extract_manual_features(full_pose)


def compute_average_pairwise_distance(feature_list):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    if n < 2:
        return float("nan")
    dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    return float(dist / (((n * n) - n) / 2))


def normalize_features(reference_features, target_features, std_epsilon=1e-6):
    reference_features = np.stack(reference_features)
    target_features = np.stack(target_features)
    mean = reference_features.mean(axis=0)
    std = reference_features.std(axis=0)
    valid_dims = std > std_epsilon

    if not np.any(valid_dims):
        normalized_reference = np.zeros((reference_features.shape[0], 0), dtype=np.float32)
        normalized_target = np.zeros((target_features.shape[0], 0), dtype=np.float32)
    else:
        mean = mean[valid_dims]
        std = std[valid_dims]
        normalized_reference = (reference_features[:, valid_dims] - mean) / std
        normalized_target = (target_features[:, valid_dims] - mean) / std

    return normalized_reference, normalized_target, valid_dims


def collect_feature_pairs(motion_path, seed=1234, sample_limit=None):
    motion_files = sorted(glob.glob(str(Path(motion_path) / "*.pkl")))
    if sample_limit is not None and len(motion_files) > sample_limit:
        rng = random.Random(seed)
        motion_files = sorted(rng.sample(motion_files, sample_limit))

    kinetic_vectors = []
    manual_vectors = []
    for motion_file in tqdm(motion_files, desc="Diversity", unit="file"):
        kinetic_feature, manual_feature = extract_diversity_features(motion_file)
        kinetic_vectors.append(kinetic_feature)
        manual_vectors.append(manual_feature)
    return motion_files, kinetic_vectors, manual_vectors


def compute_diversity_metrics(motion_path, reference_motion_path=None, seed=1234, sample_limit=None):
    motion_files, kinetic_vectors, manual_vectors = collect_feature_pairs(
        motion_path, seed=seed, sample_limit=sample_limit
    )
    if reference_motion_path:
        reference_files, reference_kinetic, reference_manual = collect_feature_pairs(
            reference_motion_path, seed=seed, sample_limit=sample_limit
        )
    else:
        reference_files = motion_files
        reference_kinetic = kinetic_vectors
        reference_manual = manual_vectors

    reference_kinetic, normalized_kinetic, kinetic_valid_dims = normalize_features(
        reference_kinetic, kinetic_vectors
    )
    reference_manual, normalized_manual, manual_valid_dims = normalize_features(
        reference_manual, manual_vectors
    )

    distk = compute_average_pairwise_distance(normalized_kinetic)
    distg = compute_average_pairwise_distance(normalized_manual)

    return {
        "Distk": distk,
        "Distg": distg,
        "Divk": distk,
        "Divm": distg,
        "Divg": distg,
        "num_motion_files": len(motion_files),
        "num_reference_files": len(reference_files),
        "reference_Distk": compute_average_pairwise_distance(reference_kinetic),
        "reference_Distg": compute_average_pairwise_distance(reference_manual),
        "zero_variance_dims_kinetic": int((~kinetic_valid_dims).sum()),
        "zero_variance_dims_manual": int((~manual_valid_dims).sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", default="eval/motions")
    parser.add_argument("--reference_motion_path", default="")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--sample_limit", default=None, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = compute_diversity_metrics(
        args.motion_path,
        reference_motion_path=args.reference_motion_path or None,
        seed=args.seed,
        sample_limit=args.sample_limit,
    )
    print(result)
