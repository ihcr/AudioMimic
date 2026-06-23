import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.audio_extraction.motion_control_v2_features import (
    CACHE_VERSION,
    LOCAL_CACHE_VERSION,
    LOCAL_FEATURE_DIR_NAME,
    LOCAL_METADATA_NAME,
    ROOT_LOCAL_FRAME,
    SUPPORT_CACHE_VERSION,
    SUPPORT_FEATURE_DIR_NAME,
    SUPPORT_METADATA_NAME,
    TARGET_FRAMES,
    WORLD_FRAME,
    _resolve_keypoint_indices,
    _weighted_fk_speed_batch,
    beatness_values_for_frames,
    extract_motion_control_v2_features,
    extract_motion_control_v4_support_features,
    parse_args,
)
from data.audio_extraction.motion_control_v3_local_features import (
    extract_motion_control_v3_local_features,
    parse_args as parse_v3_local_args,
)
from model.g1_torch_kinematics import G1TorchKinematics


def _write_motion(path, speed_scale, dof_scale=0.0):
    frames = np.arange(TARGET_FRAMES, dtype=np.float32)
    root_pos = np.zeros((TARGET_FRAMES, 3), dtype=np.float32)
    root_pos[:, 0] = frames * float(speed_scale) + 0.01 * np.sin(frames / 4.0)
    root_rot = np.zeros((TARGET_FRAMES, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    dof_pos = np.zeros((TARGET_FRAMES, 29), dtype=np.float32)
    if dof_scale:
        phase = frames[:, None] / 7.0 + np.arange(29, dtype=np.float32)[None, :] / 5.0
        dof_pos[:] = float(dof_scale) * np.sin(phase)
    q = np.concatenate((root_rot, dof_pos), axis=-1)
    payload = {
        "motion_format": "g1",
        "fps": 30,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "pos": root_pos,
        "q": q,
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _write_gaussian(path, beat_frames):
    values = np.zeros((TARGET_FRAMES, 1), dtype=np.float32)
    for frame in beat_frames:
        values[int(frame), 0] = 1.0
    np.save(path, values)


class MotionControlV2FeatureTests(unittest.TestCase):
    def test_beatness_prefers_beat_centered_local_minimum(self):
        speed = np.ones((TARGET_FRAMES,), dtype=np.float32)
        speed[44:48] = 7.0
        speed[53:57] = 8.0
        speed[50] = 0.5
        local_min_score = beatness_values_for_frames(speed, np.array([50]), 6, 3, 6)[0]

        peak_speed = np.ones((TARGET_FRAMES,), dtype=np.float32)
        peak_speed[50] = 8.0
        local_max_score = beatness_values_for_frames(peak_speed, np.array([50]), 6, 3, 6)[0]

        self.assertGreater(local_min_score, 5.0)
        self.assertLess(local_max_score, 0.5)

    def test_extract_motion_control_v2_cache_writes_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "test"):
                for folder in ("motions_sliced", "gaussian_beat_feats"):
                    (root / split / folder).mkdir(parents=True)

            for idx, speed in enumerate((0.001, 0.002, 0.003)):
                stem = f"{idx:03d}_slice000"
                _write_motion(root / "train" / "motions_sliced" / f"{stem}.pkl", speed)
                _write_gaussian(root / "train" / "gaussian_beat_feats" / f"{stem}.npy", [20, 70, 120])
            _write_motion(root / "test" / "motions_sliced" / "100_slice000.pkl", 0.002)
            _write_gaussian(root / "test" / "gaussian_beat_feats" / "100_slice000.npy", [30, 90])

            args = parse_args(
                [
                    "--data_path",
                    str(root),
                    "--device",
                    "cpu",
                    "--batch_size",
                    "2",
                    "--g1_fk_model_path",
                    "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
                ]
            )
            metadata = extract_motion_control_v2_features(args)

            self.assertEqual(metadata["cache_version"], CACHE_VERSION)
            metadata_path = root / "motion_control_v2_metadata.json"
            self.assertTrue(metadata_path.is_file())
            with open(metadata_path, "r", encoding="utf-8") as handle:
                saved_metadata = json.load(handle)
            self.assertEqual(
                saved_metadata["normalization"]["motion_intensity"]["train_peak_count"],
                9,
            )
            self.assertEqual(
                saved_metadata["normalization"]["motion_beatness"]["train_peak_count"],
                9,
            )

            feature_path = root / "test" / "motion_control_v2_feats" / "100_slice000.npz"
            self.assertTrue(feature_path.is_file())
            with np.load(feature_path) as data:
                self.assertEqual(data["motion_intensity_envelope"].shape, (TARGET_FRAMES, 1))
                self.assertEqual(data["motion_beatness_envelope"].shape, (TARGET_FRAMES, 1))
                self.assertEqual(data["weighted_fk_speed"].shape, (TARGET_FRAMES,))
                self.assertEqual(data["smoothed_weighted_fk_speed"].shape, (TARGET_FRAMES,))
                self.assertEqual(data["audio_beat_frames"].tolist(), [30, 90])
                self.assertEqual(data["intensity_peaks"].shape, (2,))
                self.assertEqual(data["beatness_peaks"].shape, (2,))
                self.assertTrue(np.isfinite(data["motion_intensity_envelope"]).all())
                self.assertTrue(np.isfinite(data["motion_beatness_envelope"]).all())

    def test_root_local_speed_removes_pure_root_yaw_motion(self):
        frames = np.arange(TARGET_FRAMES, dtype=np.float32)
        root_pos = np.zeros((1, TARGET_FRAMES, 3), dtype=np.float32)
        root_rot = np.zeros((1, TARGET_FRAMES, 4), dtype=np.float32)
        yaw = frames * 0.05
        root_rot[0, :, 2] = np.sin(yaw / 2.0)
        root_rot[0, :, 3] = np.cos(yaw / 2.0)
        dof_pos = np.zeros((1, TARGET_FRAMES, 29), dtype=np.float32)
        kinematics = G1TorchKinematics(
            "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
            root_quat_order="xyzw",
        ).to(torch.device("cpu"))
        keypoint_indices = _resolve_keypoint_indices(kinematics)

        world_speed = _weighted_fk_speed_batch(
            kinematics,
            root_pos,
            root_rot,
            dof_pos,
            keypoint_indices,
            torch.device("cpu"),
            coordinate_frame=WORLD_FRAME,
        )[0]
        local_speed = _weighted_fk_speed_batch(
            kinematics,
            root_pos,
            root_rot,
            dof_pos,
            keypoint_indices,
            torch.device("cpu"),
            coordinate_frame=ROOT_LOCAL_FRAME,
        )[0]

        self.assertGreater(float(world_speed[1:].mean()), 0.05)
        self.assertLess(float(np.max(np.abs(local_speed))), 1e-4)

    def test_extract_motion_control_v3_local_cache_uses_separate_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "test"):
                for folder in ("motions_sliced", "gaussian_beat_feats"):
                    (root / split / folder).mkdir(parents=True)

            for idx, dof_scale in enumerate((0.04, 0.06, 0.08)):
                stem = f"{idx:03d}_slice000"
                _write_motion(
                    root / "train" / "motions_sliced" / f"{stem}.pkl",
                    0.002,
                    dof_scale=dof_scale,
                )
                _write_gaussian(root / "train" / "gaussian_beat_feats" / f"{stem}.npy", [20, 70, 120])
            _write_motion(
                root / "test" / "motions_sliced" / "100_slice000.pkl",
                0.002,
                dof_scale=0.05,
            )
            _write_gaussian(root / "test" / "gaussian_beat_feats" / "100_slice000.npy", [30, 90])

            args = parse_v3_local_args(
                [
                    "--data_path",
                    str(root),
                    "--device",
                    "cpu",
                    "--batch_size",
                    "2",
                    "--g1_fk_model_path",
                    "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
                ]
            )
            metadata = extract_motion_control_v3_local_features(args)

            self.assertEqual(metadata["cache_version"], LOCAL_CACHE_VERSION)
            self.assertEqual(metadata["coordinate_frame"], ROOT_LOCAL_FRAME)
            self.assertEqual(metadata["feature_dir"], LOCAL_FEATURE_DIR_NAME)
            self.assertEqual(metadata["metadata_name"], LOCAL_METADATA_NAME)
            self.assertTrue((root / LOCAL_METADATA_NAME).is_file())
            feature_path = root / "test" / LOCAL_FEATURE_DIR_NAME / "100_slice000.npz"
            self.assertTrue(feature_path.is_file())
            with np.load(feature_path) as data:
                self.assertEqual(data["weighted_fk_speed"].shape, (TARGET_FRAMES,))
                self.assertEqual(data["motion_intensity_envelope"].shape, (TARGET_FRAMES, 1))
                self.assertEqual(data["motion_beatness_envelope"].shape, (TARGET_FRAMES, 1))

    def test_extract_motion_control_v4_support_cache_writes_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "test"):
                for folder in ("motions_sliced", "gaussian_beat_feats"):
                    (root / split / folder).mkdir(parents=True)

            for idx, dof_scale in enumerate((0.04, 0.07, 0.10)):
                stem = f"{idx:03d}_slice000"
                _write_motion(
                    root / "train" / "motions_sliced" / f"{stem}.pkl",
                    0.002,
                    dof_scale=dof_scale,
                )
                _write_gaussian(root / "train" / "gaussian_beat_feats" / f"{stem}.npy", [20, 70, 120])
            _write_motion(
                root / "test" / "motions_sliced" / "100_slice000.pkl",
                0.002,
                dof_scale=0.06,
            )
            _write_gaussian(root / "test" / "gaussian_beat_feats" / "100_slice000.npy", [30, 90])

            args = parse_args(
                [
                    "--data_path",
                    str(root),
                    "--device",
                    "cpu",
                    "--batch_size",
                    "2",
                    "--g1_fk_model_path",
                    "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
                    "--support_v6a",
                ]
            )
            metadata = extract_motion_control_v4_support_features(args)

            self.assertEqual(metadata["cache_version"], SUPPORT_CACHE_VERSION)
            self.assertEqual(metadata["coordinate_frame"], ROOT_LOCAL_FRAME)
            self.assertEqual(metadata["feature_dir"], SUPPORT_FEATURE_DIR_NAME)
            self.assertEqual(metadata["metadata_name"], SUPPORT_METADATA_NAME)
            self.assertTrue((root / SUPPORT_METADATA_NAME).is_file())
            for key in ("body_intensity", "support_beatness", "upper_beatness"):
                self.assertEqual(metadata["normalization"][key]["train_peak_count"], 9)

            feature_path = root / "test" / SUPPORT_FEATURE_DIR_NAME / "100_slice000.npz"
            self.assertTrue(feature_path.is_file())
            with np.load(feature_path) as data:
                self.assertEqual(data["body_intensity_envelope"].shape, (TARGET_FRAMES, 1))
                self.assertEqual(data["support_beatness_envelope"].shape, (TARGET_FRAMES, 1))
                self.assertEqual(data["upper_beatness_envelope"].shape, (TARGET_FRAMES, 1))
                self.assertEqual(data["support_contact"].shape, (TARGET_FRAMES, 2))
                self.assertEqual(data["body_weighted_fk_speed"].shape, (TARGET_FRAMES,))
                self.assertEqual(data["support_weighted_fk_speed"].shape, (TARGET_FRAMES,))
                self.assertEqual(data["upper_weighted_fk_speed"].shape, (TARGET_FRAMES,))
                self.assertEqual(data["lowest_foot_heights"].shape, (TARGET_FRAMES, 2))
                self.assertEqual(data["audio_beat_frames"].tolist(), [30, 90])
                self.assertEqual(data["body_intensity_peaks"].shape, (2,))
                self.assertEqual(data["support_beatness_peaks"].shape, (2,))
                self.assertEqual(data["upper_beatness_peaks"].shape, (2,))
                self.assertTrue(np.isfinite(data["body_intensity_envelope"]).all())
                self.assertTrue(np.isfinite(data["support_beatness_envelope"]).all())
                self.assertTrue(np.isfinite(data["upper_beatness_envelope"]).all())
                self.assertTrue(np.isin(data["support_contact"], [0.0, 1.0]).all())


if __name__ == "__main__":
    unittest.main()
