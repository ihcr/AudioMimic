import importlib
import pickle
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def load_module():
    module_name = "data.audio_extraction.beat_features"
    sys.modules.pop(module_name, None)
    fake_librosa = types.SimpleNamespace(
        load=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("audio unused")),
        beat=types.SimpleNamespace(
            beat_track=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("audio unused"))
        ),
    )
    fake_signal = types.SimpleNamespace(windows=types.SimpleNamespace(hann=lambda *args, **kwargs: None))
    fake_scipy = types.SimpleNamespace(signal=fake_signal)
    fake_torch = types.SimpleNamespace()
    original = {
        name: sys.modules.get(name)
        for name in ("librosa", "scipy", "scipy.signal", "torch")
    }
    sys.modules["librosa"] = fake_librosa
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.signal"] = fake_signal
    sys.modules["torch"] = fake_torch
    try:
        return importlib.import_module(module_name)
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def write_g1_motion(path):
    frames = 150
    t = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32)
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.sin(t) * 0.1
    root_pos[:, 2] = 0.8
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 0] = 1.0
    dof_pos = np.zeros((frames, 29), dtype=np.float32)
    dof_pos[:, 0] = np.sin(t * 2.0)
    dof_pos[:, 1] = np.cos(t * 2.0)
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "motion_format": "g1",
                "fps": 30.0,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "pos": root_pos,
                "q": np.concatenate([root_rot, dof_pos], axis=-1),
            },
            handle,
            pickle.HIGHEST_PROTOCOL,
        )


class G1BeatFeatureTests(unittest.TestCase):
    def test_g1_motion_beat_extraction_uses_robot_native_speed(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir, patch.object(
            module,
            "_forward_kinematics",
            side_effect=AssertionError("SMPL FK should not be used for G1 motions"),
        ):
            motion_path = Path(tmpdir) / "g1.pkl"
            write_g1_motion(motion_path)

            beat_frames, beat_mask, beat_dist, beat_spacing = (
                module.extract_motion_beats_from_motion_pkl(motion_path, fps=30, seq_len=150)
            )

        self.assertGreater(len(beat_frames), 0)
        self.assertTrue(np.all(beat_frames >= 0))
        self.assertTrue(np.all(beat_frames < 150))
        self.assertEqual(beat_mask.shape, (150,))
        self.assertEqual(beat_dist.shape, (150,))
        self.assertEqual(beat_spacing.shape, (150,))

    def test_g1_motion_beat_extraction_can_use_fk_keypoint_speed(self):
        module = load_module()

        frames = 150
        keypoints = np.zeros((frames, 2, 3), dtype=np.float32)
        keypoints[:, 0, 0] = np.sin(np.linspace(0.0, 4.0 * np.pi, frames))
        keypoints[:, 1, 0] = np.cos(np.linspace(0.0, 4.0 * np.pi, frames))

        with TemporaryDirectory() as tmpdir, patch.object(
            module,
            "forward_g1_kinematics",
            return_value={"keypoints": keypoints},
        ) as fk_mock:
            motion_path = Path(tmpdir) / "g1.pkl"
            model_path = Path(tmpdir) / "g1.xml"
            model_path.write_text("<mujoco/>", encoding="utf-8")
            write_g1_motion(motion_path)

            beat_frames, beat_mask, beat_dist, beat_spacing = (
                module.extract_motion_beats_from_motion_pkl(
                    motion_path,
                    fps=30,
                    seq_len=150,
                    g1_motion_beat_source="fk",
                    g1_fk_model_path=model_path,
                    g1_root_quat_order="xyzw",
                )
            )

        fk_mock.assert_called_once()
        self.assertGreater(len(beat_frames), 0)
        self.assertTrue(np.all(beat_frames >= 0))
        self.assertTrue(np.all(beat_frames < 150))
        self.assertEqual(beat_mask.shape, (150,))
        self.assertEqual(beat_dist.shape, (150,))
        self.assertEqual(beat_spacing.shape, (150,))


if __name__ == "__main__":
    unittest.main()
