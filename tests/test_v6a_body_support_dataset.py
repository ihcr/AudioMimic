import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from dataset.dance_dataset import AISTPPDataset
from feature_config import (
    BODY_INTENSITY_DIM,
    GAUSSIAN_BEAT_DIM,
    SUPPORT_BEATNESS_DIM,
    SUPPORT_CONTACT_DIM,
    UPPER_BEATNESS_DIM,
    WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_STFT_BEAT_DIM,
)


class V6aBodySupportDatasetTests(unittest.TestCase):
    def test_dataset_loads_body_support_controls_from_store(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wav2clip_path = tmp_path / "clip.npy"
            gaussian_path = tmp_path / "clip_gaussian.npy"
            motion_control_path = tmp_path / "clip_control.npz"
            np.save(wav2clip_path, np.ones((150, WAV2CLIP_STFT_BEAT_DIM), dtype=np.float32))
            np.save(gaussian_path, np.ones((150, GAUSSIAN_BEAT_DIM), dtype=np.float32))
            np.savez(
                motion_control_path,
                body_intensity_envelope=np.full((150, BODY_INTENSITY_DIM), 0.2, dtype=np.float32),
                support_beatness_envelope=np.full((150, SUPPORT_BEATNESS_DIM), 0.4, dtype=np.float32),
                upper_beatness_envelope=np.full((150, UPPER_BEATNESS_DIM), 0.6, dtype=np.float32),
                support_contact=np.ones((150, SUPPORT_CONTACT_DIM), dtype=np.float32),
            )
            fake_data = {
                "pos": np.zeros((1, 150, 3), dtype=np.float32),
                "q": np.zeros((1, 150, 33), dtype=np.float32),
                "filenames": [str(wav2clip_path)],
                "wavs": ["clip.wav"],
                "structured_condition_paths": [
                    {
                        "wav2clip_stft_beat": str(wav2clip_path),
                        "gaussian_beat": str(gaussian_path),
                        "motion_control": str(motion_control_path),
                    }
                ],
            }
            fake_pose = torch.zeros((1, 150, 38), dtype=torch.float32)

            with patch.object(AISTPPDataset, "load_aistpp", return_value=fake_data), patch.object(
                AISTPPDataset,
                "process_dataset",
                return_value=fake_pose,
            ):
                dataset = AISTPPDataset(
                    data_path="unused",
                    backup_path=tmpdir,
                    train=True,
                    feature_type=WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
                    force_reload=True,
                )

            self.assertTrue(Path(dataset.motion_control_store_path).is_file())
            _, cond, _, _ = dataset[0]
            self.assertEqual(tuple(cond["semantic"]["wav2clip"].shape), (150, 512))
            self.assertEqual(tuple(cond["control"]["gaussian_beat"].shape), (150, 1))
            self.assertEqual(tuple(cond["control"]["body_intensity"].shape), (150, 1))
            self.assertEqual(tuple(cond["control"]["support_beatness"].shape), (150, 1))
            self.assertEqual(tuple(cond["control"]["upper_beatness"].shape), (150, 1))
            self.assertEqual(tuple(cond["control"]["support_contact"].shape), (150, 2))
            self.assertAlmostEqual(float(cond["control"]["body_intensity"].mean()), 0.2, places=6)
            self.assertAlmostEqual(float(cond["control"]["support_beatness"].mean()), 0.4, places=6)
            self.assertAlmostEqual(float(cond["control"]["upper_beatness"].mean()), 0.6, places=6)
            self.assertAlmostEqual(float(cond["control"]["support_contact"].mean()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
