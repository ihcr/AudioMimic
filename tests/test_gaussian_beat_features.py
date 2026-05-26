import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.audio_extraction.gaussian_beat_features import extract_folder


class GaussianBeatFeatureTests(unittest.TestCase):
    def test_derives_last_channel_from_wav2clip_stft_beat_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_dir = root / "wavs"
            combined_dir = root / "wav2clip_stft_beat_feats"
            beat_dir = root / "gaussian_beat_feats"
            wav_dir.mkdir()
            combined_dir.mkdir()
            (wav_dir / "clip.wav").write_bytes(b"placeholder")

            combined = np.zeros((150, 706), dtype=np.float32)
            combined[:, -1] = np.linspace(0.0, 1.0, 150, dtype=np.float32)
            np.save(combined_dir / "clip.npy", combined)

            extract_folder(wav_dir, beat_dir, source_feature_dir=combined_dir)

            beat = np.load(beat_dir / "clip.npy")
            self.assertEqual(beat.shape, (150, 1))
            np.testing.assert_allclose(beat[:, 0], combined[:, -1])


if __name__ == "__main__":
    unittest.main()
