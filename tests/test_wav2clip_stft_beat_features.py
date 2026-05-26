import unittest

import numpy as np

from data.audio_extraction.wav2clip_stft_beat_features import (
    TARGET_FRAMES,
    _crop_or_pad,
    _coerce_wav2clip_embeddings,
    _resample_time_axis,
    gaussian_beat_from_indices,
)


class Wav2ClipStftBeatFeatureTests(unittest.TestCase):
    def test_gaussian_beat_has_peaks_and_expected_shape(self):
        beat = gaussian_beat_from_indices([10, 40], num_frames=60, alpha=0.25)
        self.assertEqual(beat.shape, (60, 1))
        self.assertAlmostEqual(float(beat[10, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(beat[40, 0]), 1.0, places=6)
        self.assertLess(float(beat[25, 0]), 1.0)

    def test_gaussian_beat_without_beats_is_zero(self):
        beat = gaussian_beat_from_indices([], num_frames=12)
        self.assertEqual(beat.shape, (12, 1))
        self.assertTrue(np.allclose(beat, 0.0))

    def test_crop_or_pad_and_resample_time_axis(self):
        short = np.ones((3, 2), dtype=np.float32)
        padded = _crop_or_pad(short, num_frames=5)
        self.assertEqual(padded.shape, (5, 2))
        self.assertTrue(np.allclose(padded[:3], 1.0))
        self.assertTrue(np.allclose(padded[3:], 0.0))

        long = np.ones((TARGET_FRAMES + 2, 2), dtype=np.float32)
        cropped = _crop_or_pad(long)
        self.assertEqual(cropped.shape, (TARGET_FRAMES, 2))

        source = np.stack([np.arange(4), np.arange(4) * 2], axis=-1).astype(np.float32)
        resampled = _resample_time_axis(source, num_frames=8)
        self.assertEqual(resampled.shape, (8, 2))
        self.assertAlmostEqual(float(resampled[0, 0]), 0.0)
        self.assertAlmostEqual(float(resampled[-1, 0]), 3.0)

    def test_wav2clip_channel_first_embeddings_are_transposed(self):
        channel_first = np.zeros((512, 7), dtype=np.float32)
        coerced = _coerce_wav2clip_embeddings(channel_first)
        self.assertEqual(coerced.shape, (7, 512))

        batched_channel_first = np.zeros((1, 512, 7), dtype=np.float32)
        coerced = _coerce_wav2clip_embeddings(batched_channel_first)
        self.assertEqual(coerced.shape, (7, 512))


if __name__ == "__main__":
    unittest.main()
