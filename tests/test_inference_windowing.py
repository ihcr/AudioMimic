import unittest
from unittest.mock import patch

import test as inference_script


class InferenceWindowingTests(unittest.TestCase):
    def test_pick_sample_window_uses_full_short_sequence(self):
        start, window = inference_script.pick_sample_window(2, 11)
        self.assertEqual(start, 0)
        self.assertEqual(window, 2)

    def test_pick_sample_window_randomizes_when_long_enough(self):
        with patch.object(inference_script.random, "randint", return_value=3) as mock_randint:
            start, window = inference_script.pick_sample_window(20, 11)
        mock_randint.assert_called_once_with(0, 9)
        self.assertEqual(start, 3)
        self.assertEqual(window, 11)

    def test_pick_sample_window_rejects_empty_sequences(self):
        with self.assertRaisesRegex(ValueError, "No audio slices were produced"):
            inference_script.pick_sample_window(0, 11)


if __name__ == "__main__":
    unittest.main()
