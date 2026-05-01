import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import vis


class VisFfmpegTests(unittest.TestCase):
    def test_uses_system_ffmpeg_when_available(self):
        with patch.object(vis.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(vis.get_ffmpeg_exe(), "/usr/bin/ffmpeg")

    def test_falls_back_to_imageio_ffmpeg_when_system_ffmpeg_missing(self):
        fake_imageio_ffmpeg = SimpleNamespace(
            get_ffmpeg_exe=lambda: "/tmp/bundled-ffmpeg"
        )

        with patch.object(vis.shutil, "which", return_value=None), patch.dict(
            sys.modules, {"imageio_ffmpeg": fake_imageio_ffmpeg}
        ):
            self.assertEqual(vis.get_ffmpeg_exe(), "/tmp/bundled-ffmpeg")


if __name__ == "__main__":
    unittest.main()
