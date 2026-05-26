import importlib
import os
import unittest
from unittest.mock import patch


class JukemirlibCacheDirTests(unittest.TestCase):
    def test_cache_dir_honors_environment_override(self):
        with patch.dict(os.environ, {"JUKE_MIRLIB_CACHE_DIR": "/tmp/juke-cache"}, clear=False):
            import jukemirlib.constants as constants

            reloaded = importlib.reload(constants)
            self.assertEqual(reloaded.CACHE_DIR, "/tmp/juke-cache")


if __name__ == "__main__":
    unittest.main()
