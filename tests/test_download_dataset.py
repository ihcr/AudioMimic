import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from data.download_dataset import (
    DRIVE_DOWNLOAD_HOST,
    GoogleDriveDownloadPageParser,
    ensure_zip_file,
)


class GoogleDriveDownloadTests(unittest.TestCase):
    def test_parser_extracts_download_form(self):
        parser = GoogleDriveDownloadPageParser()
        parser.feed(
            """
            <html><body>
              <form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
                <input type="hidden" name="id" value="abc123">
                <input type="hidden" name="export" value="download">
                <input type="hidden" name="confirm" value="t">
                <input type="hidden" name="uuid" value="uuid-1">
              </form>
            </body></html>
            """
        )

        self.assertEqual(parser.form_action, f"https://{DRIVE_DOWNLOAD_HOST}/download")
        self.assertEqual(
            parser.form_inputs,
            {"id": "abc123", "export": "download", "confirm": "t", "uuid": "uuid-1"},
        )

    def test_ensure_zip_file_rejects_html_stub(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "edge_aistpp.zip"
            path.write_text("<html>not a zip</html>", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Downloaded file is not a valid zip"):
                ensure_zip_file(path)

    def test_ensure_zip_file_accepts_valid_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "edge_aistpp.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("dataset/example.txt", "ok")

            ensure_zip_file(path)


if __name__ == "__main__":
    unittest.main()
