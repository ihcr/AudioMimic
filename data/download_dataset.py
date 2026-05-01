import shutil
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZipFile, is_zipfile


FILE_ID = "1RzqSbSnbMEwLUagV0GThfpm9JJXePGkV"
ARCHIVE_NAME = "edge_aistpp.zip"
DRIVE_ENTRY_URL = "https://docs.google.com/uc?export=download&id={file_id}"
DRIVE_DOWNLOAD_HOST = "drive.usercontent.google.com"
CHUNK_SIZE = 1024 * 1024


class GoogleDriveDownloadPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.form_action = None
        self.form_inputs = {}
        self._inside_download_form = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.form_action = attrs.get("action")
            self._inside_download_form = True
            return
        if tag == "input" and self._inside_download_form:
            name = attrs.get("name")
            if name:
                self.form_inputs[name] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form" and self._inside_download_form:
            self._inside_download_form = False


def build_opener():
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor())


def decode_body(response):
    charset = response.headers.get_content_charset() or "utf-8"
    return response.read().decode(charset, errors="replace")


def resolve_download_request(opener, file_id):
    with opener.open(DRIVE_ENTRY_URL.format(file_id=file_id)) as response:
        content_type = response.headers.get_content_type()
        if content_type != "text/html":
            return response.geturl(), {}
        body = decode_body(response)

    parser = GoogleDriveDownloadPageParser()
    parser.feed(body)
    if parser.form_action and parser.form_inputs:
        return parser.form_action, parser.form_inputs

    raise RuntimeError("Could not locate Google Drive download confirmation form.")


def stream_download(opener, url, params, output_path):
    request_url = url
    if params:
        query = urllib.parse.urlencode(params)
        request_url = f"{url}?{query}"

    with opener.open(request_url) as response, output_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file, CHUNK_SIZE)


def ensure_zip_file(path):
    if not is_zipfile(path):
        raise ValueError(f"Downloaded file is not a valid zip archive: {path}")


def extract_archive(path, output_dir):
    with ZipFile(path) as archive:
        archive.extractall(output_dir)


def download_and_extract_dataset(output_dir=None):
    output_dir = Path(output_dir or Path(__file__).resolve().parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME

    opener = build_opener()
    url, params = resolve_download_request(opener, FILE_ID)
    stream_download(opener, url, params, archive_path)
    ensure_zip_file(archive_path)
    extract_archive(archive_path, output_dir)
    return archive_path


def main():
    try:
        archive_path = download_and_extract_dataset()
    except Exception as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Downloaded and extracted {archive_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
