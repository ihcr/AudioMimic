"""Concatenate headless reference chunks and compare them with SONIC tracking."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_chunks_dir", required=True)
    parser.add_argument("--sonic_video", required=True)
    parser.add_argument("--output_mp4", required=True)
    parser.add_argument("--duration_seconds", type=float, default=60.0)
    parser.add_argument(
        "--sonic_start_seconds",
        type=float,
        default=0.0,
        help="discard this initial idle portion of the SONIC recording",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--left_label", default="Generated reference")
    parser.add_argument("--right_label", default="SONIC execution")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(args: argparse.Namespace) -> None:
    if args.sonic_start_seconds < 0:
        raise ValueError("sonic_start_seconds must be non-negative")
    chunks_dir = Path(args.reference_chunks_dir).expanduser().resolve()
    chunks = sorted(chunks_dir.glob("chunk_*.mp4"))
    if not chunks:
        raise FileNotFoundError(f"No chunk_*.mp4 files in {chunks_dir}")
    sonic_video = Path(args.sonic_video).expanduser().resolve()
    if not sonic_video.is_file():
        raise FileNotFoundError(sonic_video)
    output = Path(args.output_mp4).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sonic_reference_") as temporary_dir:
        temporary = Path(temporary_dir)
        concat_list = temporary / "chunks.txt"
        concat_list.write_text("".join(f"file '{chunk}'\n" for chunk in chunks), encoding="utf-8")
        reference_video = temporary / "reference.mp4"
        run([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(reference_video),
        ])

        width, height = int(args.width), int(args.height)
        left_text = args.left_label.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        right_text = args.right_label.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        left_label = (
            f"drawtext=fontfile={FONT}:text='{left_text}':"
            "x=18:y=18:fontsize=24:fontcolor=white:borderw=3:bordercolor=black"
        )
        right_label = (
            f"drawtext=fontfile={FONT}:text='{right_text}':"
            "x=18:y=18:fontsize=24:fontcolor=white:borderw=3:bordercolor=black"
        )
        filters = (
            f"[0:v]fps=30,scale={width}:{height},{left_label}[left];"
            f"[1:v]fps=30,trim=start={args.sonic_start_seconds:g},setpts=PTS-STARTPTS,"
            f"scale={width}:{height},{right_label}[right];"
            "[left][right]hstack=inputs=2[v]"
        )
        run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(reference_video),
            "-i", str(sonic_video), "-filter_complex", filters, "-map", "[v]",
            "-t", f"{args.duration_seconds:g}", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ])
    print(f"Wrote comparison video to {output}")


if __name__ == "__main__":
    main(parse_args())
