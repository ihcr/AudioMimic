"""Record a SONIC camera stream without opening an OpenCV window."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_host", default="localhost")
    parser.add_argument("--camera_port", default=5555, type=int)
    parser.add_argument("--camera_name", default="third_person")
    parser.add_argument("--fps", default=30.0, type=float)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--output_mp4", required=True)
    parser.add_argument("--timestamps_json", required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("duration and fps must be positive")
    output = Path(args.output_mp4).expanduser().resolve()
    timestamps_path = Path(args.timestamps_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamps_path.parent.mkdir(parents=True, exist_ok=True)

    client = ComposedCameraClientSensor(
        server_ip=args.camera_host,
        port=int(args.camera_port),
    )
    first_sample = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        sample = client.read(blocking=False)
        if sample and args.camera_name in sample.get("images", {}):
            first_sample = sample
            break
        time.sleep(0.02)
    if first_sample is None:
        client.close()
        raise TimeoutError(f"camera {args.camera_name!r} was not received within 10 seconds")

    first_image = first_sample["images"][args.camera_name]
    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        client.close()
        raise RuntimeError(f"could not open video writer for {output}")

    started_monotonic = time.monotonic()
    started_wall = time.time()
    frame_times: list[float] = []
    period = 1.0 / float(args.fps)
    next_frame = started_monotonic
    last_image = first_image
    try:
        while True:
            now = time.monotonic()
            if now - started_monotonic >= float(args.duration):
                break
            sample = client.read(blocking=False)
            if sample and args.camera_name in sample.get("images", {}):
                last_image = sample["images"][args.camera_name]
            if now < next_frame:
                time.sleep(next_frame - now)
                continue
            image = last_image
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            writer.write(image)
            frame_times.append(time.monotonic() - started_monotonic)
            next_frame += period
    finally:
        writer.release()
        client.close()

    timestamps_path.write_text(
        json.dumps(
            {
                "camera": args.camera_name,
                "fps": float(args.fps),
                "started_wall_time": started_wall,
                "started_monotonic": started_monotonic,
                "frame_times_seconds": frame_times,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Recorded {len(frame_times)} frames to {output}")


if __name__ == "__main__":
    main(parse_args())
