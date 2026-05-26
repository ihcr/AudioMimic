#!/usr/bin/env python
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path


TRAIN_EPOCH_RE = re.compile(
    r"train_epoch=(?P<epoch>\d+)\s+seconds=(?P<seconds>[0-9.]+)"
)
TRAIN_TOTAL_RE = re.compile(r"Train\s+(?P<epoch>\d+)/(?P<total>\d+)")


def format_duration(seconds):
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def parse_log(path):
    current_epoch = None
    total_epoch = None
    completed = []
    for line in Path(path).read_text(errors="replace").splitlines():
        total_match = TRAIN_TOTAL_RE.search(line)
        if total_match:
            current_epoch = int(total_match.group("epoch"))
            total_epoch = int(total_match.group("total"))
        epoch_match = TRAIN_EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group("epoch"))
            completed.append(
                (
                    int(epoch_match.group("epoch")),
                    float(epoch_match.group("seconds")),
                )
            )
    return current_epoch, total_epoch, completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    current_epoch, total_epoch, completed = parse_log(args.log_path)
    if current_epoch is None or total_epoch is None:
        raise SystemExit(f"No Train x/y progress found in {args.log_path}")
    if not completed:
        raise SystemExit(f"No completed train_epoch timing lines found in {args.log_path}")

    recent = completed[-args.window :]
    avg_epoch_seconds = sum(seconds for _, seconds in recent) / len(recent)
    last_completed_epoch = completed[-1][0]
    active_epoch = max(current_epoch, last_completed_epoch)
    remaining_epochs = max(total_epoch - last_completed_epoch, 0)
    eta_seconds = avg_epoch_seconds * remaining_epochs
    finish = datetime.now() + timedelta(seconds=eta_seconds)
    progress_percent = 100.0 * last_completed_epoch / max(total_epoch, 1)

    print(f"log={args.log_path}")
    print(f"last_completed_epoch={last_completed_epoch}/{total_epoch}")
    print(f"active_epoch={active_epoch}/{total_epoch}")
    print(f"progress_percent={progress_percent:.2f}")
    print(f"recent_avg_epoch_seconds={avg_epoch_seconds:.2f}")
    print(f"remaining_epochs={remaining_epochs}")
    print(f"eta={format_duration(eta_seconds)}")
    print(f"estimated_finish={finish.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
