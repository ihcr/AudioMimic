import argparse
import json
from pathlib import Path


METRIC_KEYS = ("PFC", "BAS", "BAP", "Distg", "Distk")
COUNT_KEYS = ("num_motion_files", "num_scored_files", "num_full_songs")


def load_row(label, metrics_path):
    metrics_path = Path(metrics_path)
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    row = {"label": label, "metrics_path": str(metrics_path)}
    for key in (*METRIC_KEYS, *COUNT_KEYS):
        row[key] = metrics.get(key)
    return row


def format_value(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "n/a"
    return str(value)


def render_markdown(rows):
    headers = ["Model", "Files", "Scored", "Songs", *METRIC_KEYS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            row["label"],
            row.get("num_motion_files"),
            row.get("num_scored_files"),
            row.get("num_full_songs"),
            *(row.get(key) for key in METRIC_KEYS),
        ]
        lines.append("| " + " | ".join(format_value(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def write_comparison(entries, json_path, markdown_path):
    rows = [load_row(label, path) for label, path in entries]
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2, sort_keys=True)
    markdown_path.write_text(render_markdown(rows), encoding="utf-8")
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Comparison entry as label=metrics.json",
    )
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--markdown_path", required=True)
    return parser.parse_args()


def parse_entry(entry):
    if "=" not in entry:
        raise ValueError(f"Expected label=path entry, got {entry!r}")
    label, path = entry.split("=", 1)
    if not label:
        raise ValueError(f"Missing label in entry {entry!r}")
    return label, path


if __name__ == "__main__":
    args = parse_args()
    rows = write_comparison(
        entries=[parse_entry(entry) for entry in args.entry],
        json_path=args.json_path,
        markdown_path=args.markdown_path,
    )
    print(render_markdown(rows), end="")
