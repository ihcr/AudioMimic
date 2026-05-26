import argparse
import json
from pathlib import Path

from eval.eval_pfc import compute_physical_score


def audit_pfc_sources(source_map):
    report = {}
    for label, motion_path in source_map.items():
        report[label] = compute_physical_score(motion_path, return_details=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Named source in the form label=path",
    )
    parser.add_argument("--output_path", default="")
    return parser.parse_args()


def parse_source_map(source_args):
    source_map = {}
    for item in source_args:
        label, path = item.split("=", 1)
        source_map[label] = path
    return source_map


if __name__ == "__main__":
    args = parse_args()
    report = audit_pfc_sources(parse_source_map(args.source))
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
