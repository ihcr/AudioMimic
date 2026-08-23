"""Build controlled corruptions for the held-out AIST++/FineDance GT suite."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from eval.build_gt_motion_corruptions import _corrupt, _copy_payload, _stable_seed


DEFAULT_GT = Path("eval/benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_metrics.json")
DEFAULT_OUTPUT = Path("eval/benchmark_v1/gt/stratified_corruptions_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-suite", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    suite_path = args.gt_suite.expanduser().resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    variants = ("clean", "jitter", "lowpass", "freeze", "repeat")
    levels = ("clean", "low", "medium", "high")

    for source_record in suite["records"]:
        sequence_id = source_record["sequence_id"]
        dataset = source_record["dataset"]
        source_motion = Path(source_record["motion_path"])
        audio_path = source_record["audio_path"]
        clean_payload = _copy_payload(source_motion, audio_path, 0.0)
        for variant in variants:
            current_levels = ("clean",) if variant == "clean" else levels[1:]
            for level in current_levels:
                payload, details = _corrupt(
                    clean_payload,
                    variant,
                    level,
                    _stable_seed(args.seed, f"{dataset}:{sequence_id}", variant, level),
                )
                destination = output_dir / dataset / variant / level / f"{sequence_id}.pkl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                records.append(
                    {
                        "dataset": dataset,
                        "sequence_id": sequence_id,
                        "style": source_record.get("style", []),
                        "variant": variant,
                        "severity": level,
                        "path": str(destination),
                        "audio_path": audio_path,
                        "frames": int(len(payload["dof_pos"])),
                        "fps": float(payload["fps"]),
                        "details": details,
                    }
                )

    manifest = {
        "schema_version": "gt_stratified_corruptions_v1",
        "source_gt_suite": str(suite_path),
        "seed": args.seed,
        "sequence_count": len(suite["records"]),
        "record_count": len(records),
        "variants": variants,
        "levels": levels[1:],
        "records": records,
    }
    (output_dir / "corruption_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(records)} stratified corruption records from {len(suite['records'])} GT sequences")
    print(f"Manifest: {output_dir / 'corruption_manifest.json'}")


if __name__ == "__main__":
    main(parse_args())
