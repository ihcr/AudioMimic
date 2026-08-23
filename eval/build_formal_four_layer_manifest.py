"""Build the four-layer formal Music-to-G1 experiment manifest.

The four layers are parallel evaluation objects under one music identity:
O-Human (SMPL/SMPLH), O-G1 (GMR target), M-ref (generator G1), and M-exec
(SONIC G1). The generator is trained in the O-G1 domain and is not retargeted
again at inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "eval/benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_metrics.json"
DEFAULT_RETARGET = ROOT / "eval/benchmark_v1/gt/retargeting_loss_v1/retargeting_loss_metrics.json"
DEFAULT_OUTPUT = ROOT / "eval/benchmark_v1/formal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-suite", type=Path, default=DEFAULT_GT)
    parser.add_argument("--retargeting", type=Path, default=DEFAULT_RETARGET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _tempo_bin(bpm: float) -> str:
    if bpm < 90.0:
        return "slow_<90"
    if bpm < 120.0:
        return "medium_90-120"
    return "fast_>=120"


def _style(record: dict) -> str:
    if record.get("style"):
        return "+".join(record["style"])
    sequence_id = record["sequence_id"]
    if sequence_id.startswith("g") and len(sequence_id) > 2:
        return f"AIST_genre_{sequence_id[1:3]}"
    return "unknown"


def _exists(path: str | None) -> bool:
    return bool(path) and Path(path).expanduser().is_file()


def _gt_records(gt_path: Path, retarget_path: Path) -> list[dict]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))["records"]
    retarget = {
        row["sequence_id"]: row
        for row in json.loads(retarget_path.read_text(encoding="utf-8"))["records"]
    }
    records = []
    for row in gt:
        target = retarget[row["sequence_id"]]
        bpm = float(row["music"]["audio_bpm"])
        records.append(
            {
                "sample_id": f"{row['dataset']}:{row['sequence_id']}",
                "dataset": row["dataset"],
                "sequence_id": row["sequence_id"],
                "tempo_bin": _tempo_bin(bpm),
                "style": _style(row),
                "audio_path": row["audio_path"],
                "duration_seconds": row["duration_seconds"],
                "layers": {
                    "O_Human": {
                        "status": "available" if _exists(target["source_motion_path"]) else "external_or_missing",
                        "format": "SMPL_or_SMPLH",
                        "motion_path": target["source_motion_path"],
                    },
                    "O_G1": {
                        "status": "available" if _exists(row["motion_path"]) else "external_or_missing",
                        "format": "G1",
                        "motion_path": row["motion_path"],
                    },
                },
                "retargeting_record": str(retarget_path),
            }
        )
    return records


def _model_records(gt_records: list[dict]) -> list[dict]:
    by_id = {row["sequence_id"]: row for row in gt_records}
    records = []

    reference_metrics = ROOT / "eval/motion_music_execution/gt_calibrated_m0_m2_m4_song098_v2/reference_metrics.json"
    execution_root = ROOT / "eval/generation_to_execution_gap"
    if reference_metrics.is_file():
        for row in json.loads(reference_metrics.read_text(encoding="utf-8")):
            if row.get("route") != "M2" or row.get("sequence_id") not in by_id:
                continue
            base = by_id[row["sequence_id"]]
            run_id = row["motion_id"]
            execution_run = {
                1234: "m2_song098_seed1234_full_rate100_aligned_r03",
            }.get(row.get("training_seed"))
            execution_dir = execution_root / execution_run if execution_run else None
            records.append(
                {
                    "sample_id": f"{base['sample_id']}:{run_id}",
                    "model": "M2",
                    "condition": "predicted_future_music",
                    "dataset": base["dataset"],
                    "sequence_id": base["sequence_id"],
                    "tempo_bin": base["tempo_bin"],
                    "style": base["style"],
                    "audio_path": base["audio_path"],
                    "training_seed": row.get("training_seed"),
                    "sampling_seed": row.get("sampling_seed"),
                    "layers": {
                        "O_Human": base["layers"]["O_Human"],
                        "O_G1": base["layers"]["O_G1"],
                        "M_ref": {
                            "status": "available" if _exists(row.get("motion_path")) else "external_or_missing",
                            "format": "G1",
                            "motion_path": row.get("motion_path"),
                        },
                        "M_exec": {
                            "status": "metrics_artifact_only" if execution_dir and execution_dir.is_dir() else "pending",
                            "format": "SONIC_G1",
                            "artifact_dir": str(execution_dir) if execution_dir else None,
                        },
                    },
                }
            )

    for path, sequence_id in (
        (ROOT / "eval/mrt2_comparison/m3_012_generator_portable.pkl", "012"),
        (ROOT / "eval/mrt2_comparison/m3_065_generator_portable.pkl", "065"),
    ):
        pair_path = ROOT / "eval/mrt2_metrics" / f"m3_{sequence_id}_pair_metrics.json"
        if sequence_id not in by_id or not pair_path.is_file():
            continue
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        base = by_id[sequence_id]
        execution_path = ROOT / "eval/mrt2_comparison" / f"m3_{sequence_id}_sonic_measured_corrected_portable.pkl"
        records.append(
            {
                "sample_id": f"{base['sample_id']}:M3-{sequence_id}",
                "model": "M3",
                "condition": "music_sidecar_paired",
                "dataset": base["dataset"],
                "sequence_id": sequence_id,
                "tempo_bin": base["tempo_bin"],
                "style": base["style"],
                "audio_path": base["audio_path"],
                "training_seed": pair.get("training_seed"),
                "sampling_seed": pair.get("sampling_seed"),
                "layers": {
                    "O_Human": base["layers"]["O_Human"],
                    "O_G1": base["layers"]["O_G1"],
                    "M_ref": {
                        "status": "available" if path.is_file() else "external_or_missing",
                        "format": "G1",
                        "motion_path": str(path),
                    },
                    "M_exec": {
                        "status": "available" if execution_path.is_file() else "pending",
                        "format": "SONIC_G1",
                        "motion_path": str(execution_path),
                    },
                },
            }
        )
    return records


def main(args: argparse.Namespace) -> None:
    gt_path = args.gt_suite.expanduser().resolve()
    retarget_path = args.retargeting.expanduser().resolve()
    gt_records = _gt_records(gt_path, retarget_path)
    model_records = _model_records(gt_records)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "formal_music_to_g1_four_layer_manifest_v1",
        "description": "O-Human/O-G1 GT and M-ref/M-exec model records under one music identity.",
        "generator_domain": "G1; generator is trained on O-G1 and is not retargeted again at inference.",
        "gt_suite": str(gt_path),
        "retargeting_audit": str(retarget_path),
        "layers": ["O_Human", "O_G1", "M_ref", "M_exec"],
        "gt_records": gt_records,
        "model_records": model_records,
        "pending_policy": "Missing M_ref/M_exec artifacts are explicit pending states and cannot enter a result table.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Formal Music-to-G1 Four-Layer Benchmark\n\n"
            "The generator directly outputs G1 motion. GMR is only used to create O-G1 GT.\n\n"
            "- `O_Human`: original SMPL/SMPLH paired motion/audio.\n"
            "- `O_G1`: GMR/retargeted G1 target and generator training domain.\n"
            "- `M_ref`: generator-produced G1 reference.\n"
            "- `M_exec`: SONIC-executed G1 motion.\n\n"
            f"GT records: {len(gt_records)}\n\n"
            f"Existing model records: {len(model_records)}\n\n"
            "Only records with all required artifacts and a fixed shared clock may enter the formal result table.\n"
        )
    print(f"wrote {len(gt_records)} GT records and {len(model_records)} model records to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
