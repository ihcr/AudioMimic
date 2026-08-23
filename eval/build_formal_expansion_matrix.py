"""Build the target matrix for the balanced multi-song model expansion."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONLINE = Path("/home/tianhup/Musics2Dance-prior-dev/onlinegeneratedmotion")
DEFAULT_OUTPUT = ROOT / "eval/benchmark_v1/formal"
MODELS = ("M0", "M2", "M3", "M4")
SONGS = ("012", "065", "098")
SEEDS = (1234, 2345, 3456)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, default=DEFAULT_ONLINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _pkl_candidates(root: Path, model: str, song: str, seed: int) -> list[Path]:
    folder = {
        "M0": root / "uncond_m0",
        "M2": root / "m2_predicted_fms",
        "M4": root / "m4_oracle_fms",
    }.get(model)
    if folder is None or not folder.is_dir():
        return []
    return sorted(folder.glob(f"*sample{seed}*song{song}.pkl"))


def _existing_exec() -> dict[tuple[str, str, int], str]:
    path = ROOT / "eval/motion_music_execution/gt_calibrated_m0_m2_m4_song098_v2/execution_metrics.json"
    result: dict[tuple[str, str, int], str] = {}
    if path.is_file():
        for row in json.loads(path.read_text(encoding="utf-8")):
            match = re.search(r"seed(\d+)", row.get("run_id", ""))
            if match:
                result[(row["route"], str(row["sequence_id"]), int(match.group(1)))] = row["run_id"]
    corrected = ROOT / "eval/mrt2_comparison"
    for song in ("012", "065"):
        p = corrected / f"m3_{song}_sonic_measured_corrected_portable.pkl"
        if p.is_file():
            result[("M3", song, 0)] = str(p)
    return result


def main() -> None:
    args = parse_args()
    online_root = args.online_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing_exec = _existing_exec()
    rows = []
    for model in MODELS:
        for song in SONGS:
            for seed in SEEDS:
                candidates = _pkl_candidates(online_root, model, song, seed)
                ref_status = "available" if candidates else "pending"
                ref_paths = ";".join(str(p) for p in candidates)
                exec_key = (model, song, seed)
                exec_status = "available" if exec_key in existing_exec else "pending"
                rows.append({
                    "model": model,
                    "sequence_id": song,
                    "sampling_seed": seed,
                    "stage": "M_ref",
                    "status": ref_status,
                    "artifact_candidates": ref_paths,
                    "paired_execution_status": exec_status,
                    "notes": "M3 pilot uses unrecorded seed; keep separate from balanced matrix." if model == "M3" else "",
                })
                rows.append({
                    "model": model,
                    "sequence_id": song,
                    "sampling_seed": seed,
                    "stage": "M_exec",
                    "status": exec_status,
                    "artifact_candidates": existing_exec.get(exec_key, ""),
                    "paired_execution_status": "requires_M_ref",
                    "notes": "M3 pilot execution is indexed separately with seed=unknown." if model == "M3" else "",
                })

    fields = list(rows[0])
    with (output / "EXPANSION_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "formal_model_expansion_matrix_v1",
        "target_models": list(MODELS),
        "target_sequences": list(SONGS),
        "target_sampling_seeds": list(SEEDS),
        "stages": ["M_ref", "M_exec"],
        "rows": rows,
        "counts": {
            "total": len(rows),
            "reference_available": sum(r["stage"] == "M_ref" and r["status"] == "available" for r in rows),
            "execution_available": sum(r["stage"] == "M_exec" and r["status"] == "available" for r in rows),
            "pending": sum(r["status"] == "pending" for r in rows),
        },
    }
    (output / "EXPANSION_MATRIX.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
