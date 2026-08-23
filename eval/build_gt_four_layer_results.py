"""Consolidate O-Human -> O-G1 GT/GMR results for the formal benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "eval/benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_metrics.json"
DEFAULT_RETARGET = ROOT / "eval/benchmark_v1/gt/retargeting_loss_v1/retargeting_loss_metrics.json"
DEFAULT_PAIRING = ROOT / "eval/benchmark_v1/gt/finedance_music_pairing_v1/pairing_metrics.json"
DEFAULT_AIST_PAIRING = ROOT / "eval/benchmark_v1/gt/aist_music_pairing_v1/pairing_metrics.json"
DEFAULT_OUTPUT = ROOT / "eval/benchmark_v1/formal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-suite", type=Path, default=DEFAULT_GT)
    parser.add_argument("--retargeting", type=Path, default=DEFAULT_RETARGET)
    parser.add_argument("--finedance-pairing", type=Path, default=DEFAULT_PAIRING)
    parser.add_argument("--aist-pairing", type=Path, default=DEFAULT_AIST_PAIRING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _metric(music: dict, key: str) -> float | None:
    value = music.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _f1(music: dict) -> float | None:
    value = music.get("event_f1", {}).get("f1")
    return float(value) if isinstance(value, (int, float)) else None


def _row(gt: dict, retarget: dict) -> dict:
    human = retarget["source_music"]
    g1 = retarget["target_music"]
    values = {
        "bas": (_metric(human, "bas_music_to_motion"), _metric(g1, "bas_music_to_motion")),
        "beat_f1": (_f1(human), _f1(g1)),
        "impact_corr": (_metric(human, "impact_best_correlation"), _metric(g1, "impact_best_correlation")),
        "abs_impact_lag": (abs(_metric(human, "impact_best_lag_seconds") or 0.0), abs(_metric(g1, "impact_best_lag_seconds") or 0.0)),
        "tempo_error": (_metric(human, "tempo_abs_error_bpm"), _metric(g1, "tempo_abs_error_bpm")),
        "phase_error": (_metric(human.get("phase", {}), "mean_phase_error_cycles"), _metric(g1.get("phase", {}), "mean_phase_error_cycles")),
    }
    row = {
        "sample_id": f"{gt['dataset']}:{gt['sequence_id']}",
        "dataset": gt["dataset"],
        "sequence_id": gt["sequence_id"],
        "audio_path": gt["audio_path"],
        "audio_bpm": float(gt["music"]["audio_bpm"]),
        "duration_seconds": float(gt["duration_seconds"]),
        "o_human_motion_path": retarget["source_motion_path"],
        "o_g1_motion_path": retarget["target_motion_path"],
        "activity_best_corr": retarget["retarget_loss"]["activity_curve"]["best"],
        "root_speed_best_corr": retarget["retarget_loss"]["root_speed_curve"]["best"],
        "activity_rms_ratio_g1_over_human": retarget["retarget_loss"]["activity_rms_ratio_target_over_source"],
        "root_speed_rms_ratio_g1_over_human": retarget["retarget_loss"]["root_speed_rms_ratio_target_over_source"],
    }
    for name, (human_value, g1_value) in values.items():
        row[f"o_human_{name}"] = human_value
        row[f"o_g1_{name}"] = g1_value
        row[f"gmr_delta_{name}"] = None if human_value is None or g1_value is None else g1_value - human_value
    row["o_g1_motion_energy"] = gt["quality"]["motion_energy_rad2_s2"]
    row["o_g1_jerk_p95"] = gt["quality"]["joint_jerk_abs_rad_s3"]["p95"]
    row["o_g1_fsr_proxy"] = gt["quality"]["physical"]["fsr_ground_calibrated_proxy"]
    row["o_g1_pfc_proxy"] = gt["quality"]["physical"]["pfc_proxy"]
    return row


def main(args: argparse.Namespace) -> None:
    gt_payload = json.loads(args.gt_suite.expanduser().resolve().read_text(encoding="utf-8"))
    retarget_payload = json.loads(args.retargeting.expanduser().resolve().read_text(encoding="utf-8"))
    retarget = {(r["dataset"], r["sequence_id"]): r for r in retarget_payload["records"]}
    rows = [_row(gt, retarget[(gt["dataset"], gt["sequence_id"])]) for gt in gt_payload["records"]]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "gt_human_to_g1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "gt_human_to_g1.json").write_text(
        json.dumps({
            "schema_version": "formal_gt_human_to_g1_results_v1",
            "gt_suite": str(args.gt_suite.expanduser().resolve()),
            "retargeting_audit": str(args.retargeting.expanduser().resolve()),
            "rows": rows,
            "interpretation": "O-Human and O-G1 are the two GT layers. Deltas are GMR/retargeting changes, not generator errors.",
        }, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# O-Human to O-G1 Formal GT Results",
        "",
        "O-Human is the original SMPL/SMPLH paired motion. O-G1 is the GMR-retargeted G1 motion",
        "and the training domain of the diffusion generator. This table measures GMR changes before",
        "any generator or SONIC execution result is introduced.",
        "",
        "| dataset | n | BAS human | BAS G1 | delta | Beat F1 human | Beat F1 G1 | delta | activity corr | root corr | activity ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        mean = lambda key: sum(float(r[key]) for r in group if r[key] is not None) / max(sum(r[key] is not None for r in group), 1)
        lines.append(
            f"| {dataset} | {len(group)} | {mean('o_human_bas'):.4f} | {mean('o_g1_bas'):.4f} | {mean('gmr_delta_bas'):+.4f} | "
            f"{mean('o_human_beat_f1'):.4f} | {mean('o_g1_beat_f1'):.4f} | {mean('gmr_delta_beat_f1'):+.4f} | "
            f"{mean('activity_best_corr'):.4f} | {mean('root_speed_best_corr'):.4f} | {mean('activity_rms_ratio_g1_over_human'):.4f} |"
        )
    pairing_paths = [
        ("FineDance", args.finedance_pairing.expanduser().resolve()),
        ("AIST++", args.aist_pairing.expanduser().resolve()),
    ]
    for pairing_name, pairing_path in pairing_paths:
        if pairing_path.is_file():
            pairing = json.loads(pairing_path.read_text(encoding="utf-8"))["summary"]
            lines += [
                "",
                f"## {pairing_name} music-pairing integrity check",
                "",
                "This is a dataset-integrity diagnostic, separate from GMR loss. The same-ID",
                "audio is compared with a cyclic wrong-song control and all other test songs.",
                "",
                f"- paired-vs-wrong best-lag correlation margin: **{pairing['mean_margin']['best_lag_corr']:.4f}**",
                f"- paired-vs-wrong event-F1 margin: **{pairing['mean_margin']['event_f1']:.4f}**",
                f"- paired correlation top-1 rate among wrong songs: **{pairing['all_wrong_song_control']['top1_rate']:.4f}**",
                "",
                "The paired distribution is a positive reference and the wrong-song",
                "distribution is a negative control; neither is a perfect human-quality score.",
                f"Source: `{pairing_path}`.",
            ]
    (output_dir / "GT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} O-Human/O-G1 records to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
