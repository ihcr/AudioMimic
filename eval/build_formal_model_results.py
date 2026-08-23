"""Consolidate existing model reference/execution metrics for the formal benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "eval/benchmark_v1/formal"
REFERENCE = ROOT / "eval/motion_music_execution/gt_calibrated_m0_m2_m4_song098_v2/reference_metrics.json"
EXECUTION = ROOT / "eval/motion_music_execution/gt_calibrated_m0_m2_m4_song098_v2/execution_metrics.json"
M3_PAIRS = [
    ROOT / "eval/mrt2_metrics/m3_012_pair_metrics.json",
    ROOT / "eval/mrt2_metrics/m3_065_pair_metrics.json",
]


FIELDS = [
    "route",
    "sequence_id",
    "seed",
    "stage",
    "run_id",
    "condition",
    "motion_energy",
    "joint_amplitude_median",
    "joint_velocity_p95",
    "joint_acceleration_p95",
    "joint_jerk_p95",
    "static_ratio",
    "repeated_pose_ratio",
    "c4_position_jump_p95",
    "c4_velocity_jump_p95",
    "root_planar_displacement",
    "root_planar_speed_p95",
    "root_height_min",
    "fsr_proxy",
    "pfc_proxy",
    "ground_penetration_ratio",
    "foot_contact_mean",
    "speed_corr",
    "speed_lag",
    "impact_zero_corr",
    "impact_corr",
    "impact_abs_lag",
    "bas_music_to_motion",
    "bas_motion_to_music",
    "audio_beats",
    "motion_beats",
    "motion_bpm",
    "motion_energy_retention",
    "median_amplitude_statistic_ratio",
    "jerk_p95_ratio",
    "fsr_ground_calibrated_proxy_change",
    "onset_impact_corr_retention",
    "onset_impact_corr_change",
    "bas_retention",
    "additional_music_response_lag_seconds",
    "source",
]


def _row(route: str, sequence_id: str, stage: str, source: str, quality: dict, music: dict,
         *, seed=None, run_id="", condition="", retention=None) -> dict:
    retention = retention or {}
    return {
        "route": route,
        "sequence_id": sequence_id,
        "seed": "" if seed is None else seed,
        "stage": stage,
        "run_id": run_id,
        "condition": condition,
        "motion_energy": quality.get("motion_energy_rad2_s2", ""),
        "joint_amplitude_median": quality.get("joint_amplitude_median_rad", ""),
        "joint_velocity_p95": quality.get("joint_velocity_abs_rad_s", {}).get("p95", ""),
        "joint_acceleration_p95": quality.get("joint_acceleration_abs_rad_s2", {}).get("p95", ""),
        "joint_jerk_p95": quality.get("joint_jerk_abs_rad_s3", {}).get("p95", ""),
        "static_ratio": quality.get("static_ratio_speed_below_008", ""),
        "repeated_pose_ratio": quality.get("repeated_pose_ratio_rms008_after2s", ""),
        "c4_position_jump_p95": quality.get("c4_boundary_position_jump_p95_rad", ""),
        "c4_velocity_jump_p95": quality.get("c4_boundary_velocity_jump_p95_rad_s", ""),
        "root_planar_displacement": quality.get("root_planar_displacement_m", ""),
        "root_planar_speed_p95": quality.get("root_planar_speed_p95_m_s", ""),
        "root_height_min": quality.get("root_height_min_m", ""),
        "fsr_proxy": quality.get("physical", {}).get("fsr_ground_calibrated_proxy", ""),
        "pfc_proxy": quality.get("physical", {}).get("pfc_proxy", ""),
        "ground_penetration_ratio": quality.get("physical", {}).get("ground_penetration_ratio", ""),
        "foot_contact_mean": (
            (quality.get("physical", {}).get("left_contact_ratio", 0.0)
             + quality.get("physical", {}).get("right_contact_ratio", 0.0)) / 2.0
            if "left_contact_ratio" in quality.get("physical", {})
            and "right_contact_ratio" in quality.get("physical", {}) else ""
        ),
        "speed_corr": music.get("speed_best_correlation", ""),
        "speed_lag": music.get("speed_best_lag_seconds", ""),
        "impact_zero_corr": music.get("impact_zero_lag_correlation", ""),
        "impact_corr": music.get("impact_best_correlation", ""),
        "impact_abs_lag": music.get("impact_best_lag_seconds", ""),
        "bas_music_to_motion": music.get("bas_music_to_motion", ""),
        "bas_motion_to_music": music.get("bas_motion_to_music", ""),
        "audio_beats": music.get("audio_beats", ""),
        "motion_beats": music.get("motion_beats", ""),
        "motion_bpm": music.get("motion_beats_per_minute", ""),
        "motion_energy_retention": retention.get("motion_energy_retention", ""),
        "median_amplitude_statistic_ratio": retention.get("median_amplitude_statistic_ratio", ""),
        "jerk_p95_ratio": retention.get("jerk_p95_ratio", ""),
        "fsr_ground_calibrated_proxy_change": retention.get("fsr_ground_calibrated_proxy_change", ""),
        "onset_impact_corr_retention": retention.get("onset_impact_correlation_retention", ""),
        "onset_impact_corr_change": retention.get("onset_impact_correlation_change", ""),
        "bas_retention": retention.get("bas_music_to_motion_retention", ""),
        "additional_music_response_lag_seconds": retention.get(
            "additional_music_response_lag_seconds", ""
        ),
        "source": source,
    }


def main() -> None:
    rows: list[dict] = []

    references = json.loads(REFERENCE.read_text())
    for item in references:
        if item.get("route") not in {"M0", "M2", "M4"}:
            continue
        quality = item["quality"]
        music = item["music"]
        rows.append(_row(
            item["route"], str(item["sequence_id"]), "M_ref", str(REFERENCE), quality, music,
            seed=item.get("sampling_seed"), run_id=item.get("motion_id", ""),
            condition="generator_reference",
        ))

    execution = json.loads(EXECUTION.read_text())
    for item in execution:
        rows.append(_row(
            item["route"], str(item["sequence_id"]), "M_exec", str(EXECUTION),
            item["execution_quality"], item["execution_music"],
            run_id=item["run_id"], condition="SONIC_repeat", retention=item["loss"],
        ))

    for pair_path in M3_PAIRS:
        item = json.loads(pair_path.read_text())
        rows.append(_row(
            "M3", item["label"].split("-")[-1], "M_ref", str(pair_path),
            item["generator"]["quality"], item["generator"]["music"],
            condition="music_sidecar_paired",
        ))
        rows.append(_row(
            "M3", item["label"].split("-")[-1], "M_exec", str(pair_path),
            item["execution"]["quality"], item["execution"]["music"],
            condition="SONIC_corrected_measured", retention=item["retention"],
        ))

    FORMAL.mkdir(parents=True, exist_ok=True)
    with (FORMAL / "model_stage_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": "formal_model_stage_results_v1",
        "description": "Existing model M_ref/M_exec results; not a final multi-song ranking.",
        "rows": rows,
        "counts": {
            "m_ref": sum(r["stage"] == "M_ref" for r in rows),
            "m_exec": sum(r["stage"] == "M_exec" for r in rows),
            "sonic_repeats": sum(r["condition"] == "SONIC_repeat" for r in rows),
        },
    }
    (FORMAL / "model_stage_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    ref_rows = [r for r in rows if r["stage"] == "M_ref"]
    exec_rows = [r for r in rows if r["stage"] == "M_exec"]
    lines = [
        "# Existing Model Stage Results",
        "",
        "This report consolidates already available artifacts. It is a progress table,",
        "not a final model ranking: routes, songs, seeds and execution protocols are not",
        "yet balanced across all models.",
        "",
        f"- M_ref rows: {len(ref_rows)}",
        f"- M_exec rows: {len(exec_rows)}",
        f"- SONIC repeat rows: {sum(r['condition'] == 'SONIC_repeat' for r in rows)}",
        "",
        "## M_ref",
        "",
        "| route | sequence | energy | jerk P95 | impact corr | abs lag | BAS | source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in ref_rows:
        lines.append(
            f"| {r['route']} | {r['sequence_id']} | {float(r['motion_energy']):.4f} "
            f"| {float(r['joint_jerk_p95']):.2f} | {float(r['impact_corr']):.4f} "
            f"| {float(r['impact_abs_lag']):.3f} | {float(r['bas_music_to_motion']):.4f} "
            f"| {Path(r['source']).name} |"
        )
    lines += [
        "",
        "## M_exec",
        "",
        "| route | sequence | condition | energy | impact corr | BAS | energy retention | BAS retention | extra lag |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in exec_rows:
        def fmt(key: str, digits: int = 4) -> str:
            value = r[key]
            return "" if value == "" else f"{float(value):.{digits}f}"
        lines.append(
            f"| {r['route']} | {r['sequence_id']} | {r['condition']} | {fmt('motion_energy')} "
            f"| {fmt('impact_corr')} | {fmt('bas_music_to_motion')} | {fmt('motion_energy_retention')} "
            f"| {fmt('bas_retention')} | {fmt('additional_music_response_lag_seconds', 3)} |"
        )
    lines += [
        "",
        "Interpretation: compare M_ref against O-G1 for generator gap, and compare",
        "M_exec against its paired M_ref for SONIC retention. The existing rows are",
        "useful for pipeline debugging, but the formal claim still requires the planned",
        "multi-song, multi-seed, tempo/style-stratified expansion.",
        "",
    ]
    (FORMAL / "MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} model-stage rows to {FORMAL}")


if __name__ == "__main__":
    main()
