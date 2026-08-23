"""Summarize the currently available formal model-stage rows by route."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "eval/benchmark_v1/formal/model_stage_results.json"
OUTPUT = ROOT / "eval/benchmark_v1/formal/MODEL_SUMMARY.md"
EXTENDED = ROOT / "eval/benchmark_v1/formal/model_music_extended/metrics.json"


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return statistics.mean(values) if values else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))["rows"]
    lines = [
        "# Formal Model Summary",
        "",
        "This is a descriptive summary of currently available artifacts. It is not a",
        "final ranking because M0/M2/M4 are primarily song098 repeats while M3 uses",
        "songs 012/065 and a different pilot collection protocol.",
        "",
        "## 1. Dance quality",
        "",
        "These metrics describe movement quality without assuming that more motion is always better.",
        "",
        "| route/stage | n | energy | amplitude | vel P95 | acc P95 | jerk P95 | static | repeat | C4 vel jump |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = {}
    for route in sorted({r["route"] for r in rows}):
        for stage in ("M_ref", "M_exec"):
            group = [r for r in rows if r["route"] == route and r["stage"] == stage]
            if not group:
                continue
            item = {
                "stage": stage,
                "route": route,
                "n": len(group),
                "energy": _mean(group, "motion_energy"),
                "amplitude": _mean(group, "joint_amplitude_median"),
                "velocity_p95": _mean(group, "joint_velocity_p95"),
                "acceleration_p95": _mean(group, "joint_acceleration_p95"),
                "jerk_p95": _mean(group, "joint_jerk_p95"),
                "static_ratio": _mean(group, "static_ratio"),
                "repeat_ratio": _mean(group, "repeated_pose_ratio"),
                "c4_velocity_jump": _mean(group, "c4_velocity_jump_p95"),
            }
            summary[f"quality:{stage}:{route}"] = item
            lines.append(
                f"| {route}/{stage} | {item['n']} | {_fmt(item['energy'])} | "
                f"{_fmt(item['amplitude'])} | {_fmt(item['velocity_p95'], 2)} | "
                f"{_fmt(item['acceleration_p95'], 2)} | {_fmt(item['jerk_p95'], 2)} | "
                f"{_fmt(item['static_ratio'])} | {_fmt(item['repeat_ratio'])} | "
                f"{_fmt(item['c4_velocity_jump'], 3)} |"
            )
    lines += [
        "",
        "## 2. Physical and execution quality",
        "",
        "FSR is a foot-sliding proxy, PFC is a foot-contact proxy, penetration measures",
        "ground violation, and root height is a stability diagnostic.",
        "",
        "| route/stage | n | FSR proxy | PFC proxy | foot contact | penetration | root height min | root displacement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in sorted({r["route"] for r in rows}):
        for stage in ("M_ref", "M_exec"):
            group = [r for r in rows if r["route"] == route and r["stage"] == stage]
            if not group:
                continue
            item = {
                "stage": stage,
                "route": route,
                "n": len(group),
                "fsr": _mean(group, "fsr_proxy"),
                "pfc": _mean(group, "pfc_proxy"),
                "foot_contact": _mean(group, "foot_contact_mean"),
                "penetration": _mean(group, "ground_penetration_ratio"),
                "root_height_min": _mean(group, "root_height_min"),
                "root_displacement": _mean(group, "root_planar_displacement"),
            }
            summary[f"physical:{stage}:{route}"] = item
            lines.append(
                f"| {route}/{stage} | {item['n']} | {_fmt(item['fsr'])} | {_fmt(item['pfc'])} | "
                f"{_fmt(item['foot_contact'])} | {_fmt(item['penetration'])} | "
                f"{_fmt(item['root_height_min'])} | {_fmt(item['root_displacement'])} |"
            )
    lines += [
        "",
        "## 3. Music and beat alignment",
        "",
        "BAS is only one item in this suite. We also report speed/impact correlation,",
        "response lag and both BAS directions. Beat F1, onset precision/recall, tempo",
        "error, phase error, semantic retrieval and human preference are reported according",
        "to artifact availability. The legacy M0/M2/M4 execution logs were exported to",
        "portable measured-motion PKLs, so their event metrics are included below; semantic",
        "retrieval and human preference still require a separate calibrated protocol.",
        "",
        "| route/stage | n | speed corr | impact corr | impact lag (s) | BAS M->A | BAS A->M | audio beats | motion beats |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in sorted({r["route"] for r in rows}):
        for stage in ("M_ref", "M_exec"):
            group = [r for r in rows if r["route"] == route and r["stage"] == stage]
            if not group:
                continue
            item = {
                "stage": stage,
                "route": route,
                "n": len(group),
                "speed_corr": _mean(group, "speed_corr"),
                "impact_corr": _mean(group, "impact_corr"),
                "impact_lag": _mean(group, "impact_abs_lag"),
                "bas_forward": _mean(group, "bas_music_to_motion"),
                "bas_reverse": _mean(group, "bas_motion_to_music"),
                "audio_beats": _mean(group, "audio_beats"),
                "motion_beats": _mean(group, "motion_beats"),
            }
            summary[f"music:{stage}:{route}"] = item
            lines.append(
                f"| {route}/{stage} | {item['n']} | {_fmt(item['speed_corr'])} | "
                f"{_fmt(item['impact_corr'])} | {_fmt(item['impact_lag'], 3)} | "
                f"{_fmt(item['bas_forward'])} | {_fmt(item['bas_reverse'])} | "
                f"{_fmt(item['audio_beats'], 1)} | {_fmt(item['motion_beats'], 1)} |"
            )
    if EXTENDED.is_file():
        extended = json.loads(EXTENDED.read_text(encoding="utf-8"))["rows"]
        lines += [
            "",
            "### Extended beat event suite",
            "",
            "| route/stage | n | event P | event R | event F1 | tempo error | phase error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for route in sorted({r["route"] for r in extended}):
            for stage in ("M_ref", "M_exec"):
                group = [r for r in extended if r["route"] == route and r["stage"] == stage]
                if not group:
                    continue
                def avg(path: tuple[str, ...]):
                    values = []
                    for item in group:
                        value = item
                        for key in path:
                            value = value[key]
                        if value is not None:
                            values.append(float(value))
                    return statistics.mean(values) if values else None
                lines.append(
                    f"| {route}/{stage} | {len(group)} | {_fmt(avg(('music','event_f1','precision')))} | "
                    f"{_fmt(avg(('music','event_f1','recall')))} | {_fmt(avg(('music','event_f1','f1')))} | "
                    f"{_fmt(avg(('music','tempo_abs_error_bpm')))} | "
                    f"{_fmt(avg(('music','phase','mean_phase_error_cycles')))} |"
                )
    lines += [
        "",
        "## 4. SONIC retention",
        "",
        "| route | n | energy retention | amplitude ratio | jerk ratio | FSR change | BAS retention | extra lag (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in sorted({r["route"] for r in rows}):
        group = [r for r in rows if r["route"] == route and r["stage"] == "M_exec"]
        if not group:
            continue
        item = {
            "stage": "M_exec",
            "route": route,
            "n": len(group),
            "energy_retention": _mean(group, "motion_energy_retention"),
            "amplitude_ratio": _mean(group, "median_amplitude_statistic_ratio"),
            "jerk_ratio": _mean(group, "jerk_p95_ratio"),
            "fsr_change": _mean(group, "fsr_ground_calibrated_proxy_change"),
            "bas_retention": _mean(group, "bas_retention"),
            "extra_lag_seconds": _mean(group, "additional_music_response_lag_seconds"),
        }
        summary[f"M_exec:{route}"] = item
        lines.append(
            f"| {route} | {item['n']} | {_fmt(item['energy_retention'])} | "
            f"{_fmt(item['amplitude_ratio'])} | {_fmt(item['jerk_ratio'])} | "
            f"{_fmt(item['fsr_change'])} | {_fmt(item['bas_retention'])} | "
            f"{_fmt(item['extra_lag_seconds'], 3)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The current table is suitable for checking the evaluator and identifying",
        "tracker retention patterns. It is not evidence that one route is better than",
        "another: song identity, training/sampling seed, audio alignment and SONIC",
        "repeat protocol are not yet balanced. The next formal claim requires the",
        "72-cell expansion matrix to be populated with the same protocol.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(summary)} route summaries to {OUTPUT}")


if __name__ == "__main__":
    main()
