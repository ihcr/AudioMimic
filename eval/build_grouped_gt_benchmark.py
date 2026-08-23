"""Group the full AIST++/FineDance G1 oracle by dataset, style and tempo."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path("eval/benchmark_v1/gt/gt_oracle_suite_all_v1/gt_oracle_suite_metrics.json")
DEFAULT_OUTPUT = Path("eval/benchmark_v1/gt/grouped_gt_all_v1")

AIST_GENRES = {
    "BR": "Breaking",
    "PO": "Popping",
    "LO": "Locking",
    "MH": "Middle_HipHop",
    "LH": "LA_HipHop",
    "HO": "House",
    "WA": "Waacking",
    "KR": "Krumping",
    "JS": "Street_Jazz",
    "JB": "Ballet_Jazz",
}

METRICS: dict[str, dict[str, Any]] = {
    "velocity_p95": {"module": "D", "submodule": "D.smoothness", "path": ("quality", "joint_velocity_abs_rad_s", "p95"), "direction": "match_gt"},
    "acceleration_p95": {"module": "D", "submodule": "D.smoothness", "path": ("quality", "joint_acceleration_abs_rad_s2", "p95"), "direction": "match_gt"},
    "jerk_p95": {"module": "D", "submodule": "D.smoothness", "path": ("quality", "joint_jerk_abs_rad_s3", "p95"), "direction": "not_above_gt"},
    "c4_position_jump_p95": {"module": "D", "submodule": "D.smoothness", "path": ("quality", "c4_boundary_position_jump_p95_rad"), "direction": "match_gt"},
    "c4_velocity_jump_p95": {"module": "D", "submodule": "D.smoothness", "path": ("quality", "c4_boundary_velocity_jump_p95_rad_s"), "direction": "match_gt"},
    "motion_energy": {"module": "D", "submodule": "D.liveliness", "path": ("quality", "motion_energy_rad2_s2"), "direction": "match_gt"},
    "static_ratio": {"module": "D", "submodule": "D.liveliness", "path": ("quality", "static_ratio_speed_below_008"), "direction": "match_gt"},
    "repeated_pose_ratio": {"module": "D", "submodule": "D.liveliness", "path": ("quality", "repeated_pose_ratio_rms008_after2s"), "direction": "match_gt"},
    "fsr_proxy": {"module": "D", "submodule": "D.physics", "path": ("quality", "physical", "fsr_ground_calibrated_proxy"), "direction": "lower"},
    "pfc_proxy": {"module": "D", "submodule": "D.physics", "path": ("quality", "physical", "pfc_proxy"), "direction": "lower"},
    "penetration_ratio": {"module": "D", "submodule": "D.physics", "path": ("quality", "physical", "ground_penetration_ratio"), "direction": "lower"},
    "root_height_min": {"module": "D", "submodule": "D.physics", "path": ("quality", "root_height_min_m"), "direction": "valid_range"},
    "root_speed_p95": {"module": "D", "submodule": "D.physics", "path": ("quality", "root_planar_speed_p95_m_s"), "direction": "match_gt"},
    "bas": {"module": "M", "submodule": "M.rhythm", "path": ("music", "bas_music_to_motion"), "direction": "higher"},
    "reverse_bas": {"module": "M", "submodule": "M.rhythm", "path": ("music", "bas_motion_to_music"), "direction": "higher"},
    "beat_precision": {"module": "M", "submodule": "M.rhythm", "path": ("music", "event_f1", "precision"), "direction": "higher"},
    "beat_recall": {"module": "M", "submodule": "M.rhythm", "path": ("music", "event_f1", "recall"), "direction": "higher"},
    "beat_f1": {"module": "M", "submodule": "M.rhythm", "path": ("music", "event_f1", "f1"), "direction": "higher"},
    "beat_timing_error": {"module": "M", "submodule": "M.rhythm", "path": ("music", "event_f1", "median_abs_timing_error_seconds"), "direction": "lower"},
    "speed_correlation": {"module": "M", "submodule": "M.dynamics", "path": ("music", "speed_best_correlation"), "direction": "higher"},
    "impact_correlation": {"module": "M", "submodule": "M.dynamics", "path": ("music", "impact_best_correlation"), "direction": "higher"},
    "response_abs_lag": {"module": "M", "submodule": "M.dynamics", "path": ("music", "impact_best_lag_seconds"), "direction": "lower", "absolute": True},
    "tempo_error": {"module": "M", "submodule": "M.dynamics", "path": ("music", "tempo_abs_error_bpm"), "direction": "lower"},
    "phase_error": {"module": "M", "submodule": "M.dynamics", "path": ("music", "phase", "mean_phase_error_cycles"), "direction": "lower"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _get(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        value = value[key]
    return value


def _canonical_style(style: str) -> str:
    normalized = str(style).strip()
    return "Jazz" if normalized.casefold() == "jazz" else normalized


def _styles(row: dict[str, Any]) -> list[str]:
    if row["dataset"] == "aistpp":
        match = re.match(r"g([A-Z]{2})_", str(row["sequence_id"]))
        code = match.group(1) if match else "UNKNOWN"
        return [AIST_GENRES.get(code, f"AIST_{code}")]
    values = sorted({_canonical_style(str(value)) for value in row.get("style", [])})
    return values or ["Unknown"]


def _tempo(row: dict[str, Any]) -> str:
    bpm = float(row["music"]["audio_bpm"])
    if bpm < 90.0:
        return "slow_<90"
    if bpm <= 130.0:
        return "medium_90-130"
    return "fast_>130"


def _groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = str(row["dataset"])
        groups[("dataset", dataset)].append(row)
        groups[("dataset_tempo", f"{dataset}/{_tempo(row)}")].append(row)
        for style in _styles(row):
            groups[("dataset_style", f"{dataset}/{style}")].append(row)
    return groups


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "q10": None, "median": None, "q90": None, "mean": None}
    return {
        "n": int(len(array)),
        "q10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "q90": float(np.percentile(array, 90)),
        "mean": float(np.mean(array)),
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (axis, group), items in sorted(_groups(rows).items()):
        for name, spec in METRICS.items():
            values = []
            for item in items:
                value = _get(item, spec["path"])
                if value is None:
                    continue
                numeric = float(value)
                values.append(abs(numeric) if spec.get("absolute") else numeric)
            output.append(
                {
                    "axis": axis,
                    "group": group,
                    "sequences": len(items),
                    "module": spec["module"],
                    "submodule": spec["submodule"],
                    "metric": name,
                    "direction": spec["direction"],
                    **_distribution(values),
                }
            )
    return output


def _fmt(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _index(summaries: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["axis"], row["group"], row["metric"]): row for row in summaries}


def _table(
    report: list[str],
    summaries: list[dict[str, Any]],
    *,
    title: str,
    submodule: str,
    metrics: list[str],
    axis_name: str = "dataset_style",
) -> None:
    report.extend(["", f"## {title}", ""])
    report.append("| 分组 | n | " + " | ".join(metrics) + " |")
    report.append("|---|---:|" + "|".join("---:" for _ in metrics) + "|")
    lookup = _index(summaries)
    groups = sorted(
        {(row["axis"], row["group"]) for row in summaries if row["axis"] == axis_name and row["submodule"] == submodule}
    )
    for axis, group in groups:
        first = lookup[(axis, group, metrics[0])]
        values = [_fmt(lookup[(axis, group, metric)]["median"]) for metric in metrics]
        report.append(f"| {group} | {first['sequences']} | " + " | ".join(values) + " |")


def main(args: argparse.Namespace) -> None:
    source = args.input.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["records"]
    counts = {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in ("aistpp", "finedance")}
    if counts != {"aistpp": 1408, "finedance": 203}:
        raise RuntimeError(f"unexpected full oracle counts: {counts}")
    summaries = _summaries(rows)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": "grouped_gt_all_v1",
        "source": str(source),
        "counts": counts,
        "tempo_bins": {"slow_<90": "BPM < 90", "medium_90-130": "90 <= BPM <= 130", "fast_>130": "BPM > 130"},
        "aist_genre_map": AIST_GENRES,
        "multi_label_policy": "FineDance sequences contribute to every declared style label.",
        "summaries": summaries,
        "unavailable_modules": {
            "D.realism": "FID extractor not frozen",
            "D.diversity": "requires model/multi-seed sample sets",
            "M.structure": "phrase/section evaluator not frozen",
            "M.semantic": "independent style/emotion/retrieval evaluator not frozen",
            "H.perception": "requires blinded human study",
        },
    }
    (output_dir / "grouped_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output_dir / "grouped_metrics_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report = [
        "# AIST++ / FineDance 全量分组 GT Benchmark v1",
        "",
        f"本报告使用全部 AIST++ `{counts['aistpp']}` 条和 FineDance `{counts['finedance']}` 条有效 paired G1 oracle。",
        "AIST++ 按文件名中的 10 个 genre 分组；FineDance 为多标签数据，每条序列进入其全部 style 组。",
        "所有数值是组内中位数，是后续 M_ref/M_exec 的条件化 reference range，不是风格排名或优美度总分。",
        "",
        "Tempo 分层单独保存在 JSON/CSV 中：slow `<90 BPM`、medium `90--130 BPM`、fast `>130 BPM`。",
    ]
    _table(report, summaries, title="D3 平滑度与连续性", submodule="D.smoothness", metrics=["velocity_p95", "acceleration_p95", "jerk_p95", "c4_position_jump_p95"])
    _table(report, summaries, title="D4 活力、冻结与重复", submodule="D.liveliness", metrics=["motion_energy", "static_ratio", "repeated_pose_ratio"])
    _table(report, summaries, title="D5 物理合理性", submodule="D.physics", metrics=["fsr_proxy", "pfc_proxy", "penetration_ratio", "root_height_min", "root_speed_p95"])
    _table(report, summaries, title="M1 节拍与节奏", submodule="M.rhythm", metrics=["bas", "reverse_bas", "beat_precision", "beat_recall", "beat_f1", "beat_timing_error"])
    _table(report, summaries, title="M2 Tempo、phase 与动态响应", submodule="M.dynamics", metrics=["speed_correlation", "impact_correlation", "response_abs_lag", "tempo_error", "phase_error"])
    _table(
        report,
        summaries,
        title="Tempo 分层：动作强度与平滑度",
        submodule="D.smoothness",
        metrics=["motion_energy", "jerk_p95", "static_ratio", "repeated_pose_ratio"],
        axis_name="dataset_tempo",
    )
    _table(
        report,
        summaries,
        title="Tempo 分层：节奏与动态适配",
        submodule="M.rhythm",
        metrics=["bas", "beat_f1", "beat_timing_error", "impact_correlation", "tempo_error", "phase_error"],
        axis_name="dataset_tempo",
    )
    report.extend(
        [
            "",
            "## 当前不能填入的模块",
            "",
            "- D1 FID/FIDg：动作 feature extractor 尚未冻结；",
            "- D2 Div：需要同音乐多 seed 的生成样本，GT 单条分组不能替代；",
            "- M3 phrase/section：段落检测和响应 evaluator 尚未冻结；",
            "- M4 style/emotion：独立 retrieval、style 和 affective evaluator 尚未冻结；",
            "- H 人评：需要正式 blinded study。",
            "",
            "## 数据解释警告",
            "",
            "- FineDance 为多标签风格，同一序列会进入多个 style 组，组间并不独立；",
            "- `n < 5` 的风格仅作诊断，不用于显著性结论；",
            "- FineDance 的 root-height 异常已单独审计：203 条中 17 条出现短时负 root z，其中 13 条属于 Breaking；这不是风格质量结论，见 `../finedance_root_height_audit_v1/REPORT_ZH.md`；",
            "- 在已有 G1 资产的 root/ground convention 核对前，D5 的 `root_height_min` 只作诊断，不用于 generator 排名；",
            "- AIST++ 与 FineDance 的时长、标注和 GMR 分布不同，不能直接用绝对值给两个数据集排名。",
            "",
            "因此本报告建立的是当前已实现指标的全量条件分布。不能用 BAS 或 jerk 推断 style/emotion/优美度，",
            "也不能用不同风格的绝对数值做高低排名。后续模型必须与相同 dataset/style/tempo 的 GT 组比较。",
        ]
    )
    (output_dir / "REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {len(summaries)} grouped metric distributions to {output_dir}")
    print(f"records: {counts}")


if __name__ == "__main__":
    main(parse_args())
