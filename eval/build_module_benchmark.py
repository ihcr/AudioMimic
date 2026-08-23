"""Build dataset-conditioned D/M module scores for GT and existing model runs.

The score is a similarity-to-GT-distribution diagnostic, not an absolute beauty
score.  A model is compared with the matched dataset, style and tempo strata so
that AIST++ and FineDance are not ranked by raw metric magnitudes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "eval/benchmark_v1/gt/gt_oracle_suite_all_v1/gt_oracle_suite_metrics.json"
DEFAULT_MANIFEST = ROOT / "eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"
DEFAULT_MODEL = ROOT / "eval/benchmark_v1/formal/model_stage_results.json"
DEFAULT_EXTENDED = ROOT / "eval/benchmark_v1/formal/model_music_extended/metrics.json"
DEFAULT_OUTPUT = ROOT / "eval/benchmark_v1/gt/module_benchmark_v1"


D_METRICS = {
    "motion_energy": ("quality", "motion_energy_rad2_s2"),
    "velocity_p95": ("quality", "joint_velocity_abs_rad_s", "p95"),
    "acceleration_p95": ("quality", "joint_acceleration_abs_rad_s2", "p95"),
    "jerk_p95": ("quality", "joint_jerk_abs_rad_s3", "p95"),
    "static_ratio": ("quality", "static_ratio_speed_below_008"),
    "repeated_pose_ratio": ("quality", "repeated_pose_ratio_rms008_after2s"),
    "fsr_proxy": ("quality", "physical", "fsr_ground_calibrated_proxy"),
    "pfc_proxy": ("quality", "physical", "pfc_proxy"),
    "penetration_ratio": ("quality", "physical", "ground_penetration_ratio"),
}

M_METRICS = {
    "bas": ("music", "bas_music_to_motion"),
    "reverse_bas": ("music", "bas_motion_to_music"),
    "event_precision": ("music", "event_f1", "precision"),
    "event_recall": ("music", "event_f1", "recall"),
    "event_f1": ("music", "event_f1", "f1"),
    "event_timing_error": ("music", "event_f1", "median_abs_timing_error_seconds"),
    "speed_correlation": ("music", "speed_best_correlation"),
    "impact_correlation": ("music", "impact_best_correlation"),
    "impact_abs_lag": ("music", "impact_best_lag_seconds"),
    "tempo_error": ("music", "tempo_abs_error_bpm"),
    "phase_error": ("music", "phase", "mean_phase_error_cycles"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--extended-model", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _get(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        value = value[key]
    return value


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _canonical_styles(values: list[str]) -> list[str]:
    output = []
    for value in values:
        normalized = str(value).strip()
        normalized = "Jazz" if normalized.casefold() == "jazz" else normalized
        if normalized and normalized not in output:
            output.append(normalized)
    return sorted(output)


def _tempo(bpm: float) -> str:
    if bpm < 90.0:
        return "slow_<90"
    if bpm <= 130.0:
        return "medium_90-130"
    return "fast_>130"


def _dataset_style(row: dict[str, Any]) -> list[str]:
    if row["dataset"] == "aistpp":
        sequence = str(row["sequence_id"])
        codes = {
            "BR": "Breaking", "PO": "Popping", "LO": "Locking",
            "MH": "Middle_HipHop", "LH": "LA_HipHop", "HO": "House",
            "WA": "Waacking", "KR": "Krumping", "JS": "Street_Jazz",
            "JB": "Ballet_Jazz",
        }
        code = sequence[1:3] if len(sequence) >= 3 else "UNKNOWN"
        return [codes.get(code, f"AIST_{code}")]
    return _canonical_styles(row.get("style", [])) or ["Unknown"]


def _flatten_gt(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "dataset": row["dataset"],
        "sequence_id": str(row["sequence_id"]),
        "styles": _dataset_style(row),
        "tempo": _tempo(float(row["music"]["audio_bpm"])),
    }
    for name, path in {**D_METRICS, **M_METRICS}.items():
        value = _finite(_get(row, path))
        if name == "impact_abs_lag" and value is not None:
            value = abs(value)
        flat[name] = value
    return flat


def _flatten_model(row: dict[str, Any], extended: dict[str, Any] | None) -> dict[str, Any]:
    flat = {
        "route": row["route"],
        "sequence_id": str(row["sequence_id"]),
        "stage": row["stage"],
        "label": row.get("run_id") or row.get("condition") or "",
        "dataset": "finedance" if str(row["sequence_id"]).isdigit() else "unknown",
    }
    for name, field in {
        "motion_energy": "motion_energy", "velocity_p95": "joint_velocity_p95",
        "acceleration_p95": "joint_acceleration_p95", "jerk_p95": "joint_jerk_p95",
        "static_ratio": "static_ratio", "repeated_pose_ratio": "repeated_pose_ratio",
        "fsr_proxy": "fsr_proxy", "pfc_proxy": "pfc_proxy",
        "penetration_ratio": "ground_penetration_ratio", "bas": "bas_music_to_motion",
        "reverse_bas": "bas_motion_to_music", "speed_correlation": "speed_corr",
        "impact_correlation": "impact_corr", "impact_abs_lag": "impact_abs_lag",
    }.items():
        value = _finite(row.get(field))
        flat[name] = abs(value) if name == "impact_abs_lag" and value is not None else value
    if extended is not None:
        music = extended["music"]
        event = music["event_f1"]
        phase = music["phase"]
        flat.update(
            {
                "event_precision": _finite(event["precision"]),
                "event_recall": _finite(event["recall"]),
                "event_f1": _finite(event["f1"]),
                "event_timing_error": _finite(event["median_abs_timing_error_seconds"]),
                "tempo_error": _finite(music["tempo_abs_error_bpm"]),
                "phase_error": _finite(phase["mean_phase_error_cycles"]),
            }
        )
    return flat


def _score(value: float | None, values: list[float]) -> float | None:
    if value is None or not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    scale = max(float(np.percentile(array, 75) - np.percentile(array, 25)), abs(median) * 0.10, 1e-6)
    return float(math.exp(-abs(value - median) / scale))


def _group_values(gt_rows: list[dict[str, Any]], key: str, metric: str) -> list[float]:
    return [float(row[metric]) for row in gt_rows if row.get(metric) is not None and (key in row.get("styles", []) or key == row.get("tempo") or key == row.get("dataset"))]


def _score_model(model: dict[str, Any], gt_rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    target = next((row for row in gt_rows if row["sequence_id"] == model["sequence_id"]), None)
    if target is None:
        return {**model, "style_score": None, "tempo_score": None, "dataset_score": None, "module_score": None}
    styles = target["styles"]
    groups = {
        "dataset": [target["dataset"]],
        "tempo": [target["tempo"]],
        "style": styles,
    }
    scores: dict[str, list[float]] = {name: [] for name in groups}
    per_metric: dict[str, Any] = {}
    for metric in metric_names:
        value = model.get(metric)
        metric_scores = {}
        for group_name, labels in groups.items():
            label_scores = []
            for label in labels:
                reference = _group_values(gt_rows, label, metric)
                score = _score(value, reference)
                if score is not None:
                    label_scores.append(score)
            if label_scores:
                metric_scores[group_name] = float(np.mean(label_scores))
                scores[group_name].append(metric_scores[group_name])
        per_metric[metric] = metric_scores
    module_scores = {name: (100.0 * float(np.mean(values)) if values else None) for name, values in scores.items()}
    valid = [value for value in module_scores.values() if value is not None]
    return {
        **model,
        "style_score": module_scores["style"],
        "tempo_score": module_scores["tempo"],
        "dataset_score": module_scores["dataset"],
        "module_score": float(np.mean(valid)) if valid else None,
        "target_style": ";".join(styles),
        "target_tempo": target["tempo"],
        "per_metric_scores": per_metric,
    }


def _distribution_rows(gt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    group_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        group_map[("dataset", row["dataset"])].append(row)
        group_map[("tempo", row["tempo"])].append(row)
        for style in row["styles"]:
            group_map[("dataset_style", f"{row['dataset']}/{style}")].append(row)
    for (axis, group), rows in sorted(group_map.items()):
        for module, metrics in (("D", D_METRICS), ("M", M_METRICS)):
            for metric in metrics:
                values = [row[metric] for row in rows if row.get(metric) is not None]
                if not values:
                    continue
                output.append({
                    "axis": axis, "group": group, "module": module, "metric": metric,
                    "n": len(values), "q10": float(np.percentile(values, 10)),
                    "median": float(np.median(values)), "q90": float(np.percentile(values, 90)),
                })
    return output


def _group_summary(distributions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "motion_energy", "jerk_p95", "fsr_proxy", "pfc_proxy", "penetration_ratio",
        "bas", "event_f1", "impact_correlation", "tempo_error", "phase_error",
    ]
    lookup = {(row["axis"], row["group"], row["metric"]): row for row in distributions}
    groups = sorted({(row["axis"], row["group"]) for row in distributions if row["axis"] in {"dataset_style", "dataset_tempo"}})
    output = []
    for axis, group in groups:
        first = next(row for row in distributions if row["axis"] == axis and row["group"] == group)
        item = {"axis": axis, "group": group}
        for metric in metrics:
            row = lookup.get((axis, group, metric))
            item[f"{metric}_n"] = row["n"] if row else 0
            item[f"{metric}_median"] = row["median"] if row else None
        output.append(item)
    return output


def _write_report(output: Path, gt_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], distributions: list[dict[str, Any]]) -> None:
    lines = [
        "# GT 分模块 Benchmark 与 M2/M3/M4 当前评分 v1",
        "",
        "本报告把指标分为 D（舞蹈动作质量）和 M（音乐-舞蹈适配）两大模块。GT 使用 AIST++ 1408 条和 FineDance 203 条 paired G1 reference；模型使用当前已有的 M2/M3/M4 reference/execution artifacts。",
        "",
        "模型分数不是‘优美度绝对分’，而是模型结果与对应 dataset/style/tempo GT 分布的接近程度：100 表示接近该分层 GT 中心，越低表示偏离越大。AIST++ 与 FineDance 不使用 pooled raw value 直接排名。",
        "",
        "## 模块定义",
        "",
        "| module | 含义 | 指标 |",
        "|---|---|---|",
        "| D | 不看音乐时的动作质量、平滑性、活力和物理合理性 | energy, velocity, acceleration, jerk, static/repetition, FSR/PFC/penetration |",
        "| M | 动作对对应音乐的节拍、动态、tempo 和 phase 适配 | BAS/reverse BAS, event P/R/F1, timing error, speed/impact correlation, lag, tempo/phase error |",
        "",
        "## GT 数据覆盖",
        "",
        f"- AIST++: `{sum(row['dataset'] == 'aistpp' for row in gt_rows)}` 条；10 个 genre。",
        f"- FineDance: `{sum(row['dataset'] == 'finedance' for row in gt_rows)}` 条；多标签 style。",
        "- 所有原始分布保存在 `module_distributions.csv`；便于直接读表的 style/tempo 模块摘要在 `gt_style_tempo_module_summary.csv`。",
    ]
    lines += [
        "",
        "## GT 数据集模块基线",
        "",
        "以下是同一实现、同一单位下的 dataset-level 中位数；它们用于描述分布，不能把 AIST++ 的绝对值当作 FineDance 的排名标准。完整 style/tempo 表见 `grouped_gt_all_v1/REPORT_ZH.md`。",
        "",
        "| dataset | n | D energy | D jerk P95 | D FSR | D PFC | D penetration | M BAS | M event F1 | M impact corr | M tempo error | M phase error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("aistpp", "finedance"):
        group = [row for row in gt_rows if row["dataset"] == dataset]

        def median(metric: str) -> float:
            values = [row[metric] for row in group if row.get(metric) is not None]
            return float(np.median(values)) if values else float("nan")

        lines.append(
            f"| {dataset} | {len(group)} | {median('motion_energy'):.4f} | {median('jerk_p95'):.2f} | "
            f"{median('fsr_proxy'):.4f} | {median('pfc_proxy'):.4f} | {median('penetration_ratio'):.4f} | "
            f"{median('bas'):.4f} | {median('event_f1'):.4f} | {median('impact_correlation'):.4f} | "
            f"{median('tempo_error'):.2f} | {median('phase_error'):.4f} |"
        )
    lines += [
        "",
        "## 当前 M2/M3/M4 模块评分",
        "",
        "| route | stage | n | target style | target tempo | D dataset | D style | D tempo | D overall | M dataset | M style | M tempo | M overall |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ("M2", "M3", "M4"):
        for stage in ("M_ref", "M_exec"):
            rows = [r for r in model_rows if r["route"] == route and r["stage"] == stage]
            if not rows:
                continue
            # Scores are grouped by module in the raw rows below; this table is filled later.
            d = [r.get("d_module_score") for r in rows if r.get("d_module_score") is not None]
            m = [r.get("m_module_score") for r in rows if r.get("m_module_score") is not None]
            styles = ";".join(
                sorted(
                    {
                        style
                        for r in rows
                        for style in r.get("target_style", "").split(";")
                        if style
                    }
                )
            )
            tempos = sorted({r.get("target_tempo", "") for r in rows})
            tempo_label = tempos[0] if len(tempos) == 1 else "mixed"
            lines.append(
                f"| {route} | {stage} | {len(rows)} | {styles} | {tempo_label} | "
                f"{np.mean([r.get('d_dataset_score', np.nan) for r in rows]):.1f} | {np.mean([r.get('d_style_score', np.nan) for r in rows]):.1f} | {np.mean([r.get('d_tempo_score', np.nan) for r in rows]):.1f} | {np.mean(d) if d else float('nan'):.1f} | "
                f"{np.mean([r.get('m_dataset_score', np.nan) for r in rows]):.1f} | {np.mean([r.get('m_style_score', np.nan) for r in rows]):.1f} | {np.mean([r.get('m_tempo_score', np.nan) for r in rows]):.1f} | {np.mean(m) if m else float('nan'):.1f} |"
            )
    lines += [
        "",
        "## 论文指标对照",
        "",
        "| 工作 | 其主要指标 | 在本 benchmark 的对应模块 | 可否直接比较 |",
        "|---|---|---|---|",
        "| FACT / AIST++ | FIDk/FIDg, Distk/Distg, BeatAlign, user study | D.realism/diversity, M.rhythm, H | 仅同 extractor、split、长度和 beat detector 时可比较 |",
        "| EDGE | PFC, Beat Align, Distk/Distg, Elo/win rate | D.physics, M.rhythm, H | PFC/BAS 可复现，论文数值不能跨数据集直接搬用 |",
        "| Lodge | FIDk/FIDg, FSR, Div, BAS, runtime, user study | D.realism/physics/diversity, M.rhythm, R/H | 需要统一 FineDance/AIST++ 协议 |",
        "| Beat-It | PFC, BAS, Div, BAP, KPD, user study | D, M.rhythm, H | BAP/KPD 只有显式 beat/keypose target 才启用 |",
        "| RoboPerform | R@1/2/3, MMDist, success, EMPJPE/EMPKPE | M.semantic, X.tracking/safety | 需要独立 audio-motion encoder 和相同 simulator |",
        "",
        "## 论文中的典型报告值",
        "",
        "下表只记录本地论文原文中的代表性结果，用于检查量纲和评估习惯；由于数据集、序列长度、feature extractor、beat detector 和统计协议不同，不能直接当作本项目的排名阈值。",
        "",
        "| 来源/数据集 | 方法或 GT | PFC/FSR | BAS/BeatAlign | FID/Div 或其他 |",
        "|---|---|---:|---:|---|",
        "| EDGE / AIST++ | GT | PFC 1.332 | 0.24 | Distk 10.61, Distg 7.48 |",
        "| EDGE / AIST++ | EDGE (w=2) | PFC 1.5363 | 0.26 | Distk 9.48, Distg 5.72 |",
        "| Lodge / FineDance | GT | FSR 6.22% | 0.2120 | Divk 9.73, Divg 7.44 |",
        "| Lodge / FineDance | Lodge (DDPM) | FSR 5.01% | 0.2397 | FIDk 45.56, FIDg 34.29 |",
        "| Beat-It / AIST++ | GT | PFC 1.338 | 0.384 | Divk 9.773, Divm 7.212 |",
        "| Beat-It / AIST++ | Ours (beat + keyframes) | PFC 0.966 | 0.661 | BAP 0.793 |",
        "| RoboPerform / FineDance | Music-Motion retrieval | n/a | n/a | R@1 66.7, R@2 78.8, R@3 83.5, MM-Dist 1.154 |",
        "",
        "当前 AudioMimic M_ref 的 BAS 均值约为：M2 `0.247`、M3 `0.263`、M4 `0.242`；它们接近部分 FineDance/AIST++ 论文的 BAS 量级，但不能据此宣称优于论文方法。",
    ]
    lines += [
        "",
        "## 当前解释与限制",
        "",
        "- GT 不是每项指标都应为 1；不同风格有不同动作强度和节奏策略。",
        "- BAS 仍是 M.rhythm 核心指标，但必须和 event F1、tempo/phase、correlation 和 lag 一起报告。",
        "- 当前 M2/M3/M4 的歌曲、seed、长度和 execution protocol 尚未完全平衡，因此评分是 pipeline 诊断，不是最终论文排名。",
        "- M3 只有 012/065，M2/M4 主要是 098；下一轮正式实验要扩展到多歌曲、多 seed、跨 tempo/style。",
        "- FID/Div、retrieval、人评和 X/R 模块仍需独立冻结 extractor/protocol，不能从当前 D/M 分数推断。",
    ]
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_payload = json.loads(args.gt.read_text(encoding="utf-8"))
    gt_rows = [_flatten_gt(row) for row in gt_payload["records"]]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # The current model artifacts are FineDance song IDs; manifest is used to
    # attach the exact style/tempo target to each model row in the report.
    fine_rows = {str(row["sequence_id"]): row for row in manifest["datasets"]["finedance"]["records"]}
    gt_by_id = {row["sequence_id"]: row for row in gt_rows if row["dataset"] == "finedance"}
    model_payload = json.loads(args.model.read_text(encoding="utf-8"))
    extended_payload = json.loads(args.extended_model.read_text(encoding="utf-8"))
    extended = {}
    for row in extended_payload["rows"]:
        key = (row["route"], str(row["sequence_id"]), row["stage"], row["label"])
        extended[key] = row
    model_rows = []
    for source_row in model_payload["rows"]:
        if source_row["route"] not in {"M2", "M3", "M4"}:
            continue
        sequence_id = str(source_row["sequence_id"])
        if sequence_id not in fine_rows or sequence_id not in gt_by_id:
            continue
        label = source_row.get("run_id") or source_row.get("condition") or ""
        key = (source_row["route"], sequence_id, source_row["stage"], label)
        ext = extended.get(key)
        # M3 formal rows have an empty run_id; extended rows use M3-012/M3-065.
        if ext is None and source_row["route"] == "M3":
            ext = next((value for ekey, value in extended.items() if ekey[:3] == ("M3", sequence_id, source_row["stage"])), None)
        model_rows.append(_flatten_model(source_row, ext))

    distributions = _distribution_rows(gt_rows)
    group_summary = _group_summary(distributions)
    for row in model_rows:
        d = _score_model(row, gt_rows, list(D_METRICS))
        m = _score_model(row, gt_rows, list(M_METRICS))
        row.update({f"d_{key}": value for key, value in d.items() if key.endswith("score")})
        row.update({f"m_{key}": value for key, value in m.items() if key.endswith("score")})
        row["target_style"] = ";".join(
            _canonical_styles(fine_rows[row["sequence_id"]].get("style", []))
        )
        row["target_tempo"] = gt_by_id[row["sequence_id"]]["tempo"]

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "module_distributions.json").write_text(json.dumps({"gt_records": len(gt_rows), "rows": distributions}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output / "module_distributions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distributions[0]))
        writer.writeheader()
        writer.writerows(distributions)
    with (output / "gt_style_tempo_module_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_summary[0]))
        writer.writeheader()
        writer.writerows(group_summary)
    (output / "model_module_scores.json").write_text(json.dumps({"rows": model_rows}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    fields = sorted({key for row in model_rows for key in row if not isinstance(row[key], dict)})
    with (output / "model_module_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(model_rows)
    _write_report(output, gt_rows, model_rows, distributions)
    print(f"GT records: {len(gt_rows)}; model rows: {len(model_rows)}; distributions: {len(distributions)}")


if __name__ == "__main__":
    main()
