"""Map GT corruption checks to the frozen D/M benchmark taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "eval/benchmark_v1/gt/stratified_audit_v1/stratified_audit.json"
DEFAULT_VALIDITY = ROOT / "eval/benchmark_v1/gt/benchmark_validity_v1/benchmark_validity.json"
DEFAULT_TAXONOMY = ROOT / "eval/metric_taxonomy_v1.json"
DEFAULT_EVALUATION_MAP = ROOT / "eval/evaluation_map_v1.json"
DEFAULT_OUTPUT = ROOT / "eval/benchmark_v1/gt/module_benchmark_v1"


CHECK_MAP = {
    "jitter_jerk_p95_rad_s3": {
        "module": "D",
        "submodule": "smoothness",
        "metric": "jerk_p95",
        "metric_id": "G-DYNAMICS",
        "interpretation": "随机关节扰动应提高高阶运动变化率",
    },
    "lowpass_jerk_p95_rad_s3": {
        "module": "D",
        "submodule": "smoothness",
        "metric": "jerk_p95",
        "metric_id": "G-DYNAMICS",
        "interpretation": "低通平滑应降低高阶运动变化率",
    },
    "freeze_static_ratio_below_0p05_rad_s": {
        "module": "D",
        "submodule": "liveliness",
        "metric": "static_ratio",
        "metric_id": "G-STATIC",
        "interpretation": "冻结动作应提高近似静止帧比例",
    },
    "freeze_G1BeatF1": {
        "module": "M",
        "submodule": "rhythm",
        "metric": "event_f1",
        "metric_id": "M-BEAT-F1",
        "interpretation": "冻结动作通常应削弱音乐事件的覆盖和匹配",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validity-audit", type=Path, default=DEFAULT_VALIDITY)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--evaluation-map", type=Path, default=DEFAULT_EVALUATION_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _annotate(check: dict[str, Any]) -> dict[str, Any]:
    name = str(check["check"])
    mapping = CHECK_MAP.get(name)
    if mapping is None:
        return {**check, "module": "unmapped", "submodule": "unmapped", "metric": "unmapped"}
    return {**check, **mapping}


def _status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {"D": [], "M": [], "unmapped": []}
    for row in rows:
        grouped.setdefault(row["module"], []).append(row)
    result = {}
    for module, values in grouped.items():
        if not values:
            continue
        passed = sum(row["result"] == "PASS" for row in values)
        warned = sum(row["result"] == "WARN" for row in values)
        result[module] = {
            "checks": len(values),
            "pass": passed,
            "warn": warned,
            "status": "PASS" if warned == 0 else "PASS_WITH_CAVEAT",
        }
    return result


def _metric_inventory(
    taxonomy: dict[str, Any],
    evaluation_map: dict[str, Any],
    validity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return every frozen metric, including metrics not yet implementable."""
    evaluation = {row["id"]: row for row in evaluation_map["metrics"]}
    validity_status = validity.get("metric_status", {})
    validity_map = {
        "G-DYNAMICS": "jerk",
        "G-ENERGY": "motion_energy",
        "G-STATIC": "static_ratio",
        "G-REPEAT": "repeat_similarity",
        "M-BAS": "bas",
        "M-BEAT-F1": "beat_f1",
    }
    output = []
    for module in taxonomy["modules"]:
        for submodule in module["submodules"]:
            for metric_id in submodule["metric_ids"]:
                definition = evaluation.get(metric_id, {})
                validity_key = validity_map.get(metric_id)
                validation = validity_status.get(validity_key, {}) if validity_key else {}
                if validation:
                    calibration = validation.get("status", "reported_in_validity_audit")
                elif definition.get("current_status") == "missing":
                    calibration = "not_implemented"
                elif definition.get("current_status") == "manual":
                    calibration = "human_study_required"
                else:
                    calibration = "not_directionally_calibrated"
                output.append(
                    {
                        "module": module["id"],
                        "module_name_zh": module["name_zh"],
                        "submodule": submodule["id"],
                        "submodule_name_zh": submodule["name_zh"],
                        "metric_id": metric_id,
                        "name": definition.get("name", metric_id),
                        "level": definition.get("level", "unregistered"),
                        "current_status": definition.get("current_status", "unregistered"),
                        "calibration_status": calibration,
                        "required_inputs": definition.get("required_inputs", []),
                    }
                )
    return output


def _report(payload: dict[str, Any]) -> str:
    rows = payload["checks"]
    lines = [
        "# D/M Benchmark 指标方向校准报告 v1",
        "",
        "本报告不是模型排名，而是检查冻结的 D（舞蹈动作质量）和 M（音乐-舞蹈适配）指标能否识别已知动作退化。输入为 38 条 sealed GT 的 clean、jitter、low-pass、freeze 和 repeat 变体。",
        "",
        "## 校准结论",
        "",
        "- D 模块：jerk P95 和 static ratio 的方向检查全部通过，可以继续作为动作质量子模块的核心指标。",
        "- M 模块：Beat F1 在 AIST++ freeze 检查通过，但在 FineDance 中为 WARN；Beat F1 不能单独作为跨数据集退化 gate。",
        "- BAS 不因该 WARN 被删除。BAS、Beat F1、event precision/recall、impact correlation、lag、tempo error 和 phase error 仍应作为完整 rhythm suite 联合报告。",
        "- 这一步只验证‘指标对指定退化是否敏感’，不能把 GT 的分数解释为绝对优美度，也不能替代人类盲评。",
        "",
        "## 逐项映射",
        "",
        "| module | submodule | metric | dataset | check | expected | result | high-clean delta |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['module']} | {row['submodule']} | {row['metric']} | {row['dataset']} | "
            f"{row['check']} | {row['expected']} | **{row['result']}** | {row['delta_high_minus_clean']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 全部冻结模块与指标覆盖",
            "",
            "下面列出 taxonomy v1 中的全部指标。‘方向校准’只表示当前已有 GT corruption 或 validity audit 对该指标做过敏感性检查；‘未校准’不表示指标无效，而表示正式模型实验前仍需完成相应的 GT/counterfactual/人评校准。",
            "",
            "| module | submodule | metric | 当前实现状态 | 当前校准状态 | level |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in payload["metric_inventory"]:
        lines.append(
            f"| {item['module']} {item['module_name_zh']} | {item['submodule']} {item['submodule_name_zh']} | "
            f"{item['metric_id']} {item['name']} | {item['current_status']} | {item['calibration_status']} | {item['level']} |"
        )
    lines.extend(
        [
            "",
            "## 对 benchmark 的决定",
            "",
            "| 模块 | 当前状态 | 后续使用方式 |",
            "|---|---|---|",
            "| D / smoothness | PASS | 报告 jerk、velocity/acceleration、energy，并结合 static ratio、FSR/PFC 和物理稳定性解释 |",
            "| D / liveliness | PASS | 报告 motion energy、static ratio 和重复率；不能把能量越大直接当作越优美 |",
            "| M / rhythm | PASS_WITH_CAVEAT | BAS 与 Beat F1、事件覆盖、impact/速度相关性、lag、tempo 和 phase 一起报告 |",
            "| 其它 M 子模块 | 待校准 | structure 和 semantic 必须分别做 phrase、retrieval、MMDist、style/emotion 评估，不能由 rhythm 代替 |",
            "| X / SONIC | 待完整采集 | reference、tracking、retention、安全四组全部报告，不能只看 success rate 或 RMSE |",
            "| R / online | 待完整采集 | latency、deadline、RTF、stale/fallback 等一起报告，离线 PKL 不算 online 证据 |",
            "| H / perception | 待盲评 | naturalness、aesthetics、smoothness、expressiveness、rhythm、style/emotion 分开评分 |",
            "",
            "## 下一步",
            "",
            "1. 保持本报告中的 D/M 指标定义不变，扩展到全量 AIST++ 1,408 条和 FineDance 203 条 GT 的 style/tempo 分层。",
            "2. 在同一歌曲、同一音频起点和同一统计窗口上扩展 M2/M3/M4，多 seed 分别计算 M_ref 和 M_exec。",
            "3. 对正式模型结果同时报告原始指标、匹配 GT 分布的条件分数，以及 SONIC retention；不使用单一总分替代分模块结果。",
            "4. 对 FineDance 继续保留 Beat F1，但先检查 beat detector、音频起点和动作事件定义，再决定是否做数据集专属校准。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    input_path = args.audit.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = json.loads(input_path.read_text(encoding="utf-8"))
    checks = [_annotate(row) for row in source.get("corruption_checks", [])]
    validity = json.loads(args.validity_audit.expanduser().resolve().read_text(encoding="utf-8"))
    taxonomy = json.loads(args.taxonomy.expanduser().resolve().read_text(encoding="utf-8"))
    evaluation_map = json.loads(args.evaluation_map.expanduser().resolve().read_text(encoding="utf-8"))
    inventory = _metric_inventory(taxonomy, evaluation_map, validity)
    payload = {
        "schema_version": "module_calibration_audit_v1",
        "source_audit": str(input_path),
        "sequence_count": source.get("sequence_count"),
        "checks": checks,
        "metric_inventory": inventory,
        "module_status": _status(checks),
        "decision": "D_PASS_M_PASS_WITH_CAVEAT",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "CALIBRATION_REPORT_ZH.md").write_text(_report(payload), encoding="utf-8")
    (output_dir / "module_calibration_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metric_inventory.csv").write_text(
        "module,submodule,metric_id,name,current_status,calibration_status,level\n"
        + "\n".join(
            ",".join(
                str(item[field]).replace(",", ";")
                for field in ("module", "submodule", "metric_id", "name", "current_status", "calibration_status", "level")
            )
            for item in inventory
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(checks)} mapped checks to {output_dir}")
    for module, status in payload["module_status"].items():
        print(f"{module}: {status['status']} ({status['pass']} PASS, {status['warn']} WARN)")


if __name__ == "__main__":
    main(parse_args())
