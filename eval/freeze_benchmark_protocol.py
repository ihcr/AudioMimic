"""Freeze the current GT benchmark roles without inventing a final test split."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "eval/benchmark_v1/gt/gt_oracle_suite_v2/gt_oracle_suite_metrics.json"
FULL_REFERENCE = ROOT / "eval/benchmark_v1/gt/gt_oracle_suite_all_v1/gt_oracle_suite_metrics.json"
MANIFEST = ROOT / "eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"
OUTPUT = ROOT / "eval/benchmark_v1/gt/benchmark_protocol_v1"


def _counts(records: list[dict]) -> dict[str, int]:
    datasets = sorted({str(row["dataset"]) for row in records})
    return {dataset: sum(str(row["dataset"]) == dataset for row in records) for dataset in datasets}


def _ids(records: list[dict]) -> set[tuple[str, str]]:
    return {(str(row["dataset"]), str(row["sequence_id"])) for row in records}


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["records"]
    full_reference = json.loads(FULL_REFERENCE.read_text(encoding="utf-8"))["records"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["datasets"]

    calibration_ids = _ids(calibration)
    full_ids = _ids(full_reference)
    if not calibration_ids <= full_ids:
        raise RuntimeError("Calibration records are not a subset of the full reference suite")

    manifest_counts = {
        dataset: sum(bool(row.get("paired_valid")) for row in payload.get("records", []))
        for dataset, payload in manifest.items()
    }
    full_counts = _counts(full_reference)
    expected_counts = {"aistpp": 1408, "finedance": 203}
    if full_counts != expected_counts:
        raise RuntimeError(f"Unexpected full reference counts: {full_counts}")
    if manifest_counts.get("aistpp") != 1408 or manifest_counts.get("finedance") != 203:
        raise RuntimeError(f"Manifest paired counts do not match: {manifest_counts}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": "benchmark_protocol_v1",
        "status": "calibration_and_reference_frozen_final_test_pending",
        "roles": {
            "calibration_38": {
                "source": str(CALIBRATION),
                "purpose": "freeze metric definitions, corruption severity and direction checks",
                "counts": _counts(calibration),
                "total": len(calibration),
                "allowed_for_threshold_tuning": True,
                "allowed_for_final_paper_ranking": False,
            },
            "full_reference_1611": {
                "source": str(FULL_REFERENCE),
                "purpose": "estimate dataset, tempo and style-conditioned GT reference distributions",
                "counts": full_counts,
                "total": len(full_reference),
                "allowed_for_threshold_tuning": False,
                "allowed_for_final_paper_ranking": False,
            },
            "final_test": {
                "status": "not_frozen",
                "purpose": "unbiased final comparison of M2/M3/M_exec",
                "counts": None,
                "allowed_for_threshold_tuning": False,
                "allowed_for_final_paper_ranking": True,
            },
        },
        "checks": {
            "calibration_is_subset_of_full_reference": True,
            "manifest_matches_full_reference": True,
            "calibration_ids": sorted(f"{dataset}:{sequence_id}" for dataset, sequence_id in calibration_ids),
        },
        "evaluation_rule": (
            "Use the calibration set only to validate metric directions and corruption sensitivity. "
            "Use the full reference suite only to report conditional GT ranges. "
            "Freeze a separate final test split before selecting or ranking a model."
        ),
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        "# GT Benchmark Protocol v1（中文）",
        "",
        "本协议把‘指标校准’、‘GT 分布估计’和‘论文最终测试’分开，避免把全量数据混作一个分数。",
        "",
        "## 当前数据状态",
        "",
        "| 数据角色 | AIST++ | FineDance | 总数 | 用途 |",
        "|---|---:|---:|---:|---|",
        f"| calibration_38 | {_counts(calibration).get('aistpp', 0)} | {_counts(calibration).get('finedance', 0)} | {len(calibration)} | 指标方向和 corruption 校准 |",
        f"| full_reference_1611 | {full_counts.get('aistpp', 0)} | {full_counts.get('finedance', 0)} | {len(full_reference)} | 条件化 GT 分布和风格/tempo 分析 |",
        "| final_test | 待冻结 | 待冻结 | 待冻结 | M2/M3/M_exec 最终论文比较 |",
        "",
        "## 使用规则",
        "",
        "1. `calibration_38` 可以用于确定指标方向、corruption 严重程度和实现错误。",
        "2. `full_reference_1611` 只用于估计 dataset/style/tempo 条件下的 GT reference range，不能用于挑选最佳 checkpoint。",
        "3. `final_test` 必须在模型选择、阈值和特征方案冻结后单独划分；在它冻结前，不能声称论文最终结果。",
        "4. 生成动作和 SONIC execution 都必须与相同条件的 GT reference 比较，不能把不同风格或不同 tempo 的 GT 合并成一个理想分数。",
        "",
        "## 当前 benchmark 结论",
        "",
        "当前 benchmark 已经可以做诊断性 motion-generation 评估，但仍处于 calibration/reference 阶段。",
        "jerk、低通能量、static ratio 的方向校准较稳定；FineDance 的 Beat F1 对 freeze corruption 未表现出稳定下降，",
        "因此 Beat F1 必须和 BAS、coverage、impact correlation、lag、tempo、phase 联合报告，不能单独作为质量 gate。",
        "repeat similarity、FID/Div、retrieval 和人类审美评分仍需独立校准。",
        "",
        "## 生成模型评估入口",
        "",
        "模型结果应报告三层：`M_ref`、`M_exec`、`M_exec / M_ref retention`。每一层同时报告动作质量、",
        "音乐适配和运行/跟踪指标；‘优美度’和风格真实性需要人工盲评，不从 BAS 反推。",
        "",
        "机器可读协议：`protocol.json`。",
    ]
    (OUTPUT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote benchmark protocol to {OUTPUT}")
    print(f"calibration={_counts(calibration)} total={len(calibration)}")
    print(f"full_reference={full_counts} total={len(full_reference)}")


if __name__ == "__main__":
    main()
