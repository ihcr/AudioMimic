"""Audit FineDance-G1 root-height anomalies without changing the benchmark.

The audit separates a transient root-z excursion from a broader retargeting or
coordinate-convention problem.  These labels are diagnostics only; they are
not quality scores and do not remove clips from the sealed benchmark.
    """

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MANIFEST = Path(
    "eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"
)
DEFAULT_SUITE = Path(
    "eval/benchmark_v1/gt/gt_oracle_suite_all_v1/gt_oracle_suite_metrics.json"
)
DEFAULT_OUTPUT = Path("eval/benchmark_v1/gt/finedance_root_height_audit_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--median-height-warning-m", type=float, default=0.60)
    parser.add_argument("--max-transient-negative-fraction", type=float, default=0.05)
    parser.add_argument("--max-penetration-ratio", type=float, default=0.02)
    parser.add_argument("--min-foot-height-warning-m", type=float, default=-0.12)
    return parser.parse_args()


def _load_root_z(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: unexpected root_pos shape {root_pos.shape}")
    if not np.isfinite(root_pos).all():
        raise ValueError(f"{path}: non-finite root_pos")
    return root_pos[:, 2]


def _metric_map(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["sequence_id"]): row for row in suite["records"]}


def _canonical_styles(values: list[str]) -> str:
    styles = []
    for value in values:
        normalized = str(value).strip()
        normalized = "Jazz" if normalized.casefold() == "jazz" else normalized
        styles.append(normalized)
    return ";".join(sorted(set(styles)))


def classify_root_height(
    *,
    root_min: float,
    root_median: float,
    negative_fraction: float,
    penetration_ratio: float,
    foot_min: float,
    median_height_warning_m: float,
    max_transient_negative_fraction: float,
    max_penetration_ratio: float,
    min_foot_height_warning_m: float,
) -> str:
    if root_min >= 0.0:
        return "valid_root_z"
    if (
        root_median >= median_height_warning_m
        and negative_fraction <= max_transient_negative_fraction
        and penetration_ratio <= max_penetration_ratio
        and foot_min >= min_foot_height_warning_m
    ):
        return "transient_root_drop_feet_near_ground"
    return "root_drop_with_physical_warning"


def _record(
    manifest_row: dict[str, Any],
    suite_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    motion_path = Path(manifest_row["g1_motion"]["path"])
    z = _load_root_z(motion_path)
    quality = suite_row["quality"]
    physical = quality["physical"]
    negative_fraction = float(np.mean(z < 0.0))
    record = {
        "sequence_id": str(manifest_row["sequence_id"]),
        "style": _canonical_styles(manifest_row.get("style", [])),
        "motion_path": str(motion_path),
        "frames": int(z.size),
        "root_height_min_m": float(np.min(z)),
        "root_height_p01_m": float(np.percentile(z, 1.0)),
        "root_height_median_m": float(np.median(z)),
        "root_height_max_m": float(np.max(z)),
        "negative_root_fraction": negative_fraction,
        "minimum_foot_height_relative_m": float(
            physical["minimum_foot_height_relative_m"]
        ),
        "ground_penetration_ratio": float(physical["ground_penetration_ratio"]),
        "estimated_ground_height_m": float(physical["estimated_ground_height_m"]),
    }
    record["diagnostic_label"] = classify_root_height(
        root_min=record["root_height_min_m"],
        root_median=record["root_height_median_m"],
        negative_fraction=negative_fraction,
        penetration_ratio=record["ground_penetration_ratio"],
        foot_min=record["minimum_foot_height_relative_m"],
        median_height_warning_m=args.median_height_warning_m,
        max_transient_negative_fraction=args.max_transient_negative_fraction,
        max_penetration_ratio=args.max_penetration_ratio,
        min_foot_height_warning_m=args.min_foot_height_warning_m,
    )
    return record


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = Counter(row["diagnostic_label"] for row in records)
    by_style: dict[str, dict[str, int]] = {}
    for row in records:
        for style in filter(None, row["style"].split(";")):
            by_style.setdefault(style, Counter())[row["diagnostic_label"]] += 1
    return {
        "clips": len(records),
        "labels": dict(sorted(by_label.items())),
        "style_labels": {
            style: dict(sorted(values.items()))
            for style, values in sorted(by_style.items())
        },
    }


def _write_report(path: Path, records: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# FineDance-G1 Root Height Audit v1",
        "",
        "这是一项已有 G1 资产/retargeting/坐标约定诊断，不是舞蹈质量评分，也不修改或删除 benchmark 数据。",
        "`root_height_min_m` 只看 root_pos 的 z 轴；脚部穿透和接触使用 FK 后的地面相对高度单独统计。",
        "",
        "## 判定规则",
        "",
        f"- `valid_root_z`: root z 全部不小于 0。",
        f"- `transient_root_drop_feet_near_ground`: root z 有负值，但中位数 >= {args.median_height_warning_m:.2f} m、负值比例 <= {args.max_transient_negative_fraction:.2f}、脚部穿透比例 <= {args.max_penetration_ratio:.2f} 且最低脚高 >= {args.min_foot_height_warning_m:.2f} m。",
        "- `root_drop_with_physical_warning`: 负 root z 同时伴随较低根部中位高度或明显脚部物理异常。",
        "- 以上阈值仅用于定位问题，不能作为训练集删选或论文质量分数。",
        "",
        "## 总体",
        "",
        "| diagnostic label | clips |",
        "|---|---:|",
    ]
    for label, count in summary["labels"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## 风格分布", "", "| style | clips | labels |", "|---|---:|---|"])
    for style, labels in summary["style_labels"].items():
        count = sum(labels.values())
        compact = ", ".join(f"{label}: {value}" for label, value in labels.items())
        lines.append(f"| {style} | {count} | {compact} |")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "当前应把负 root z 视为已有 G1 资产的 root/ground convention 待核查信号，不能直接解释为 Breaking 或其他风格的动作质量下降。",
            "若 root z 出现负值但脚部穿透很低，优先检查 root 轨迹的轴定义、全局平移与地面参考；若两者同时异常，再检查上游 retargeting 的姿态/地面约束。",
            "在问题修正前，D5 的 `root_height_min` 只作为诊断字段，generator 与 SONIC execution 的正式比较应优先使用 foot penetration、FSR/PFC proxy 及其他冻结指标。",
            "",
            "## 文件",
            "",
            "- `root_height_audit.csv`: 每条 FineDance 序列的诊断记录。",
            "- `root_height_audit.json`: 阈值、原始路径、完整记录和汇总。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    dataset = manifest["datasets"]["finedance"]
    suite_rows = _metric_map(suite)
    records = []
    for row in dataset["records"]:
        if not row.get("paired_valid"):
            continue
        sequence_id = str(row["sequence_id"])
        if sequence_id not in suite_rows:
            raise KeyError(f"Missing suite metrics for FineDance sequence {sequence_id}")
        records.append(_record(row, suite_rows[sequence_id], args))
    if len(records) != 203:
        raise RuntimeError(f"Expected 203 FineDance records, found {len(records)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(records)
    payload = {
        "schema_version": "finedance_root_height_audit_v1",
        "manifest": str(args.manifest.resolve()),
        "suite": str(args.suite.resolve()),
        "thresholds": {
            "median_height_warning_m": args.median_height_warning_m,
            "max_transient_negative_fraction": args.max_transient_negative_fraction,
            "max_penetration_ratio": args.max_penetration_ratio,
            "min_foot_height_warning_m": args.min_foot_height_warning_m,
        },
        "summary": summary,
        "records": records,
    }
    (output_dir / "root_height_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    fields = list(records[0])
    with (output_dir / "root_height_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    _write_report(output_dir / "REPORT_ZH.md", records, summary, args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
