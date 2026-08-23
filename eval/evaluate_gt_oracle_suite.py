"""Evaluate the sealed AIST++ and FineDance paired G1 data as one oracle suite.

The suite is model-independent.  It does not train a model and it does not
declare a single aesthetic ground truth.  It creates per-dataset and pooled
reference distributions that can be reused when scoring generator outputs and
SONIC executions.

The default scope is the audited held-out test split of each dataset:
20 AIST++ sequences and 18 FineDance cross-genre sequences.  ``--scope all``
is available for a broader descriptive calibration, but must not replace the
held-out test report in a paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_motion_music_execution import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    compute_motion_quality,
)
from eval.evaluate_finedance_gt_oracle import _music_extension  # noqa: E402


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "motion_energy": ("quality", "motion_energy_rad2_s2"),
    "joint_jerk_p95": ("quality", "joint_jerk_abs_rad_s3", "p95"),
    "static_ratio": ("quality", "static_ratio_speed_below_008"),
    "repeated_pose_ratio": ("quality", "repeated_pose_ratio_rms008_after2s"),
    "fsr_proxy": ("quality", "physical", "fsr_ground_calibrated_proxy"),
    "pfc_proxy": ("quality", "physical", "pfc_proxy"),
    "root_height_min": ("quality", "root_height_min_m"),
    "speed_corr": ("music", "speed_best_correlation"),
    "impact_corr": ("music", "impact_best_correlation"),
    "impact_abs_lag": ("music", "impact_best_lag_seconds"),
    "bas": ("music", "bas_music_to_motion"),
    "event_f1": ("music", "event_f1", "f1"),
    "event_precision": ("music", "event_f1", "precision"),
    "event_recall": ("music", "event_f1", "recall"),
    "event_timing_error": (
        "music",
        "event_f1",
        "median_abs_timing_error_seconds",
    ),
    "audio_bpm": ("music", "audio_bpm"),
    "motion_bpm": ("music", "motion_impact_bpm"),
    "tempo_error": ("music", "tempo_abs_error_bpm"),
    "phase_error": ("music", "phase", "mean_phase_error_cycles"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aist-manifest",
        type=Path,
        default=Path("eval/benchmark_v1/gt/manifest_v1/gt_benchmark_manifest.json"),
    )
    parser.add_argument(
        "--finedance-manifest",
        type=Path,
        default=Path("eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/benchmark_v1/gt/gt_oracle_suite_v2"),
    )
    parser.add_argument("--scope", choices=("test", "all"), default="test")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--event-tolerance-seconds", type=float, default=0.20)
    parser.add_argument("--max-lag-seconds", type=float, default=1.0)
    return parser.parse_args()


def _load_motion(path: Path) -> dict[str, np.ndarray | float]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "fps": float(payload.get("fps", 30.0)),
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
    }


def _records_from_manifest(manifest_path: Path, dataset_name: str, scope: str) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = payload["datasets"][dataset_name]
    if scope == "all":
        records = dataset["records"]
    else:
        test_ids = set(dataset["split"]["test_assets_available"])
        records = [record for record in dataset["records"] if record["sequence_id"] in test_ids]
    return [record for record in records if record.get("paired_valid")]


def _dataset_paths(dataset_name: str, record: dict[str, Any]) -> tuple[Path, Path, float | None, str]:
    if dataset_name == "aistpp":
        motion = record["motion"]
        audio = record["audio"]
        duration = min(float(motion["duration_seconds"]), float(audio["duration_seconds"]))
        return Path(motion["path"]), Path(audio["path"]), duration, motion["root_quat_order"]
    if dataset_name == "finedance":
        motion = record["g1_motion"]
        audio = record["audio"]
        return (
            Path(motion["path"]),
            Path(audio["path"]),
            float(record["common_duration_seconds"]),
            motion["root_quat_order"],
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _get_path(row: dict[str, Any], path: tuple[str, ...]) -> object:
    value: object = row
    for key in path:
        value = value[key]  # type: ignore[index]
    return value


def _finite_values(rows: list[dict[str, Any]], name: str, path: tuple[str, ...]) -> np.ndarray:
    values = []
    for row in rows:
        value = _get_path(row, path)
        if value is not None and np.isfinite(float(value)):
            numeric = float(value)
            if name == "impact_abs_lag":
                numeric = abs(numeric)
            values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def _distribution(values: np.ndarray, direction: str) -> dict[str, Any]:
    if not len(values):
        return {"count": 0, "direction": direction}
    return {
        "count": int(len(values)),
        "direction": direction,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q10": float(np.percentile(values, 10)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "q90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _evaluate_record(
    dataset_name: str,
    record: dict[str, Any],
    *,
    model_path: Path,
    audio_cache: dict,
    event_tolerance_seconds: float,
    max_lag_seconds: float,
) -> dict[str, Any]:
    motion_path, audio_path, common_duration, quat_order = _dataset_paths(dataset_name, record)
    motion = _load_motion(motion_path)
    fps = float(motion["fps"])
    frames = min(len(motion["dof_pos"]), int(round(common_duration * fps))) if common_duration else len(motion["dof_pos"])
    for key in ("root_pos", "root_rot", "dof_pos"):
        motion[key] = np.asarray(motion[key])[:frames]
    quality, speed_curve = compute_motion_quality(
        motion,
        model_path=model_path,
        quat_order=quat_order,
    )
    music = _music_extension(
        speed_curve,
        fps,
        audio_path,
        audio_cache,
        event_tolerance_seconds,
        max_lag_seconds,
    )
    return {
        "dataset": dataset_name,
        "sequence_id": record["sequence_id"],
        "frames": frames,
        "duration_seconds": float(frames / fps),
        "style": record.get("style", []),
        "motion_path": str(motion_path),
        "audio_path": str(audio_path),
        "quality": quality,
        "music": music,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    distributions = {}
    directions = {
        "motion_energy": "reference_only",
        "joint_jerk_p95": "lower_is_better",
        "static_ratio": "lower_is_better",
        "repeated_pose_ratio": "lower_is_better",
        "fsr_proxy": "lower_is_better",
        "pfc_proxy": "lower_is_better",
        "root_height_min": "higher_is_better_within_valid_range",
        "speed_corr": "higher_is_better",
        "impact_corr": "higher_is_better",
        "impact_abs_lag": "lower_is_better",
        "bas": "higher_is_better",
        "event_f1": "higher_is_better",
        "event_precision": "higher_is_better",
        "event_recall": "higher_is_better",
        "event_timing_error": "lower_is_better",
        "audio_bpm": "reference_only",
        "motion_bpm": "reference_only",
        "tempo_error": "lower_is_better",
        "phase_error": "lower_is_better",
    }
    for name, path in METRIC_PATHS.items():
        distributions[name] = _distribution(_finite_values(rows, name, path), directions[name])
    return {
        "count": len(rows),
        "datasets": {name: sum(row["dataset"] == name for row in rows) for name in ("aistpp", "finedance")},
        "metrics": distributions,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["dataset", "sequence_id", "frames", "duration_seconds", "style"] + list(METRIC_PATHS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {
                "dataset": row["dataset"],
                "sequence_id": row["sequence_id"],
                "frames": row["frames"],
                "duration_seconds": row["duration_seconds"],
                "style": ";".join(row["style"]) if isinstance(row["style"], list) else row["style"],
            }
            for name, metric_path in METRIC_PATHS.items():
                flat[name] = _get_path(row, metric_path)
            writer.writerow(flat)


def _write_report(payload: dict[str, Any], path: Path) -> None:
    report = [
        "# Multi-dataset GT Oracle Suite",
        "",
        "This report calibrates the common evaluation protocol on paired G1 motion and audio.",
        "The default held-out suite combines AIST++ crossmodal test and FineDance cross-genre test.",
        "It is a reference distribution, not a single aesthetic ground-truth score.",
        "",
        f"Scope: **{payload['scope']}**; sequences: **{payload['summary']['count']}**.",
        "",
        "## Dataset coverage",
        "",
        "| dataset | sequences |",
        "|---|---:|",
    ]
    for name, count in payload["summary"]["datasets"].items():
        report.append(f"| `{name}` | {count} |")
    report.extend(["", "## Pooled reference distributions", "", "| metric | direction | median | q10 | q90 | mean | std |", "|---|---|---:|---:|---:|---:|---:|"])
    for name, values in payload["summary"]["metrics"].items():
        if not values.get("count"):
            report.append(f"| `{name}` | {values['direction']} | n/a | n/a | n/a | n/a | n/a |")
            continue
        report.append(
            f"| `{name}` | {values['direction']} | {values['median']:.6f} | "
            f"{values['q10']:.6f} | {values['q90']:.6f} | {values['mean']:.6f} | {values['std']:.6f} |"
        )
    report.extend(
        [
            "",
            "## Use in generator and execution evaluation",
            "",
            "The same metrics must be computed for generator reference and SONIC execution. "
            "The oracle suite supplies calibration ranges; it does not replace paired comparison, "
            "execution retention, or blinded human evaluation.",
            "",
            "Raw values must be reported per dataset before any pooled summary because AIST++ "
            "and FineDance differ in style distribution, duration, and retargeting statistics.",
            "",
            "For a metric with direction `higher_is_better`, a generated result is calibrated "
            "against the oracle quantiles directly. For `lower_is_better`, the inequality is reversed. "
            "`reference_only` metrics describe the data distribution and are not quality gates.",
        ]
    )
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_chinese_report(payload: dict[str, Any], path: Path) -> None:
    report = [
        "# 多数据集 GT Oracle Benchmark（中文）",
        "",
        "本报告使用 AIST++ 和 FineDance 的 paired 音乐-舞蹈数据，建立生成器和 SONIC",
        "执行结果的参考分布。它不是把每条真实舞蹈压缩成一个必须达到 1.0 的分数，",
        "而是按数据集、速度和风格提供可比较的 calibration range。",
        "",
        f"- 评估范围：`{payload['scope']}`",
        f"- 总序列数：**{payload['summary']['count']}**",
        f"- AIST++：**{payload['summary']['datasets'].get('aistpp', 0)}** 条",
        f"- FineDance：**{payload['summary']['datasets'].get('finedance', 0)}** 条",
        "- beat 指标使用固定 audio clock、固定 beat detector 和固定事件容差；模型和 SONIC 必须复用同一协议。",
        "",
        "## 数据集分层的核心音乐指标",
        "",
        "表中为每个数据集的中位数。BAS、Event F1、相关性越高越好；tempo/phase/lag 越接近 0 越好。",
        "",
        "| 数据集 | n | BAS | Event F1 | Tempo error (BPM) | Phase error | Impact corr | Lag (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("aistpp", "finedance"):
        summary = payload["per_dataset"].get(dataset, {})
        metrics = summary.get("metrics", {})

        def median(name: str) -> str:
            value = metrics.get(name, {}).get("median")
            return "n/a" if value is None else f"{float(value):.4f}"

        report.append(
            f"| {dataset} | {summary.get('count', 0)} | {median('bas')} | {median('event_f1')} "
            f"| {median('tempo_error')} | {median('phase_error')} | {median('impact_corr')} | {median('impact_abs_lag')} |"
        )
    report += [
        "",
        "## 指标含义",
        "",
        "- **BAS**：动作 kinematic beat 到最近音乐 beat 的时间接近程度，是 beat-alignment 的核心指标，但不是唯一指标。",
        "- **Beat Event F1**：动作 impact 对音乐 beat/onset 的命中与覆盖，联合 Precision、Recall 解释，避免少量动作 beat 造成虚高 BAS。",
        "- **Tempo error**：动作 impact 速度与音乐节奏速度的 BPM 差异。",
        "- **Phase error**：动作 accent 相对于音乐周期的位置误差。",
        "- **Impact correlation**：动作 impact 强度与音乐 onset/能量曲线的相关性。",
        "- **Lag**：两者最佳相关对应的时间偏移；只有相关性达到可靠性阈值时才解释。",
        "",
        "## 如何用于模型和 tracker",
        "",
        "1. `O-Human` 用于观察原始 SMPL/SMPLH paired 数据的自然分布。",
        "2. `O-G1` 用于报告 GMR/retargeting 后的变化；这不是 diffusion 的生成误差。",
        "3. `M_ref` 与匹配的 O-G1/GT 分层比较，报告 generator gap。",
        "4. `M_exec` 使用同一条 reference 经过 SONIC 的测量轨迹，报告 tracking retention。",
        "",
        "不同数据集的数值不能直接混合排名。最终论文还需要 paired/wrong-song、time-shifted、",
        "tempo-preserved counterfactual，以及 R@K/MMDist 和人工盲评；自动指标不能单独等同于",
        "‘舞蹈优美’或‘音乐语义匹配’。",
        "",
        f"完整逐条数据见 `{path.with_name('gt_oracle_suite_metrics.json').name}` 和 `gt_oracle_suite_summary.csv`。",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset_name, manifest_path in (("aistpp", args.aist_manifest), ("finedance", args.finedance_manifest)):
        dataset_records = _records_from_manifest(manifest_path, dataset_name, args.scope)
        print(f"{dataset_name}: {len(dataset_records)} paired {args.scope} records", flush=True)
        records.extend((dataset_name, record) for record in dataset_records)
    if not records:
        raise RuntimeError("No paired records found")

    audio_cache: dict = {}
    rows = []
    for index, (dataset_name, record) in enumerate(sorted(records, key=lambda item: (item[0], item[1]["sequence_id"])), start=1):
        row = _evaluate_record(
            dataset_name,
            record,
            model_path=args.model_path,
            audio_cache=audio_cache,
            event_tolerance_seconds=args.event_tolerance_seconds,
            max_lag_seconds=args.max_lag_seconds,
        )
        rows.append(row)
        print(
            f"[{index}/{len(records)}] {dataset_name}:{row['sequence_id']} "
            f"event_f1={row['music']['event_f1']['f1']:.3f} "
            f"bas={row['music']['bas_music_to_motion']:.3f}",
            flush=True,
        )

    per_dataset = {
        name: _summary([row for row in rows if row["dataset"] == name])
        for name in ("aistpp", "finedance")
    }
    payload = {
        "schema_version": "gt_oracle_suite_v1",
        "scope": args.scope,
        "model_independent": True,
        "manifests": {
            "aistpp": str(args.aist_manifest.resolve()),
            "finedance": str(args.finedance_manifest.resolve()),
        },
        "event_tolerance_seconds": args.event_tolerance_seconds,
        "max_lag_seconds": args.max_lag_seconds,
        "summary": _summary(rows),
        "per_dataset": per_dataset,
        "records": rows,
    }
    (args.output_dir / "gt_oracle_suite_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(rows, args.output_dir / "gt_oracle_suite_summary.csv")
    _write_report(payload, args.output_dir / "REPORT.md")
    _write_chinese_report(payload, args.output_dir / "REPORT_ZH.md")
    print(f"Saved oracle suite to {args.output_dir}")


if __name__ == "__main__":
    main()
