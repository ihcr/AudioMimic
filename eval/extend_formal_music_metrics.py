"""Add event/tempo/phase beat metrics to available formal model artifacts."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_motion_music_execution import (
    DEFAULT_MODEL_PATH,
    compute_motion_quality,
)
from eval.evaluate_finedance_gt_oracle import _music_extension


MODEL_METRICS = ROOT / "eval/motion_music_execution/gt_calibrated_m0_m2_m4_song098_v2/reference_metrics.json"
EXECUTION_METRICS = ROOT / "eval/motion_music_execution/gt_calibrated_m0_m2_m4_song098_v2/execution_metrics.json"
SONIC_PORTABLE = ROOT / "eval/benchmark_v1/formal/sonic_portable"
M3_PAIRS = [
    ROOT / "eval/mrt2_metrics/m3_012_pair_metrics.json",
    ROOT / "eval/mrt2_metrics/m3_065_pair_metrics.json",
]
OUTPUT = ROOT / "eval/benchmark_v1/formal/model_music_extended"
REPORT_EN = OUTPUT / "REPORT.md"
REPORT_ZH = OUTPUT / "REPORT_ZH.md"


def _load_motion(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "fps": float(payload.get("fps", 30.0)),
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
    }


def _evaluate(route: str, sequence_id: str, stage: str, motion_path: Path, audio_path: Path, label: str) -> dict:
    motion = _load_motion(motion_path)
    quality, speed_curve = compute_motion_quality(
        motion, model_path=DEFAULT_MODEL_PATH, quat_order="xyzw"
    )
    music = _music_extension(
        speed_curve,
        motion["fps"],
        audio_path,
        {},
        tolerance=0.20,
        max_lag_seconds=1.0,
    )
    return {
        "route": route,
        "sequence_id": sequence_id,
        "stage": stage,
        "label": label,
        "motion_path": str(motion_path),
        "audio_path": str(audio_path),
        "quality": quality,
        "music": music,
    }


def main() -> None:
    rows = []
    for item in json.loads(MODEL_METRICS.read_text(encoding="utf-8")):
        rows.append(_evaluate(
            item["route"], str(item["sequence_id"]), "M_ref",
            Path(item["motion_path"]), Path(item["audio_path"]), item["motion_id"],
        ))
    for item in json.loads(EXECUTION_METRICS.read_text(encoding="utf-8")):
        portable = SONIC_PORTABLE / f"{item['run_id']}.pkl"
        if not portable.is_file():
            continue
        audio_path = ROOT / "../Musics2Dance-prior-dev/onlinegeneratedmotion/audio" / f"{item['sequence_id']}_t000128_60s.wav"
        rows.append(_evaluate(
            item["route"], str(item["sequence_id"]), "M_exec", portable,
            audio_path.resolve(), item["run_id"],
        ))
    for pair_path in M3_PAIRS:
        item = json.loads(pair_path.read_text(encoding="utf-8"))
        sequence_id = item["label"].split("-")[-1]
        rows.append(_evaluate(
            "M3", sequence_id, "M_ref", Path(item["reference_pkl"]),
            Path(item["audio"]), item["label"],
        ))
        rows.append(_evaluate(
            "M3", sequence_id, "M_exec", Path(item["execution_pkl"]),
            Path(item["audio"]), item["label"] + "_execution",
        ))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps({
        "schema_version": "formal_model_music_metrics_v1",
        "description": "Extended beat suite for artifacts with complete motion and aligned audio.",
        "rows": rows,
        "missing_policy": "Only artifacts with complete motion and aligned audio are included; future matrix cells remain pending.",
    }, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    fields = [
        "route", "sequence_id", "stage", "label", "event_precision", "event_recall",
        "event_f1", "event_timing_error_seconds", "audio_bpm", "motion_bpm",
        "tempo_error_bpm", "phase_error_cycles", "speed_corr", "impact_corr",
        "impact_lag_seconds", "bas_music_to_motion", "bas_motion_to_music",
    ]
    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            music = row["music"]
            event = music["event_f1"]
            phase = music["phase"]
            writer.writerow({
                "route": row["route"],
                "sequence_id": row["sequence_id"],
                "stage": row["stage"],
                "label": row["label"],
                "event_precision": event["precision"],
                "event_recall": event["recall"],
                "event_f1": event["f1"],
                "event_timing_error_seconds": event["median_abs_timing_error_seconds"],
                "audio_bpm": music["audio_bpm"],
                "motion_bpm": music["motion_impact_bpm"],
                "tempo_error_bpm": music["tempo_abs_error_bpm"],
                "phase_error_cycles": phase["mean_phase_error_cycles"],
                "speed_corr": music["speed_best_correlation"],
                "impact_corr": music["impact_best_correlation"],
                "impact_lag_seconds": music["impact_best_lag_seconds"],
                "bas_music_to_motion": music["bas_music_to_motion"],
                "bas_motion_to_music": music["bas_motion_to_music"],
            })
    lines = [
        "# Extended Music Metrics",
        "",
        "This report adds the beat event suite to artifacts with both complete motion",
        "and aligned audio. It is complementary to BAS, not a replacement for it.",
        "",
        "| route | stage | n | event P | event R | event F1 | tempo error | phase error | speed corr | impact corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in sorted({r["route"] for r in rows}):
        for stage in ("M_ref", "M_exec"):
            group = [r for r in rows if r["route"] == route and r["stage"] == stage]
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
                return float(np.mean(values)) if values else None
            lines.append(
                f"| {route} | {stage} | {len(group)} | {avg(('music','event_f1','precision')):.4f} "
                f"| {avg(('music','event_f1','recall')):.4f} | {avg(('music','event_f1','f1')):.4f} "
                f"| {avg(('music','tempo_abs_error_bpm')):.4f} | {avg(('music','phase','mean_phase_error_cycles')):.4f} "
                f"| {avg(('music','speed_best_correlation')):.4f} | {avg(('music','impact_best_correlation')):.4f} |"
            )
    lines += [
        "",
        "The previously recorded M0/M2/M4 SONIC runs were recovered from their feedback logs",
        "as portable measured-motion PKLs and are included above. Future matrix cells remain pending.",
    ]
    REPORT_EN.write_text("\n".join(lines) + "\n", encoding="utf-8")

    zh_lines = [
        "# 扩展音乐与节奏指标报告",
        "",
        "本报告在动作质量和 SONIC retention 指标之外，补充音乐-动作对应关系的事件级、",
        "速度级和相位级评估。BAS 是 beat alignment 的核心指标，但不能单独代表舞蹈质量；",
        "因此必须和事件覆盖、tempo、phase、onset/impact correlation 以及 response lag 一起解释。",
        "",
        "## 指标定义",
        "",
        "- **Event Precision / Recall / F1**：音乐 beat/onset 与动作 impact 事件在容差窗口内的精确率、召回率和 F1。",
        "- **Tempo error (BPM)**：动作 impact 的估计速度与音乐 tempo 的绝对差，越低越好。",
        "- **Phase error (cycles)**：动作 impact 相对音乐 beat 的周期相位误差，越低越好。",
        "- **Speed / impact correlation**：动作速度包络或 impact 强度与音乐对应曲线的最佳相关性，越高越好。",
        "- **BAS / reverse BAS**：音乐 beat 到动作 beat、以及动作 beat 到音乐 beat 的双向对齐分数。",
        "",
        "## 当前结果",
        "",
        "以下为同一路线内的平均值；`M_ref` 是生成器参考轨迹，`M_exec` 是 SONIC 执行后的测量轨迹。",
        "这些结果用于检查评估链路和 execution retention，不作为未平衡矩阵上的最终模型排名。",
        "",
        "| 路线 | 阶段 | n | Event P | Event R | Event F1 | Tempo error (BPM) | Phase error (cycles) | Speed corr | Impact corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in sorted({r["route"] for r in rows}):
        for stage in ("M_ref", "M_exec"):
            group = [r for r in rows if r["route"] == route and r["stage"] == stage]
            if not group:
                continue

            def avg_zh(path: tuple[str, ...]):
                values = []
                for item in group:
                    value = item
                    for key in path:
                        value = value[key]
                    if value is not None:
                        values.append(float(value))
                return float(np.mean(values)) if values else float("nan")

            zh_lines.append(
                f"| {route} | {stage} | {len(group)} | {avg_zh(('music','event_f1','precision')):.4f} "
                f"| {avg_zh(('music','event_f1','recall')):.4f} | {avg_zh(('music','event_f1','f1')):.4f} "
                f"| {avg_zh(('music','tempo_abs_error_bpm')):.4f} | {avg_zh(('music','phase','mean_phase_error_cycles')):.4f} "
                f"| {avg_zh(('music','speed_best_correlation')):.4f} | {avg_zh(('music','impact_best_correlation')):.4f} |"
            )
    zh_lines += [
        "",
        "## 数据状态",
        "",
        "旧的 M0/M2/M4 SONIC 记录已经从 feedback log 导出为完整的 measured-motion PKL，",
        "因此本报告已经包含这些记录的 `M_exec` 事件指标，不需要为这 9 次旧实验重新录制。",
        "尚未采集的 72-cell 正式矩阵单元仍然标记为 pending；后续只有这些新单元需要重新运行",
        "SONIC 并按相同协议记录。",
        "",
        "本报告不把 BAS 当作唯一判断标准，也不把自动指标直接等同于‘舞蹈优美’。最终结论应结合",
        "GT 分层分布、paired/wrong-song counterfactual、动作质量、音乐匹配和人工盲评。",
    ]
    REPORT_ZH.write_text("\n".join(zh_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} extended music rows to {OUTPUT}")


if __name__ == "__main__":
    main()
