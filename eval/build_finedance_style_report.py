"""Build the Chinese FineDance style/tempo and multi-metric report."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval/benchmark_v1/gt/manifest_v2_finedance/gt_benchmark_manifest.json"
WINDOWED = ROOT / "eval/benchmark_v1/gt/finedance_windowed_v1/windowed_metrics.csv"
ORACLE = ROOT / "eval/benchmark_v1/gt/gt_oracle_suite_all_v1/gt_oracle_suite_metrics.json"
RETARGET = ROOT / "eval/benchmark_v1/gt/retargeting_loss_all_v1/retargeting_loss_metrics.json"
OUTPUT = ROOT / "eval/benchmark_v1/gt/finedance_stratified_v1"


def _canonical_style(value: str) -> str:
    aliases = {"jazz": "Jazz"}
    return aliases.get(value, value)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _read_rows() -> tuple[dict[str, dict], dict[str, dict]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["datasets"]["finedance"]
    paired = [r for r in manifest["records"] if r.get("paired_valid")]
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    oracle_rows = {str(r["sequence_id"]): r for r in oracle["records"] if r["dataset"] == "finedance"}
    retarget = json.loads(RETARGET.read_text(encoding="utf-8"))
    retarget_rows = {str(r["sequence_id"]): r for r in retarget["records"] if r["dataset"] == "finedance"}
    meta = {}
    for row in paired:
        sid = str(row["sequence_id"])
        bpm = oracle_rows[sid]["music"].get("audio_bpm")
        tempo = "unknown" if bpm is None else ("slow" if bpm < 90 else "medium" if bpm <= 130 else "fast")
        meta[sid] = {
            "styles": sorted({_canonical_style(str(s)) for s in row.get("style", [])}) or ["Unknown"],
            "tempo": tempo,
            "audio_bpm": bpm,
            "oracle": oracle_rows[sid],
            "retarget": retarget_rows[sid],
        }
    window_rows = {}
    with WINDOWED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["window"] != "full":
                continue
            sid = str(row["sequence_id"])
            window_rows[(sid, row["stage"])] = {
                key: (float(value) if value not in ("", None) else None)
                for key, value in row.items()
                if key not in {"sequence_id", "stage", "window", "start_seconds"}
            }
    return meta, window_rows


def _metric_row(meta: dict, window_rows: dict, sid: str, stage: str) -> dict:
    row = window_rows[(sid, stage)]
    return {
        "bas": row["bas_music_to_motion"],
        "reverse_bas": row["bas_motion_to_music"],
        "event_f1": row["event_f1"],
        "impact_corr": row["impact_corr"],
        "abs_lag": abs(row["impact_lag_seconds"]),
        "tempo_error": row["tempo_error_bpm"],
        "phase_error": row["phase_error_cycles"],
    }


def _style_counts(meta: dict) -> dict[str, int]:
    styles = sorted({style for item in meta.values() for style in item["styles"]})
    return {style: sum(style in item["styles"] for item in meta.values()) for style in styles}


def _group_summary(meta: dict, window_rows: dict, dimension: str, value: str, stage: str) -> dict | None:
    sids = [sid for sid, item in meta.items() if value in item["styles"]] if dimension == "style" else [sid for sid, item in meta.items() if item["tempo"] == value]
    rows = [_metric_row(meta, window_rows, sid, stage) for sid in sids if (sid, stage) in window_rows]
    if not rows:
        return None
    output = {"dimension": dimension, "group": value, "stage": stage, "n_sequences": len(rows)}
    for key in rows[0]:
        values = [float(row[key]) for row in rows if row[key] is not None and np.isfinite(float(row[key]))]
        output[f"{key}_median"] = _median(values)
        output[f"{key}_mean"] = _mean(values)
    return output


def _quality_row(meta: dict, sid: str) -> dict:
    quality = meta[sid]["oracle"]["quality"]
    physical = quality.get("physical", {})
    return {
        "motion_energy": quality.get("motion_energy_rad2_s2"),
        "velocity_p95": quality.get("joint_velocity_abs_rad_s", {}).get("p95"),
        "acceleration_p95": quality.get("joint_acceleration_abs_rad_s2", {}).get("p95"),
        "jerk_p95": quality.get("joint_jerk_abs_rad_s3", {}).get("p95"),
        "static_ratio": quality.get("static_ratio_speed_below_008"),
        "repeated_pose_ratio": quality.get("repeated_pose_ratio_rms008_after2s"),
        "boundary_jump_p95": quality.get("c4_boundary_position_jump_p95_rad"),
        "root_displacement": quality.get("root_planar_displacement_m"),
        "root_speed_p95": quality.get("root_planar_speed_p95_m_s"),
        "root_height_min": quality.get("root_height_min_m"),
        "contact_proxy": physical.get("fsr_ground_calibrated_proxy"),
        "penetration_ratio": physical.get("ground_penetration_ratio"),
        "pfc_proxy": physical.get("pfc_proxy"),
    }


def _quality_summary(meta: dict, dimension: str, value: str) -> dict | None:
    sids = [
        sid
        for sid, item in meta.items()
        if (value in item["styles"] if dimension == "style" else item["tempo"] == value)
    ]
    rows = [_quality_row(meta, sid) for sid in sids]
    if not rows:
        return None
    output = {"dimension": dimension, "group": value, "n_sequences": len(rows)}
    for key in rows[0]:
        values = [float(row[key]) for row in rows if row[key] is not None and np.isfinite(float(row[key]))]
        output[f"{key}_median"] = _median(values)
    return output


def _feature_vectors(meta: dict, window_rows: dict, stage: str, group: str) -> tuple[list[str], np.ndarray]:
    feature_names = (
        ("bas", "reverse_bas", "event_f1", "impact_corr", "abs_lag", "tempo_error", "phase_error")
        if group == "music"
        else ("motion_energy", "joint_jerk_p95", "static_ratio", "repeated_pose_ratio", "fsr_proxy", "pfc_proxy", "root_height_min")
    )
    ids = [sid for sid in meta if (sid, stage) in window_rows]
    vectors = []
    for sid in ids:
        if group == "music":
            values = _metric_row(meta, window_rows, sid, stage)
        else:
            quality = meta[sid]["oracle"]["quality"]
            physical = quality.get("physical", {})
            values = {
                "motion_energy": quality.get("motion_energy_rad2_s2"),
                "joint_jerk_p95": quality.get("joint_jerk_abs_rad_s3", {}).get("p95"),
                "static_ratio": quality.get("static_ratio_speed_below_008"),
                "repeated_pose_ratio": quality.get("repeated_pose_ratio_rms008_after2s"),
                "fsr_proxy": physical.get("fsr_ground_calibrated_proxy"),
                "pfc_proxy": physical.get("pfc_proxy"),
                "root_height_min": quality.get("root_height_min_m"),
            }
        vectors.append([float(values[name]) if values[name] is not None else np.nan for name in feature_names])
    array = np.asarray(vectors, dtype=np.float64)
    for col in range(array.shape[1]):
        median = np.nanmedian(array[:, col])
        array[np.isnan(array[:, col]), col] = median
        scale = np.nanpercentile(array[:, col], 75) - np.nanpercentile(array[:, col], 25)
        array[:, col] = (array[:, col] - median) / max(float(scale), 1e-8)
    return ids, array


def _style_similarity(meta: dict, window_rows: dict, stage: str, group: str, permutations: int = 500) -> dict:
    ids, vectors = _feature_vectors(meta, window_rows, stage, group)
    labels = [set(meta[sid]["styles"]) for sid in ids]
    pairs = list(combinations(range(len(ids)), 2))

    def effect(current: list[set[str]]) -> tuple[float, float, float, int, int]:
        same = []
        different = []
        for left, right in pairs:
            distance = float(np.linalg.norm(vectors[left] - vectors[right]))
            if current[left] & current[right]:
                same.append(distance)
            else:
                different.append(distance)
        within = float(np.mean(same))
        between = float(np.mean(different))
        return within, between, between - within, len(same), len(different)

    within, between, observed, same_count, different_count = effect(labels)
    rng = np.random.default_rng(20260823)
    shuffled_effects = []
    for _ in range(permutations):
        shuffled = [labels[index] for index in rng.permutation(len(labels))]
        shuffled_effects.append(effect(shuffled)[2])
    p_value = float((1 + sum(value >= observed for value in shuffled_effects)) / (permutations + 1))
    return {
        "stage": stage,
        "feature_group": group,
        "n_sequences": len(ids),
        "same_style_pairs": same_count,
        "different_style_pairs": different_count,
        "within_style_distance": within,
        "between_style_distance": between,
        "between_minus_within": observed,
        "permutation_p_value": p_value,
        "interpretation": "same-style more similar" if observed > 0 and p_value < 0.05 else "not established",
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    meta, window_rows = _read_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    quality_summaries = []
    styles = sorted(_style_counts(meta))
    for dimension, values in (("style", styles), ("tempo", ["slow", "medium", "fast", "unknown"])):
        for value in values:
            for stage in ("source", "g1_target"):
                row = _group_summary(meta, window_rows, dimension, value, stage)
                if row:
                    summaries.append(row)
            quality_row = _quality_summary(meta, dimension, value)
            if quality_row:
                quality_summaries.append(quality_row)
    similarity = [
        _style_similarity(meta, window_rows, "g1_target", "music"),
        _style_similarity(meta, window_rows, "g1_target", "dance_quality"),
    ]
    (OUTPUT / "style_similarity.json").write_text(json.dumps(similarity, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUTPUT / "style_summary.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "quality_summaries": quality_summaries,
                "style_counts": _style_counts(meta),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = list(summaries[0])
    with (OUTPUT / "style_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    report = [
        "# FineDance 风格化音乐-舞蹈指标报告",
        "",
        "本报告对全部 203 条 FineDance paired 序列进行舞蹈风格和音乐 tempo 分层。",
        "每个序列可以属于多个风格标签，因此风格样本数会重叠；小于 5 条的风格只做诊断，不作为论文强结论。",
        "",
        "## 一、评估维度",
        "",
        "### 音乐-舞蹈适配",
        "BAS 双向、Beat Event Precision/Recall/F1、Impact correlation、response lag、",
        "tempo error 和 phase error。它们描述节奏、动态和相位，不等同于舞蹈美观度。",
        "",
        "### 舞蹈动作质量",
        "motion energy、jerk、static ratio、repeated pose ratio、FSR/PFC proxy、root stability",
        "和 ground/contact 统计。当前这些主要是 G1 动作质量和可执行性代理，不是完整的审美评分。",
        "",
        "### 风格相似性验证",
        "使用完整序列的标准化特征向量，比较同风格序列和不同风格序列的平均距离。",
        "通过置换风格标签计算探索性 p-value；同风格距离更小且 p<0.05 才认为数据支持",
        "‘同风格更相似’，否则记录为未建立。多标签归属使该检验属于探索性分析。",
        "",
        "## 二、风格覆盖",
        "",
        "| 舞蹈风格 | 序列数 |",
        "|---|---:|",
    ]
    counts = _style_counts(meta)
    for style, count in counts.items():
        report.append(f"| {style} | {count} |")
    report += [
        "",
        "## 三、不同舞蹈风格的音乐匹配（完整序列中位数）",
        "",
        "| 风格 | 层 | n | BAS | Event F1 | Impact corr | Tempo error | Phase error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["dimension"] != "style":
            continue
        report.append(
            f"| {row['group']} | {row['stage']} | {row['n_sequences']} | {_fmt(row['bas_median'])} "
            f"| {_fmt(row['event_f1_median'])} | {_fmt(row['impact_corr_median'])} "
            f"| {_fmt(row['tempo_error_median'])} | {_fmt(row['phase_error_median'])} |"
        )
    report += [
        "",
        "## 四、不同音乐速度的匹配",
        "",
        "速度分组：slow `<90 BPM`，medium `90--130 BPM`，fast `>130 BPM`。",
        "",
        "| Tempo | 层 | n | BAS | Event F1 | Impact corr | Tempo error | Phase error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["dimension"] != "tempo":
            continue
        report.append(
            f"| {row['group']} | {row['stage']} | {row['n_sequences']} | {_fmt(row['bas_median'])} "
            f"| {_fmt(row['event_f1_median'])} | {_fmt(row['impact_corr_median'])} "
            f"| {_fmt(row['tempo_error_median'])} | {_fmt(row['phase_error_median'])} |"
        )
    report += [
        "",
        "## 五、舞蹈动作质量（G1 oracle，完整序列中位数）",
        "",
        "以下指标在 GMR 后的 G1 reference 上计算，作为自然配对动作的动作质量基准。",
        "它们用于判断动作是否有足够活动、是否过于僵硬或重复、是否平滑且可执行；数值本身不是审美分数。",
        "jerk、边界跳变、静止率和重复姿态率通常越低越平滑，但 motion energy 过低也可能意味着平均化。",
        "",
        "| 风格 | n | Energy | Vel p95 | Acc p95 | Jerk p95 | Static | Repeat | Boundary jump | Root speed p95 | Contact | Penetration | PFC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in quality_summaries:
        if row["dimension"] != "style" or row["n_sequences"] < 5:
            continue
        report.append(
            f"| {row['group']} | {row['n_sequences']} | {_fmt(row['motion_energy_median'])} "
            f"| {_fmt(row['velocity_p95_median'])} | {_fmt(row['acceleration_p95_median'])} "
            f"| {_fmt(row['jerk_p95_median'])} | {_fmt(row['static_ratio_median'])} "
            f"| {_fmt(row['repeated_pose_ratio_median'])} | {_fmt(row['boundary_jump_p95_median'])} "
            f"| {_fmt(row['root_speed_p95_median'])} | {_fmt(row['contact_proxy_median'])} "
            f"| {_fmt(row['penetration_ratio_median'])} | {_fmt(row['pfc_proxy_median'])} |"
        )
    report += [
        "",
        "### Tempo 分层的质量基准",
        "",
        "| Tempo | n | Energy | Jerk p95 | Static | Repeat | Root speed p95 | Contact | Penetration | PFC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in quality_summaries:
        if row["dimension"] != "tempo":
            continue
        report.append(
            f"| {row['group']} | {row['n_sequences']} | {_fmt(row['motion_energy_median'])} "
            f"| {_fmt(row['jerk_p95_median'])} | {_fmt(row['static_ratio_median'])} "
            f"| {_fmt(row['repeated_pose_ratio_median'])} | {_fmt(row['root_speed_p95_median'])} "
            f"| {_fmt(row['contact_proxy_median'])} | {_fmt(row['penetration_ratio_median'])} "
            f"| {_fmt(row['pfc_proxy_median'])} |"
        )
    report += [
        "",
        "## 六、同风格是否更相似",
        "",
        "距离使用每条序列的标准化指标向量；数值越大表示组间差异越大。",
        "",
        "| 特征组 | 层 | 同风格距离 | 不同风格距离 | 差值 | p-value | 结论 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in similarity:
        report.append(
            f"| {row['feature_group']} | {row['stage']} | {row['within_style_distance']:.4f} "
            f"| {row['between_style_distance']:.4f} | {row['between_minus_within']:.4f} "
            f"| {row['permutation_p_value']:.4f} | {row['interpretation']} |"
        )
    report += [
        "",
        "## 七、为什么 G1 的 BAS 可能高于原始 SMPL",
        "",
        "FineDance 全量 retargeting audit 中，source 的 BAS/Event F1/Impact corr 中位数约为",
        "`0.1201/0.4771/0.0282`，G1 target 约为 `0.2267/0.6829/0.0392`。这不表示 G1 舞蹈",
        "更优，原因包括：",
        "",
        "1. source 和 G1 使用不同的运动信号：source 是 SMPLH 6D rotation speed，G1 是 29-DoF joint speed；",
        "2. GMR 的平滑、重采样和关节映射会改变局部极值，可能产生更容易被 detector 命中的 motion beats；",
        "3. FineDance source→G1 activity correlation 中位数只有 `0.0495`，root-speed correlation 只有 `0.0517`；",
        "4. G1/source activity event count ratio 中位数约 `1.66`，说明 event 数量已经发生系统性变化；",
        "5. 所以 G1 BAS 上升可能是 detector/representation effect，而不是舞蹈变得更优美。",
        "",
        "## 八、优美度、丝滑度和节奏感",
        "",
        "- **节奏感**：可由 BAS、Event F1、tempo、phase、impact correlation 和 lag 联合评价。",
        "- **丝滑度**：可由 velocity/acceleration/jerk、局部 discontinuity、FSR 和 foot contact 评价；",
        "  jerk 过低也可能意味着动作过平或平均化，不能只追求更低。",
        "- **舞蹈质量**：需要动作质量、物理可执行性、音乐适配和多样性共同判断。",
        "- **优美度/表现力/风格真实性**：当前没有可靠的自动 oracle，必须加入同风格人工盲评，",
        "  例如 naturalness、expressiveness、rhythm、style consistency 四个维度。",
        "",
        "本报告的自动指标不能单独证明‘优美’。后续 M2/M3/M_exec 必须使用相同的分层 GT",
        "分布，并同时报告 generator 与 SONIC execution 的绝对值和 retention。",
    ]
    (OUTPUT / "REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote Chinese style report to {OUTPUT / 'REPORT_ZH.md'}")


if __name__ == "__main__":
    main()
