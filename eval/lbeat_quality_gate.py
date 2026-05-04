import json
import math
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "pfc_ratio": 1.5,
    "distg_ratio": 1.2,
    "distk_ratio": 1.5,
    "bas_drop": 0.02,
    "bap_gain": 0.02,
    "bas_gain": 0.02,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def metric_value(metrics, *names):
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def evaluate_lbeat_acceptance(candidate, reference, thresholds=None):
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    failed_rules = []

    pfc = metric_value(candidate, "PFC")
    ref_pfc = metric_value(reference, "PFC")
    if pfc is None or ref_pfc is None or pfc > ref_pfc * thresholds["pfc_ratio"]:
        failed_rules.append("PFC")

    distg = metric_value(candidate, "Distg")
    ref_distg = metric_value(reference, "Distg")
    if distg is None or ref_distg is None or distg > ref_distg * thresholds["distg_ratio"]:
        failed_rules.append("Distg")

    distk = metric_value(candidate, "Distk")
    ref_distk = metric_value(reference, "Distk")
    if distk is None or ref_distk is None or distk > ref_distk * thresholds["distk_ratio"]:
        failed_rules.append("Distk")

    bas = metric_value(candidate, "BAS", "Beat Align.")
    ref_bas = metric_value(reference, "BAS", "Beat Align.")
    if bas is None or ref_bas is None or bas < ref_bas - thresholds["bas_drop"]:
        failed_rules.append("BAS")

    bap = metric_value(candidate, "BAP")
    ref_bap = metric_value(reference, "BAP")
    beat_gain_ok = False
    if bap is not None and ref_bap is not None:
        beat_gain_ok = bap >= ref_bap + thresholds["bap_gain"]
    if bas is not None and ref_bas is not None:
        beat_gain_ok = beat_gain_ok or bas >= ref_bas + thresholds["bas_gain"]
    if not beat_gain_ok:
        failed_rules.append("beat_gain")

    return {
        "accepted": not failed_rules,
        "failed_rules": failed_rules,
    }


def candidate_score(candidate, reference):
    metrics = candidate["metrics"]
    acceptance = evaluate_lbeat_acceptance(metrics, reference)
    bas = metric_value(metrics, "BAS", "Beat Align.") or 0.0
    bap = metric_value(metrics, "BAP") or 0.0
    pfc = metric_value(metrics, "PFC") or 1e9
    return (
        1 if acceptance["accepted"] else 0,
        bas + bap,
        -pfc,
    )


def select_best_checkpoint(candidates, reference):
    if not candidates:
        raise ValueError("No checkpoint candidates were provided.")
    enriched = []
    for candidate in candidates:
        item = dict(candidate)
        item["acceptance"] = evaluate_lbeat_acceptance(item["metrics"], reference)
        enriched.append(item)
    return max(enriched, key=lambda candidate: candidate_score(candidate, reference))


def exit_code_for_lbeat_report(report, fail_on_rejection=False):
    if fail_on_rejection and not report.get("accepted", False):
        return 1
    return 0


def load_reference_metrics(eval_dir):
    eval_dir = Path(eval_dir)
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.is_file():
        return load_json(metrics_path)
    edge_table_path = eval_dir / "edge_table.json"
    if edge_table_path.is_file():
        return load_json(edge_table_path)
    raise FileNotFoundError(f"No reference metrics found under {eval_dir}")
