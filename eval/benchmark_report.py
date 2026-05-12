import json
from pathlib import Path


def build_benchmark_report(metrics, method_name, use_beats, beat_rep):
    edge_table = {
        "Method": method_name,
        "PFC": metrics.get("PFC"),
        "Beat Align.": metrics.get("BAS"),
        "Distk": metrics.get("Distk"),
        "Distg": metrics.get("Distg"),
    }
    beatit_table = {
        "Methods": method_name,
        "PFC": metrics.get("PFC"),
        "BAS": metrics.get("BAS"),
        "Divk": metrics.get("Divk"),
        "Divm": metrics.get("Divm"),
        "KPD": None,
        "BAP": metrics.get("BAP") if use_beats else None,
    }
    notes = {
        "pfc_internal_only": bool(metrics.get("PFC_internal_only", False)),
        "beat_rep": beat_rep if use_beats else "none",
        "keyframe_metrics": "not_applicable",
    }
    return {
        "edge_table": edge_table,
        "beatit_table": beatit_table,
        "notes": notes,
    }


def render_paper_report(report, metrics):
    lines = [
        "# Evaluation Report",
        "",
        f"- PFC (internal-only): {metrics.get('PFC')}",
        f"- BAS: {metrics.get('BAS')}",
        f"- BAP: {metrics.get('BAP')}",
        f"- BAP precision: {metrics.get('BAP_precision')}",
        f"- BAP recall: {metrics.get('BAP_recall')}",
        f"- Distk: {metrics.get('Distk')}",
        f"- Distg: {metrics.get('Distg')}",
        f"- Divk: {metrics.get('Divk')}",
        f"- Divm: {metrics.get('Divm')}",
        "",
        "## EDGE Table Row",
        "",
        "```json",
        json.dumps(report["edge_table"], indent=2, sort_keys=True),
        "```",
        "",
        "## Beat-It Table Row",
        "",
        "```json",
        json.dumps(report["beatit_table"], indent=2, sort_keys=True),
        "```",
        "",
        "## Notes",
        "",
        f"- PFC internal only: {report['notes']['pfc_internal_only']}",
        f"- Beat representation: {report['notes']['beat_rep']}",
        f"- Keyframe metrics: {report['notes']['keyframe_metrics']}",
    ]
    return "\n".join(lines) + "\n"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
