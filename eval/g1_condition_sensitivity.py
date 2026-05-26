import argparse
import json
from pathlib import Path

import numpy as np

from eval.g1_kinematics import EXPECTED_G1_29DOF_JOINTS
from eval.g1_metrics import load_g1_motion


DEFAULT_VARIANTS = (
    "real",
    "shift_p10",
    "random",
    "constant",
    "no_beat_uncond",
)

DEFAULT_LABEL_PREFIXES = (
    "no_beat_uncond",
    "shift_p10",
    "g1_eval",
    "constant",
    "random",
    "real",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare G1 ablation outputs clip-by-clip against an unconditional "
            "baseline to measure condition sensitivity and DOF-level changes."
        )
    )
    parser.add_argument(
        "--ablation_root",
        default="eval/EXP-20260522-gaussian-beat-condition-ablation",
    )
    parser.add_argument("--baseline_variant", default="no_beat_uncond")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_markdown", default="")
    parser.add_argument("--active_dof_threshold", type=float, default=0.05)
    parser.add_argument("--top_dof_count", type=int, default=8)
    return parser.parse_args()


def json_safe(payload):
    if isinstance(payload, dict):
        return {key: json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [json_safe(value) for value in payload]
    if isinstance(payload, tuple):
        return [json_safe(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, Path):
        return str(payload)
    return payload


def infer_motion_key(path, label_prefixes=DEFAULT_LABEL_PREFIXES):
    stem = Path(path).stem
    if stem.endswith("_g1"):
        stem = stem[:-3]
    for label in sorted(label_prefixes, key=len, reverse=True):
        prefix = f"{label}_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix) :]
            batch_index, separator, clip_key = remainder.partition("_")
            if batch_index.isdigit() and separator and clip_key:
                return clip_key
    raise ValueError(f"Could not infer clip key from motion filename: {path}")


def index_motion_dir(motion_dir):
    motion_dir = Path(motion_dir)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
    indexed = {}
    for path in sorted(motion_dir.glob("*.pkl")):
        key = infer_motion_key(path)
        if key in indexed:
            raise ValueError(f"Duplicate clip key {key!r} in {motion_dir}")
        indexed[key] = path
    if not indexed:
        raise FileNotFoundError(f"No G1 motion pkl files found in {motion_dir}")
    return indexed


def finite_mean(values, default=0.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(values.mean())


def pearson_correlation(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def root_rot_angle_degrees(left_quat, right_quat):
    dots = np.abs(np.sum(left_quat * right_quat, axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def path_length(root_pos):
    if root_pos.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(root_pos, axis=0), axis=-1).sum())


def require_matching_motion_shapes(variant_motion, baseline_motion, clip_key):
    for field in ("root_pos", "root_rot", "dof_pos"):
        if variant_motion[field].shape != baseline_motion[field].shape:
            raise ValueError(
                f"Shape mismatch for {clip_key} field {field}: "
                f"{variant_motion[field].shape} vs {baseline_motion[field].shape}"
            )


def compare_motion_pair(variant_motion, baseline_motion, clip_key, active_dof_threshold):
    require_matching_motion_shapes(variant_motion, baseline_motion, clip_key)
    dof_delta = variant_motion["dof_pos"] - baseline_motion["dof_pos"]
    root_delta = variant_motion["root_pos"] - baseline_motion["root_pos"]

    variant_velocity = np.diff(variant_motion["dof_pos"], axis=0)
    baseline_velocity = np.diff(baseline_motion["dof_pos"], axis=0)
    dof_velocity_delta = variant_velocity - baseline_velocity

    dof_rmse_by_joint = np.sqrt(np.mean(np.square(dof_delta), axis=0))
    dof_mae_by_joint = np.mean(np.abs(dof_delta), axis=0)
    variant_std_by_joint = variant_motion["dof_pos"].std(axis=0)
    baseline_std_by_joint = baseline_motion["dof_pos"].std(axis=0)
    variant_range_by_joint = (
        variant_motion["dof_pos"].max(axis=0) - variant_motion["dof_pos"].min(axis=0)
    )
    baseline_range_by_joint = (
        baseline_motion["dof_pos"].max(axis=0) - baseline_motion["dof_pos"].min(axis=0)
    )
    baseline_std_mean = float(baseline_std_by_joint.mean())

    return {
        "clip_key": clip_key,
        "dof_rmse": float(np.sqrt(np.mean(np.square(dof_delta)))),
        "dof_mae": float(np.mean(np.abs(dof_delta))),
        "dof_velocity_rmse": float(np.sqrt(np.mean(np.square(dof_velocity_delta)))),
        "dof_pearson": pearson_correlation(variant_motion["dof_pos"], baseline_motion["dof_pos"]),
        "dof_delta_over_baseline_std": float(
            np.sqrt(np.mean(np.square(dof_delta))) / max(baseline_std_mean, 1e-8)
        ),
        "active_dof_delta_count": int(np.sum(dof_rmse_by_joint > active_dof_threshold)),
        "root_rmse": float(np.sqrt(np.mean(np.square(root_delta)))),
        "root_flat_rmse": float(np.sqrt(np.mean(np.square(root_delta[:, :2])))),
        "root_rot_angle_deg": finite_mean(
            root_rot_angle_degrees(variant_motion["root_rot"], baseline_motion["root_rot"])
        ),
        "root_path_length_ratio": float(
            path_length(variant_motion["root_pos"])
            / max(path_length(baseline_motion["root_pos"]), 1e-8)
        ),
        "dof_std_ratio": float(
            variant_std_by_joint.mean() / max(baseline_std_by_joint.mean(), 1e-8)
        ),
        "dof_range_ratio": float(
            variant_range_by_joint.mean() / max(baseline_range_by_joint.mean(), 1e-8)
        ),
        "dof_rmse_by_joint": dof_rmse_by_joint,
        "dof_mae_by_joint": dof_mae_by_joint,
        "variant_std_by_joint": variant_std_by_joint,
        "baseline_std_by_joint": baseline_std_by_joint,
        "variant_range_by_joint": variant_range_by_joint,
        "baseline_range_by_joint": baseline_range_by_joint,
    }


def aggregate_joint_metrics(pair_metrics):
    joint_payload = []
    for joint_index, joint_name in enumerate(EXPECTED_G1_29DOF_JOINTS):
        rmse = [entry["dof_rmse_by_joint"][joint_index] for entry in pair_metrics]
        mae = [entry["dof_mae_by_joint"][joint_index] for entry in pair_metrics]
        variant_std = [entry["variant_std_by_joint"][joint_index] for entry in pair_metrics]
        baseline_std = [entry["baseline_std_by_joint"][joint_index] for entry in pair_metrics]
        variant_range = [entry["variant_range_by_joint"][joint_index] for entry in pair_metrics]
        baseline_range = [entry["baseline_range_by_joint"][joint_index] for entry in pair_metrics]
        joint_payload.append(
            {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "dof_rmse_mean": finite_mean(rmse),
                "dof_mae_mean": finite_mean(mae),
                "variant_std_mean": finite_mean(variant_std),
                "baseline_std_mean": finite_mean(baseline_std),
                "std_ratio": finite_mean(variant_std) / max(finite_mean(baseline_std), 1e-8),
                "variant_range_mean": finite_mean(variant_range),
                "baseline_range_mean": finite_mean(baseline_range),
                "range_ratio": finite_mean(variant_range) / max(finite_mean(baseline_range), 1e-8),
            }
        )
    return joint_payload


def summarize_variant(variant, variant_dir, baseline_dir, active_dof_threshold):
    variant_index = index_motion_dir(variant_dir)
    baseline_index = index_motion_dir(baseline_dir)
    common_keys = sorted(set(variant_index) & set(baseline_index))
    if not common_keys:
        raise ValueError(f"No paired clips found for {variant_dir} vs {baseline_dir}")

    pair_metrics = []
    for clip_key in common_keys:
        variant_motion = load_g1_motion(variant_index[clip_key])
        baseline_motion = load_g1_motion(baseline_index[clip_key])
        pair_metrics.append(
            compare_motion_pair(
                variant_motion,
                baseline_motion,
                clip_key,
                active_dof_threshold=active_dof_threshold,
            )
        )

    joint_metrics = aggregate_joint_metrics(pair_metrics)
    top_joints = sorted(
        joint_metrics,
        key=lambda entry: entry["dof_rmse_mean"],
        reverse=True,
    )

    return {
        "variant": variant,
        "variant_motion_dir": str(variant_dir),
        "baseline_motion_dir": str(baseline_dir),
        "num_variant_files": len(variant_index),
        "num_baseline_files": len(baseline_index),
        "num_pairs": len(common_keys),
        "num_missing_from_variant": len(set(baseline_index) - set(variant_index)),
        "num_missing_from_baseline": len(set(variant_index) - set(baseline_index)),
        "DofRMSEMean": finite_mean([entry["dof_rmse"] for entry in pair_metrics]),
        "DofMAEMean": finite_mean([entry["dof_mae"] for entry in pair_metrics]),
        "DofVelocityRMSEMean": finite_mean(
            [entry["dof_velocity_rmse"] for entry in pair_metrics]
        ),
        "DofPearsonMean": finite_mean([entry["dof_pearson"] for entry in pair_metrics]),
        "DofDeltaOverBaselineStdMean": finite_mean(
            [entry["dof_delta_over_baseline_std"] for entry in pair_metrics]
        ),
        "ActiveDofDeltaCountMean": finite_mean(
            [entry["active_dof_delta_count"] for entry in pair_metrics]
        ),
        "RootRMSEMean": finite_mean([entry["root_rmse"] for entry in pair_metrics]),
        "RootFlatRMSEMean": finite_mean([entry["root_flat_rmse"] for entry in pair_metrics]),
        "RootRotAngleDegMean": finite_mean(
            [entry["root_rot_angle_deg"] for entry in pair_metrics]
        ),
        "RootPathLengthRatioMean": finite_mean(
            [entry["root_path_length_ratio"] for entry in pair_metrics]
        ),
        "DofStdRatioMean": finite_mean([entry["dof_std_ratio"] for entry in pair_metrics]),
        "DofRangeRatioMean": finite_mean([entry["dof_range_ratio"] for entry in pair_metrics]),
        "top_changed_dofs": top_joints,
    }


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_motion_dir(ablation_root, manifest, variant):
    result = manifest.get("results", {}).get(variant, {})
    motion_dir = result.get("motion_dir")
    if motion_dir:
        return Path(motion_dir)
    return Path(ablation_root) / variant / "motions"


def format_float(value):
    return f"{float(value):.4f}"


def write_markdown(summary, markdown_path, top_dof_count):
    lines = [
        "# G1 Condition Sensitivity",
        "",
        f"Baseline variant: `{summary['baseline_variant']}`",
        "",
        "| Variant | Pairs | DofRMSE | DofMAE | DofPearson | DofDelta/BaselineStd | ActiveDOFs | DofStdRatio | DofRangeRatio | RootRMSE | RootRotDeg | RootPathRatio | Top Changed DOFs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in summary["variants"]:
        top_names = ", ".join(
            joint["joint_name"] for joint in entry["top_changed_dofs"][:top_dof_count]
        )
        lines.append(
            "| {variant} | {pairs} | {dof_rmse} | {dof_mae} | {dof_corr} | "
            "{dof_norm} | {active} | {std_ratio} | {range_ratio} | {root_rmse} | "
            "{root_rot} | {path_ratio} | {top} |".format(
                variant=entry["variant"],
                pairs=entry["num_pairs"],
                dof_rmse=format_float(entry["DofRMSEMean"]),
                dof_mae=format_float(entry["DofMAEMean"]),
                dof_corr=format_float(entry["DofPearsonMean"]),
                dof_norm=format_float(entry["DofDeltaOverBaselineStdMean"]),
                active=format_float(entry["ActiveDofDeltaCountMean"]),
                std_ratio=format_float(entry["DofStdRatioMean"]),
                range_ratio=format_float(entry["DofRangeRatioMean"]),
                root_rmse=format_float(entry["RootRMSEMean"]),
                root_rot=format_float(entry["RootRotAngleDegMean"]),
                path_ratio=format_float(entry["RootPathLengthRatioMean"]),
                top=top_names,
            )
        )

    lines.extend(
        [
            "",
            "## Top DOF Deltas",
            "",
        ]
    )
    for entry in summary["variants"]:
        lines.extend(
            [
                f"### {entry['variant']}",
                "",
                "| Joint | DofRMSE | VariantStd | UncondStd | StdRatio | VariantRange | UncondRange | RangeRatio |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for joint in entry["top_changed_dofs"][:top_dof_count]:
            lines.append(
                "| {joint} | {rmse} | {vstd} | {bstd} | {sratio} | {vrange} | {brange} | {rratio} |".format(
                    joint=joint["joint_name"],
                    rmse=format_float(joint["dof_rmse_mean"]),
                    vstd=format_float(joint["variant_std_mean"]),
                    bstd=format_float(joint["baseline_std_mean"]),
                    sratio=format_float(joint["std_ratio"]),
                    vrange=format_float(joint["variant_range_mean"]),
                    brange=format_float(joint["baseline_range_mean"]),
                    rratio=format_float(joint["range_ratio"]),
                )
            )
        lines.append("")

    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_condition_sensitivity(args):
    ablation_root = Path(args.ablation_root)
    manifest_path = Path(args.manifest) if args.manifest else ablation_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    variants = args.variant or list(DEFAULT_VARIANTS)
    baseline_variant = args.baseline_variant
    baseline_dir = resolve_motion_dir(ablation_root, manifest, baseline_variant)

    summary = {
        "ablation_root": str(ablation_root),
        "manifest": str(manifest_path),
        "baseline_variant": baseline_variant,
        "baseline_motion_dir": str(baseline_dir),
        "active_dof_threshold": args.active_dof_threshold,
        "variants": [],
    }

    for variant in variants:
        if variant == baseline_variant:
            continue
        variant_dir = resolve_motion_dir(ablation_root, manifest, variant)
        summary["variants"].append(
            summarize_variant(
                variant,
                variant_dir,
                baseline_dir,
                active_dof_threshold=args.active_dof_threshold,
            )
        )

    output_json = (
        Path(args.output_json)
        if args.output_json
        else ablation_root / f"condition_sensitivity_vs_{baseline_variant}.json"
    )
    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown
        else ablation_root / f"condition_sensitivity_vs_{baseline_variant}.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
    write_markdown(summary, output_markdown, args.top_dof_count)
    return summary


if __name__ == "__main__":
    run_condition_sensitivity(parse_args())
