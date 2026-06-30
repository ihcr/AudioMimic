#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.g1_metrics import run_g1_motion_evaluation


DEFAULT_VARIANTS = ("real_beat8d", "shifted_beat8d", "random_beat8d", "zero_beat8d")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--audio_data_path", default="data/finedance_aistpp", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), type=str)
    parser.add_argument("--diagnostic_count", default=8, type=int)
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        type=str,
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    return parser.parse_args()


def write_json(payload, path):
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def set_audio_path(path, audio_path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload["audio_path"] = str(audio_path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle)
    tmp_path.replace(path)


def audio_path_for_stem(args, stem):
    audio_path = Path(args.audio_data_path) / args.split / "wavs_sliced" / f"{stem}.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(f"missing audio slice for {stem}: {audio_path}")
    return audio_path


def repair_variant(args, variant):
    variant_dir = Path(args.output_dir) / variant
    motion_dir = variant_dir / "motions"
    target_dir = variant_dir / "targets"
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"missing motion dir for {variant}: {motion_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"missing target dir for {variant}: {target_dir}")

    motion_files = sorted(motion_dir.glob("*.pkl"))
    if not motion_files:
        raise FileNotFoundError(f"no generated motions found in {motion_dir}")
    for motion_path in motion_files:
        stem = motion_path.stem
        audio_path = audio_path_for_stem(args, stem)
        set_audio_path(motion_path, audio_path)
        target_path = target_dir / motion_path.name
        if target_path.is_file():
            set_audio_path(target_path, audio_path)

    return run_g1_motion_evaluation(
        motion_path=motion_dir,
        reference_motion_path=target_dir,
        metrics_path=variant_dir / "metrics.json",
        g1_table_path=variant_dir / "g1_table.json",
        motion_audit_path=variant_dir / "motion_audit.json",
        paper_report_path=variant_dir / "paper_report.md",
        render_dir=variant_dir / "diagnostics",
        diagnostic_count=args.diagnostic_count,
        checkpoint=args.checkpoint,
        feature_type=f"g1_latent_diffusion_{variant}",
        use_beats=True,
        beat_rep="beat_features_8d",
        seed=1234,
        sample_limit=None,
        enable_fk_metrics=True,
        fk_model_path=args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
        failure_panel_path=variant_dir / "failure_panel.json",
    )


def main():
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    summary_path = Path(args.output_dir) / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "checkpoint": args.checkpoint,
            "output_dir": args.output_dir,
            "variants": {},
        }
    summary["audio_data_path"] = args.audio_data_path
    summary["audio_metrics_repaired"] = True
    for variant in variants:
        metrics = repair_variant(args, variant)
        summary.setdefault("variants", {})[variant] = {
            "metrics_path": str(Path(args.output_dir) / variant / "metrics.json"),
            "G1FKBAS": metrics.get("G1FKBAS"),
            "G1BeatF1": metrics.get("G1BeatF1"),
            "G1BeatRecall": metrics.get("G1BeatRecall"),
            "G1BeatPrecision": metrics.get("G1BeatPrecision"),
            "G1Dist": metrics.get("G1Dist"),
            "G1Div": metrics.get("G1Div"),
            "G1NoNearSupportRate": metrics.get("G1NoNearSupportRate"),
            "G1FootHighLiftRate": metrics.get("G1FootHighLiftRate"),
            "G1GroundPenetration": metrics.get("G1GroundPenetration"),
            "G1FootSliding": metrics.get("G1FootSliding"),
            "num_fk_audio_beats": metrics.get("num_fk_audio_beats"),
        }
        write_json(summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
