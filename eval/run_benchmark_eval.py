import argparse
import glob
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.benchmark_report import (build_benchmark_report, render_paper_report,
                                   write_json, write_text)
from eval.eval_bas_bap import evaluate_motion_dir
from eval.eval_diversity import compute_diversity_metrics
from eval.eval_pfc import compute_physical_score, load_full_pose


def prepare_motion_dir_for_benchmark(motion_path, wav_dir="", motion_source="generated"):
    motion_files = sorted(glob.glob(str(Path(motion_path) / "*.pkl")))
    if not motion_files:
        raise FileNotFoundError(f"No motion pickle files found in {motion_path}")

    if motion_source == "generated":
        for motion_file in motion_files:
            with open(motion_file, "rb") as handle:
                payload = pickle.load(handle)
            if "audio_path" not in payload:
                raise KeyError(
                    f"Generated motion file {motion_file} is missing audio_path. "
                    "Generated benchmark inputs must include saved joint motion and audio metadata."
                )
            load_full_pose(
                payload,
                source_preference="full_pose",
                validate_consistency=True,
            )
        return motion_path, None

    if motion_source != "dataset":
        raise ValueError(f"Unsupported motion_source {motion_source!r}")

    if not wav_dir:
        raise ValueError("Dataset benchmark preparation requires --wav_dir.")

    temp_dir = TemporaryDirectory()
    prepared_dir = Path(temp_dir.name)
    wav_root = Path(wav_dir)
    for motion_file in motion_files:
        with open(motion_file, "rb") as handle:
            payload = pickle.load(handle)
        wav_candidate = wav_root / (Path(motion_file).stem + ".wav")
        if not wav_candidate.is_file():
            raise FileNotFoundError(
                f"Missing wav file for dataset motion {motion_file}: {wav_candidate}"
            )
        prepared_payload = {
            "full_pose": load_full_pose(payload, source_preference="smpl_or_legacy"),
            "audio_path": str(wav_candidate),
        }
        with open(prepared_dir / Path(motion_file).name, "wb") as handle:
            pickle.dump(prepared_payload, handle)
    return str(prepared_dir), temp_dir


def run_benchmark_evaluation(
    motion_path,
    metrics_path,
    method_name,
    feature_type,
    use_beats,
    beat_rep,
    reference_motion_path="",
    seed=1234,
    checkpoint="",
    beat_source="none",
    wav_dir="",
    motion_source="generated",
    edge_table_path="",
    beatit_table_path="",
    paper_report_path="",
    pfc_audit_path="",
    extra_metadata=None,
):
    prepared_motion_path, temp_dir = prepare_motion_dir_for_benchmark(
        motion_path, wav_dir=wav_dir, motion_source=motion_source
    )
    try:
        bas_bap = evaluate_motion_dir(prepared_motion_path)
        pfc_details = compute_physical_score(
            prepared_motion_path, return_details=True, seed=seed
        )
        diversity = compute_diversity_metrics(
            prepared_motion_path,
            reference_motion_path=reference_motion_path or None,
            seed=seed,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    metrics = {
        "checkpoint": checkpoint,
        "feature_type": feature_type,
        "use_beats": use_beats,
        "beat_rep": beat_rep if use_beats else "none",
        "beat_source": beat_source if use_beats else "none",
        "seed": seed,
        "PFC": pfc_details["PFC"],
        "PFC_internal_only": True,
        "PFC_audit": pfc_details,
        "BAS": bas_bap["BAS"],
        "BAP": bas_bap["BAP"],
        "BAP_precision": bas_bap["BAP_precision"],
        "BAP_recall": bas_bap["BAP_recall"],
        "Distk": diversity["Distk"],
        "Distg": diversity["Distg"],
        "Divk": diversity["Divk"],
        "Divm": diversity["Divm"],
        "Divg": diversity["Divg"],
        "num_motion_files": diversity["num_motion_files"],
        "num_reference_files": diversity["num_reference_files"],
        "zero_variance_dims_kinetic": diversity.get("zero_variance_dims_kinetic", 0),
        "zero_variance_dims_manual": diversity.get("zero_variance_dims_manual", 0),
        "num_scored_files": bas_bap["num_scored_files"],
        "num_skipped_files": bas_bap["num_skipped_files"],
        "num_generated_beats": bas_bap["num_generated_beats"],
        "num_designated_beats": bas_bap["num_designated_beats"],
    }
    if extra_metadata:
        metrics.update(extra_metadata)

    report = build_benchmark_report(
        metrics=metrics,
        method_name=method_name,
        use_beats=use_beats,
        beat_rep=beat_rep,
    )
    metrics["edge_table"] = report["edge_table"]
    metrics["beatit_table"] = report["beatit_table"]
    metrics["report_notes"] = report["notes"]

    write_json(metrics_path, metrics)
    if edge_table_path:
        write_json(edge_table_path, report["edge_table"])
    if beatit_table_path:
        write_json(beatit_table_path, report["beatit_table"])
    if paper_report_path:
        write_text(paper_report_path, render_paper_report(report, metrics))
    if pfc_audit_path:
        write_json(pfc_audit_path, pfc_details)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved benchmark metrics to {metrics_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", required=True)
    parser.add_argument("--metrics_path", default="eval/metrics.json")
    parser.add_argument("--edge_table_path", default="")
    parser.add_argument("--beatit_table_path", default="")
    parser.add_argument("--paper_report_path", default="")
    parser.add_argument("--pfc_audit_path", default="")
    parser.add_argument("--method_name", default="evaluation")
    parser.add_argument("--feature_type", default="jukebox")
    parser.add_argument("--reference_motion_path", default="")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--use_beats", action="store_true")
    parser.add_argument("--beat_rep", choices=("distance", "pulse"), default="distance")
    parser.add_argument("--beat_source", default="none")
    parser.add_argument("--wav_dir", default="")
    parser.add_argument("--motion_source", choices=("generated", "dataset"), default="generated")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark_evaluation(
        motion_path=args.motion_path,
        metrics_path=args.metrics_path,
        method_name=args.method_name,
        feature_type=args.feature_type,
        use_beats=args.use_beats,
        beat_rep=args.beat_rep,
        reference_motion_path=args.reference_motion_path,
        seed=args.seed,
        checkpoint=args.checkpoint,
        beat_source=args.beat_source,
        wav_dir=args.wav_dir,
        motion_source=args.motion_source,
        edge_table_path=args.edge_table_path,
        beatit_table_path=args.beatit_table_path,
        paper_report_path=args.paper_report_path,
        pfc_audit_path=args.pfc_audit_path,
    )
