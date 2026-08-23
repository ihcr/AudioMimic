"""Evaluate one generated motion and its recorded SONIC execution."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_motion_music_execution import (
    DEFAULT_MODEL_PATH,
    _execution_loss,
    compute_motion_quality,
    compute_music_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_pkl", required=True)
    parser.add_argument("--execution_pkl", required=True)
    parser.add_argument("--audio", required=True, help="audio already aligned to motion frame zero")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--label", default="motion_execution_pair")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def _load_motion(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "fps": float(payload.get("fps", 30.0)),
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
    }


def main() -> None:
    args = parse_args()
    reference = _load_motion(Path(args.reference_pkl).expanduser().resolve())
    execution = _load_motion(Path(args.execution_pkl).expanduser().resolve())
    if reference["fps"] != execution["fps"]:
        raise ValueError("reference and execution FPS must match")
    frames = min(len(reference["dof_pos"]), len(execution["dof_pos"]))
    for motion in (reference, execution):
        for field in ("root_pos", "root_rot", "dof_pos"):
            motion[field] = motion[field][:frames]

    model_path = Path(args.model_path).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    audio_cache: dict = {}
    reference_quality, reference_speed = compute_motion_quality(
        reference, model_path=model_path, quat_order="xyzw"
    )
    execution_quality, execution_speed = compute_motion_quality(
        execution, model_path=model_path, quat_order="xyzw"
    )
    reference_music = compute_music_metrics(
        reference_speed,
        fps=reference["fps"],
        audio_path=audio_path,
        audio_cache=audio_cache,
    )
    execution_music = compute_music_metrics(
        execution_speed,
        fps=execution["fps"],
        audio_path=audio_path,
        audio_cache=audio_cache,
    )
    result = {
        "schema_version": "motion_execution_pair_metrics_v1",
        "label": args.label,
        "reference_pkl": str(Path(args.reference_pkl).expanduser().resolve()),
        "execution_pkl": str(Path(args.execution_pkl).expanduser().resolve()),
        "audio": str(audio_path),
        "frames": frames,
        "fps": reference["fps"],
        "generator": {"quality": reference_quality, "music": reference_music},
        "execution": {"quality": execution_quality, "music": execution_music},
        "retention": _execution_loss(
            reference_quality, execution_quality, reference_music, execution_music
        ),
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        f"{args.label}: energy {reference_quality['motion_energy_rad2_s2']:.3f} -> "
        f"{execution_quality['motion_energy_rad2_s2']:.3f}, "
        f"BAS {reference_music['bas_music_to_motion']:.3f} -> "
        f"{execution_music['bas_music_to_motion']:.3f}, "
        f"impact corr {reference_music['impact_best_correlation']:.3f} -> "
        f"{execution_music['impact_best_correlation']:.3f}"
    )


if __name__ == "__main__":
    main()
