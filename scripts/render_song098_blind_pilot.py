"""Render the first matched song098 M0/M2/M4 blind-study pilot assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.render_blind_g1_video import render


ROUTE_SPECS = {
    "M0": {
        "motion_pkl": Path(
            "~/Musics2Dance-prior-dev/onlinegeneratedmotion/uncond_m0/"
            "m0_train1234_sample1234_u200000_best_song098.pkl"
        ),
        "tracking_run": Path(
            "eval/generation_to_execution_gap/"
            "m0_song098_seed1234_full_rate100_aligned_r01"
        ),
    },
    "M2": {
        "motion_pkl": Path(
            "~/Musics2Dance-prior-dev/onlinegeneratedmotion/m2_predicted_fms/"
            "m2_train1234_sample1234_u100000_best_song098.pkl"
        ),
        "tracking_run": Path(
            "eval/generation_to_execution_gap/"
            "m2_song098_seed1234_full_rate100_aligned_r01"
        ),
    },
    "M4": {
        "motion_pkl": Path(
            "~/Musics2Dance-prior-dev/onlinegeneratedmotion/m4_oracle_fms/"
            "m4_train1234_sample1234_u100000_best_song098.pkl"
        ),
        "tracking_run": Path(
            "eval/generation_to_execution_gap/"
            "m4_song098_seed1234_full_rate100_aligned_r01"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("eval/human_study/media/song098_seed1234_pilot"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/human_study/assets_song098_seed1234_pilot.json"),
    )
    parser.add_argument("--start_seconds", type=float, default=4.0)
    parser.add_argument("--duration_seconds", type=float, default=16.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _render_args(
    *,
    motion_pkl: Path | None,
    tracking_run: Path | None,
    output_mp4: Path,
    audio_wav: Path,
    start_seconds: float,
    duration_seconds: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        motion_pkl=motion_pkl,
        tracking_run=tracking_run,
        tracking_representation="execution",
        output_mp4=output_mp4,
        audio_wav=audio_wav,
        audio_start_seconds=0.0,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        output_fps=30.0,
        tracking_analysis_fps=50.0,
        width=640,
        height=480,
        azimuth=180.0,
        elevation=-12.0,
        distance=3.0,
        model_xml=REPO_ROOT / "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )


def main(args: argparse.Namespace) -> None:
    output_dir = (REPO_ROOT / args.output_dir).expanduser().resolve()
    manifest_path = (REPO_ROOT / args.manifest).expanduser().resolve()
    audio_wav = Path(
        "~/Musics2Dance-prior-dev/onlinegeneratedmotion/audio/098_t000128_60s.wav"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    window_id = f"w{int(args.start_seconds):02d}_{int(args.start_seconds + args.duration_seconds):02d}"
    for route, spec in ROUTE_SPECS.items():
        for representation in ("reference", "execution"):
            asset_id = f"song098_seed1234_{window_id}_{route.lower()}_{representation}"
            output_path = output_dir / f"{asset_id}.mp4"
            if args.overwrite or not output_path.is_file():
                render(
                    _render_args(
                        motion_pkl=(
                            spec["motion_pkl"].expanduser().resolve()
                            if representation == "reference"
                            else None
                        ),
                        tracking_run=(
                            (REPO_ROOT / spec["tracking_run"]).resolve()
                            if representation == "execution"
                            else None
                        ),
                        output_mp4=output_path,
                        audio_wav=audio_wav,
                        start_seconds=float(args.start_seconds),
                        duration_seconds=float(args.duration_seconds),
                    )
                )
            relative_path = output_path.relative_to(REPO_ROOT)
            asset = {
                "asset_id": asset_id,
                "path": str(relative_path),
                "route": route,
                "representation": representation,
                "song_id": "098",
                "generation_seed": 1234,
                "window_id": window_id,
                "duration_seconds": float(args.duration_seconds),
                "camera_profile": "front_follow_az180_el-12_d3",
                "render_profile": "g1_mujoco_640x480_30fps_v1",
                "label_free": True,
                "audio_embedded": True,
                "eligible": True,
            }
            if representation == "execution":
                asset["tracker_repeat"] = 1
            assets.append(asset)

    payload = {
        "schema_version": "audiomimic_human_assets_v1",
        "inventory_date": "2026-08-20",
        "scope": "song098 seed1234 pilot only; not paper-complete",
        "minimum_paper_design": {
            "songs": 3,
            "generation_seeds_per_song": 3,
            "routes": ["M0", "M2", "M4"],
            "tracker_repeats_per_reference": 3,
            "participants": 24
        },
        "assets": assets,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path} with {len(assets)} assets")


if __name__ == "__main__":
    main(parse_args())
