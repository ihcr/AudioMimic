#!/usr/bin/env python3
"""Upload compact EDGE G1 artifacts to one Hugging Face repo."""

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_ID = "wyksdsg/edge-g1-beatdistance"
DEFAULT_REPO_TYPE = "model"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIFFUSION_ROOT = REPO_ROOT.parent / "diffusion"
for _parent in (REPO_ROOT, *REPO_ROOT.parents):
    if _parent.name == ".worktrees":
        DEFAULT_SOURCE_ROOT = _parent.parent
        break
else:
    DEFAULT_SOURCE_ROOT = REPO_ROOT

FINEDANCE_SOURCE_REL = Path("data/finedance")
FINEDANCE_G1_RETARGETED_REL = Path("data/finedance-g1-retargeted")
WAV2CLIP_CHECKPOINT_REL = Path(
    "runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/"
    "weights/train-500.pt"
)
DIFFUSION_CHECKPOINT_RELS = (
    Path("runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt"),
    Path(
        "runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/"
        "weights/train-2000.pt"
    ),
)
PRUNE_PATHS = (
    "data/finedance_g1_fkbeats",
    "data/finedance_g1_wav2clip_stft_beat_concat_norm_dataset_backups",
    "data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups",
    "data/finedance_g1_fkbeats_dataset_backups_fkbeat1000",
    "data/finedance_g1_librosa35_fullctx_motiondist_cond_dataset_backups",
    "docs/experiments/artifacts",
)


@dataclass(frozen=True)
class UploadItem:
    source_root: Path
    local_path: Path
    path_in_repo: Path
    kind: str
    note: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Upload only compact EDGE G1 artifacts to a single Hugging Face "
            "repo: raw FineDance source data, retargeted G1 motions, and "
            "checkpoints. Feature folders and tensor caches are intentionally "
            "not uploaded; rebuild them on the target 4090 machine."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=("model", "dataset"),
        help=(
            "HF repo type. Default: model, matching "
            "https://huggingface.co/wyksdsg/edge-g1-beatdistance."
        ),
    )
    parser.add_argument("--revision", default="main", help="Target branch/revision.")
    parser.add_argument("--private", action="store_true", help="Create repo as private.")
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help=(
            "Repo root containing shared source data/finedance and "
            "data/finedance-g1-retargeted. Default: parent of .worktrees."
        ),
    )
    parser.add_argument(
        "--diffusion-root",
        default=str(DEFAULT_DIFFUSION_ROOT),
        help="Path to the diffusion branch checkout/worktree.",
    )
    parser.add_argument(
        "--skip-finedance-source",
        action="store_true",
        help="Do not upload data/finedance.",
    )
    parser.add_argument(
        "--skip-retargeted-g1",
        action="store_true",
        help="Do not upload data/finedance-g1-retargeted.",
    )
    parser.add_argument(
        "--skip-wav2clip-checkpoint",
        action="store_true",
        help="Do not upload wav2clip r02 stream-adapter train-500.pt.",
    )
    parser.add_argument(
        "--skip-diffusion-checkpoints",
        action="store_true",
        help="Do not upload diffusion anchor checkpoint .pt files.",
    )
    parser.add_argument(
        "--prune-large-feature-paths",
        action="store_true",
        help=(
            "Delete older HF uploads of feature folders, caches, and evidence "
            "paths before uploading compact artifacts."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker count for resumable large-folder uploads.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=int,
        default=30,
        help="How often upload_large_folder prints a progress report. Default: 30.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned uploads only.")
    return parser.parse_args()


def git_commit(root):
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def count_files(path, limit=20000):
    if path.is_file():
        return "1"
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            if count >= limit:
                return f">={limit}"
    return str(count)


def require_item(item):
    full_path = item.source_root / item.local_path
    if not full_path.exists():
        raise SystemExit(f"Missing required artifact: {full_path}")
    return full_path


def hf_repo_type_arg(repo_type):
    return None if repo_type == "model" else repo_type


def planned_uploads(args):
    source_root = Path(args.source_root).resolve()
    diffusion_root = Path(args.diffusion_root).resolve()
    uploads = []

    if not args.skip_finedance_source:
        uploads.append(
            UploadItem(
                source_root,
                FINEDANCE_SOURCE_REL,
                FINEDANCE_SOURCE_REL,
                "folder",
                "raw FineDance source data for local preprocessing",
            )
        )
    if not args.skip_retargeted_g1:
        uploads.append(
            UploadItem(
                source_root,
                FINEDANCE_G1_RETARGETED_REL,
                FINEDANCE_G1_RETARGETED_REL,
                "folder",
                "FineDance retargeted G1 sequence motions",
            )
        )

    if not args.skip_wav2clip_checkpoint:
        uploads.append(
            UploadItem(
                REPO_ROOT,
                WAV2CLIP_CHECKPOINT_REL,
                WAV2CLIP_CHECKPOINT_REL,
                "file",
                "wav2clip stream-adapter checkpoint",
            )
        )
    if not args.skip_diffusion_checkpoints:
        for rel in DIFFUSION_CHECKPOINT_RELS:
            uploads.append(
                UploadItem(
                    diffusion_root,
                    rel,
                    rel,
                    "file",
                    "diffusion anchor checkpoint",
                )
            )
    return uploads


def write_manifest(args, uploads):
    source_root = Path(args.source_root).resolve()
    diffusion_root = Path(args.diffusion_root).resolve()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "layout": "repo-relative",
        "source_commits": {
            "source_root": git_commit(source_root),
            "wav2clip": git_commit(REPO_ROOT),
            "diffusion": git_commit(diffusion_root),
        },
        "policy": "HF stores compact source/checkpoint artifacts only; features and caches are rebuilt locally.",
        "artifacts": [],
    }
    for item in uploads:
        full_path = require_item(item)
        manifest["artifacts"].append(
            {
                "source_root": str(item.source_root),
                "local_path": str(item.local_path),
                "path_in_repo": str(item.path_in_repo),
                "kind": item.kind,
                "file_count": count_files(full_path),
                "note": item.note,
            }
        )

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="edge_g1_hf_manifest_",
        delete=False,
    )
    with handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    path_in_repo = Path("hf_manifest.json")
    return Path(handle.name), path_in_repo


def upload_items_resumable(api, args, uploads):
    grouped = defaultdict(list)
    for item in uploads:
        require_item(item)
        grouped[item.source_root].append(item)

    ignore_patterns = ["__pycache__/**", "*.pyc", ".DS_Store", ".git/**"]
    for source_root, items in grouped.items():
        allow_patterns = []
        for item in items:
            if item.kind == "folder":
                allow_patterns.append(f"{item.local_path.as_posix()}/**")
            else:
                allow_patterns.append(item.local_path.as_posix())

        print(f"Uploading from {source_root} with resumable large-folder uploader.")
        print("Allow patterns:")
        for pattern in allow_patterns:
            print(f"  {pattern}")

        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            private=args.private,
            folder_path=str(source_root),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=args.progress_seconds,
        )


def prune_large_feature_paths(api, args):
    for path in PRUNE_PATHS:
        print(f"Pruning HF path if present: {path}")
        try:
            api.delete_folder(
                repo_id=args.repo_id,
                repo_type=hf_repo_type_arg(args.repo_type),
                revision=args.revision,
                path_in_repo=path,
                commit_message=f"Prune {path}",
            )
        except Exception as exc:
            print(f"  skipped {path}: {exc}")


def main():
    args = parse_args()
    uploads = planned_uploads(args)
    manifest_path, manifest_path_in_repo = write_manifest(args, uploads)

    print(f"Target HF {args.repo_type} repo: {args.repo_id}")
    print("Planned HF uploads:")
    for item in uploads:
        full_path = require_item(item)
        print(
            f"  {item.kind:6s} {item.path_in_repo} "
            f"({count_files(full_path)} files; source={item.source_root})"
        )
    print(f"  file   {manifest_path_in_repo} (generated manifest)")

    if args.dry_run:
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install it with `pip install huggingface_hub` "
            "or run scripts/bootstrap_finedance_g1_4090.sh first."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=hf_repo_type_arg(args.repo_type),
        private=args.private,
        exist_ok=True,
    )

    if args.prune_large_feature_paths:
        prune_large_feature_paths(api, args)

    upload_items_resumable(api, args, uploads)

    api.upload_file(
        repo_id=args.repo_id,
        repo_type=hf_repo_type_arg(args.repo_type),
        revision=args.revision,
        path_or_fileobj=str(manifest_path),
        path_in_repo=str(manifest_path_in_repo),
        commit_message="Upload HF artifact manifest",
    )

    print("Upload complete.")


if __name__ == "__main__":
    main()
