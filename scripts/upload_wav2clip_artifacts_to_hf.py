#!/usr/bin/env python3
"""Upload runtime artifacts for the wav2clip-stft-beat branch to HF Datasets."""

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_REL = Path("data/finedance_g1_fkbeats")
CACHE_REL = Path("data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups")
CHECKPOINT_REL = Path(
    "runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/"
    "weights/train-500.pt"
)
EVIDENCE_REL = Path(
    "docs/experiments/artifacts/EXP-20260513-finedance-g1-wav2clip-stft-beat"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Upload the FineDance+G1 Wav2CLIP/STFT/Beat runtime artifacts to a "
            "Hugging Face dataset repo. The uploaded layout preserves repo-relative "
            "paths so scripts/setup_new_server.sh can download it directly."
        )
    )
    parser.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. user/repo")
    parser.add_argument("--revision", default="main", help="Target branch/revision.")
    parser.add_argument("--private", action="store_true", help="Create repo as private.")
    parser.add_argument("--include-cache", action="store_true", help="Upload tensor/cache backup.")
    parser.add_argument("--include-checkpoint", action="store_true", help="Upload r02 train-500.pt.")
    parser.add_argument("--include-evidence", action="store_true", help="Upload curated Slurm evidence.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned uploads only.")
    return parser.parse_args()


def git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def count_files(path):
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def require_path(path):
    full_path = REPO_ROOT / path
    if not full_path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    return full_path


def planned_uploads(args):
    uploads = [(DATA_REL, "folder", True)]
    if args.include_cache:
        uploads.append((CACHE_REL, "folder", False))
    if args.include_checkpoint:
        uploads.append((CHECKPOINT_REL, "file", False))
    if args.include_evidence:
        uploads.append((EVIDENCE_REL, "folder", False))
    return uploads


def write_manifest(args, uploads):
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": git_commit(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "layout": "repo-relative",
        "artifacts": [],
    }
    for rel, kind, follows_symlinks in uploads:
        full_path = require_path(rel)
        manifest["artifacts"].append(
            {
                "path": str(rel),
                "kind": kind,
                "file_count": count_files(full_path),
                "source_upload_follows_symlinks": follows_symlinks,
            }
        )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="edge_wav2clip_hf_manifest_",
        delete=False,
    )
    with handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    path_in_repo = Path(
        "docs/experiments/artifacts/"
        "EXP-20260513-finedance-g1-wav2clip-stft-beat/hf_manifest.json"
    )
    return Path(handle.name), path_in_repo


def main():
    args = parse_args()
    uploads = planned_uploads(args)
    manifest_path, manifest_path_in_repo = write_manifest(args, uploads)

    print("Planned HF uploads:")
    for rel, kind, _ in uploads:
        full_path = require_path(rel)
        print(f"  {kind:6s} {rel} ({count_files(full_path)} files)")
    print(f"  file   {manifest_path_in_repo} (generated manifest)")

    if args.dry_run:
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install it with `pip install huggingface_hub` "
            "or run scripts/setup_new_server.sh first."
        ) from exc

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    for rel, kind, _ in uploads:
        full_path = require_path(rel)
        if kind == "folder":
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                folder_path=str(full_path),
                path_in_repo=str(rel),
                commit_message=f"Upload {rel}",
                ignore_patterns=["__pycache__/**", "*.pyc", ".DS_Store"],
            )
        else:
            api.upload_file(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                path_or_fileobj=str(full_path),
                path_in_repo=str(rel),
                commit_message=f"Upload {rel}",
            )

    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        path_or_fileobj=str(manifest_path),
        path_in_repo=str(manifest_path_in_repo),
        commit_message="Upload HF artifact manifest",
    )

    print("Upload complete.")


if __name__ == "__main__":
    main()
