#!/usr/bin/env python3
"""Upload EDGE G1 runtime artifacts to one Hugging Face dataset repo."""

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_ID = "wyksdsg/edge-g1-beatdistance"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIFFUSION_ROOT = REPO_ROOT.parent / "diffusion"

WAV2CLIP_FEATURE_RELS = (
    Path("data/finedance_g1_fkbeats/train/wav2clip_stft_beat_feats"),
    Path("data/finedance_g1_fkbeats/test/wav2clip_stft_beat_feats"),
)
WAV2CLIP_CACHE_REL = Path(
    "data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups"
)
WAV2CLIP_CHECKPOINT_REL = Path(
    "runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/"
    "weights/train-500.pt"
)
WAV2CLIP_EVIDENCE_REL = Path(
    "docs/experiments/artifacts/EXP-20260513-finedance-g1-wav2clip-stft-beat"
)

DIFFUSION_DATA_REL = Path("data/finedance_g1_fkbeats")
DIFFUSION_CACHE_RELS = (
    Path("data/finedance_g1_fkbeats_dataset_backups_fkbeat1000"),
    Path("data/finedance_g1_librosa35_fullctx_motiondist_cond_dataset_backups"),
)
DIFFUSION_CHECKPOINT_RELS = (
    Path("runs/train/finedance_g1_fkbeatdistance_1000/weights/train-1000.pt"),
    Path(
        "runs/train/finedance_g1_librosa35_fullctx_motiondist_cond_2000/"
        "weights/train-2000.pt"
    ),
)
DIFFUSION_EVIDENCE_REL = Path(
    "docs/experiments/artifacts/EXP-20260512-diffusion-baseline-evidence"
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
            "Upload the EDGE G1 runtime artifacts to a single Hugging Face "
            "dataset repo. The uploaded layout preserves repo-relative paths so "
            "scripts/setup_new_server.sh can download them directly."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument("--revision", default="main", help="Target branch/revision.")
    parser.add_argument("--private", action="store_true", help="Create repo as private.")
    parser.add_argument(
        "--diffusion-root",
        default=str(DEFAULT_DIFFUSION_ROOT),
        help="Path to the diffusion branch checkout/worktree.",
    )
    parser.add_argument(
        "--skip-diffusion-data",
        action="store_true",
        help="Do not upload diffusion's real data/finedance_g1_fkbeats tree.",
    )
    parser.add_argument(
        "--include-cache",
        action="store_true",
        help="Upload wav2clip stream-adapter tensor/cache backup.",
    )
    parser.add_argument(
        "--include-checkpoint",
        action="store_true",
        help="Upload wav2clip r02 stream-adapter train-500.pt.",
    )
    parser.add_argument(
        "--include-diffusion-caches",
        action="store_true",
        help="Upload diffusion baseline tensor/cache backups.",
    )
    parser.add_argument(
        "--include-diffusion-checkpoints",
        action="store_true",
        help="Upload diffusion anchor checkpoint .pt files.",
    )
    parser.add_argument(
        "--skip-evidence",
        action="store_true",
        help="Do not upload curated GitHub-safe evidence folders.",
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


def planned_uploads(args):
    diffusion_root = Path(args.diffusion_root).resolve()
    uploads = []

    if not args.skip_diffusion_data:
        uploads.append(
            UploadItem(
                diffusion_root,
                DIFFUSION_DATA_REL,
                DIFFUSION_DATA_REL,
                "folder",
                "real FineDance+G1 prepared data from diffusion",
            )
        )

    for rel in WAV2CLIP_FEATURE_RELS:
        uploads.append(
            UploadItem(
                REPO_ROOT,
                rel,
                rel,
                "folder",
                "wav2clip_stft_beat feature shard",
            )
        )

    if args.include_cache:
        uploads.append(
            UploadItem(
                REPO_ROOT,
                WAV2CLIP_CACHE_REL,
                WAV2CLIP_CACHE_REL,
                "folder",
                "wav2clip stream-adapter tensor cache",
            )
        )
    if args.include_checkpoint:
        uploads.append(
            UploadItem(
                REPO_ROOT,
                WAV2CLIP_CHECKPOINT_REL,
                WAV2CLIP_CHECKPOINT_REL,
                "file",
                "wav2clip stream-adapter checkpoint",
            )
        )
    if args.include_diffusion_caches:
        for rel in DIFFUSION_CACHE_RELS:
            uploads.append(
                UploadItem(
                    diffusion_root,
                    rel,
                    rel,
                    "folder",
                    "diffusion baseline tensor cache",
                )
            )
    if args.include_diffusion_checkpoints:
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
    if not args.skip_evidence:
        uploads.append(
            UploadItem(
                REPO_ROOT,
                WAV2CLIP_EVIDENCE_REL,
                WAV2CLIP_EVIDENCE_REL,
                "folder",
                "wav2clip curated evidence",
            )
        )
        uploads.append(
            UploadItem(
                diffusion_root,
                DIFFUSION_EVIDENCE_REL,
                DIFFUSION_EVIDENCE_REL,
                "folder",
                "diffusion curated evidence",
            )
        )
    return uploads


def write_manifest(args, uploads):
    diffusion_root = Path(args.diffusion_root).resolve()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "layout": "repo-relative",
        "source_commits": {
            "wav2clip": git_commit(REPO_ROOT),
            "diffusion": git_commit(diffusion_root),
        },
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
    path_in_repo = Path("docs/experiments/artifacts/hf_manifest.json")
    return Path(handle.name), path_in_repo


def upload_item(api, args, item):
    full_path = require_item(item)
    if item.kind == "folder":
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            folder_path=str(full_path),
            path_in_repo=str(item.path_in_repo),
            commit_message=f"Upload {item.path_in_repo}",
            ignore_patterns=["__pycache__/**", "*.pyc", ".DS_Store"],
        )
    else:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            path_or_fileobj=str(full_path),
            path_in_repo=str(item.path_in_repo),
            commit_message=f"Upload {item.path_in_repo}",
        )


def main():
    args = parse_args()
    uploads = planned_uploads(args)
    manifest_path, manifest_path_in_repo = write_manifest(args, uploads)

    print(f"Target HF dataset repo: {args.repo_id}")
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
            "or run scripts/setup_new_server.sh first."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    for item in uploads:
        upload_item(api, args, item)

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
