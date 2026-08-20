"""Validate blinded video assets and build balanced pairwise study manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


ROUTES = ("M0", "M2", "M4")
TASK_QUESTIONS = {
    "generator_quality": ("H-NATURAL", "H-DANCE", "H-EXPRESS"),
    "music_match": ("H-RHYTHM", "H-STYLE"),
    "execution_retention": ("H-NATURAL", "H-DANCE", "H-EXPRESS"),
}
MATCH_FIELDS = (
    "song_id",
    "generation_seed",
    "window_id",
    "duration_seconds",
    "camera_profile",
    "render_profile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("eval/human_study/assets_current_20260820.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("eval/human_study/design_20260820"),
    )
    parser.add_argument("--participants", type=int, default=24)
    parser.add_argument("--trials_per_task", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def _load(path: Path):
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _resolve_asset_path(asset: dict, repo_root: Path) -> Path:
    path = Path(asset["path"]).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _blind_id(asset_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{asset_id}".encode()).hexdigest()[:10]
    return f"clip_{digest}"


def _asset_issues(asset: dict, repo_root: Path) -> list[str]:
    issues = []
    required = {
        "asset_id",
        "path",
        "route",
        "representation",
        "song_id",
        "generation_seed",
        "window_id",
        "duration_seconds",
        "camera_profile",
        "render_profile",
        "label_free",
        "audio_embedded",
        "eligible",
    }
    missing = sorted(required - set(asset))
    if missing:
        return [f"missing fields: {', '.join(missing)}"]
    path = _resolve_asset_path(asset, repo_root)
    if not path.is_file():
        issues.append("video file missing")
    if asset["route"] not in ROUTES:
        issues.append("route must be M0, M2, or M4")
    if asset["representation"] not in ("reference", "execution"):
        issues.append("representation must be reference or execution")
    if not asset["label_free"]:
        issues.append("video contains identifying labels or debug composition")
    if float(asset["duration_seconds"]) <= 0:
        issues.append("duration must be positive")
    if asset["representation"] == "execution" and asset.get("tracker_repeat") is None:
        issues.append("execution asset requires tracker_repeat")
    if not asset["eligible"]:
        issues.extend(asset.get("ineligible_reasons", ["asset marked ineligible"]))
    return list(dict.fromkeys(issues))


def _match_key(asset: dict) -> tuple:
    return tuple(asset[field] for field in MATCH_FIELDS)


def _candidate(
    task: str,
    first: dict,
    second: dict,
    index: int,
) -> dict:
    questions = list(TASK_QUESTIONS[task])
    if task == "generator_quality" and float(first["duration_seconds"]) >= 30.0:
        questions.append("H-COHERENCE")
    return {
        "trial_id": f"{task}_{index:04d}",
        "task": task,
        "asset_a": first["asset_id"],
        "asset_b": second["asset_id"],
        "questions": questions,
        "match": {field: first[field] for field in MATCH_FIELDS},
    }


def build_candidates(assets: list[dict]) -> list[dict]:
    candidates = []
    references = defaultdict(dict)
    executions = defaultdict(list)
    for asset in assets:
        if asset["representation"] == "reference":
            references[_match_key(asset)][asset["route"]] = asset
        else:
            executions[(_match_key(asset), asset["route"])].append(asset)

    for key in sorted(references, key=str):
        by_route = references[key]
        for route_a, route_b in combinations(ROUTES, 2):
            if route_a not in by_route or route_b not in by_route:
                continue
            for task in ("generator_quality", "music_match"):
                if task == "music_match" and not (
                    by_route[route_a]["audio_embedded"]
                    and by_route[route_b]["audio_embedded"]
                ):
                    continue
                candidates.append(
                    _candidate(task, by_route[route_a], by_route[route_b], len(candidates) + 1)
                )

    for key in sorted(references, key=str):
        for route, reference in references[key].items():
            for execution in sorted(
                executions.get((key, route), []),
                key=lambda item: int(item["tracker_repeat"]),
            ):
                candidates.append(
                    _candidate(
                        "execution_retention", reference, execution, len(candidates) + 1
                    )
                )
    return candidates


def build_assignments(
    candidates: list[dict],
    *,
    participants: int,
    trials_per_task: int,
    seed: int,
) -> list[dict]:
    by_task = defaultdict(list)
    for trial in candidates:
        by_task[trial["task"]].append(trial)
    assignments = []
    for participant_index in range(participants):
        rng = random.Random(seed + participant_index)
        trials = []
        for task in TASK_QUESTIONS:
            bank = list(by_task[task])
            rng.shuffle(bank)
            selected = bank[: min(trials_per_task, len(bank))]
            for local_index, trial in enumerate(selected):
                swap = (participant_index + local_index) % 2 == 1
                left = trial["asset_b"] if swap else trial["asset_a"]
                right = trial["asset_a"] if swap else trial["asset_b"]
                trials.append(
                    {
                        "trial_id": trial["trial_id"],
                        "task": task,
                        "left_asset": left,
                        "right_asset": right,
                        "questions": trial["questions"],
                    }
                )
        rng.shuffle(trials)
        assignments.append(
            {
                "participant_id": f"P{participant_index + 1:03d}",
                "trials": trials,
            }
        )
    return assignments


def _coverage(candidates: list[dict]) -> dict:
    output = {}
    for task in TASK_QUESTIONS:
        selected = [trial for trial in candidates if trial["task"] == task]
        output[task] = {
            "candidate_trials": len(selected),
            "songs": sorted({trial["match"]["song_id"] for trial in selected}),
            "generation_seeds": sorted(
                {str(trial["match"]["generation_seed"]) for trial in selected}
            ),
        }
    return output


def _readiness(assets: list[dict], candidates: list[dict], minimum: dict) -> dict:
    coverage = _coverage(candidates)
    pilot_ready = all(coverage[task]["candidate_trials"] > 0 for task in TASK_QUESTIONS)
    complete_reference_groups = defaultdict(set)
    execution_repeats = defaultdict(set)
    for asset in assets:
        key = (
            asset["song_id"],
            str(asset["generation_seed"]),
            asset["window_id"],
        )
        if asset["representation"] == "reference" and asset["audio_embedded"]:
            complete_reference_groups[key].add(asset["route"])
        elif asset["representation"] == "execution":
            execution_repeats[(*key, asset["route"])].add(int(asset["tracker_repeat"]))
    complete = {
        key for key, routes in complete_reference_groups.items() if set(ROUTES).issubset(routes)
    }
    songs = sorted({key[0] for key in complete})
    seeds_per_song = {
        song: len({key[1] for key in complete if key[0] == song}) for song in songs
    }
    required_songs = int(minimum.get("songs", 3))
    required_seeds = int(minimum.get("generation_seeds_per_song", 3))
    required_repeats = int(minimum.get("tracker_repeats_per_reference", 3))
    reasons = []
    if len(songs) < required_songs:
        reasons.append(f"need {required_songs} songs with complete M0/M2/M4 references; found {len(songs)}")
    if any(seeds_per_song[song] < required_seeds for song in songs) or not songs:
        reasons.append(f"need {required_seeds} generation seeds per song")
    missing_tracker = [
        (*key, route)
        for key in complete
        for route in ROUTES
        if len(execution_repeats.get((*key, route), set())) < required_repeats
    ]
    if missing_tracker:
        reasons.append(
            f"need {required_repeats} tracker repeats for every matched reference; "
            f"{len(missing_tracker)} route-window groups incomplete"
        )
    if not pilot_ready:
        reasons.append("at least one pairwise study has no candidate trials")
    return {
        "pilot_ready": pilot_ready,
        "paper_ready": not reasons,
        "paper_blockers": reasons,
        "complete_songs": songs,
        "complete_generation_seeds_per_song": seeds_per_song,
    }


def main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _load(args.assets)
    assets = payload["assets"]
    audit = {
        asset["asset_id"]: _asset_issues(asset, repo_root)
        for asset in assets
    }
    eligible = [asset for asset in assets if not audit[asset["asset_id"]]]
    candidates = build_candidates(eligible)
    minimum = payload.get("minimum_paper_design", {})
    readiness = _readiness(eligible, candidates, minimum)
    assignments = (
        build_assignments(
            candidates,
            participants=args.participants,
            trials_per_task=args.trials_per_task,
            seed=args.seed,
        )
        if candidates
        else []
    )
    blind_ids = {asset["asset_id"]: _blind_id(asset["asset_id"], args.seed) for asset in eligible}

    public_assignments = []
    for assignment in assignments:
        public_assignments.append(
            {
                "participant_id": assignment["participant_id"],
                "trials": [
                    {
                        **trial,
                        "left_asset": blind_ids[trial["left_asset"]],
                        "right_asset": blind_ids[trial["right_asset"]],
                    }
                    for trial in assignment["trials"]
                ],
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(exist_ok=True)
    public_media = {}
    for asset in eligible:
        blind_id = blind_ids[asset["asset_id"]]
        target = media_dir / f"{blind_id}.mp4"
        source = _resolve_asset_path(asset, repo_root)
        if target.exists() and not os.path.samefile(source, target):
            target.unlink()
        if not target.exists():
            os.link(source, target)
        public_media[blind_id] = f"media/{blind_id}.mp4"
    coverage = _coverage(candidates)
    design = {
        "schema_version": "audiomimic_human_pairwise_v1",
        "seed": args.seed,
        "participants_planned": args.participants,
        "trials_per_task_cap": args.trials_per_task,
        "eligible_assets": len(eligible),
        "candidate_trials": len(candidates),
        "coverage": coverage,
        **readiness,
        "ready": readiness["paper_ready"],
        "media": public_media,
        "public_assignments": public_assignments,
    }
    private_key = {
        "blind_asset_key": {
            blind_ids[asset["asset_id"]]: {
                "asset_id": asset["asset_id"],
                "path": asset["path"],
                "route": asset["route"],
                "representation": asset["representation"],
                "tracker_repeat": asset.get("tracker_repeat"),
            }
            for asset in eligible
        },
        "canonical_trials": candidates,
    }
    audit_payload = {
        "assets_total": len(assets),
        "eligible_assets": len(eligible),
        "ineligible_assets": {
            asset_id: issues for asset_id, issues in audit.items() if issues
        },
        "minimum_paper_design": minimum,
    }
    (output_dir / "public_study.json").write_text(
        json.dumps(design, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "private_key.json").write_text(
        json.dumps(private_key, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "asset_audit.json").write_text(
        json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Audited {len(assets)} assets: {len(eligible)} eligible, "
        f"{len(candidates)} candidate trials, pilot_ready={design['pilot_ready']}, "
        f"paper_ready={design['paper_ready']}"
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main(parse_args())
