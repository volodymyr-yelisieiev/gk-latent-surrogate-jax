"""Write the tracked, path-sanitized release manifest for a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _digest(value: object, *, label: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"aggregate {label} must be a lowercase hexadecimal digest")
    return value


def _fold_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "outer_fold",
        "selected_family",
        "num_test_trajectories",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "learned_flux_rmse",
        "observed_flux_rmse",
        "latent_persistence_flux_rmse",
        "diagnostic_head_oracle_flux_rmse",
        "latent_mse_difference",
        "learned_latent_mse",
        "latent_persistence_latent_mse",
        "learned_latent_mse_by_step",
        "latent_persistence_latent_mse_by_step",
        "learned_flux_rmse_by_step",
        "observed_flux_rmse_by_step",
        "diagnostic_head_oracle_flux_rmse_by_step",
        "learned_flux_mae",
        "observed_flux_mae",
        "latent_persistence_flux_mae",
    }
    return [{key: row[key] for key in sorted(keep) if key in row} for row in rows]


def _stage_config_hashes(ledger: list[dict[str, Any]]) -> dict[str, str]:
    """Return a stable, path-free map of resolved configuration hashes."""
    result: dict[str, str] = {}
    for entry in ledger:
        if entry.get("status") != "accepted" or "config_resolved_sha256" not in entry:
            continue
        outer_fold = entry.get("outer_fold")
        seed = entry.get("training_seed")
        stage = entry.get("stage")
        digest = entry.get("config_resolved_sha256")
        if not isinstance(outer_fold, int) or not isinstance(seed, int) or not isinstance(stage, str):
            raise ValueError("accepted stage ledger entry has an invalid identity")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("accepted stage ledger entry has an invalid config hash")
        result[f"outer_fold_{outer_fold}/seed_{seed}/{stage}"] = digest
    return {key: result[key] for key in sorted(result)}


def build_manifest(
    aggregate_path: Path,
    *,
    repo_root: Path,
    output_path: Path,
    postflight_maintenance: bool = False,
) -> dict[str, Any]:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if not isinstance(aggregate, dict) or aggregate.get("status") != "accepted":
        raise ValueError("aggregate result is missing accepted status")
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("aggregate result is missing artifact hashes")
    source_commit = aggregate.get("source_commit")
    head = _git(repo_root, "rev-parse", "HEAD")
    _digest(source_commit, label="source_commit", length=40)
    analysis_commit = aggregate.get("analysis_commit", source_commit)
    _digest(analysis_commit, label="analysis_commit", length=40)
    source_tag = aggregate.get("source_tag")
    if not isinstance(source_tag, str) or not source_tag.strip():
        raise ValueError("aggregate source_tag is missing")
    tag_commit = _git(repo_root, "rev-parse", "--verify", f"refs/tags/{source_tag}^{{commit}}")
    if tag_commit != source_commit:
        raise ValueError("aggregate source_tag does not resolve to the aggregate source commit")
    if not postflight_maintenance and source_commit != head:
        raise ValueError("release must be generated at the frozen aggregate source commit")
    if not postflight_maintenance and analysis_commit != head:
        raise ValueError("release analysis_commit must resolve to the current HEAD")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("release must be generated from a clean worktree")
    tracked_diff = _digest(
        aggregate.get("analysis_tracked_diff_sha256"),
        label="analysis_tracked_diff_sha256",
        length=64,
    )
    ledger = aggregate.get("stage_ledger", [])
    if not isinstance(ledger, list):
        raise ValueError("aggregate result is missing the stage ledger")
    provenance = {
        "raw_aggregate_sha256": _sha256(aggregate_path),
        "source_commit_verified_against_HEAD": not postflight_maintenance,
        "source_commit_verified_at_release_generation": True,
        "postflight_maintenance_after_release": postflight_maintenance,
        "raw_stage_ledger_not_published": True,
        "trajectory_ids_not_published": True,
        "server_paths_not_published": True,
    }
    if postflight_maintenance:
        provenance["postflight_head_commit"] = head
    manifest = {
        "release_manifest_schema_version": "1.0.0",
        "status": "accepted",
        "protocol_id": aggregate["protocol_id"],
        "protocol_sha256": aggregate["protocol_sha256"],
        "source_tag": source_tag,
        "source_commit": source_commit,
        "analysis_commit": analysis_commit,
        "analysis_tracked_diff_sha256": tracked_diff,
        "dataset_revision": aggregate["dataset_revision"],
        "universe_manifest_sha256": aggregate["universe_manifest_sha256"],
        "universe_trajectory_ids_sha256": aggregate.get("universe_trajectory_ids_sha256"),
        "outer_fold_manifest_sha256": aggregate["outer_fold_manifest_sha256"],
        "config_sha256": aggregate.get("config_sha256", {}),
        "stage_config_resolved_sha256": _stage_config_hashes(ledger),
        "selection": aggregate["selection"],
        "primary": aggregate["primary"],
        "secondary": aggregate["secondary"],
        "seed_variability": aggregate.get("seed_variability", []),
        "secondary_scalar_metrics": aggregate.get("secondary_scalar_metrics", {}),
        "spectra_metrics": aggregate.get("spectra_metrics", {}),
        "bootstrap": aggregate["bootstrap"],
        "outer_fold_summary": _fold_summary(aggregate["outer_fold_summary"]),
        "stage_counts": {
            "planned_slots": len(aggregate.get("stage_ledger", [])),
            "accepted_slots": sum(entry.get("status") == "accepted" for entry in aggregate.get("stage_ledger", [])),
            "skipped_unselected_slots": sum(
                entry.get("status") == "skipped_unselected" for entry in aggregate.get("stage_ledger", [])
            ),
        },
        "artifact_sha256": {
            key: value
            for key, value in artifacts.items()
            if key.endswith("_sha256")
        },
        "wandb_status": aggregate.get("wandb_status", {"enabled": False, "requested": False, "mode": "disabled"}),
        "provenance": provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--postflight-maintenance",
        action="store_true",
        help=(
            "regenerate a release whose evidence source is an immutable tag but whose checkout "
            "contains documented post-run maintenance"
        ),
    )
    args = parser.parse_args()
    build_manifest(
        args.aggregate,
        repo_root=args.repo_root,
        output_path=args.output,
        postflight_maintenance=args.postflight_maintenance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
