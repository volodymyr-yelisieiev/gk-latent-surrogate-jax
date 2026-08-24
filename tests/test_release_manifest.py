from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/write_multiseed_release_manifest.py"
    spec = importlib.util.spec_from_file_location("write_multiseed_release_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate(source_commit: str) -> dict[str, object]:
    digest = "a" * 64
    horizon = {
        "learned_latent_mse_by_step": [0.1],
        "latent_persistence_latent_mse_by_step": [0.2],
        "learned_flux_rmse_by_step": [1.0],
        "observed_flux_rmse_by_step": [0.8],
        "diagnostic_head_oracle_flux_rmse_by_step": [0.9],
    }
    ledger = [
        {
            "outer_fold": 0,
            "training_seed": 52,
            "stage": "encoder",
            "status": "accepted",
            "config_resolved_sha256": digest,
            "metrics_path": "/server/private/metrics.json",
        },
        {"outer_fold": 0, "training_seed": 52, "stage": "selection", "status": "accepted"},
    ]
    return {
        "status": "accepted",
        "protocol_id": "multiseed-v1",
        "protocol_sha256": digest,
        "source_tag": "protocol/multiseed-v1",
        "source_commit": source_commit,
        "analysis_commit": source_commit,
        "analysis_tracked_diff_sha256": "c" * 64,
        "dataset_revision": "dataset-v1",
        "universe_manifest_sha256": digest,
        "universe_trajectory_ids_sha256": digest,
        "outer_fold_manifest_sha256": digest,
        "config_sha256": {"encoder": digest},
        "selection": {},
        "primary": {},
        "secondary": {},
        "secondary_scalar_metrics": {},
        "spectra_metrics": {},
        "bootstrap": {},
        "outer_fold_summary": [dict(horizon) for _ in range(5)],
        "stage_ledger": ledger,
        "artifacts": {"outer_fold_summary_sha256": digest},
        "wandb_status": {"enabled": False, "requested": False, "mode": "disabled"},
    }


def test_release_manifest_is_sanitized_and_tracks_stage_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    commit = "b" * 40
    def fake_git(*args: object) -> str:
        return "" if "status" in args else commit

    monkeypatch.setattr(module, "_git", fake_git)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(_aggregate(commit)), encoding="utf-8")
    output = tmp_path / "release.json"
    manifest = module.build_manifest(aggregate_path, repo_root=tmp_path, output_path=output)
    assert manifest["release_manifest_schema_version"] == "1.0.0"
    assert manifest["stage_config_resolved_sha256"] == {"outer_fold_0/seed_52/encoder": "a" * 64}
    assert manifest["outer_fold_summary"][0]["learned_flux_rmse_by_step"] == [1.0]
    assert "stage_ledger" not in manifest
    assert "/server/private" not in json.dumps(manifest)


def test_release_manifest_rejects_source_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_git", lambda *_args: "c" * 40)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(_aggregate("b" * 40)), encoding="utf-8")
    with pytest.raises(ValueError, match="source commit"):
        module.build_manifest(aggregate_path, repo_root=tmp_path, output_path=tmp_path / "release.json")


def test_release_manifest_can_reproduce_explicit_postflight_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    source_commit = "b" * 40
    head = "d" * 40

    def fake_git(*args: object) -> str:
        if "status" in args:
            return ""
        if any("refs/tags/" in str(arg) for arg in args):
            return source_commit
        return head

    monkeypatch.setattr(module, "_git", fake_git)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(_aggregate(source_commit)), encoding="utf-8")
    manifest = module.build_manifest(
        aggregate_path,
        repo_root=tmp_path,
        output_path=tmp_path / "release.json",
        postflight_maintenance=True,
    )
    provenance = manifest["provenance"]
    assert provenance["source_commit_verified_against_HEAD"] is False
    assert provenance["source_commit_verified_at_release_generation"] is True
    assert provenance["postflight_maintenance_after_release"] is True
    assert provenance["postflight_generation_commit"] == head


def test_release_manifest_schema_is_distinct_from_raw_result_schema() -> None:
    schema = json.loads((Path(__file__).parents[1] / "experiment_protocols/release_manifest.schema.json").read_text())
    assert schema["required"][0] == "release_manifest_schema_version"
    assert "stage_ledger" not in schema["required"]
