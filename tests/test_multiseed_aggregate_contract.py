from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gk_surrogate import multiseed_aggregate as aggregate_module
from gk_surrogate import protocol_runner
from gk_surrogate.multiseed_aggregate import (
    AggregationError,
    _correlation,
    _fold_manifests,
    _metric_series,
    _nonnegative,
    _validate_result_contract,
    _validate_stage_wandb_status,
    _validate_wandb_contract,
    _write_csv,
)


def test_scalar_domain_guards_cover_nonnegative_and_correlation() -> None:
    with pytest.raises(AggregationError, match="non-negative"):
        _nonnegative(-1.0, label="error")
    with pytest.raises(AggregationError, match=r"\[-1, 1\]"):
        _correlation(1.1, label="correlation")
    assert _nonnegative(0.0, label="error") == 0.0
    assert _correlation(-1.0, label="correlation") == -1.0


def test_metrics_and_csv_contracts_cover_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = protocol_runner.Stage("embed", 52, (), tmp_path / "embed")
    stage.output_dir.mkdir()
    (stage.output_dir / "metrics.json").write_text('{"artifact_role":"latent_cache"}', encoding="utf-8")
    status_path = stage.output_dir / "wandb_status.json"
    status_path.write_text(
        json.dumps({"enabled": False, "requested": False, "mode": "disabled"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        protocol_runner,
        "_load_stage_evidence",
        lambda _: {
            "metrics_sha256": "a" * 64,
            "latent_cache_sha256": "b" * 64,
            "wandb_status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        },
    )
    payload, evidence = aggregate_module._metrics_and_evidence(stage)
    assert payload["artifact_role"] == "latent_cache"
    assert evidence["metrics_sha256"] == "a" * 64
    with pytest.raises(AggregationError, match="empty CSV"):
        _write_csv(tmp_path / "empty.csv", [])


def test_wandb_contract_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs = {role: tmp_path / f"{role}.yaml" for role in protocol_runner.CONFIG_ROLES}
    for path in configs.values():
        path.write_text("role: test\n", encoding="utf-8")
    disabled = SimpleNamespace(logging=SimpleNamespace(wandb=SimpleNamespace(enabled=False, mode="disabled")))
    monkeypatch.setattr(protocol_runner, "load_config", lambda *_args, **_kwargs: disabled)
    _validate_wandb_contract(configs)
    enabled = SimpleNamespace(logging=SimpleNamespace(wandb=SimpleNamespace(enabled=True, mode="online")))
    monkeypatch.setattr(protocol_runner, "load_config", lambda *_args, **_kwargs: enabled)
    with pytest.raises(AggregationError, match="W&B"):
        _validate_wandb_contract(configs)
    stage = protocol_runner.Stage("encoder", 52, (), tmp_path / "encoder")
    stage.output_dir.mkdir()
    status_path = stage.output_dir / "wandb_status.json"
    status_path.write_text(json.dumps({"enabled": True, "requested": True, "mode": "online"}), encoding="utf-8")
    with pytest.raises(AggregationError, match="non-disabled"):
        _validate_stage_wandb_status(
            stage,
            {"wandb_status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest()},
        )


def test_metric_series_rejects_negative_values_and_wrong_count() -> None:
    base = {
        "selected_trajectory_ids": ["a"],
        "flux_trajectory_ids": ["a"],
        "flux_rmse_by_trajectory": [1.0],
        "num_trajectories": 1,
    }
    with pytest.raises(AggregationError, match="non-negative"):
        _metric_series(
            {**base, "flux_rmse_by_trajectory": [-1.0]},
            value_key="flux_rmse_by_trajectory",
            id_key="flux_trajectory_ids",
            label="test",
        )
    with pytest.raises(AggregationError, match="num_trajectories"):
        _metric_series(
            {**base, "num_trajectories": 2},
            value_key="flux_rmse_by_trajectory",
            id_key="flux_trajectory_ids",
            label="test",
        )


def test_result_contract_rejects_malformed_release_fields() -> None:
    result = {
        "result_schema_version": "1.0.0",
        "status": "accepted",
        "protocol_id": "multiseed-v1",
        "protocol_sha256": "a" * 64,
        "source_tag": "freeze",
        "source_commit": "b" * 40,
        "dataset_revision": "dataset-v1",
        "universe_manifest_sha256": "c" * 64,
        "outer_fold_manifest_sha256": "d" * 64,
        "selection": {},
        "primary_estimand": "estimand",
        "bootstrap": {},
        "primary": {},
        "secondary": {},
        "seed_summary": [{} for _ in range(25)],
        "seed_variability": [
            {
                "training_seed": seed,
                "mean_difference": 0.0,
                "std_across_outer_folds": 0.0,
                "min_across_outer_folds": 0.0,
                "max_across_outer_folds": 0.0,
            }
            for seed in (52, 53, 54, 55, 56)
        ],
        "outer_fold_summary": [{} for _ in range(5)],
        "stage_ledger": [{} for _ in range(255)],
        "artifacts": {
            "outer_fold_summary_sha256": "e" * 64,
            "paired_trajectory_results_sha256": "f" * 64,
            "primary_difference_figure_sha256": "1" * 64,
        },
        "wandb_status": {"enabled": False, "requested": False, "mode": "disabled", "config_verified": True},
    }
    for field, value, message in (
        ("protocol_sha256", "bad", "SHA-256"),
        ("source_commit", "bad", "commit"),
        ("outer_fold_summary", [], "exactly five"),
        ("stage_ledger", [], "255"),
        ("artifacts", [], "artifacts"),
    ):
        invalid = {**result, field: value}
        with pytest.raises(AggregationError, match=message):
            _validate_result_contract(invalid)
    with pytest.raises(AggregationError, match="commit"):
        _validate_result_contract({**result, "source_commit": "g" * 40})
    with pytest.raises(AggregationError, match="SHA-256"):
        _validate_result_contract({**result, "protocol_sha256": "g" * 64})
    invalid = {**result, "artifacts": {**result["artifacts"], "outer_fold_summary_sha256": "bad"}}
    with pytest.raises(AggregationError, match="artifact"):
        _validate_result_contract(invalid)


def test_aggregate_cli_reports_failure_and_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    args = [
        "--protocol",
        str(tmp_path / "protocol.json"),
        "--fold-manifest",
        str(tmp_path / "folds.json"),
        "--universe-manifest",
        str(tmp_path / "universe.json"),
        "--output-root",
        str(tmp_path / "outputs"),
        "--repo-root",
        str(tmp_path),
    ]
    for role in protocol_runner.CONFIG_ROLES:
        args.extend(("--config", f"{role}=config.yaml"))
    monkeypatch.setattr(
        aggregate_module,
        "aggregate",
        lambda *_, **__: {
            "status": "accepted",
            "protocol_id": "multiseed-v1",
            "source_tag": "freeze",
            "primary": {},
            "bootstrap": {},
        },
    )
    assert aggregate_module.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "accepted"

    def fail_aggregate(*_: object, **__: object) -> dict[str, object]:
        raise AggregationError("bad evidence")

    monkeypatch.setattr(aggregate_module, "aggregate", fail_aggregate)
    assert aggregate_module.main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_validation_stage_payload_rejects_frozen_id_mismatches(tmp_path: Path) -> None:
    cli = tmp_path / "fold.json"
    cli.write_text("fold", encoding="utf-8")
    stage = protocol_runner.Stage(
        "gru_validation",
        52,
        ("fake", "--split-manifest", str(cli)),
        tmp_path / "validation",
        phase="validation",
        family="gru",
    )
    fold = {"fold_id": "outer-0", "splits": {"train": ["a"], "val": ["b"], "test": ["c"]}}
    payload = {
        "protocol_version": 1,
        "data_split": "val",
        "data_split_seed": 52,
        "training_seed": 52,
        "selected_trajectory_ids": ["b"],
        "trajectory_manifest_sha256": protocol_runner._trajectory_manifest_sha256(["b"]),
        "universe_trajectory_ids": ["a", "b", "c"],
        "universe_manifest_sha256": protocol_runner._trajectory_manifest_sha256(["a", "b", "c"]),
        "split_fold_id": "outer-0",
        "split_manifest_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "flux_trajectory_ids": ["b"],
        "flux_rmse_by_trajectory": [1.0],
        "num_trajectories": 1,
        "trajectory_balanced_flux_rmse": 1.0,
        "flux_rmse": 1.0,
    }
    bad_selected = {**payload, "selected_trajectory_ids": ["a"]}
    with pytest.raises(AggregationError, match="selected IDs"):
        aggregate_module._validate_stage_payload(
            bad_selected,
            stage=stage,
            fold=fold,
            seed=52,
            expected_split="val",
        )
    bad_universe = {**payload, "universe_trajectory_ids": ["c", "b", "a"]}
    with pytest.raises(AggregationError, match="universe IDs"):
        aggregate_module._validate_stage_payload(
            bad_universe,
            stage=stage,
            fold=fold,
            seed=52,
            expected_split="val",
            canonical_universe_ids=["a", "b", "c"],
        )


def test_selection_contract_rejects_missing_family_hash_and_nonminimum(tmp_path: Path) -> None:
    validation = {
        family: {52: {"trajectory_balanced_flux_rmse": 1.0 if family == "gru" else 2.0}}
        for family in ("gru", "transformer")
    }
    hashes = {family: {52: f"{family}-hash"} for family in ("gru", "transformer")}
    base = {
        "protocol_version": 1,
        "outer_fold": 0,
        "selection_split": "val",
        "primary_metric": "trajectory_balanced_flux_rmse",
        "matched_training_seeds": [52],
        "candidate_mean_validation_trajectory_balanced_flux_rmse": {"gru": 1.0, "transformer": 2.0},
        "candidate_validation_metrics_sha256": {"gru": {"52": "gru-hash"}},
        "selected_family": "gru",
        "test_evidence_opened": False,
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(AggregationError, match="lacks hashes"):
        aggregate_module._selection(
            path,
            fold_id=0,
            seeds=(52,),
            validation_payloads=validation,
            validation_hashes=hashes,
        )
    nonminimum = {
        **base,
        "candidate_validation_metrics_sha256": {
            "gru": {"52": "gru-hash"},
            "transformer": {"52": "transformer-hash"},
        },
        "selected_family": "transformer",
    }
    path.write_text(json.dumps(nonminimum), encoding="utf-8")
    with pytest.raises(AggregationError, match="deterministic minimum"):
        aggregate_module._selection(
            path,
            fold_id=0,
            seeds=(52,),
            validation_payloads=validation,
            validation_hashes=hashes,
        )


def _valid_aggregate_inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(
        json.dumps({"dataset_revision": "dataset-v1", "trajectory_ids": ["a", "b", "c", "d", "e"]}),
        encoding="utf-8",
    )
    fold_path = tmp_path / "folds.json"
    fold_path.write_text("fold-index", encoding="utf-8")
    protocol = copy.deepcopy(
        json.loads((Path(__file__).parents[1] / "experiment_protocols/multiseed_v1.json").read_text())
    )
    protocol.update(
        status="frozen",
        source={
            "repository": "https://example.invalid/repo",
            "commit": "0" * 40,
            "tag": "freeze",
            "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )
    protocol["data"]["dataset_revision"] = "dataset-v1"
    protocol["data"]["universe_manifest_sha256"] = hashlib.sha256(universe_path.read_bytes()).hexdigest()
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = hashlib.sha256(fold_path.read_bytes()).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol_path, universe_path, fold_path, protocol


def test_fold_manifest_reader_rejects_determinism_and_artifact_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol_path, universe_path, fold_path, protocol = _valid_aggregate_inputs(tmp_path)
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    generated = protocol_runner.generate_outer_fold_manifest(protocol, universe)

    def write_index(index: dict[str, Any]) -> None:
        fold_path.write_bytes(protocol_runner._canonical_json_bytes(index))
        protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = hashlib.sha256(
            fold_path.read_bytes()
        ).hexdigest()
        for fold in index["folds"]:
            cli_path = protocol_runner._fold_cli_manifest_path(fold_path, int(fold["outer_fold"]))
            cli_payload = protocol_runner._fold_cli_manifest_payload(protocol["protocol_id"], fold)
            cli_path.write_bytes(protocol_runner._canonical_json_bytes(cli_payload))

    # A hash-consistent but semantically changed index must be rejected against
    # deterministic regeneration before any CLI files are trusted.
    altered = copy.deepcopy(generated)
    altered["dataset_revision"] = "wrong"
    write_index(altered)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(AggregationError, match="deterministic regeneration"):
        _fold_manifests(protocol, universe_path, fold_path)

    # The remaining checks are isolated by making the runner's regeneration
    # return the intentionally malformed index verbatim.
    monkeypatch.setattr(protocol_runner, "generate_outer_fold_manifest", lambda *_: current_index)
    current_index: dict[str, Any] = copy.deepcopy(generated)
    current_index["folds"][0]["outer_fold"] = 9
    write_index(current_index)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(AggregationError, match="invalid outer fold"):
        _fold_manifests(protocol, universe_path, fold_path)

    current_index = copy.deepcopy(generated)
    write_index(current_index)
    protocol_runner._fold_cli_manifest_path(fold_path, 2).unlink()
    with pytest.raises(AggregationError, match="missing CLI"):
        _fold_manifests(protocol, universe_path, fold_path)

    current_index = copy.deepcopy(generated)
    write_index(current_index)
    noncanonical = protocol_runner._fold_cli_manifest_path(fold_path, 0)
    noncanonical.write_text("not canonical", encoding="utf-8")
    current_index["folds"][0]["cli_split_manifest_sha256"] = hashlib.sha256(noncanonical.read_bytes()).hexdigest()
    write_index(current_index)
    noncanonical.write_text("not canonical", encoding="utf-8")
    with pytest.raises(AggregationError, match="not canonical"):
        _fold_manifests(protocol, universe_path, fold_path)

    current_index = copy.deepcopy(generated)
    current_index["folds"][1] = copy.deepcopy(current_index["folds"][0])
    write_index(current_index)
    with pytest.raises(AggregationError, match="cover folds"):
        _fold_manifests(protocol, universe_path, fold_path)


def _patch_valid_aggregate_plumbing(
    monkeypatch: pytest.MonkeyPatch,
    protocol_path: Path,
    folds: dict[int, dict[str, object]],
    *,
    stage_plan: list[protocol_runner.Stage] | None = None,
    tagged_bytes: bytes | None = None,
) -> None:
    monkeypatch.setattr(aggregate_module, "_source_tag_commit", lambda *args, **kwargs: ("freeze", "0" * 40))
    monkeypatch.setattr(aggregate_module, "_fold_manifests", lambda *args, **kwargs: folds)
    if stage_plan is not None:
        monkeypatch.setattr(aggregate_module.protocol_runner, "build_stages", lambda *args, **kwargs: stage_plan)
    payload = tagged_bytes
    monkeypatch.setattr(
        aggregate_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=protocol_path.read_bytes() if payload is None else payload,
        ),
    )


def test_aggregate_rejects_early_protocol_and_source_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol_path, universe_path, fold_path, protocol = _valid_aggregate_inputs(tmp_path)
    kwargs = {
        "fold_manifest_path": fold_path,
        "universe_manifest_path": universe_path,
        "output_root": tmp_path / "result",
        "configs": {},
        "repo_root": tmp_path,
    }
    planned = {**protocol, "status": "planned"}
    protocol_path.write_text(json.dumps(planned), encoding="utf-8")
    with pytest.raises(AggregationError, match="requires the immutable frozen"):
        aggregate_module.aggregate(protocol_path, **kwargs)

    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    invalid_data = copy.deepcopy(protocol)
    invalid_data["data"]["normalization_fit_split"] = "all"
    protocol_path.write_text(json.dumps(invalid_data), encoding="utf-8")
    with pytest.raises(AggregationError, match="frozen protocol contract"):
        aggregate_module.aggregate(protocol_path, **kwargs)

    monkeypatch.setattr(aggregate_module.protocol_runner, "_validate_protocol_schema_contract", lambda *_: None)
    monkeypatch.setattr(aggregate_module.protocol_runner, "_validate_protocol", lambda *_args, **_kwargs: None)
    accepted = copy.deepcopy(protocol)
    accepted["accepted_runs"] = [{"stage": "x"}]
    protocol_path.write_text(json.dumps(accepted), encoding="utf-8")
    with pytest.raises(AggregationError, match="empty accepted_runs"):
        aggregate_module.aggregate(protocol_path, **kwargs)

    missing_id = copy.deepcopy(protocol)
    missing_id["protocol_id"] = ""
    protocol_path.write_text(json.dumps(missing_id), encoding="utf-8")
    with pytest.raises(AggregationError, match="protocol_id is missing"):
        aggregate_module.aggregate(protocol_path, **kwargs)

    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    _patch_valid_aggregate_plumbing(monkeypatch, protocol_path, {})
    with pytest.raises(AggregationError, match="inside repo_root"):
        aggregate_module.aggregate(
            protocol_path,
            **{**kwargs, "repo_root": tmp_path / "different-root"},
        )
    _patch_valid_aggregate_plumbing(monkeypatch, protocol_path, {}, tagged_bytes=b"mismatch")
    with pytest.raises(AggregationError, match="current protocol bytes"):
        aggregate_module.aggregate(
            protocol_path,
            **{**kwargs, "repo_root": tmp_path},
        )


def test_aggregate_rejects_model_and_stage_plan_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol_path, universe_path, fold_path, protocol = _valid_aggregate_inputs(tmp_path)
    kwargs = {
        "fold_manifest_path": fold_path,
        "universe_manifest_path": universe_path,
        "output_root": tmp_path / "result",
        "configs": {},
        "repo_root": tmp_path,
    }
    monkeypatch.setattr(aggregate_module.protocol_runner, "_validate_protocol_schema_contract", lambda *_: None)
    monkeypatch.setattr(aggregate_module.protocol_runner, "_validate_protocol", lambda *_args, **_kwargs: [52])
    monkeypatch.setattr(aggregate_module, "_validate_wandb_contract", lambda *_args, **_kwargs: None)
    _patch_valid_aggregate_plumbing(monkeypatch, protocol_path, {})
    no_models = copy.deepcopy(protocol)
    no_models["models"] = None
    protocol_path.write_text(json.dumps(no_models), encoding="utf-8")
    with pytest.raises(AggregationError, match="models is missing"):
        aggregate_module.aggregate(protocol_path, **kwargs)

    wrong_seeds = copy.deepcopy(protocol)
    wrong_seeds["models"]["training_seeds"] = [52]
    protocol_path.write_text(json.dumps(wrong_seeds), encoding="utf-8")
    with pytest.raises(AggregationError, match="matched seeds"):
        aggregate_module.aggregate(protocol_path, **kwargs)

    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    folds = {
        fold: {"fold_id": f"outer-{fold}", "splits": {"train": ["a"], "val": ["b"], "test": ["c"]}}
        for fold in range(5)
    }
    _patch_valid_aggregate_plumbing(monkeypatch, protocol_path, folds, stage_plan=[])
    with pytest.raises(AggregationError, match="stage plan length"):
        aggregate_module.aggregate(protocol_path, **kwargs)
