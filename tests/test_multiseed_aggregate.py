from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gk_surrogate import multiseed_aggregate as aggregate_module
from gk_surrogate import protocol_runner
from gk_surrogate.multiseed_aggregate import (
    AggregationError,
    _bootstrap,
    _bootstrap_within_fold,
    _config_arg,
    _finite,
    _finite_array,
    _hierarchical_mean,
    _metric_series,
    _read_json,
    _require_close,
    _require_equal,
    _sha256,
    _source_tag_commit,
    _string_array,
    _validate_result_contract,
)


def test_hierarchical_mean_weights_folds_and_seeds_equally() -> None:
    groups = {
        0: {52: [0.0, 0.0], 53: [2.0, 2.0]},
        1: {52: [10.0], 53: [10.0]},
    }
    # Fold means are 1 and 10, so the equal-fold estimand is 5.5 rather than
    # the flattened-row mean (which would give a different weight to fold 0).
    assert _hierarchical_mean(groups) == pytest.approx(5.5)


def test_hierarchical_bootstrap_is_deterministic_and_finite() -> None:
    groups = {
        0: {52: [1.0, 2.0], 53: [2.0, 3.0]},
        1: {52: [4.0, 5.0], 53: [5.0, 6.0]},
    }
    first = _bootstrap(groups, replicates=500, seed=17)
    second = _bootstrap(groups, replicates=500, seed=17)
    assert first == second
    assert first["ci_lower"] <= first["mean"] <= first["ci_upper"]


def test_metric_series_rejects_unpaired_trajectory_arrays() -> None:
    payload = {
        "selected_trajectory_ids": ["a", "b"],
        "flux_trajectory_ids": ["a", "c"],
        "flux_rmse_by_trajectory": [1.0, 2.0],
        "num_trajectories": 2,
    }
    with pytest.raises(AggregationError, match="same order"):
        _metric_series(payload, value_key="flux_rmse_by_trajectory", id_key="flux_trajectory_ids", label="test")


def test_result_contract_fails_closed_before_artifact_release() -> None:
    with pytest.raises(AggregationError, match="missing required fields"):
        _validate_result_contract({})


def test_aggregation_scalar_helpers_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(AggregationError, match="cannot read JSON"):
        _read_json(tmp_path / "missing.json")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(AggregationError, match="must be an object"):
        _read_json(non_object)
    with pytest.raises(AggregationError, match="cannot hash evidence"):
        _sha256(tmp_path / "missing.bin")
    with pytest.raises(AggregationError, match="finite number"):
        _finite("bad", label="value")
    with pytest.raises(AggregationError, match="finite"):
        _finite(float("nan"), label="value")
    with pytest.raises(AggregationError, match="must be an array"):
        _finite_array("bad", label="values")
    with pytest.raises(AggregationError, match="length"):
        _finite_array([1.0], label="values", length=2)
    with pytest.raises(AggregationError, match="must not be empty"):
        _finite_array([], label="values")
    with pytest.raises(AggregationError, match="non-empty string array"):
        _string_array([], label="ids")
    with pytest.raises(AggregationError, match="unique"):
        _string_array(["a", "a"], label="ids")
    with pytest.raises(AggregationError, match="length"):
        _string_array(["a"], label="ids", length=2)
    with pytest.raises(AggregationError, match="mismatch"):
        _require_equal(1, 2, label="value")
    with pytest.raises(AggregationError, match="mismatch"):
        _require_close(1.0, 2.0, label="value")
    # JAX writes float32 reductions while the independent pass recomputes
    # means from decimal JSON values; a few ulps are accepted, but not a
    # substantive scalar discrepancy.
    _require_close(13.978230476379395, 13.978228855133057, label="float32 reduction")
    with pytest.raises(AggregationError, match="mismatch"):
        _require_close(1.0, 1.00001, label="substantive discrepancy")
    with pytest.raises(AggregationError, match="empty groups"):
        _hierarchical_mean({})
    with pytest.raises(AggregationError, match="at least 100"):
        _bootstrap({0: {52: [1.0]}}, replicates=99, seed=1)
    assert _bootstrap_within_fold({52: [1.0], 53: [2.0]}, replicates=10, seed=1)[0] <= 2.0
    with pytest.raises(argparse.ArgumentTypeError):
        _config_arg("not-a-role=path")


def test_source_tag_contract_rejects_and_accepts_provenance_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest = hashlib.sha256(b"").hexdigest()
    base = {"source": {"tag": "freeze", "tracked_diff_sha256": digest}}
    with pytest.raises(AggregationError, match="source is missing"):
        _source_tag_commit({}, tmp_path)
    with pytest.raises(AggregationError, match="source tag"):
        _source_tag_commit({"source": {"tag": "", "tracked_diff_sha256": digest}}, tmp_path)

    def run_sequence(*results: Any):
        iterator = iter(results)
        return lambda *args, **kwargs: next(iterator)

    result = type("Result", (), {})
    failed_tag = result()
    failed_tag.returncode, failed_tag.stdout = 1, ""
    monkeypatch.setattr(aggregate_module.subprocess, "run", run_sequence(failed_tag))
    with pytest.raises(AggregationError, match="does not resolve"):
        _source_tag_commit(base, tmp_path)

    tag_ok = result()
    tag_ok.returncode, tag_ok.stdout = 0, "a" * 40
    head_bad = result()
    head_bad.returncode, head_bad.stdout = 0, "b" * 40
    monkeypatch.setattr(aggregate_module.subprocess, "run", run_sequence(tag_ok, head_bad))
    with pytest.raises(AggregationError, match="exact frozen"):
        _source_tag_commit(base, tmp_path)

    head_ok = result()
    head_ok.returncode, head_ok.stdout = 0, "a" * 40
    dirty = result()
    dirty.returncode, dirty.stdout = 0, " M file"
    monkeypatch.setattr(aggregate_module.subprocess, "run", run_sequence(tag_ok, head_ok, dirty))
    with pytest.raises(AggregationError, match="clean worktree"):
        _source_tag_commit(base, tmp_path)

    clean = result()
    clean.returncode, clean.stdout = 0, ""
    diff_failed = result()
    diff_failed.returncode, diff_failed.stdout = 1, b""
    monkeypatch.setattr(aggregate_module.subprocess, "run", run_sequence(tag_ok, head_ok, clean, diff_failed))
    with pytest.raises(AggregationError, match="inspect frozen source diff"):
        _source_tag_commit(base, tmp_path)

    diff_wrong = result()
    diff_wrong.returncode, diff_wrong.stdout = 0, b"different"
    monkeypatch.setattr(aggregate_module.subprocess, "run", run_sequence(tag_ok, head_ok, clean, diff_wrong))
    with pytest.raises(AggregationError, match="tracked source diff"):
        _source_tag_commit(base, tmp_path)

    diff_ok = result()
    diff_ok.returncode, diff_ok.stdout = 0, b""
    with_commit = {"source": {**base["source"], "commit": "b" * 40}}
    monkeypatch.setattr(aggregate_module.subprocess, "run", run_sequence(tag_ok, head_ok, clean, diff_ok))
    with pytest.raises(AggregationError, match="commit and tag"):
        _source_tag_commit(with_commit, tmp_path)
    monkeypatch.setattr(aggregate_module.subprocess, "run", run_sequence(tag_ok, head_ok, clean, diff_ok))
    assert _source_tag_commit(base, tmp_path) == ("freeze", "a" * 40)


def test_fold_manifest_contract_rejects_malformed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    universe_path = tmp_path / "universe.json"
    fold_path = tmp_path / "folds.json"
    universe_path.write_text(json.dumps({"trajectory_ids": ["a", "b"]}), encoding="utf-8")
    universe_hash = hashlib.sha256(universe_path.read_bytes()).hexdigest()
    protocol = {"data": {"universe_manifest_sha256": universe_hash, "fallback_rule": {}}}
    with pytest.raises(AggregationError, match="protocol.data"):
        aggregate_module._fold_manifests({}, universe_path, fold_path)
    fold_path.write_text(json.dumps({"folds": []}), encoding="utf-8")
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = hashlib.sha256(
        fold_path.read_bytes()
    ).hexdigest()
    with pytest.raises(AggregationError, match="exactly five"):
        aggregate_module._fold_manifests(protocol, universe_path, fold_path)
    bad_index = {"folds": [None] * 5}
    fold_path.write_text(json.dumps(bad_index), encoding="utf-8")
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = hashlib.sha256(
        fold_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(protocol_runner, "generate_outer_fold_manifest", lambda *_: bad_index)
    with pytest.raises(AggregationError, match="entry is not an object"):
        aggregate_module._fold_manifests(protocol, universe_path, fold_path)


def test_metrics_and_selection_barriers_fail_closed(tmp_path: Path) -> None:
    stage = protocol_runner.Stage("validation", 52, (), tmp_path / "validation", phase="validation", family="gru")
    with pytest.raises(AggregationError, match="missing or invalid"):
        aggregate_module._metrics_and_evidence(stage)
    validation = {
        family: {seed: {"trajectory_balanced_flux_rmse": 1.0 if family == "gru" else 2.0} for seed in (52, 53)}
        for family in ("gru", "transformer")
    }
    hashes = {family: {seed: f"hash-{family}-{seed}" for seed in (52, 53)} for family in ("gru", "transformer")}
    base = {
        "protocol_version": 1,
        "outer_fold": 0,
        "selection_split": "val",
        "primary_metric": "trajectory_balanced_flux_rmse",
        "matched_training_seeds": [52, 53],
        "candidate_mean_validation_trajectory_balanced_flux_rmse": {"gru": 1.0, "transformer": 2.0},
        "candidate_validation_metrics_sha256": hashes,
        "selected_family": "gru",
        "test_evidence_opened": False,
    }
    selection_path = tmp_path / "selection.json"
    for field, value, message in (
        ("test_evidence_opened", True, "test evidence"),
        ("candidate_mean_validation_trajectory_balanced_flux_rmse", None, "candidate means"),
        ("candidate_validation_metrics_sha256", None, "validation hashes"),
        ("selected_family", "invalid", "invalid family"),
    ):
        payload = dict(base)
        payload[field] = value
        selection_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(AggregationError, match=message):
            aggregate_module._selection(
                selection_path,
                fold_id=0,
                seeds=(52, 53),
                validation_payloads=validation,
                validation_hashes=hashes,
            )
    mismatch = dict(base)
    mismatch["candidate_mean_validation_trajectory_balanced_flux_rmse"] = {"gru": 9.0, "transformer": 2.0}
    selection_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(AggregationError, match="mean mismatch"):
        aggregate_module._selection(
            selection_path,
            fold_id=0,
            seeds=(52, 53),
            validation_payloads=validation,
            validation_hashes=hashes,
        )
    mismatch = dict(base)
    mismatch["candidate_validation_metrics_sha256"] = {
        "gru": {"52": "wrong", "53": hashes["gru"][53]},
        "transformer": hashes["transformer"],
    }
    selection_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(AggregationError, match="hash"):
        aggregate_module._selection(
            selection_path,
            fold_id=0,
            seeds=(52, 53),
            validation_payloads=validation,
            validation_hashes=hashes,
        )


def test_aggregate_cli_reports_missing_config_roles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = aggregate_module.main(
        [
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--fold-manifest",
            str(tmp_path / "folds.json"),
            "--universe-manifest",
            str(tmp_path / "universe.json"),
            "--output-root",
            str(tmp_path / "result"),
            "--config",
            "encoder=encoder.yaml",
        ]
    )
    assert result == 2
    assert "missing config roles" in capsys.readouterr().out


def test_fold_manifest_reader_accepts_runner_canonical_bytes_and_all_split(tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    universe_payload = {
        "dataset_revision": "dataset-v1",
        "trajectory_ids": [f"iteration_{index}" for index in range(51)],
    }
    universe.write_text(json.dumps(universe_payload, sort_keys=True), encoding="utf-8")
    protocol = {
        "protocol_id": "multiseed-v1",
        "data": {
            "dataset_revision": "dataset-v1",
            "universe_manifest_sha256": hashlib.sha256(universe.read_bytes()).hexdigest(),
            "development_split_seed": 52,
            "evaluation_route": "nested_group_holdout_cross_validation",
            "fallback_rule": {
                "method": "nested_group_holdout_cross_validation",
                "outer_folds": 5,
            },
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    fold_manifest = tmp_path / "folds.json"
    protocol_runner.write_outer_fold_manifest(protocol_path, universe, fold_manifest)
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = hashlib.sha256(
        fold_manifest.read_bytes()
    ).hexdigest()
    folds = protocol_runner.generate_outer_fold_manifest(protocol, universe_payload)
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = hashlib.sha256(
        fold_manifest.read_bytes()
    ).hexdigest()
    from gk_surrogate.multiseed_aggregate import _fold_manifests

    loaded = _fold_manifests(protocol, universe, fold_manifest)
    assert tuple(sorted(loaded)) == (0, 1, 2, 3, 4)
    assert set(loaded[0]["splits"]) == {"train", "val", "test"}
    assert folds["folds"][0]["outer_fold"] == 0
