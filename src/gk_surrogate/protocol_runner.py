"""Fail-closed preflight and execution for frozen multi-seed protocols."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gk_surrogate.config.load import load_config

CONFIG_ROLES: dict[str, str] = {
    "encoder": "train-encoder",
    "embed": "embed-dataset",
    "gru_train": "train-sequence",
    "transformer_train": "train-sequence",
    "gru_eval": "evaluate-rollout",
    "transformer_eval": "evaluate-rollout",
    "latent_persistence_eval": "evaluate-rollout",
    "observed_persistence_eval": "evaluate-rollout",
}

_NESTED_ROUTE = "nested_group_holdout_cross_validation"
_TRANSFORMER_TYPES = {"causal_transformer", "guppy_latent_transformer"}


@dataclass(frozen=True)
class Stage:
    name: str
    seed: int | None
    command: tuple[str, ...]
    output_dir: Path
    dependencies: tuple[str, ...] = ()
    outer_fold: int | None = None
    phase: str = "run"
    family: str | None = None
    data_seed: int | None = None


@dataclass
class PreflightReport:
    protocol_id: str | None = None
    mode: str = "preflight"
    ready: bool = False
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, object] = field(default_factory=dict)
    stages: list[dict[str, object]] = field(default_factory=list)
    execution: list[dict[str, object]] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_manifest_sha256(trajectory_ids: Sequence[str]) -> str:
    payload = json.dumps(list(trajectory_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def _fold_cli_manifest_payload(protocol_id: object, fold: Mapping[str, Any]) -> dict[str, Any]:
    outer_fold = int(fold["outer_fold"])
    return {
        "schema_version": "1.0.0",
        "protocol_id": protocol_id,
        "fold_id": f"outer-{outer_fold}",
        "splits": {
            "train": fold["train_trajectory_ids"],
            "val": fold["validation_trajectory_ids"],
            "test": fold["test_trajectory_ids"],
        },
    }


def _fold_cli_manifest_path(index_path: Path, outer_fold: int) -> Path:
    return index_path.with_name(f"{index_path.stem}.outer_fold_{outer_fold}{index_path.suffix}")


def generate_outer_fold_manifest(
    protocol: Mapping[str, Any],
    universe_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive deterministic, disjoint outer-test and inner-validation assignments."""

    data = protocol.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("protocol.data is required to generate folds")
    fallback = data.get("fallback_rule")
    if not isinstance(fallback, Mapping) or fallback.get("method") != _NESTED_ROUTE:
        raise ValueError(f"fold generation requires the {_NESTED_ROUTE} fallback")
    outer_folds = fallback.get("outer_folds")
    if outer_folds != 5:
        raise ValueError("fold generation requires exactly five outer folds")
    seed = data.get("development_split_seed")
    if not isinstance(seed, int):
        raise ValueError("protocol.data.development_split_seed must be an integer")
    raw_ids = universe_manifest.get("trajectory_ids")
    if not isinstance(raw_ids, list) or not raw_ids or any(not isinstance(item, str) or not item for item in raw_ids):
        raise ValueError("universe manifest trajectory_ids must be non-empty strings")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("universe manifest contains duplicate trajectory IDs")
    if len(raw_ids) < outer_folds:
        raise ValueError("trajectory universe must contain at least one trajectory per outer fold")
    ranked = sorted(raw_ids, key=lambda item: (hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(), item))
    groups = [ranked[index::outer_folds] for index in range(outer_folds)]
    folds = []
    for outer_fold in range(outer_folds):
        validation_fold = (outer_fold + 1) % outer_folds
        test_ids = sorted(groups[outer_fold])
        validation_ids = sorted(groups[validation_fold])
        train_ids = sorted(
            trajectory_id
            for group_index, group in enumerate(groups)
            if group_index not in {outer_fold, validation_fold}
            for trajectory_id in group
        )
        fold = {
            "outer_fold": outer_fold,
            "inner_validation_fold": validation_fold,
            "train_trajectory_ids": train_ids,
            "validation_trajectory_ids": validation_ids,
            "test_trajectory_ids": test_ids,
        }
        cli_payload = _fold_cli_manifest_payload(protocol.get("protocol_id"), fold)
        fold["cli_split_manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(cli_payload)).hexdigest()
        folds.append(fold)
    return {
        "schema_version": "1.0.0",
        "protocol_id": protocol.get("protocol_id"),
        "dataset_revision": data.get("dataset_revision"),
        "universe_manifest_sha256": data.get("universe_manifest_sha256"),
        "development_split_seed": seed,
        "assignment_algorithm": "sha256_rank_round_robin_cyclic_inner_validation_v1",
        "outer_folds": outer_folds,
        "folds": folds,
    }


def write_outer_fold_manifest(
    protocol_path: Path,
    universe_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Explicitly create a fold artifact which can then be hashed into the protocol."""

    protocol = _read_json(protocol_path)
    universe = _read_json(universe_path)
    data = protocol.get("data")
    if not isinstance(data, Mapping) or not data.get("dataset_revision"):
        raise ValueError("protocol dataset_revision must be frozen before fold generation")
    if data.get("universe_manifest_sha256") != _sha256(universe_path):
        raise ValueError("universe manifest SHA-256 must be frozen before fold generation")
    payload = generate_outer_fold_manifest(protocol, universe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json_bytes(payload))
    for fold in payload["folds"]:
        outer_fold = int(fold["outer_fold"])
        cli_payload = _fold_cli_manifest_payload(payload["protocol_id"], fold)
        _fold_cli_manifest_path(output_path, outer_fold).write_bytes(_canonical_json_bytes(cli_payload))
    return payload


def inspect_environment() -> dict[str, object]:
    """Report accelerator and optional data-stack availability without mutating state."""

    platforms: list[str] = []
    error: str | None = None
    try:
        import jax

        platforms = sorted({device.platform for device in jax.devices()})
    except Exception as exc:  # pragma: no cover - depends on the host JAX installation
        error = f"{type(exc).__name__}: {exc}"
    result: dict[str, object] = {
        "jax_platforms": platforms,
        "gpu_available": "gpu" in platforms,
        "kvikio_available": importlib.util.find_spec("kvikio") is not None,
        "cupy_available": importlib.util.find_spec("cupy") is not None,
    }
    if error:
        result["jax_error"] = error
    return result


def _git_output(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_source(repo_root: Path) -> dict[str, object]:
    head = _git_output(repo_root, "rev-parse", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    diff_sha = hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
    return {
        "commit": head,
        "tracked_diff_sha256": diff_sha,
        "untracked_files": [line[3:] for line in (status or "").splitlines() if line.startswith("?? ")],
    }


def _validate_protocol(payload: Mapping[str, Any], report: PreflightReport, *, resume: bool) -> list[int]:
    _validate_protocol_schema_contract(payload, report)
    report.protocol_id = str(payload.get("protocol_id")) if payload.get("protocol_id") else None
    if payload.get("schema_version") != "1.0.0":
        report.blockers.append("protocol.schema_version must be 1.0.0")
    if payload.get("status") != "frozen":
        report.blockers.append("protocol.status must be 'frozen' before execution")
    if payload.get("created_before_runs") is not True:
        report.blockers.append("protocol must attest created_before_runs=true")
    source = payload.get("source")
    data = payload.get("data")
    models = payload.get("models")
    selection = payload.get("selection")
    if not isinstance(source, Mapping):
        report.blockers.append("protocol.source is missing")
    elif not source.get("commit") and not source.get("tag"):
        report.blockers.append("protocol.source must freeze either commit or tag")
    if not isinstance(data, Mapping):
        report.blockers.append("protocol.data is missing")
    else:
        if not data.get("dataset_revision"):
            report.blockers.append("protocol.data.dataset_revision is not frozen")
        if not data.get("universe_manifest_sha256"):
            report.blockers.append("protocol.data.universe_manifest_sha256 is not frozen")
        if data.get("normalization_fit_split") != "train_only":
            report.blockers.append("normalization_fit_split must be train_only")
    if not isinstance(selection, Mapping) or selection.get("test_used_for_selection") is not False:
        report.blockers.append("protocol must prohibit test use for model selection")
    elif (
        selection.get("checkpoint_rule") != "lowest_validation_trajectory_balanced_latent_rmse"
        or selection.get("primary_metric") != "trajectory_balanced_flux_rmse"
        or selection.get("architecture_rule")
        != "within_each_outer_fold_lowest_mean_inner_validation_primary_metric_over_all_matched_training_seeds"
        or selection.get("architecture_aggregation") != "arithmetic_mean_over_matched_training_seeds_within_outer_fold"
    ):
        report.blockers.append("architecture selection must aggregate validation performance over all matched seeds")
    if not isinstance(models, Mapping):
        report.blockers.append("protocol.models is missing")
        return []
    seeds = models.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        report.blockers.append("protocol.models.training_seeds must be a non-empty integer list")
        return []
    if len(set(seeds)) != len(seeds):
        report.blockers.append("protocol training seeds must be unique")
    expected_families = [
        "observed_diagnostic_persistence",
        "latent_state_persistence_decoded",
        "gru",
        "transformer",
    ]
    if models.get("families") != expected_families:
        report.blockers.append(
            "protocol model families must distinguish observed-diagnostic persistence, "
            "latent-state persistence decoded, GRU, and Transformer"
        )
    accepted = payload.get("accepted_runs")
    if not resume and accepted:
        report.blockers.append("new execution requires an empty accepted_runs list; use --resume for existing runs")
    return seeds


def _validate_protocol_schema_contract(payload: Mapping[str, Any], report: PreflightReport) -> None:
    """Enforce the execution-critical subset of ``protocol.schema.json`` without an optional validator."""

    required = {
        "schema_version",
        "protocol_id",
        "status",
        "created_before_runs",
        "source",
        "data",
        "models",
        "selection",
        "evaluation",
        "reporting",
        "accepted_runs",
    }
    missing = sorted(required.difference(payload))
    if missing:
        report.blockers.append(f"protocol schema missing required fields: {', '.join(missing)}")
    status = payload.get("status")
    source = payload.get("source")
    if status in {"frozen", "completed", "invalidated"} and isinstance(source, Mapping):
        commit = source.get("commit")
        tag = source.get("tag")
        valid_commit = isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None
        valid_tag = isinstance(tag, str) and bool(tag.strip())
        if not (valid_commit or valid_tag):
            report.blockers.append("protocol schema requires a valid source commit or tag when frozen")
    data = payload.get("data")
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("primary_estimand") != (
        "paired_selected_model_minus_observed_diagnostic_persistence_per_trajectory_flux_rmse_"
        "mean_over_outer_folds_seeds_and_trajectories"
    ):
        report.blockers.append(
            "protocol schema requires the selected-model-minus-observed-diagnostic-persistence estimand"
        )
    if isinstance(evaluation, Mapping) and evaluation.get("secondary_estimand") != (
        "paired_selected_model_minus_latent_state_persistence_decoded_per_trajectory_latent_mse_"
        "mean_over_outer_folds_seeds_and_trajectories"
    ):
        report.blockers.append("protocol schema requires the latent-state-persistence secondary estimand")
    accepted = payload.get("accepted_runs")
    if accepted is not None and not isinstance(accepted, list):
        report.blockers.append("protocol schema requires accepted_runs to be an array")
        return
    nested = isinstance(data, Mapping) and data.get("evaluation_route") == _NESTED_ROUTE
    for index, run in enumerate(accepted or []):
        if not isinstance(run, Mapping):
            report.blockers.append(f"accepted_runs[{index}] must be an object")
            continue
        for field_name in ("stage", "wandb_run_id", "artifact_manifest_sha256", "status"):
            if field_name not in run:
                report.blockers.append(f"accepted_runs[{index}] is missing {field_name}")
        outer_fold = run.get("outer_fold")
        if nested and (not isinstance(outer_fold, int) or isinstance(outer_fold, bool) or not 0 <= outer_fold <= 4):
            report.blockers.append(f"accepted_runs[{index}].outer_fold must be an integer from 0 to 4")
        if not nested and "outer_fold" in run:
            report.blockers.append(f"accepted_runs[{index}] must not define outer_fold for final-test execution")


def _validate_held_out_manifest(
    protocol: Mapping[str, Any],
    manifest_path: Path | None,
    report: PreflightReport,
    *,
    resume: bool,
    universe_ids: set[str] | None,
) -> None:
    data = protocol.get("data")
    if not isinstance(data, Mapping):
        return
    rule = data.get("final_test_rule")
    if not isinstance(rule, Mapping):
        report.blockers.append("protocol.data.final_test_rule is missing")
        return
    route = data.get("evaluation_route")
    if route == _NESTED_ROUTE:
        if manifest_path is not None:
            report.blockers.append("nested-CV route must not supply or open a final-test manifest")
        return
    if route != "final_unseen_test":
        report.blockers.append("protocol.data.evaluation_route is invalid or missing")
        return
    expected_hash = rule.get("manifest_sha256")
    if not expected_hash:
        report.blockers.append("final-test manifest hash is not frozen in the protocol")
    if manifest_path is None:
        report.blockers.append("--held-out-manifest is required to prove final-test eligibility")
        return
    if not manifest_path.is_file():
        report.blockers.append(f"held-out manifest does not exist: {manifest_path}")
        return
    actual_hash = _sha256(manifest_path)
    if expected_hash and actual_hash != expected_hash:
        report.blockers.append("held-out manifest SHA-256 does not match the frozen protocol")
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        report.blockers.append(str(exc))
        return
    trajectory_ids = manifest.get("trajectory_ids")
    if not isinstance(trajectory_ids, list) or any(not isinstance(item, str) or not item for item in trajectory_ids):
        report.blockers.append("held-out manifest trajectory_ids must be non-empty strings")
        return
    minimum = rule.get("minimum_unseen_trajectories")
    if not isinstance(minimum, int) or len(set(trajectory_ids)) < minimum:
        report.blockers.append(f"held-out manifest does not prove at least {minimum} unique trajectories")
    if len(set(trajectory_ids)) != len(trajectory_ids):
        report.blockers.append("held-out manifest contains duplicate trajectory IDs")
    if universe_ids is not None and not set(trajectory_ids).issubset(universe_ids):
        report.blockers.append("held-out trajectories are not all present in the frozen universe manifest")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        report.blockers.append("held-out manifest protocol_id does not match")
    if manifest.get("dataset_revision") != data.get("dataset_revision"):
        report.blockers.append("held-out manifest dataset_revision does not match")
    if manifest.get("universe_manifest_sha256") != data.get("universe_manifest_sha256"):
        report.blockers.append("held-out manifest universe hash does not match")
    if manifest.get("not_used_for_training") is not True or manifest.get("not_used_for_model_selection") is not True:
        report.blockers.append("held-out manifest lacks explicit non-use attestations")
    if not isinstance(manifest.get("attested_by"), str) or not manifest["attested_by"].strip():
        report.blockers.append("held-out manifest requires a non-empty attested_by")
    opened_at = rule.get("opened_at_utc")
    if resume and not opened_at:
        report.blockers.append("resume requires protocol.final_test_rule.opened_at_utc")
    if not resume and opened_at:
        report.blockers.append("new execution cannot use an already-opened final-test protocol; use --resume")


def _validate_universe_manifest(
    protocol: Mapping[str, Any],
    manifest_path: Path | None,
    report: PreflightReport,
) -> set[str] | None:
    data = protocol.get("data")
    expected_hash = data.get("universe_manifest_sha256") if isinstance(data, Mapping) else None
    if manifest_path is None:
        report.blockers.append("--universe-manifest is required to verify the dataset universe")
        return None
    if not manifest_path.is_file():
        report.blockers.append(f"universe manifest does not exist: {manifest_path}")
        return None
    if expected_hash and _sha256(manifest_path) != expected_hash:
        report.blockers.append("universe manifest SHA-256 does not match the frozen protocol")
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        report.blockers.append(str(exc))
        return None
    ids = manifest.get("trajectory_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids):
        report.blockers.append("universe manifest trajectory_ids must be non-empty strings")
        return None
    if len(set(ids)) != len(ids):
        report.blockers.append("universe manifest contains duplicate trajectory IDs")
    if isinstance(data, Mapping) and manifest.get("dataset_revision") != data.get("dataset_revision"):
        report.blockers.append("universe manifest dataset_revision does not match")
    if isinstance(data, Mapping):
        route = data.get("evaluation_route")
        fallback = data.get("fallback_rule")
        if len(set(ids)) == 51 and route != _NESTED_ROUTE:
            report.blockers.append(
                "the historical 51-trajectory universe cannot prove an unseen final test; "
                f"freeze evaluation_route={_NESTED_ROUTE}"
            )
        if route == _NESTED_ROUTE:
            if not isinstance(fallback, Mapping) or fallback.get("method") != _NESTED_ROUTE:
                report.blockers.append(f"nested holdout-CV route requires the frozen {_NESTED_ROUTE} fallback")
            elif fallback.get("outer_folds") != 5:
                report.blockers.append("nested-CV route requires exactly five frozen outer folds")
    return set(ids)


def _validate_outer_fold_manifest(
    protocol: Mapping[str, Any],
    manifest_path: Path | None,
    universe_path: Path | None,
    report: PreflightReport,
) -> tuple[int, ...]:
    data = protocol.get("data")
    if not isinstance(data, Mapping):
        return ()
    route = data.get("evaluation_route")
    if route != _NESTED_ROUTE:
        if manifest_path is not None:
            report.blockers.append("final-test route must not supply an outer-fold manifest")
        return ()
    fallback = data.get("fallback_rule")
    expected_hash = fallback.get("outer_fold_manifest_sha256") if isinstance(fallback, Mapping) else None
    if not expected_hash:
        report.blockers.append("nested-CV outer-fold manifest hash is not frozen in the protocol")
    if manifest_path is None:
        report.blockers.append("--fold-manifest is required for nested group CV")
        return ()
    if not manifest_path.is_file():
        report.blockers.append(f"outer-fold manifest does not exist: {manifest_path}")
        return ()
    if expected_hash and _sha256(manifest_path) != expected_hash:
        report.blockers.append("outer-fold manifest SHA-256 does not match the frozen protocol")
    if universe_path is None or not universe_path.is_file():
        return ()
    try:
        actual = _read_json(manifest_path)
        universe = _read_json(universe_path)
        expected = generate_outer_fold_manifest(protocol, universe)
    except ValueError as exc:
        report.blockers.append(str(exc))
        return ()
    if actual != expected:
        report.blockers.append("outer-fold assignments do not match deterministic regeneration")
        return ()
    folds = actual.get("folds")
    assert isinstance(folds, list)
    validated_folds = []
    for fold in folds:
        outer_fold = int(fold["outer_fold"])
        cli_path = _fold_cli_manifest_path(manifest_path, outer_fold)
        if not cli_path.is_file():
            report.blockers.append(f"outer fold {outer_fold} CLI split manifest does not exist: {cli_path}")
            continue
        expected_cli = _fold_cli_manifest_payload(actual.get("protocol_id"), fold)
        expected_cli_bytes = _canonical_json_bytes(expected_cli)
        if _sha256(cli_path) != fold.get("cli_split_manifest_sha256"):
            report.blockers.append(f"outer fold {outer_fold} CLI split manifest SHA-256 does not match")
            continue
        try:
            cli_payload = _read_json(cli_path)
        except ValueError as exc:
            report.blockers.append(str(exc))
            continue
        if cli_payload != expected_cli or cli_path.read_bytes() != expected_cli_bytes:
            report.blockers.append(f"outer fold {outer_fold} CLI split manifest is not canonical")
            continue
        validated_folds.append(outer_fold)
    return tuple(validated_folds) if len(validated_folds) == len(folds) else ()


def exact_fold_cli_supported() -> bool:
    """Detect the explicit CLI contract needed to consume exact fold assignments."""

    try:
        from gk_surrogate.cli import _build_parser

        parser = _build_parser()
        subparsers_action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        required = {"--split-manifest"}
        commands = (
            "train-encoder",
            "embed-dataset",
            "train-sequence",
            "evaluate-rollout",
        )
        return all(
            required.issubset(
                option for action in subparsers_action.choices[command]._actions for option in action.option_strings
            )
            for command in commands
        )
    except (KeyError, StopIteration):
        return False


def _validate_source(protocol: Mapping[str, Any], repo_root: Path, report: PreflightReport) -> None:
    actual = inspect_source(repo_root)
    report.environment["source"] = actual
    source = protocol.get("source")
    if not isinstance(source, Mapping):
        return
    expected_commit = source.get("commit")
    expected_tag = source.get("tag")
    if expected_commit and expected_commit != actual["commit"]:
        report.blockers.append("checked-out Git commit does not match protocol.source.commit")
    if expected_tag:
        if not isinstance(expected_tag, str) or not expected_tag.strip():
            report.blockers.append("protocol.source.tag must be a non-empty Git tag name")
        else:
            resolved_tag = _git_output(repo_root, "rev-parse", "--verify", f"refs/tags/{expected_tag}^{{commit}}")
            if resolved_tag is None:
                report.blockers.append(f"protocol.source.tag does not resolve locally: {expected_tag}")
            elif resolved_tag != actual["commit"]:
                report.blockers.append("protocol.source.tag does not resolve exactly to checked-out HEAD")
    if expected_commit and expected_tag:
        resolved_tag = _git_output(repo_root, "rev-parse", "--verify", f"refs/tags/{expected_tag}^{{commit}}")
        if resolved_tag is not None and resolved_tag != expected_commit:
            report.blockers.append("protocol.source.commit and protocol.source.tag resolve to different commits")
    expected_diff = source.get("tracked_diff_sha256")
    if expected_diff is None:
        report.blockers.append("protocol.source.tracked_diff_sha256 is not frozen")
    elif expected_diff != actual["tracked_diff_sha256"]:
        report.blockers.append("tracked Git diff does not match protocol.source.tracked_diff_sha256")
    if actual["untracked_files"]:
        report.blockers.append("untracked files are not covered by protocol source provenance")


def _config_arg(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or role not in CONFIG_ROLES or not raw_path:
        choices = ", ".join(CONFIG_ROLES)
        raise argparse.ArgumentTypeError(f"config must be ROLE=PATH where ROLE is one of: {choices}")
    return role, Path(raw_path)


def _validate_configs(configs: Mapping[str, Path], report: PreflightReport) -> None:
    for role in CONFIG_ROLES:
        if role not in configs:
            report.blockers.append(f"missing experiment config role: {role}")
    loaded: dict[str, Any] = {}
    for role, path in configs.items():
        if not path.is_file():
            report.blockers.append(f"config for {role} does not exist: {path}")
            continue
        try:
            loaded[role] = load_config(path, command=CONFIG_ROLES[role])
        except Exception as exc:
            report.blockers.append(f"config for {role} failed validation: {exc}")
    expected_splits = {
        "encoder": "train",
        "embed": "all",
        "gru_train": "train",
        "transformer_train": "train",
        "gru_eval": "test",
        "transformer_eval": "test",
        "latent_persistence_eval": "test",
        "observed_persistence_eval": "test",
    }
    for role, split in expected_splits.items():
        if role in loaded and loaded[role].data.split != split:
            report.blockers.append(f"config {role} must use data.split={split}")
    expected_baselines = {
        "latent_persistence_eval": "latent_state_persistence_decoded",
        "observed_persistence_eval": "observed_diagnostic_persistence",
    }
    for role, baseline in expected_baselines.items():
        if role in loaded and getattr(loaded[role].evaluation, "baseline_mode", None) != baseline:
            report.blockers.append(f"config {role} must use evaluation.baseline_mode={baseline}")
    for role in ("gru_eval", "transformer_eval"):
        if role in loaded and getattr(loaded[role].evaluation, "baseline_mode", "none") != "none":
            report.blockers.append(f"config {role} must use evaluation.baseline_mode=none")
    if any(
        config.data.backend == "cyclone_kvikio" and config.data.cyclone and config.data.cyclone.use_kvikio
        for config in loaded.values()
    ) and not report.environment.get("kvikio_available"):
        report.blockers.append("a config requires KvikIO, but the kvikio package is unavailable")
    if set(CONFIG_ROLES).issubset(loaded):
        _validate_config_semantics(loaded, report)


def _dump_config_part(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in set(keys)}


def _require_equal_config_parts(
    loaded: Mapping[str, Any],
    roles: Sequence[str],
    getter: Callable[[Any], Any],
    label: str,
    report: PreflightReport,
) -> None:
    baseline_role = roles[0]
    baseline = getter(loaded[baseline_role])
    mismatches = [role for role in roles[1:] if getter(loaded[role]) != baseline]
    if mismatches:
        report.blockers.append(
            f"config compatibility mismatch for {label}: {baseline_role} differs from {', '.join(mismatches)}"
        )


def _validate_config_semantics(loaded: Mapping[str, Any], report: PreflightReport) -> None:
    roles = tuple(CONFIG_ROLES)

    def data_contract(config: Any) -> Any:
        raw = _dump_config_part(config.data)
        # Training-time augmentation, ordering, and batching are stage-specific; the physical
        # dataset view and target contract must remain identical across all roles.
        return _without(raw, "split", "split_manifest", "batch_size", "shuffle", "seed", "augmentations")

    def representation_contract(config: Any) -> Any:
        model = _dump_config_part(config.model)
        return {"encoder": model.get("encoder"), "diagnostics": model.get("diagnostics")}

    def diagnostic_loss_contract(config: Any) -> Any:
        loss = _dump_config_part(config.loss)
        return {key: loss.get(key) for key in ("flux_weight", "spectra_weight", "use_log_spectra", "spectra_epsilon")}

    _require_equal_config_parts(loaded, roles, data_contract, "data preprocessing", report)
    _require_equal_config_parts(loaded, roles, representation_contract, "encoder/diagnostic model", report)
    _require_equal_config_parts(loaded, roles, diagnostic_loss_contract, "diagnostic loss", report)

    for family, expected_types in (("gru", {"gru"}), ("transformer", _TRANSFORMER_TYPES)):
        train_role = f"{family}_train"
        eval_role = f"{family}_eval"
        train_sequence = _dump_config_part(loaded[train_role].model.sequence)
        eval_sequence = _dump_config_part(loaded[eval_role].model.sequence)
        train_type = train_sequence.get("type") if isinstance(train_sequence, Mapping) else None
        eval_type = eval_sequence.get("type") if isinstance(eval_sequence, Mapping) else None
        if train_type not in expected_types or eval_type not in expected_types:
            report.blockers.append(
                f"{family} roles require sequence type in {sorted(expected_types)}, "
                f"got train={train_type!r}, eval={eval_type!r}"
            )
        if train_sequence != eval_sequence:
            report.blockers.append(f"{family} train/eval sequence model configs must match exactly")

    learned_roles = ("gru_train", "transformer_train")
    _require_equal_config_parts(
        loaded,
        learned_roles,
        lambda config: {
            "latent_dim": config.model.sequence.latent_dim,
            "context_length": config.model.sequence.context_length,
        },
        "learned-model latent/context dimensions",
        report,
    )
    eval_roles = ("gru_eval", "transformer_eval", "latent_persistence_eval", "observed_persistence_eval")
    _require_equal_config_parts(
        loaded,
        eval_roles,
        lambda config: _without(_dump_config_part(config.evaluation), "baseline_mode"),
        "evaluation horizon/metrics",
        report,
    )
    latent_roles = ("gru_train", "transformer_train", *eval_roles)
    _require_equal_config_parts(
        loaded,
        latent_roles,
        lambda config: _without(
            _dump_config_part(config.latent_cache),
            "path",
            "encoder_checkpoint_path",
            "sequence_checkpoint_path",
        ),
        "latent normalization",
        report,
    )


def _verify_dataset_bytes(
    *,
    dataset_root: Path | None,
    universe_manifest: Path | None,
    configs: Mapping[str, Path],
    execute: bool,
    resume: bool,
    report: PreflightReport,
) -> None:
    if not (execute or resume) or dataset_root is None or universe_manifest is None:
        return
    if not dataset_root.is_dir() or not universe_manifest.is_file() or "encoder" not in configs:
        return
    try:
        from gk_surrogate.data.universe_manifest import verify_cyclone_universe_manifest

        config = load_config(
            configs["encoder"],
            overrides=[f"data.root={dataset_root}"],
            command="train-encoder",
        )
        if config.data.cyclone is None:
            raise ValueError("encoder config does not define the Cyclone dataset contract")
        expected = _read_json(universe_manifest)
        verified = verify_cyclone_universe_manifest(expected, dataset_root, config.data.cyclone)
        report.environment["verified_dataset_revision"] = verified["dataset_revision"]
    except Exception as exc:
        report.blockers.append(f"frozen dataset byte verification failed: {exc}")


def build_stages(
    configs: Mapping[str, Path],
    output_root: Path,
    seeds: Sequence[int],
    *,
    fold_manifest: Path | None = None,
    outer_folds: Sequence[int] = (),
    data_seed: int | None = None,
) -> list[Stage]:
    """Create a two-phase plan with validation-complete selection before any test command."""

    stages: list[Stage] = []
    fold_values: tuple[int | None, ...] = tuple(outer_folds) if outer_folds else (None,)
    for outer_fold in fold_values:
        fold_root = output_root if outer_fold is None else output_root / f"outer_fold_{outer_fold}"

        def command(
            role: str,
            output: Path,
            *overrides: str,
            run_seed: int,
            run_outer_fold: int | None = outer_fold,
        ) -> tuple[str, ...]:
            split_seed = data_seed if data_seed is not None else run_seed
            argv = [
                sys.executable,
                "-m",
                "gk_surrogate.cli",
                CONFIG_ROLES[role],
                "--config",
                str(configs[role]),
                "--seed",
                str(split_seed),
                "--training-seed",
                str(run_seed),
                "--output-dir",
                str(output),
            ]
            if run_outer_fold is not None:
                assert fold_manifest is not None
                argv.extend(
                    (
                        "--split-manifest",
                        str(_fold_cli_manifest_path(fold_manifest, run_outer_fold)),
                    )
                )
            for override in overrides:
                argv.extend(("--override", override))
            return tuple(argv)

        # Phase 1: produce representations, both learned candidates, and validation-only metrics.
        for seed in seeds:
            seed_root = fold_root / f"seed_{seed}"
            encoder_out = seed_root / "encoder"
            embed_out = seed_root / "embed"
            cache = embed_out / "latent_cache.h5"
            encoder_ref = "latent_cache.encoder_checkpoint_path={encoder.checkpoint}"
            stages.append(
                Stage(
                    "encoder",
                    seed,
                    command("encoder", encoder_out, run_seed=seed),
                    encoder_out,
                    outer_fold=outer_fold,
                    data_seed=data_seed,
                )
            )
            stages.append(
                Stage(
                    "embed",
                    seed,
                    command("embed", embed_out, encoder_ref, run_seed=seed),
                    embed_out,
                    ("encoder",),
                    outer_fold,
                    data_seed=data_seed,
                )
            )
            for family in ("gru", "transformer"):
                train_role = f"{family}_train"
                train_out = seed_root / train_role
                stages.append(
                    Stage(
                        train_role,
                        seed,
                        command(
                            train_role,
                            train_out,
                            f"latent_cache.path={cache}",
                            encoder_ref,
                            run_seed=seed,
                        ),
                        train_out,
                        ("encoder", "embed"),
                        outer_fold,
                        data_seed=data_seed,
                    )
                )
                validation_name = f"{family}_validation"
                validation_out = seed_root / validation_name
                stages.append(
                    Stage(
                        validation_name,
                        seed,
                        command(
                            f"{family}_eval",
                            validation_out,
                            f"latent_cache.path={cache}",
                            encoder_ref,
                            f"latent_cache.sequence_checkpoint_path={{{train_role}.checkpoint}}",
                            "data.split=val",
                            run_seed=seed,
                        ),
                        validation_out,
                        ("encoder", "embed", train_role),
                        outer_fold,
                        "validation",
                        family,
                        data_seed=data_seed,
                    )
                )

        selection_out = fold_root / "selection"
        stages.append(
            Stage(
                "architecture_selection",
                None,
                (),
                selection_out,
                tuple(f"{family}_validation" for family in ("gru", "transformer")),
                outer_fold,
                "selection",
                data_seed=data_seed,
            )
        )

        # Phase 2: the executor skips the unselected learned family. Persistence is always paired.
        for seed in seeds:
            seed_root = fold_root / f"seed_{seed}"
            cache = seed_root / "embed" / "latent_cache.h5"
            encoder_ref = "latent_cache.encoder_checkpoint_path={encoder.checkpoint}"
            for family in ("gru", "transformer"):
                train_role = f"{family}_train"
                eval_role = f"{family}_eval"
                eval_out = seed_root / eval_role
                stages.append(
                    Stage(
                        eval_role,
                        seed,
                        command(
                            eval_role,
                            eval_out,
                            f"latent_cache.path={cache}",
                            encoder_ref,
                            f"latent_cache.sequence_checkpoint_path={{{train_role}.checkpoint}}",
                            run_seed=seed,
                        ),
                        eval_out,
                        ("architecture_selection", "encoder", "embed", train_role),
                        outer_fold,
                        "test",
                        family,
                        data_seed=data_seed,
                    )
                )
            for role, family in (
                ("observed_persistence_eval", "observed_diagnostic_persistence"),
                ("latent_persistence_eval", "latent_state_persistence_decoded"),
            ):
                baseline_out = seed_root / role
                stages.append(
                    Stage(
                        role,
                        seed,
                        command(
                            role,
                            baseline_out,
                            f"latent_cache.path={cache}",
                            encoder_ref,
                            f"evaluation.baseline_mode={family}",
                            "latent_cache.sequence_checkpoint_path=null",
                            run_seed=seed,
                        ),
                        baseline_out,
                        ("encoder", "embed"),
                        outer_fold,
                        "test",
                        family,
                        data_seed=data_seed,
                    )
                )
    return stages


def _stage_payload(stage: Stage) -> dict[str, object]:
    return {
        "name": stage.name,
        "seed": stage.seed,
        "command": list(stage.command),
        "output_dir": str(stage.output_dir),
        "dependencies": list(stage.dependencies),
        "outer_fold": stage.outer_fold,
        "phase": stage.phase,
        "family": stage.family,
        "data_seed": stage.data_seed,
        "status": "planned",
    }


def _checkpoint_file(path: Path) -> Path:
    return path / "checkpoint.pkl" if path.is_dir() else path


def _stage_split(stage: Stage) -> str:
    if stage.name == "embed":
        return "all"
    if stage.phase == "validation":
        return "val"
    if stage.phase == "test":
        return "test"
    return "train"


def _stage_fold_manifest(stage: Stage) -> tuple[dict[str, Any], Path] | None:
    if "--split-manifest" not in stage.command:
        return None
    index = stage.command.index("--split-manifest")
    path = Path(stage.command[index + 1])
    return _read_json(path), path


def _load_stage_evidence(stage: Stage) -> dict[str, str] | None:
    if stage.phase == "selection":
        return None
    metrics_path = stage.output_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        payload = _read_json(metrics_path)
    except ValueError:
        return None
    expected_split = _stage_split(stage)
    required = {
        "protocol_version",
        "data_split",
        "data_split_seed",
        "training_seed",
        "selected_trajectory_ids",
        "trajectory_manifest_sha256",
        "universe_trajectory_ids",
        "universe_manifest_sha256",
    }
    if required.difference(payload):
        return None
    expected_data_seed = stage.data_seed if stage.data_seed is not None else stage.seed
    if (
        payload.get("protocol_version") != 1
        or payload.get("data_split") != expected_split
        or payload.get("data_split_seed") != expected_data_seed
        or payload.get("training_seed") != stage.seed
    ):
        return None
    selected_ids = payload.get("selected_trajectory_ids")
    universe_ids = payload.get("universe_trajectory_ids")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or any(not isinstance(item, str) or not item for item in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
        or not isinstance(universe_ids, list)
        or not universe_ids
        or any(not isinstance(item, str) or not item for item in universe_ids)
        or len(set(universe_ids)) != len(universe_ids)
        or not set(selected_ids).issubset(universe_ids)
        or payload.get("trajectory_manifest_sha256") != _trajectory_manifest_sha256(selected_ids)
        or payload.get("universe_manifest_sha256") != _trajectory_manifest_sha256(universe_ids)
    ):
        return None
    fold_artifact = _stage_fold_manifest(stage)
    if fold_artifact is not None:
        manifest, manifest_path = fold_artifact
        splits = manifest.get("splits")
        if not isinstance(splits, Mapping) or any(
            not isinstance(splits.get(role), list | tuple) or not splits.get(role)
            for role in ("train", "val", "test")
        ):
            return None
        expected_selected = (
            list(splits.get("train", []))
            if expected_split == "train"
            else list(splits.get("val", []))
            if expected_split == "val"
            else list(splits.get("test", []))
            if expected_split == "test"
            else [*splits.get("train", []), *splits.get("val", []), *splits.get("test", [])]
        )
        expected_universe = {str(item) for role in ("train", "val", "test") for item in splits.get(role, [])}
        if set(universe_ids) != expected_universe or set(selected_ids) != set(expected_selected):
            return None
        if payload.get("split_fold_id") != manifest.get("fold_id") or payload.get("split_manifest_sha256") != _sha256(
            manifest_path
        ):
            return None

    evidence = {
        "metrics": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
    }
    wandb_status_path = stage.output_dir / "wandb_status.json"
    if wandb_status_path.is_file():
        evidence["wandb_status_sha256"] = _sha256(wandb_status_path)
    expected_artifact_role = {
        "encoder": "encoder_checkpoint",
        "embed": "latent_cache",
        "gru_train": "sequence_checkpoint",
        "transformer_train": "sequence_checkpoint",
    }.get(stage.name)
    if expected_artifact_role is not None and payload.get("artifact_role") != expected_artifact_role:
        return None
    for key in ("checkpoint", "latent_cache"):
        raw = payload.get(key)
        if isinstance(raw, str) and Path(raw).exists():
            evidence[key] = raw
    if stage.name in {"encoder", "gru_train", "transformer_train"} and "checkpoint" not in evidence:
        return None
    if "checkpoint" in evidence:
        checkpoint_file = _checkpoint_file(Path(evidence["checkpoint"]))
        if not checkpoint_file.is_file() or payload.get("checkpoint_sha256") != _sha256(checkpoint_file):
            return None
        evidence["checkpoint_sha256"] = str(payload["checkpoint_sha256"])
    if stage.name == "embed" and "latent_cache" not in evidence:
        return None
    if "latent_cache" in evidence:
        actual_cache_sha256 = _sha256(Path(evidence["latent_cache"]))
        declared_cache_sha256 = payload.get("latent_cache_sha256")
        if (
            not isinstance(declared_cache_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", declared_cache_sha256)
            or declared_cache_sha256 != actual_cache_sha256
        ):
            return None
        evidence["latent_cache_sha256"] = actual_cache_sha256
    for artifact_key in ("encoder_checkpoint", "sequence_checkpoint"):
        raw = payload.get(artifact_key)
        if not isinstance(raw, str) or not raw:
            continue
        artifact_path = Path(raw)
        if not artifact_path.exists():
            continue
        declared_hash = payload.get(f"{artifact_key}_sha256")
        if artifact_key == "sequence_checkpoint" and payload.get("rollout_method") != "learned_sequence_model":
            continue
        checkpoint_file = _checkpoint_file(artifact_path)
        if not checkpoint_file.is_file():
            return None
        actual_hash = _sha256(checkpoint_file)
        if (
            not isinstance(declared_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", declared_hash)
            or declared_hash != actual_hash
        ):
            return None
    if stage.phase in {"validation", "test"}:
        if "latent_cache" in payload:
            cache_path = payload.get("latent_cache")
            if not isinstance(cache_path, str) or not cache_path or not Path(cache_path).is_file():
                return None
            if "latent_cache_sha256" not in evidence:
                return None
            encoder_path = payload.get("encoder_checkpoint")
            encoder_hash = payload.get("encoder_checkpoint_sha256")
            if (
                not isinstance(encoder_path, str)
                or not encoder_path
                or not Path(encoder_path).exists()
                or not isinstance(encoder_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", encoder_hash)
                or not _checkpoint_file(Path(encoder_path)).is_file()
                or _sha256(_checkpoint_file(Path(encoder_path))) != encoder_hash
            ):
                return None
            if payload.get("rollout_method") == "learned_sequence_model":
                sequence_path = payload.get("sequence_checkpoint")
                sequence_hash = payload.get("sequence_checkpoint_sha256")
                if (
                    not isinstance(sequence_path, str)
                    or not Path(sequence_path).exists()
                    or not isinstance(sequence_hash, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", sequence_hash)
                    or not _checkpoint_file(Path(sequence_path)).is_file()
                    or _sha256(_checkpoint_file(Path(sequence_path))) != sequence_hash
                ):
                    return None
        expected_baseline_mode = {
            "latent_persistence_eval": "latent_state_persistence_decoded",
            "observed_persistence_eval": "observed_diagnostic_persistence",
        }.get(stage.name, "none")
        flux_rmse = payload.get("flux_rmse")
        trajectory_balanced_flux_rmse = payload.get("trajectory_balanced_flux_rmse")
        require_trajectory_balanced = stage.phase == "validation" and stage.family in {"gru", "transformer"}
        if (
            not isinstance(flux_rmse, int | float)
            or isinstance(flux_rmse, bool)
            or not math.isfinite(float(flux_rmse))
            or not isinstance(payload.get("stable"), bool)
            or payload.get("stable") is not True
            or not isinstance(payload.get("num_trajectories"), int)
            or payload["num_trajectories"] != len(selected_ids)
            or payload.get("baseline_mode") != expected_baseline_mode
            or (
                require_trajectory_balanced
                and (
                    not isinstance(trajectory_balanced_flux_rmse, int | float)
                    or isinstance(trajectory_balanced_flux_rmse, bool)
                    or not math.isfinite(float(trajectory_balanced_flux_rmse))
                    or not math.isclose(
                        float(trajectory_balanced_flux_rmse),
                        float(flux_rmse),
                        rel_tol=1e-7,
                        abs_tol=1e-7,
                    )
                )
            )
        ):
            return None
        evidence["flux_rmse"] = repr(float(flux_rmse))
        if isinstance(trajectory_balanced_flux_rmse, int | float) and not isinstance(
            trajectory_balanced_flux_rmse, bool
        ):
            evidence["trajectory_balanced_flux_rmse"] = repr(float(trajectory_balanced_flux_rmse))
    return evidence


def _select_architecture(
    stage: Stage,
    by_fold_seed: Mapping[tuple[int | None, int], Mapping[str, Mapping[str, str]]],
    seeds: Sequence[int],
) -> tuple[str, dict[str, object]]:
    means: dict[str, float] = {}
    metric_hashes: dict[str, dict[str, str]] = {}
    for family in ("gru", "transformer"):
        values = []
        hashes: dict[str, str] = {}
        evidence_name = f"{family}_validation"
        for seed in seeds:
            evidence = by_fold_seed.get((stage.outer_fold, seed), {}).get(evidence_name)
            if (
                evidence is None
                or "trajectory_balanced_flux_rmse" not in evidence
                or "metrics_sha256" not in evidence
            ):
                raise RuntimeError(
                    f"selection barrier lacks verified {family} validation evidence for "
                    f"outer fold {stage.outer_fold}, seed {seed}"
                )
            values.append(float(evidence["trajectory_balanced_flux_rmse"]))
            hashes[str(seed)] = evidence["metrics_sha256"]
        means[family] = float(sum(values) / len(values))
        metric_hashes[family] = hashes
    selected = min(means, key=lambda family: (means[family], family))
    payload: dict[str, object] = {
        "protocol_version": 1,
        "outer_fold": stage.outer_fold,
        "selection_split": "val",
        "primary_metric": "trajectory_balanced_flux_rmse",
        "matched_training_seeds": list(seeds),
        "candidate_mean_validation_trajectory_balanced_flux_rmse": means,
        "candidate_validation_metrics_sha256": metric_hashes,
        "tie_break_rule": "lexicographic_family_name",
        "selected_family": selected,
        "test_evidence_opened": False,
    }
    stage.output_dir.mkdir(parents=True, exist_ok=True)
    path = stage.output_dir / "metrics.json"
    path.write_bytes(_canonical_json_bytes(payload))
    return selected, {**payload, "metrics": str(path), "metrics_sha256": _sha256(path)}


def _resolve_command(command: Sequence[str], evidence: Mapping[str, Mapping[str, str]]) -> list[str]:
    resolved = []
    for argument in command:
        value = argument
        for stage, fields in evidence.items():
            for evidence_field, replacement in fields.items():
                value = value.replace(f"{{{stage}.{evidence_field}}}", replacement)
        if "{" in value or "}" in value:
            raise ValueError(f"unresolved stage dependency in command argument: {value}")
        resolved.append(value)
    return resolved


def execute_stages(
    stages: Sequence[Stage],
    *,
    repo_root: Path,
    resume: bool,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    by_fold_seed: dict[tuple[int | None, int], dict[str, dict[str, str]]] = {}
    selected_by_fold: dict[int | None, str] = {}
    seeds_by_fold = {
        fold: sorted({int(stage.seed) for stage in stages if stage.outer_fold == fold and stage.seed is not None})
        for fold in {stage.outer_fold for stage in stages}
    }
    for stage in stages:
        if stage.phase == "selection":
            selected, selection = _select_architecture(
                stage,
                by_fold_seed,
                seeds_by_fold[stage.outer_fold],
            )
            selected_by_fold[stage.outer_fold] = selected
            results.append(
                {
                    "name": stage.name,
                    "seed": None,
                    "outer_fold": stage.outer_fold,
                    "status": "completed_selection_barrier",
                    **selection,
                }
            )
            continue
        if stage.phase == "test":
            selected = selected_by_fold.get(stage.outer_fold)
            if selected is None:
                raise RuntimeError(f"outer fold {stage.outer_fold} reached test before selection barrier")
            if stage.family in {"gru", "transformer"} and stage.family != selected:
                results.append(
                    {
                        "name": stage.name,
                        "seed": stage.seed,
                        "outer_fold": stage.outer_fold,
                        "status": "skipped_unselected",
                        "selected_family": selected,
                    }
                )
                continue
        if stage.seed is None:
            raise RuntimeError(f"non-selection stage {stage.name} is missing a training seed")
        evidence = by_fold_seed.setdefault((stage.outer_fold, stage.seed), {})
        existing = _load_stage_evidence(stage) if resume else None
        if existing is not None:
            evidence[stage.name] = existing
            results.append(
                {
                    "name": stage.name,
                    "seed": stage.seed,
                    "outer_fold": stage.outer_fold,
                    "status": "resumed_existing",
                    "phase": stage.phase,
                    **existing,
                }
            )
            continue
        command = _resolve_command(stage.command, evidence)
        if command_runner is subprocess.run:
            completed = command_runner(
                command,
                cwd=repo_root,
                check=False,
                text=True,
                capture_output=True,
            )
            stage.output_dir.mkdir(parents=True, exist_ok=True)
            (stage.output_dir / "command.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
            (stage.output_dir / "command.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        else:
            # Keep the injectable runner contract small for tests and callers
            # that provide their own command executor.
            completed = command_runner(command, cwd=repo_root, check=False, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"stage {stage.name} seed {stage.seed} failed with exit code {completed.returncode}")
        produced = _load_stage_evidence(stage)
        if produced is None:
            raise RuntimeError(f"stage {stage.name} seed {stage.seed} returned success without valid evidence")
        evidence[stage.name] = produced
        results.append(
            {
                "name": stage.name,
                "seed": stage.seed,
                "outer_fold": stage.outer_fold,
                "status": "completed",
                "phase": stage.phase,
                **produced,
            }
        )
    return results


def preflight(
    protocol_path: Path,
    *,
    configs: Mapping[str, Path],
    output_root: Path,
    held_out_manifest: Path | None,
    universe_manifest: Path | None,
    fold_manifest: Path | None,
    repo_root: Path,
    execute: bool,
    resume: bool,
    environment: Mapping[str, object] | None = None,
    dataset_root: Path | None = None,
) -> tuple[PreflightReport, list[Stage]]:
    protocol = _read_json(protocol_path)
    report = PreflightReport(mode="resume" if resume else "execute" if execute else "preflight")
    report.environment.update(environment or inspect_environment())
    report.environment["dataset_root"] = str(dataset_root.resolve()) if dataset_root is not None else None
    if (execute or resume) and dataset_root is None:
        report.blockers.append("--dataset-root is required for execute/resume so frozen dataset bytes can be verified")
    elif dataset_root is not None and not dataset_root.is_dir():
        report.blockers.append(f"dataset root does not exist or is not a directory: {dataset_root}")
    seeds = _validate_protocol(protocol, report, resume=resume)
    _validate_source(protocol, repo_root, report)
    universe_ids = _validate_universe_manifest(protocol, universe_manifest, report)
    outer_folds = _validate_outer_fold_manifest(protocol, fold_manifest, universe_manifest, report)
    _validate_held_out_manifest(
        protocol,
        held_out_manifest,
        report,
        resume=resume,
        universe_ids=universe_ids,
    )
    _validate_configs(configs, report)
    _verify_dataset_bytes(
        dataset_root=dataset_root,
        universe_manifest=universe_manifest,
        configs=configs,
        execute=execute,
        resume=resume,
        report=report,
    )
    data = protocol.get("data")
    if isinstance(data, Mapping) and data.get("evaluation_route") == _NESTED_ROUTE:
        fold_cli = exact_fold_cli_supported()
        report.environment["exact_fold_cli_supported"] = fold_cli
        if not fold_cli:
            message = (
                "nested group CV manifest is valid, but the training CLI lacks "
                "--split-manifest; do not substitute its ordinary split"
            )
            if execute or resume:
                report.blockers.append(message)
            else:
                report.warnings.append(message)
    if not report.environment.get("gpu_available"):
        message = "JAX reports no GPU; the accepted multi-seed protocol cannot execute on CPU only"
        if execute or resume:
            report.blockers.append(message)
        else:
            report.warnings.append(message)
    if not report.environment.get("kvikio_available"):
        report.warnings.append("KvikIO is unavailable")
    if not report.environment.get("cupy_available"):
        report.warnings.append("CuPy is unavailable")
    stages = (
        build_stages(
            configs,
            output_root,
            seeds,
            fold_manifest=fold_manifest,
            outer_folds=outer_folds,
            data_seed=(data.get("development_split_seed") if isinstance(data, Mapping) else None),
        )
        if not set(CONFIG_ROLES).difference(configs)
        else []
    )
    report.stages = [_stage_payload(stage) for stage in stages]
    report.ready = not report.blockers
    return report, stages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gks-protocol")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiment_protocols/multiseed_v1.json"),
    )
    parser.add_argument("--config", action="append", default=[], type=_config_arg, metavar="ROLE=PATH")
    parser.add_argument("--held-out-manifest", type=Path)
    parser.add_argument("--universe-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument(
        "--generate-fold-manifest",
        type=Path,
        metavar="OUTPUT",
        help="Generate deterministic outer/inner fold assignments and exit without running stages.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multiseed-v1"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Execute a ready frozen protocol.")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Execute while reusing stages with verified local evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configs = dict(args.config)
    try:
        if args.generate_fold_manifest is not None:
            if args.universe_manifest is None:
                raise ValueError("--generate-fold-manifest requires --universe-manifest")
            payload = write_outer_fold_manifest(
                args.protocol,
                args.universe_manifest,
                args.generate_fold_manifest,
            )
            print(
                json.dumps(
                    {
                        "generated": str(args.generate_fold_manifest),
                        "sha256": _sha256(args.generate_fold_manifest),
                        "outer_folds": payload["outer_folds"],
                        "execution": [],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        report, stages = preflight(
            args.protocol,
            configs=configs,
            output_root=args.output_root,
            held_out_manifest=args.held_out_manifest,
            universe_manifest=args.universe_manifest,
            fold_manifest=args.fold_manifest,
            repo_root=args.repo_root,
            execute=args.execute,
            resume=args.resume,
            dataset_root=args.dataset_root,
        )
        if (args.execute or args.resume) and report.ready:
            report.execution = execute_stages(
                stages,
                repo_root=args.repo_root,
                resume=args.resume,
            )
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
