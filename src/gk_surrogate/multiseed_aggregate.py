"""Fail-closed aggregation for the frozen multi-seed nested-CV protocol.

The protocol runner owns execution and stage ordering.  This module owns the
independent evidence pass: it checks every expected artifact, reconstructs the
selection barrier from validation metrics, pairs outer-test trajectories across
the selected model and direct observed-flux persistence, and writes a complete
analysis record.  Raw per-trajectory values stay in the ignored output tree;
the thesis consumes only the sanitized summary manifest produced afterwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gk_surrogate import protocol_runner

BOOTSTRAP_DEFAULT_SEED = 20260813
BOOTSTRAP_DEFAULT_REPLICATES = 10_000
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_FAMILIES = ("gru", "transformer")
# Stage scalars are accumulated in JAX float32, while the independent pass
# recomputes them from decimal JSON trajectory values (NumPy float64).  A few
# ulps of reduction-order error are therefore expected; this remains tight
# enough to reject substantive evidence mismatches.
_SCALAR_RECOMPUTE_REL_TOL = 1e-6
_SCALAR_RECOMPUTE_ABS_TOL = 3e-6


class AggregationError(ValueError):
    """Raised when the execution tree cannot support a defensible result."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AggregationError(f"JSON evidence must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AggregationError(f"cannot hash evidence {path}: {exc}") from exc


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return protocol_runner._canonical_json_bytes(payload)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AggregationError(f"{label} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise AggregationError(f"{label} must be finite, got {value!r}")
    return result


def _finite_array(value: object, *, label: str, length: int | None = None) -> list[float]:
    if not isinstance(value, list | tuple):
        raise AggregationError(f"{label} must be an array")
    values = [_finite(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if length is not None and len(values) != length:
        raise AggregationError(f"{label} length {len(values)} does not match expected {length}")
    if not values:
        raise AggregationError(f"{label} must not be empty")
    return values


def _nonnegative(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0.0:
        raise AggregationError(f"{label} must be non-negative, got {result!r}")
    return result


def _correlation(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if not -1.0 <= result <= 1.0:
        raise AggregationError(f"{label} must lie in [-1, 1], got {result!r}")
    return result


def _string_array(value: object, *, label: str, length: int | None = None) -> list[str]:
    if not isinstance(value, list | tuple) or not value:
        raise AggregationError(f"{label} must be a non-empty string array")
    values = [item for item in value if isinstance(item, str) and item]
    if len(values) != len(value) or len(set(values)) != len(values):
        raise AggregationError(f"{label} must contain unique non-empty strings")
    if length is not None and len(values) != length:
        raise AggregationError(f"{label} length {len(values)} does not match expected {length}")
    return values


def _require_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise AggregationError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_close(actual: object, expected: float, *, label: str) -> None:
    value = _finite(actual, label=label)
    if not math.isclose(
        value,
        expected,
        rel_tol=_SCALAR_RECOMPUTE_REL_TOL,
        abs_tol=_SCALAR_RECOMPUTE_ABS_TOL,
    ):
        raise AggregationError(f"{label} mismatch: expected {expected!r}, got {value!r}")


def _fold_manifests(
    protocol: Mapping[str, Any],
    universe_path: Path,
    fold_manifest_path: Path,
) -> dict[int, dict[str, Any]]:
    data = protocol.get("data")
    if not isinstance(data, Mapping):
        raise AggregationError("protocol.data is missing")
    expected_universe_hash = data.get("universe_manifest_sha256")
    _require_equal(_sha256(universe_path), expected_universe_hash, label="universe manifest SHA-256")
    index = _read_json(fold_manifest_path)
    fallback = data.get("fallback_rule")
    if not isinstance(fallback, Mapping):
        raise AggregationError("protocol.data.fallback_rule is missing")
    expected_fold_hash = fallback.get("outer_fold_manifest_sha256")
    _require_equal(_sha256(fold_manifest_path), expected_fold_hash, label="outer-fold manifest SHA-256")
    folds = index.get("folds")
    if not isinstance(folds, list) or len(folds) != len(EXPECTED_FOLDS):
        raise AggregationError("outer-fold manifest must contain exactly five folds")
    universe = _read_json(universe_path)
    regenerated = protocol_runner.generate_outer_fold_manifest(protocol, universe)
    if index != regenerated:
        raise AggregationError("outer-fold manifest differs from deterministic regeneration")
    result: dict[int, dict[str, Any]] = {}
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise AggregationError("outer-fold entry is not an object")
        outer_fold = fold.get("outer_fold")
        if not isinstance(outer_fold, int) or outer_fold not in EXPECTED_FOLDS:
            raise AggregationError(f"invalid outer fold identifier: {outer_fold!r}")
        cli_path = protocol_runner._fold_cli_manifest_path(fold_manifest_path, outer_fold)
        if not cli_path.is_file():
            raise AggregationError(f"missing CLI fold manifest: {cli_path}")
        cli_payload = protocol_runner._fold_cli_manifest_payload(protocol.get("protocol_id"), fold)
        _require_equal(
            _sha256(cli_path),
            fold.get("cli_split_manifest_sha256"),
            label=f"outer fold {outer_fold} CLI hash",
        )
        if cli_path.read_bytes() != _canonical_json(cli_payload):
            raise AggregationError(f"outer fold {outer_fold} CLI manifest is not canonical")
        result[outer_fold] = dict(cli_payload)
    if tuple(sorted(result)) != EXPECTED_FOLDS:
        raise AggregationError("outer-fold manifest does not cover folds 0..4 exactly once")
    return result


def _source_tag_commit(protocol: Mapping[str, Any], repo_root: Path) -> tuple[str, str]:
    source = protocol.get("source")
    if not isinstance(source, Mapping):
        raise AggregationError("protocol.source is missing")
    tag = source.get("tag")
    if not isinstance(tag, str) or not tag.strip():
        raise AggregationError("frozen protocol must record a source tag")
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AggregationError(f"protocol source tag does not resolve locally: {tag}")
    tag_commit = result.stdout.strip()
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if head_result.returncode != 0 or head_result.stdout.strip() != tag_commit:
        raise AggregationError("aggregation must run at the exact frozen protocol source tag")
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status_result.returncode != 0 or status_result.stdout.strip():
        raise AggregationError("aggregation requires a clean worktree at the frozen source tag")
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if diff_result.returncode != 0:
        raise AggregationError("cannot inspect frozen source diff")
    actual_diff = hashlib.sha256(diff_result.stdout).hexdigest()
    if actual_diff != source.get("tracked_diff_sha256"):
        raise AggregationError("tracked source diff does not match the frozen protocol")
    expected_commit = source.get("commit")
    if expected_commit is not None and expected_commit != tag_commit:
        raise AggregationError("protocol source commit and tag resolve to different commits")
    return tag, tag_commit


def _metrics_and_evidence(stage: protocol_runner.Stage) -> tuple[dict[str, Any], dict[str, str]]:
    evidence = protocol_runner._load_stage_evidence(stage)
    if evidence is None:
        raise AggregationError(f"missing or invalid stage evidence: {stage.output_dir}")
    _validate_stage_wandb_status(stage, evidence)
    metrics_path = stage.output_dir / "metrics.json"
    return _read_json(metrics_path), evidence


def _validate_wandb_contract(configs: Mapping[str, Path]) -> None:
    """Prove that every canonical role is configured without live W&B logging."""
    for role in protocol_runner.CONFIG_ROLES:
        path = configs.get(role)
        if path is None or not path.is_file():
            raise AggregationError(f"missing canonical config for W&B verification: {role}")
        try:
            config = protocol_runner.load_config(path, command=protocol_runner.CONFIG_ROLES[role])
        except Exception as exc:
            raise AggregationError(f"cannot validate W&B config for {role}: {exc}") from exc
        wandb = config.logging.wandb
        if wandb.enabled or wandb.mode != "disabled":
            raise AggregationError(f"canonical config {role} requests live or offline W&B logging")


def _validate_stage_wandb_status(stage: protocol_runner.Stage, evidence: Mapping[str, str]) -> None:
    """Reject a stage if its durable W&B decision says logging was requested."""
    status_path = stage.output_dir / "wandb_status.json"
    if not status_path.is_file():
        raise AggregationError(f"stage {stage.name} is missing durable W&B status: {status_path}")
    status = _read_json(status_path)
    if status.get("enabled") is not False or status.get("requested") is not False or status.get("mode") != "disabled":
        raise AggregationError(f"stage {stage.name} has non-disabled W&B status: {status_path}")
    _require_equal(evidence.get("wandb_status_sha256"), _sha256(status_path), label=f"{stage.name} W&B status hash")


def _validate_stage_payload(
    payload: Mapping[str, Any],
    *,
    stage: protocol_runner.Stage,
    fold: Mapping[str, Any],
    seed: int,
    expected_split: str,
    canonical_universe_ids: Sequence[str] | None = None,
    expected_family: str | None = None,
) -> None:
    if expected_split == "all":
        expected_ids = [
            *fold["splits"]["train"],
            *fold["splits"]["val"],
            *fold["splits"]["test"],
        ]
    else:
        expected_ids = list(fold["splits"][expected_split])
    expected_universe = list(canonical_universe_ids) if canonical_universe_ids is not None else [
        *fold["splits"]["train"],
        *fold["splits"]["val"],
        *fold["splits"]["test"],
    ]
    _require_equal(payload.get("protocol_version"), 1, label=f"{stage.name} protocol version")
    _require_equal(payload.get("data_split"), expected_split, label=f"{stage.name} data split")
    _require_equal(payload.get("data_split_seed"), 52, label=f"{stage.name} data seed")
    _require_equal(payload.get("training_seed"), seed, label=f"{stage.name} training seed")
    selected_ids = _string_array(payload.get("selected_trajectory_ids"), label=f"{stage.name} selected IDs")
    universe_ids = _string_array(payload.get("universe_trajectory_ids"), label=f"{stage.name} universe IDs")
    if set(selected_ids) != set(expected_ids):
        raise AggregationError(f"{stage.name} selected IDs do not match the frozen fold")
    if universe_ids != expected_universe:
        raise AggregationError(f"{stage.name} universe IDs do not match the frozen fold")
    _require_equal(
        payload.get("trajectory_manifest_sha256"),
        protocol_runner._trajectory_manifest_sha256(selected_ids),
        label=f"{stage.name} selected ID hash",
    )
    _require_equal(
        payload.get("universe_manifest_sha256"),
        protocol_runner._trajectory_manifest_sha256(universe_ids),
        label=f"{stage.name} universe ID hash",
    )
    _require_equal(payload.get("split_fold_id"), fold["fold_id"], label=f"{stage.name} fold ID")
    cli_path = stage.command[stage.command.index("--split-manifest") + 1]
    _require_equal(payload.get("split_manifest_sha256"), _sha256(Path(cli_path)), label=f"{stage.name} split hash")
    if expected_family is not None:
        _require_equal(payload.get("baseline_mode"), expected_family, label=f"{stage.name} baseline mode")
    if stage.phase in {"validation", "test"} and "flux_rmse_by_trajectory" in payload:
        flux_series = _metric_series(
            payload,
            value_key="flux_rmse_by_trajectory",
            id_key="flux_trajectory_ids",
            label=f"{stage.name} validation flux",
        )
        expected_metric = float(np.mean(list(flux_series.values())))
        _require_close(
            payload.get("trajectory_balanced_flux_rmse"),
            expected_metric,
            label=f"{stage.name} trajectory-balanced flux RMSE",
        )
        _require_close(
            payload.get("flux_rmse"),
            expected_metric,
            label=f"{stage.name} flux RMSE scalar",
        )


def _metric_series(payload: Mapping[str, Any], *, value_key: str, id_key: str, label: str) -> dict[str, float]:
    ids = _string_array(payload.get(id_key), label=f"{label}.{id_key}")
    values = _finite_array(payload.get(value_key), label=f"{label}.{value_key}", length=len(ids))
    if any(value < 0.0 for value in values):
        raise AggregationError(f"{label}.{value_key} must contain non-negative errors")
    if payload.get("num_trajectories") != len(ids):
        raise AggregationError(f"{label}.num_trajectories does not match trajectory arrays")
    selected_ids = _string_array(payload.get("selected_trajectory_ids"), label=f"{label}.selected_trajectory_ids")
    if selected_ids != ids:
        raise AggregationError(f"{label} trajectory IDs are not in the same order as selected IDs")
    return dict(zip(ids, values, strict=True))


def _per_step(payload: Mapping[str, Any], key: str, *, label: str, expected_length: int | None = None) -> list[float]:
    return _finite_array(payload.get(key), label=f"{label}.{key}", length=expected_length)


def _selection(
    path: Path,
    *,
    fold_id: int,
    seeds: Sequence[int],
    validation_payloads: Mapping[str, Mapping[int, Mapping[str, Any]]],
    validation_hashes: Mapping[str, Mapping[int, str]],
) -> tuple[str, dict[str, Any]]:
    payload = _read_json(path)
    _require_equal(payload.get("protocol_version"), 1, label=f"selection fold {fold_id} protocol version")
    _require_equal(payload.get("outer_fold"), fold_id, label=f"selection fold {fold_id} ID")
    _require_equal(payload.get("selection_split"), "val", label=f"selection fold {fold_id} split")
    _require_equal(
        payload.get("primary_metric"),
        "trajectory_balanced_flux_rmse",
        label=f"selection fold {fold_id} primary metric",
    )
    _require_equal(payload.get("matched_training_seeds"), list(seeds), label=f"selection fold {fold_id} seeds")
    if payload.get("test_evidence_opened") is not False:
        raise AggregationError(f"selection fold {fold_id} records test evidence as opened")
    means = payload.get("candidate_mean_validation_trajectory_balanced_flux_rmse")
    if not isinstance(means, Mapping):
        raise AggregationError(f"selection fold {fold_id} is missing candidate means")
    declared_hashes = payload.get("candidate_validation_metrics_sha256")
    if not isinstance(declared_hashes, Mapping):
        raise AggregationError(f"selection fold {fold_id} is missing validation hashes")
    for family in EXPECTED_FAMILIES:
        values = [
            _finite(
                validation_payloads[family][seed].get("trajectory_balanced_flux_rmse"),
                label=f"{family} validation trajectory-balanced flux RMSE",
            )
            for seed in seeds
        ]
        mean = float(np.mean(values))
        declared_mean = _finite(means.get(family), label=f"selection fold {fold_id} {family} mean")
        if not math.isclose(declared_mean, mean, rel_tol=1e-12, abs_tol=1e-12):
            raise AggregationError(
                f"selection fold {fold_id} {family} mean mismatch: expected {mean}, got {declared_mean}"
            )
        declared_family_hashes = declared_hashes.get(family)
        if not isinstance(declared_family_hashes, Mapping):
            raise AggregationError(f"selection fold {fold_id} lacks hashes for {family}")
        for seed in seeds:
            _require_equal(
                declared_family_hashes.get(str(seed)),
                validation_hashes[family][seed],
                label=f"selection fold {fold_id} {family} seed {seed} hash",
            )
    selected = payload.get("selected_family")
    if selected not in EXPECTED_FAMILIES:
        raise AggregationError(f"selection fold {fold_id} chose invalid family {selected!r}")
    expected_selected = min(
        EXPECTED_FAMILIES,
        key=lambda family: (
            _finite(means.get(family), label=f"selection fold {fold_id} {family} mean"),
            family,
        ),
    )
    if selected != expected_selected:
        raise AggregationError(
            "selection fold "
            f"{fold_id} violates deterministic minimum rule: expected {expected_selected}, got {selected}"
        )
    return str(selected), payload


def _hierarchical_mean(groups: Mapping[int, Mapping[int, Sequence[float]]]) -> float:
    fold_means = []
    for fold in sorted(groups):
        seed_means = [float(np.mean(values)) for _, values in sorted(groups[fold].items())]
        fold_means.append(float(np.mean(seed_means)))
    if not fold_means:
        raise AggregationError("cannot aggregate empty groups")
    return float(np.mean(fold_means))


def _bootstrap(
    groups: Mapping[int, Mapping[int, Sequence[float]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if replicates < 100:
        raise AggregationError("bootstrap requires at least 100 replicates")
    fold_ids = tuple(sorted(groups))
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        fold_draw = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        fold_values = []
        for fold_id in fold_draw:
            seed_ids = tuple(sorted(groups[int(fold_id)]))
            seed_draw = rng.choice(seed_ids, size=len(seed_ids), replace=True)
            seed_values = []
            for seed_id in seed_draw:
                values = np.asarray(groups[int(fold_id)][int(seed_id)], dtype=np.float64)
                trajectory_draw = rng.choice(values, size=len(values), replace=True)
                seed_values.append(float(np.mean(trajectory_draw)))
            fold_values.append(float(np.mean(seed_values)))
        samples[index] = float(np.mean(fold_values))
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(np.mean(samples)),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
    }


def _bootstrap_within_fold(
    values_by_seed: Mapping[int, Sequence[float]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    seed_ids = tuple(sorted(values_by_seed))
    samples = []
    for _ in range(replicates):
        seed_draw = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_means = []
        for seed_id in seed_draw:
            values = np.asarray(values_by_seed[int(seed_id)], dtype=np.float64)
            seed_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        samples.append(float(np.mean(seed_means)))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise AggregationError("cannot write an empty CSV")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, fold_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional rendering environment
        raise AggregationError(f"cannot import matplotlib for result plot: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray([row["outer_fold"] for row in fold_rows], dtype=np.float64)
    y = np.asarray([row["mean_difference"] for row in fold_rows], dtype=np.float64)
    lower = y - np.asarray([row["ci_lower"] for row in fold_rows], dtype=np.float64)
    upper = np.asarray([row["ci_upper"] for row in fold_rows], dtype=np.float64) - y
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    axis.errorbar(
        x,
        y,
        yerr=np.vstack((lower, upper)),
        fmt="o",
        color="#2f5d8a",
        ecolor="#2f5d8a",
        capsize=4,
        linewidth=1.2,
        markersize=5,
        label="Selected learned model − observed-flux persistence",
    )
    axis.axhline(0.0, color="#4a4a4a", linewidth=1.0, linestyle="--", label="No difference")
    axis.set_title("Outer-fold test flux RMSE difference")
    axis.set_xlabel("Outer fold")
    axis.set_ylabel("RMSE difference (preprocessed target units)")
    axis.set_xticks(x)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.legend(frameon=False, loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _validate_result_contract(result: Mapping[str, Any]) -> None:
    required = {
        "result_schema_version",
        "status",
        "protocol_id",
        "protocol_sha256",
        "source_tag",
        "source_commit",
        "dataset_revision",
        "universe_manifest_sha256",
        "outer_fold_manifest_sha256",
        "selection",
        "primary_estimand",
        "bootstrap",
        "primary",
        "secondary",
        "outer_fold_summary",
        "seed_summary",
        "seed_variability",
        "stage_ledger",
        "artifacts",
        "wandb_status",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise AggregationError(f"result manifest is missing required fields: {', '.join(missing)}")
    _require_equal(result.get("result_schema_version"), "1.0.0", label="result schema version")
    _require_equal(result.get("status"), "accepted", label="result status")
    if result.get("protocol_id") != "multiseed-v1":
        raise AggregationError("result protocol_id must be multiseed-v1")
    if not isinstance(result.get("source_tag"), str) or not result["source_tag"].strip():
        raise AggregationError("result source_tag must be a non-empty tag")
    for key in ("protocol_sha256", "universe_manifest_sha256", "outer_fold_manifest_sha256"):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise AggregationError(f"result {key} must be a lowercase SHA-256 digest")
    source_commit = result.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(char not in "0123456789abcdef" for char in source_commit)
    ):
        raise AggregationError("result source_commit must be a full Git commit hash")
    if not isinstance(result.get("outer_fold_summary"), list) or len(result["outer_fold_summary"]) != 5:
        raise AggregationError("result outer_fold_summary must contain exactly five folds")
    seed_summary = result.get("seed_summary")
    if not isinstance(seed_summary, list) or len(seed_summary) != 25:
        raise AggregationError("result seed_summary must contain exactly 25 fold-seed rows")
    seed_variability = result.get("seed_variability")
    if not isinstance(seed_variability, list) or len(seed_variability) != 5:
        raise AggregationError("result seed_variability must contain exactly five seeds")
    if {row.get("training_seed") for row in seed_variability if isinstance(row, Mapping)} != {52, 53, 54, 55, 56}:
        raise AggregationError("result seed_variability must cover training seeds 52..56")
    for row in seed_variability:
        if not isinstance(row, Mapping):
            raise AggregationError("result seed_variability rows must be objects")
        for key in ("mean_difference", "std_across_outer_folds", "min_across_outer_folds", "max_across_outer_folds"):
            _finite(row.get(key), label=f"seed_variability.{key}")
    if not isinstance(result.get("stage_ledger"), list) or len(result["stage_ledger"]) != 255:
        raise AggregationError("result stage_ledger must contain all 255 planned slots")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AggregationError("result artifacts must be an object")
    for key in (
        "outer_fold_summary_sha256",
        "paired_trajectory_results_sha256",
        "primary_difference_figure_sha256",
    ):
        value = artifacts.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise AggregationError(f"result artifact {key} must be a SHA-256 digest")
    wandb_status = result.get("wandb_status")
    if (
        not isinstance(wandb_status, Mapping)
        or wandb_status.get("enabled") is not False
        or wandb_status.get("requested") is not False
        or wandb_status.get("mode") != "disabled"
        or wandb_status.get("config_verified") is not True
    ):
        raise AggregationError("result W&B status must prove disabled logging")


def aggregate(
    protocol_path: Path,
    *,
    fold_manifest_path: Path,
    universe_manifest_path: Path,
    output_root: Path,
    configs: Mapping[str, Path],
    repo_root: Path,
    bootstrap_replicates: int = BOOTSTRAP_DEFAULT_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_DEFAULT_SEED,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("status") != "frozen":
        raise AggregationError("aggregation requires the immutable frozen protocol snapshot")
    protocol_report = protocol_runner.PreflightReport()
    protocol_runner._validate_protocol_schema_contract(protocol, protocol_report)
    protocol_runner._validate_protocol(protocol, protocol_report, resume=True)
    if protocol_report.blockers:
        raise AggregationError("frozen protocol contract failed: " + "; ".join(protocol_report.blockers))
    if protocol.get("accepted_runs") != []:
        raise AggregationError("frozen protocol snapshot must retain an empty accepted_runs list")
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise AggregationError("protocol_id is missing")
    source_tag, source_commit = _source_tag_commit(protocol, repo_root)
    try:
        protocol_relpath = protocol_path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise AggregationError("protocol path must be inside repo_root") from exc
    tagged_protocol = subprocess.run(
        ["git", "show", f"{source_tag}:{protocol_relpath.as_posix()}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if tagged_protocol.returncode != 0 or hashlib.sha256(tagged_protocol.stdout).hexdigest() != _sha256(protocol_path):
        raise AggregationError("current protocol bytes do not match the frozen source tag")
    folds = _fold_manifests(protocol, universe_manifest_path, fold_manifest_path)
    universe_payload = _read_json(universe_manifest_path)
    canonical_universe_ids = _string_array(
        universe_payload.get("trajectory_ids"),
        label="frozen universe trajectory_ids",
    )
    models = protocol.get("models")
    if not isinstance(models, Mapping):
        raise AggregationError("protocol.models is missing")
    seeds = models.get("training_seeds")
    if not isinstance(seeds, list) or tuple(seeds) != (52, 53, 54, 55, 56):
        raise AggregationError("aggregator requires the frozen matched seeds [52, 53, 54, 55, 56]")
    stage_plan = protocol_runner.build_stages(
        configs,
        output_root,
        seeds,
        fold_manifest=fold_manifest_path,
        outer_folds=EXPECTED_FOLDS,
        data_seed=52,
    )
    _validate_wandb_contract(configs)
    if len(stage_plan) != 255:
        raise AggregationError(f"unexpected stage plan length: {len(stage_plan)}")
    by_key = {(stage.outer_fold, stage.seed, stage.name): stage for stage in stage_plan}
    fold_results: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    difference_groups: dict[int, dict[int, list[float]]] = {}
    learned_groups: dict[int, dict[int, list[float]]] = {}
    observed_groups: dict[int, dict[int, list[float]]] = {}
    latent_groups: dict[int, dict[int, list[float]]] = {}
    oracle_groups: dict[int, dict[int, list[float]]] = {}
    secondary_difference_groups: dict[int, dict[int, list[float]]] = {}
    learned_latent_mse_groups: dict[int, dict[int, list[float]]] = {}
    latent_latent_mse_groups: dict[int, dict[int, list[float]]] = {}
    scalar_groups: dict[str, dict[int, dict[int, list[float]]]] = {
        "learned_flux_mae": {},
        "observed_flux_mae": {},
        "latent_persistence_flux_mae": {},
    }
    spectral_groups: dict[str, dict[str, dict[int, dict[int, list[float]]]]] = {}
    stage_ledger: list[dict[str, Any]] = []

    for outer_fold in EXPECTED_FOLDS:
        fold = folds[outer_fold]
        validation_payloads: dict[str, dict[int, dict[str, Any]]] = {family: {} for family in EXPECTED_FAMILIES}
        validation_hashes: dict[str, dict[int, str]] = {family: {} for family in EXPECTED_FAMILIES}
        seed_artifacts: dict[int, dict[str, dict[str, str]]] = {seed: {} for seed in seeds}
        for seed in seeds:
            for family in EXPECTED_FAMILIES:
                stage = by_key[(outer_fold, seed, f"{family}_validation")]
                payload, evidence = _metrics_and_evidence(stage)
                _validate_stage_payload(
                    payload,
                    stage=stage,
                    fold=fold,
                    seed=seed,
                    expected_split="val",
                    canonical_universe_ids=canonical_universe_ids,
                )
                validation_payloads[family][seed] = payload
                validation_hashes[family][seed] = evidence["metrics_sha256"]
                seed_artifacts[seed][f"{family}_validation"] = evidence
                stage_ledger.append(
                    {
                        "outer_fold": outer_fold,
                        "training_seed": seed,
                        "stage": stage.name,
                        "phase": stage.phase,
                        "status": "accepted",
                        "metrics_path": str(stage.output_dir / "metrics.json"),
                        "metrics_sha256": evidence["metrics_sha256"],
                    }
                )
        selected_family, selection = _selection(
            output_root / f"outer_fold_{outer_fold}" / "selection" / "metrics.json",
            fold_id=outer_fold,
            seeds=seeds,
            validation_payloads=validation_payloads,
            validation_hashes=validation_hashes,
        )
        selection_path = output_root / f"outer_fold_{outer_fold}" / "selection" / "metrics.json"
        stage_ledger.append(
            {
                "outer_fold": outer_fold,
                "training_seed": None,
                "stage": "architecture_selection",
                "phase": "selection",
                "status": "accepted",
                "selected_family": selected_family,
                "metrics_path": str(selection_path),
                "metrics_sha256": _sha256(selection_path),
            }
        )
        difference_groups[outer_fold] = {}
        learned_groups[outer_fold] = {}
        observed_groups[outer_fold] = {}
        latent_groups[outer_fold] = {}
        oracle_groups[outer_fold] = {}
        secondary_difference_groups[outer_fold] = {}
        learned_latent_mse_groups[outer_fold] = {}
        latent_latent_mse_groups[outer_fold] = {}
        for values in scalar_groups.values():
            values[outer_fold] = {}
        fold_step_learned: list[list[float]] = []
        fold_step_observed: list[list[float]] = []
        fold_step_oracle: list[list[float]] = []
        fold_step_learned_latent: list[list[float]] = []
        fold_step_latent_latent: list[list[float]] = []
        for seed in seeds:
            for role in ("encoder", "embed", "gru_train", "transformer_train"):
                stage = by_key[(outer_fold, seed, role)]
                payload, evidence = _metrics_and_evidence(stage)
                _validate_stage_payload(
                    payload,
                    stage=stage,
                    fold=fold,
                    seed=seed,
                    expected_split="all" if role == "embed" else "train",
                    canonical_universe_ids=canonical_universe_ids,
                )
                if role == "embed":
                    cache_hash = payload.get("latent_cache_sha256")
                    actual_cache_hash = evidence.get("latent_cache_sha256")
                    _require_equal(
                        cache_hash,
                        actual_cache_hash,
                        label=f"outer fold {outer_fold} seed {seed} cache hash",
                    )
                    _require_equal(
                        payload.get("encoder_checkpoint_sha256"),
                        seed_artifacts[seed]["encoder"].get("checkpoint_sha256"),
                        label=f"outer fold {outer_fold} seed {seed} cache encoder lineage",
                    )
                if role == "encoder":
                    _require_equal(
                        payload.get("checkpoint_selection"),
                        "minimum_validation_trajectory_balanced_flux_rmse",
                        label=f"outer fold {outer_fold} seed {seed} encoder checkpoint selection",
                    )
                    _finite(
                        payload.get("best_validation_metric"),
                        label=f"outer fold {outer_fold} seed {seed} encoder best validation metric",
                    )
                if role in {"gru_train", "transformer_train"}:
                    _require_equal(
                        payload.get("checkpoint_selection"),
                        "minimum_validation_trajectory_balanced_latent_rmse",
                        label=f"outer fold {outer_fold} seed {seed} {role} checkpoint selection",
                    )
                    _finite(
                        payload.get("best_validation_metric"),
                        label=f"outer fold {outer_fold} seed {seed} {role} best validation metric",
                    )
                seed_artifacts[seed][role] = evidence
                stage_ledger.append(
                    {
                        "outer_fold": outer_fold,
                        "training_seed": seed,
                        "stage": stage.name,
                        "phase": stage.phase,
                        "status": "accepted",
                        "metrics_path": str(stage.output_dir / "metrics.json"),
                        **{key: value for key, value in evidence.items() if key.endswith("_sha256")},
                    }
                )
            for family in EXPECTED_FAMILIES:
                validation_payload = validation_payloads[family][seed]
                train_evidence = seed_artifacts[seed][f"{family}_train"]
                _require_equal(
                    validation_payload.get("sequence_checkpoint_sha256"),
                    train_evidence.get("checkpoint_sha256"),
                    label=f"outer {outer_fold} seed {seed} {family} validation checkpoint lineage",
                )
                _require_equal(
                    validation_payload.get("latent_cache_sha256"),
                    seed_artifacts[seed]["embed"].get("latent_cache_sha256"),
                    label=f"outer {outer_fold} seed {seed} {family} validation cache lineage",
                )
                _require_equal(
                    validation_payload.get("encoder_checkpoint_sha256"),
                    seed_artifacts[seed]["encoder"].get("checkpoint_sha256"),
                    label=f"outer {outer_fold} seed {seed} {family} validation encoder lineage",
                )
            selected_stage = by_key[(outer_fold, seed, f"{selected_family}_eval")]
            learned_payload, learned_evidence = _metrics_and_evidence(selected_stage)
            _validate_stage_payload(
                learned_payload,
                stage=selected_stage,
                fold=fold,
                seed=seed,
                expected_split="test",
                canonical_universe_ids=canonical_universe_ids,
                expected_family="none",
            )
            _require_equal(
                learned_payload.get("rollout_method"),
                "learned_sequence_model",
                label="learned rollout method",
            )
            _require_equal(
                learned_payload.get("encoder_checkpoint_sha256"),
                seed_artifacts[seed]["encoder"].get("checkpoint_sha256"),
                label=f"outer {outer_fold} seed {seed} learned encoder lineage",
            )
            _require_equal(
                learned_payload.get("latent_cache_sha256"),
                seed_artifacts[seed]["embed"].get("latent_cache_sha256"),
                label=f"outer {outer_fold} seed {seed} learned cache lineage",
            )
            _require_equal(
                learned_payload.get("sequence_checkpoint_sha256"),
                seed_artifacts[seed][f"{selected_family}_train"].get("checkpoint_sha256"),
                label=f"outer {outer_fold} seed {seed} learned sequence lineage",
            )
            observed_stage = by_key[(outer_fold, seed, "observed_persistence_eval")]
            observed_payload, observed_evidence = _metrics_and_evidence(observed_stage)
            _validate_stage_payload(
                observed_payload,
                stage=observed_stage,
                fold=fold,
                seed=seed,
                expected_split="test",
                canonical_universe_ids=canonical_universe_ids,
                expected_family="observed_diagnostic_persistence",
            )
            _require_equal(
                observed_payload.get("rollout_method"),
                "observed_diagnostic_persistence",
                label="observed baseline rollout method",
            )
            _require_equal(
                observed_payload.get("encoder_checkpoint_sha256"),
                seed_artifacts[seed]["encoder"].get("checkpoint_sha256"),
                label=f"outer {outer_fold} seed {seed} observed encoder lineage",
            )
            _require_equal(
                observed_payload.get("latent_cache_sha256"),
                seed_artifacts[seed]["embed"].get("latent_cache_sha256"),
                label=f"outer {outer_fold} seed {seed} observed cache lineage",
            )
            latent_stage = by_key[(outer_fold, seed, "latent_persistence_eval")]
            latent_payload, latent_evidence = _metrics_and_evidence(latent_stage)
            _validate_stage_payload(
                latent_payload,
                stage=latent_stage,
                fold=fold,
                seed=seed,
                expected_split="test",
                canonical_universe_ids=canonical_universe_ids,
                expected_family="latent_state_persistence_decoded",
            )
            _require_equal(
                latent_payload.get("rollout_method"),
                "latent_state_persistence_decoded",
                label="latent baseline rollout method",
            )
            _require_equal(
                latent_payload.get("encoder_checkpoint_sha256"),
                seed_artifacts[seed]["encoder"].get("checkpoint_sha256"),
                label=f"outer {outer_fold} seed {seed} latent encoder lineage",
            )
            _require_equal(
                latent_payload.get("latent_cache_sha256"),
                seed_artifacts[seed]["embed"].get("latent_cache_sha256"),
                label=f"outer {outer_fold} seed {seed} latent cache lineage",
            )
            learned = _metric_series(
                learned_payload,
                value_key="flux_rmse_by_trajectory",
                id_key="flux_trajectory_ids",
                label=f"outer {outer_fold} seed {seed} learned",
            )
            observed = _metric_series(
                observed_payload,
                value_key="flux_rmse_by_trajectory",
                id_key="flux_trajectory_ids",
                label=f"outer {outer_fold} seed {seed} observed",
            )
            embedded_observed = _metric_series(
                learned_payload,
                value_key="observed_diagnostic_persistence_flux_rmse_by_trajectory",
                id_key="flux_trajectory_ids",
                label=f"outer {outer_fold} seed {seed} embedded observed",
            )
            if any(
                not math.isclose(embedded_observed[trajectory_id], observed[trajectory_id], rel_tol=1e-6, abs_tol=1e-6)
                for trajectory_id in observed
            ):
                raise AggregationError(
                    "learned-stage observed baseline disagrees with dedicated baseline in "
                    f"outer {outer_fold}, seed {seed}"
                )
            latent = _metric_series(
                latent_payload,
                value_key="flux_rmse_by_trajectory",
                id_key="flux_trajectory_ids",
                label=f"outer {outer_fold} seed {seed} latent",
            )
            oracle = _metric_series(
                learned_payload,
                value_key="diagnostic_head_oracle_flux_rmse_by_trajectory",
                id_key="flux_trajectory_ids",
                label=f"outer {outer_fold} seed {seed} oracle",
            )
            learned_latent_mse = _metric_series(
                learned_payload,
                value_key="mse_by_trajectory",
                id_key="latent_trajectory_ids",
                label=f"outer {outer_fold} seed {seed} learned latent",
            )
            latent_latent_mse = _metric_series(
                latent_payload,
                value_key="mse_by_trajectory",
                id_key="latent_trajectory_ids",
                label=f"outer {outer_fold} seed {seed} latent baseline",
            )
            if (
                tuple(learned) != tuple(observed)
                or tuple(learned) != tuple(latent)
                or tuple(learned) != tuple(oracle)
                or tuple(learned) != tuple(learned_latent_mse)
                or tuple(learned) != tuple(latent_latent_mse)
            ):
                raise AggregationError(f"trajectory pairing mismatch in outer fold {outer_fold}, seed {seed}")
            horizon = len(_per_step(learned_payload, "flux_rmse_by_step", label="learned"))
            learned_step = _per_step(learned_payload, "flux_rmse_by_step", label="learned", expected_length=horizon)
            observed_step = _per_step(observed_payload, "flux_rmse_by_step", label="observed", expected_length=horizon)
            oracle_step = _per_step(
                learned_payload,
                "diagnostic_head_oracle_flux_rmse_by_step",
                label="oracle",
                expected_length=horizon,
            )
            learned_latent_step = _per_step(
                learned_payload,
                "mse_by_step",
                label="learned latent",
                expected_length=horizon,
            )
            latent_latent_step = _per_step(
                latent_payload,
                "mse_by_step",
                label="latent baseline",
                expected_length=horizon,
            )
            fold_step_learned.append(learned_step)
            fold_step_observed.append(observed_step)
            fold_step_oracle.append(oracle_step)
            fold_step_learned_latent.append(learned_latent_step)
            fold_step_latent_latent.append(latent_latent_step)
            difference_groups[outer_fold][seed] = []
            learned_groups[outer_fold][seed] = []
            observed_groups[outer_fold][seed] = []
            latent_groups[outer_fold][seed] = []
            oracle_groups[outer_fold][seed] = []
            secondary_difference_groups[outer_fold][seed] = []
            learned_latent_mse_groups[outer_fold][seed] = []
            latent_latent_mse_groups[outer_fold][seed] = []
            for metric_name, payload, prefix in (
                ("learned_flux_mae", learned_payload, ""),
                ("observed_flux_mae", observed_payload, ""),
                ("latent_persistence_flux_mae", latent_payload, ""),
            ):
                scalar_groups[metric_name][outer_fold][seed] = [
                    _nonnegative(
                        payload.get(f"{prefix}flux_mae"),
                        label=f"{metric_name} outer {outer_fold} seed {seed}",
                    )
                ]
            for target_key in sorted(
                {
                    key.removeprefix("spectra_").removesuffix("_relative_l2")
                    for key in learned_payload
                    if key.startswith("spectra_")
                    and key != "spectra_relative_l2"
                    and key.endswith("_relative_l2")
                }
            ):
                spectral_groups.setdefault(target_key, {})
                for metric_name, payload, prefix in (
                    ("learned_relative_l2", learned_payload, "spectra_"),
                    ("latent_relative_l2", latent_payload, "spectra_"),
                    ("observed_relative_l2", observed_payload, "observed_diagnostic_persistence_spectra_"),
                ):
                    key = f"{prefix}{target_key}_relative_l2"
                    if key not in payload:
                        raise AggregationError(f"missing spectral metric {key} in outer {outer_fold} seed {seed}")
                    spectral_groups[target_key].setdefault(metric_name, {}).setdefault(outer_fold, {})[seed] = [
                        _nonnegative(payload[key], label=f"{key} outer {outer_fold} seed {seed}")
                    ]
                for metric_name, payload, prefix in (
                    ("learned_shape_corr", learned_payload, "spectra_"),
                    ("latent_shape_corr", latent_payload, "spectra_"),
                    ("observed_shape_corr", observed_payload, "observed_diagnostic_persistence_spectra_"),
                ):
                    key = f"{prefix}{target_key}_shape_corr"
                    if key not in payload:
                        raise AggregationError(f"missing spectral metric {key} in outer {outer_fold} seed {seed}")
                    spectral_groups[target_key].setdefault(metric_name, {}).setdefault(outer_fold, {})[seed] = [
                        _correlation(payload[key], label=f"{key} outer {outer_fold} seed {seed}")
                    ]
            for trajectory_id in learned:
                diff = learned[trajectory_id] - observed[trajectory_id]
                secondary_diff = learned_latent_mse[trajectory_id] - latent_latent_mse[trajectory_id]
                difference_groups[outer_fold][seed].append(diff)
                learned_groups[outer_fold][seed].append(learned[trajectory_id])
                observed_groups[outer_fold][seed].append(observed[trajectory_id])
                latent_groups[outer_fold][seed].append(latent[trajectory_id])
                oracle_groups[outer_fold][seed].append(oracle[trajectory_id])
                secondary_difference_groups[outer_fold][seed].append(secondary_diff)
                learned_latent_mse_groups[outer_fold][seed].append(learned_latent_mse[trajectory_id])
                latent_latent_mse_groups[outer_fold][seed].append(latent_latent_mse[trajectory_id])
                paired_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "training_seed": seed,
                        "selected_family": selected_family,
                        "trajectory_id": trajectory_id,
                        "learned_flux_rmse": learned[trajectory_id],
                        "observed_flux_rmse": observed[trajectory_id],
                        "latent_persistence_flux_rmse": latent[trajectory_id],
                        "diagnostic_head_oracle_flux_rmse": oracle[trajectory_id],
                        "paired_difference": diff,
                        "learned_latent_mse": learned_latent_mse[trajectory_id],
                        "latent_persistence_latent_mse": latent_latent_mse[trajectory_id],
                        "latent_mse_difference": secondary_diff,
                    }
                )
            for stage, evidence, phase, status in (
                (selected_stage, learned_evidence, "test", "accepted"),
                (observed_stage, observed_evidence, "test", "accepted"),
                (latent_stage, latent_evidence, "test", "accepted"),
            ):
                stage_ledger.append(
                    {
                        "outer_fold": outer_fold,
                        "training_seed": seed,
                        "stage": stage.name,
                        "phase": phase,
                        "status": status,
                        "selected_family": selected_family,
                        "metrics_path": str(stage.output_dir / "metrics.json"),
                        **{key: value for key, value in evidence.items() if key.endswith("_sha256")},
                    }
                )
            for family in EXPECTED_FAMILIES:
                unselected = by_key[(outer_fold, seed, f"{family}_eval")]
                if family != selected_family:
                    if unselected.output_dir.exists():
                        raise AggregationError(f"unselected test stage unexpectedly exists: {unselected.output_dir}")
                    stage_ledger.append(
                        {
                            "outer_fold": outer_fold,
                            "training_seed": seed,
                            "stage": unselected.name,
                            "phase": "test",
                            "status": "skipped_unselected",
                            "selected_family": selected_family,
                        }
                    )
        fold_mean = _hierarchical_mean({outer_fold: difference_groups[outer_fold]})
        fold_ci = _bootstrap_within_fold(
            difference_groups[outer_fold],
            replicates=max(2000, bootstrap_replicates // 5),
            seed=bootstrap_seed + outer_fold,
        )
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "selected_family": selected_family,
                "num_test_trajectories": len(fold["splits"]["test"]),
                "mean_difference": fold_mean,
                "ci_lower": fold_ci[0],
                "ci_upper": fold_ci[1],
                "learned_flux_rmse": _hierarchical_mean({outer_fold: learned_groups[outer_fold]}),
                "observed_flux_rmse": _hierarchical_mean({outer_fold: observed_groups[outer_fold]}),
                "latent_persistence_flux_rmse": _hierarchical_mean({outer_fold: latent_groups[outer_fold]}),
                "diagnostic_head_oracle_flux_rmse": _hierarchical_mean({outer_fold: oracle_groups[outer_fold]}),
                "latent_mse_difference": _hierarchical_mean({outer_fold: secondary_difference_groups[outer_fold]}),
                "learned_latent_mse": _hierarchical_mean({outer_fold: learned_latent_mse_groups[outer_fold]}),
                "latent_persistence_latent_mse": _hierarchical_mean({outer_fold: latent_latent_mse_groups[outer_fold]}),
                "learned_flux_mae": _hierarchical_mean({outer_fold: scalar_groups["learned_flux_mae"][outer_fold]}),
                "observed_flux_mae": _hierarchical_mean({outer_fold: scalar_groups["observed_flux_mae"][outer_fold]}),
                "latent_persistence_flux_mae": _hierarchical_mean(
                    {outer_fold: scalar_groups["latent_persistence_flux_mae"][outer_fold]}
                ),
                "learned_flux_rmse_by_step": np.mean(np.asarray(fold_step_learned), axis=0).tolist(),
                "observed_flux_rmse_by_step": np.mean(np.asarray(fold_step_observed), axis=0).tolist(),
                "diagnostic_head_oracle_flux_rmse_by_step": np.mean(np.asarray(fold_step_oracle), axis=0).tolist(),
                "learned_latent_mse_by_step": np.mean(np.asarray(fold_step_learned_latent), axis=0).tolist(),
                "latent_persistence_latent_mse_by_step": np.mean(np.asarray(fold_step_latent_latent), axis=0).tolist(),
                **{
                    f"spectra_{target}_{metric_name}": _hierarchical_mean(
                        {outer_fold: metric_groups[outer_fold]}
                    )
                    for target, target_metrics in spectral_groups.items()
                    for metric_name, metric_groups in target_metrics.items()
                    if outer_fold in metric_groups
                },
            }
        )
    for entry in stage_ledger:
        if entry["phase"] == "selection":
            continue
        stage = by_key[(entry["outer_fold"], entry["training_seed"], entry["stage"])]
        entry["command"] = list(stage.command)
        config_path = stage.output_dir / "config_resolved.json"
        if entry["status"] == "accepted":
            if not config_path.is_file():
                raise AggregationError(f"accepted stage is missing resolved config: {config_path}")
            entry["config_resolved_path"] = str(config_path)
            entry["config_resolved_sha256"] = _sha256(config_path)
    if len(stage_ledger) != 255:
        raise AggregationError(f"stage ledger must contain all 255 planned slots, got {len(stage_ledger)}")
    overall_difference = _hierarchical_mean(difference_groups)
    bootstrap = _bootstrap(
        difference_groups,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    secondary_bootstrap = _bootstrap(
        secondary_difference_groups,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1,
    )
    seed_summary = [
        {
            "outer_fold": outer_fold,
            "training_seed": seed,
            "mean_difference": float(np.mean(difference_groups[outer_fold][seed])),
            "learned_flux_rmse": float(np.mean(learned_groups[outer_fold][seed])),
            "observed_flux_rmse": float(np.mean(observed_groups[outer_fold][seed])),
            "latent_persistence_flux_rmse": float(np.mean(latent_groups[outer_fold][seed])),
            "learned_latent_mse": float(np.mean(learned_latent_mse_groups[outer_fold][seed])),
            "latent_persistence_latent_mse": float(np.mean(latent_latent_mse_groups[outer_fold][seed])),
        }
        for outer_fold in EXPECTED_FOLDS
        for seed in seeds
    ]
    seed_variability = []
    for seed in seeds:
        seed_diff_values: list[float] = [
            float(row["mean_difference"]) for row in seed_summary if row["training_seed"] == seed
        ]
        seed_variability.append(
            {
                "training_seed": seed,
                "mean_difference": float(np.mean(np.asarray(seed_diff_values, dtype=np.float64))),
                "std_across_outer_folds": float(np.std(np.asarray(seed_diff_values, dtype=np.float64), ddof=1)),
                "min_across_outer_folds": float(np.min(np.asarray(seed_diff_values, dtype=np.float64))),
                "max_across_outer_folds": float(np.max(np.asarray(seed_diff_values, dtype=np.float64))),
            }
        )
    fraction_groups = {
        fold: {seed: [float(value < 0.0) for value in values] for seed, values in seed_values.items()}
        for fold, seed_values in difference_groups.items()
    }
    result: dict[str, Any] = {
        "result_schema_version": "1.0.0",
        "status": "accepted",
        "protocol_id": protocol_id,
        "protocol_sha256": _sha256(protocol_path),
        "source_tag": source_tag,
        "source_commit": source_commit,
        "analysis_commit": source_commit,
        "analysis_tracked_diff_sha256": protocol["source"]["tracked_diff_sha256"],
        "dataset_revision": protocol["data"]["dataset_revision"],
        "universe_manifest_sha256": protocol["data"]["universe_manifest_sha256"],
        "universe_trajectory_ids_sha256": protocol_runner._trajectory_manifest_sha256(canonical_universe_ids),
        "outer_fold_manifest_sha256": protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"],
        "config_sha256": {role: _sha256(path) for role, path in sorted(configs.items())},
        "selection": {
            "scope": "inner_validation_only",
            "families": list(EXPECTED_FAMILIES),
            "training_seeds": list(seeds),
            "checkpoint_rule": protocol["selection"]["checkpoint_rule"],
            "primary_metric": protocol["selection"]["primary_metric"],
            "test_used_for_selection": False,
            "selected_family_by_outer_fold": {str(row["outer_fold"]): row["selected_family"] for row in fold_results},
        },
        "primary_estimand": protocol["evaluation"]["primary_estimand"],
        "aggregation_weighting": "equal_outer_fold_then_training_seed_then_test_trajectory",
        "bootstrap": {
            "method": "paired_hierarchical_bootstrap_outer_folds_training_seeds_trajectories",
            **bootstrap,
        },
        "primary": {
            "mean_difference": overall_difference,
            "ci_lower": bootstrap["ci_lower"],
            "ci_upper": bootstrap["ci_upper"],
            "learned_flux_rmse": _hierarchical_mean(learned_groups),
            "observed_flux_rmse": _hierarchical_mean(observed_groups),
            "latent_persistence_flux_rmse": _hierarchical_mean(latent_groups),
            "diagnostic_head_oracle_flux_rmse": _hierarchical_mean(oracle_groups),
            "fraction_trajectories_improved": _hierarchical_mean(fraction_groups),
            "num_outer_folds": len(EXPECTED_FOLDS),
            "num_training_seeds": len(seeds),
            "num_paired_trajectory_runs": len(paired_rows),
            "negative_result_rule": protocol["reporting"]["negative_result_rule"],
        },
        "secondary": {
            "estimand": protocol["evaluation"]["secondary_estimand"],
            "mean_latent_mse_difference": _hierarchical_mean(secondary_difference_groups),
            "ci_lower": secondary_bootstrap["ci_lower"],
            "ci_upper": secondary_bootstrap["ci_upper"],
            "learned_latent_mse": _hierarchical_mean(learned_latent_mse_groups),
            "latent_persistence_latent_mse": _hierarchical_mean(latent_latent_mse_groups),
            "bootstrap": secondary_bootstrap,
        },
        "seed_summary": seed_summary,
        "seed_variability": seed_variability,
        "secondary_scalar_metrics": {
            metric_name: _hierarchical_mean(groups)
            for metric_name, groups in scalar_groups.items()
        },
        "spectra_metrics": {
            target: {
                metric_name: _hierarchical_mean(groups)
                for metric_name, groups in metric_groups.items()
            }
            for target, metric_groups in spectral_groups.items()
        },
        "outer_fold_summary": fold_results,
        "stage_ledger": stage_ledger,
        "paired_rows": paired_rows,
        "inputs": {
            "protocol_path": str(protocol_path),
            "fold_manifest_path": str(fold_manifest_path),
            "universe_manifest_path": str(universe_manifest_path),
            "output_root": str(output_root),
        },
        "wandb_status": {
            "enabled": False,
            "requested": False,
            "mode": "disabled",
            "config_verified": True,
            "provenance": "all canonical protocol configs and any stage status files prove W&B is disabled",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    csv_rows = [
        {
            key: row[key]
            for key in (
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
                "learned_flux_mae",
                "observed_flux_mae",
                "latent_persistence_flux_mae",
            )
            if key in row
        }
        for row in fold_results
    ]
    for target, target_metrics in spectral_groups.items():
        for metric_name in target_metrics:
            column = f"spectra_{target}_{metric_name}"
            for row, fold_row in zip(csv_rows, fold_results, strict=True):
                if column in fold_row:
                    row[column] = fold_row[column]
    outer_fold_summary_path = output_root / "outer_fold_summary.csv"
    paired_rows_path = output_root / "paired_trajectory_results.csv"
    figure_path = output_root / "primary_difference_by_fold.png"
    _write_csv(outer_fold_summary_path, csv_rows)
    _write_csv(paired_rows_path, paired_rows)
    _write_plot(figure_path, fold_results)
    result["artifacts"] = {
        "aggregate_results_json": str(output_root / "aggregate_results.json"),
        "outer_fold_summary_csv": str(outer_fold_summary_path),
        "outer_fold_summary_sha256": _sha256(outer_fold_summary_path),
        "paired_trajectory_results_csv": str(paired_rows_path),
        "paired_trajectory_results_sha256": _sha256(paired_rows_path),
        "primary_difference_figure": str(figure_path),
        "primary_difference_figure_sha256": _sha256(figure_path),
    }
    _validate_result_contract(result)
    output_path = output_root / "aggregate_results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _config_arg(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or role not in protocol_runner.CONFIG_ROLES or not raw_path:
        choices = ", ".join(protocol_runner.CONFIG_ROLES)
        raise argparse.ArgumentTypeError(f"config must be ROLE=PATH where ROLE is one of: {choices}")
    return role, Path(raw_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gks-aggregate")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", action="append", required=True, type=_config_arg, metavar="ROLE=PATH")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configs = dict(args.config)
    missing = sorted(set(protocol_runner.CONFIG_ROLES).difference(configs))
    if missing:
        print(json.dumps({"status": "error", "error": f"missing config roles: {', '.join(missing)}"}, indent=2))
        return 2
    try:
        result = aggregate(
            args.protocol,
            fold_manifest_path=args.fold_manifest,
            universe_manifest_path=args.universe_manifest,
            output_root=args.output_root,
            configs=configs,
            repo_root=args.repo_root,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except AggregationError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "protocol_id": result["protocol_id"],
                "source_tag": result["source_tag"],
                "primary": result["primary"],
                "bootstrap": result["bootstrap"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
