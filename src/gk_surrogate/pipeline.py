"""Small CPU-safe end-to-end pipeline helpers used by the CLI and tests."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from gk_surrogate.config.load import config_to_yaml
from gk_surrogate.config.schema import ExperimentConfig
from gk_surrogate.data.collate import collate_snapshots
from gk_surrogate.data.factory import build_dataset
from gk_surrogate.data.latent_cache import LatentCacheDataset, LatentCacheWriter
from gk_surrogate.data.normalization import (
    NormalizationStats,
    estimate_dataset_stats,
    estimate_trajectory_stats,
    normalize_snapshot,
)
from gk_surrogate.data.sequence_dataset import valid_sequence_starts
from gk_surrogate.data.split import (
    TrajectorySplits,
    resolve_trajectory_splits,
)
from gk_surrogate.data.split import (
    split_trajectory_ids as _seeded_split_trajectory_ids,
)
from gk_surrogate.evaluation.flux_head import evaluate_flux_head as evaluate_linear_flux_head
from gk_surrogate.evaluation.reports import save_rollout_plots, save_rollout_report
from gk_surrogate.evaluation.representation import evaluate_representation
from gk_surrogate.evaluation.rollout import (
    autoregressive_rollout,
    observed_diagnostic_persistence,
    persistence_rollout,
    trajectory_balanced_rollout_metrics,
)
from gk_surrogate.factory import (
    build_diagnostic_heads,
    build_direct_diagnostic_baseline,
    build_encoder_with_diagnostics,
    build_sequence_model,
    build_simsiam_encoder_with_diagnostics,
)
from gk_surrogate.losses.diagnostics import diagnostic_prediction_loss
from gk_surrogate.models.diagnostics import DiagnosticPredictions
from gk_surrogate.parallel.batch import drop_or_pad_to_multiple, shard_batch
from gk_surrogate.parallel.devices import ParallelPlan, get_local_devices, resolve_parallel_mode, write_device_report
from gk_surrogate.parallel.pmap_steps import (
    make_pmap_encoder_eval_step,
    make_pmap_encoder_train_step,
    make_pmap_sequence_eval_step,
    make_pmap_sequence_train_step,
)
from gk_surrogate.parallel.replicate import replicate_state, unreplicate_state, unreplicate_tree
from gk_surrogate.training.checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from gk_surrogate.training.embed_dataset import encode_snapshots
from gk_surrogate.training.logging import MetricsLogger, write_json, write_run_metadata
from gk_surrogate.training.optimizer import build_optimizer, learning_rate_schedule
from gk_surrogate.training.state import TrainState
from gk_surrogate.training.train_encoder import eval_encoder_step, train_encoder_step
from gk_surrogate.training.train_sequence import eval_sequence_step, train_sequence_step
from gk_surrogate.utils.paths import ensure_dir

_PROTOCOL_VERSION = 1
_UNSET = object()

# Backward-compatible test/helper surface; new pipeline code uses the manifest-aware resolver.
split_trajectory_ids = _seeded_split_trajectory_ids


def _output_dir(config: ExperimentConfig) -> Path:
    return ensure_dir(config.output_dir)


def _relative_artifact_path(path: str | Path) -> str:
    """Represent a local artifact without embedding a machine-specific absolute root."""

    return os.path.relpath(Path(path).resolve(), start=Path.cwd().resolve())


def _loss_config(config: ExperimentConfig) -> dict[str, Any]:
    return config.loss.model_dump(mode="json")


def _metrics_logger(output_dir: Path, config: ExperimentConfig) -> MetricsLogger:
    return MetricsLogger(
        output_dir,
        wandb_config=config.logging.wandb.model_dump(mode="json"),
        run_config=config.model_dump(mode="json"),
    )


def _parallel_plan(config: ExperimentConfig) -> ParallelPlan:
    return resolve_parallel_mode(config.parallel, batch_size=config.data.batch_size)


def _config_with_parallel_optimizer(config: ExperimentConfig, plan: ParallelPlan) -> ExperimentConfig:
    if not plan.uses_pmap or not plan.auto_scale_learning_rate or plan.num_devices <= 1:
        return config
    training = config.training.model_copy(update={"learning_rate": config.training.learning_rate * plan.num_devices})
    return config.model_copy(update={"training": training})


def _replicated_if_needed(state: TrainState, plan: ParallelPlan) -> TrainState:
    if not plan.uses_pmap:
        return state
    devices = get_local_devices()[: plan.num_devices]
    return replicate_state(state, devices)


def _host_state(state: TrainState, plan: ParallelPlan) -> TrainState:
    return unreplicate_state(state) if plan.uses_pmap else state


def _host_metrics(metrics: Mapping[str, Any], plan: ParallelPlan) -> Mapping[str, Any]:
    return unreplicate_tree(metrics) if plan.uses_pmap else metrics


def _prepare_parallel_batch(batch: Mapping[str, Any], plan: ParallelPlan) -> dict[str, Any] | None:
    if not plan.uses_pmap:
        return dict(batch)
    prepared = drop_or_pad_to_multiple(
        batch,
        plan.num_devices,
        drop_remainder=plan.drop_remainder,
    )
    if prepared is None:
        return None
    return shard_batch(prepared, plan.num_devices)


def _batch_size(batch: Mapping[str, Any]) -> int:
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(batch) if leaf is not None and hasattr(leaf, "shape")]
    if not leaves or not leaves[0].shape:
        raise ValueError("evaluation batch has no batched array leaves")
    return int(leaves[0].shape[0])


def _aggregate_eval_metrics(
    state: TrainState,
    batches: Iterator[dict[str, Any]],
    *,
    plan: ParallelPlan,
    step_fn: Any,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    total_examples = 0
    for batch in batches:
        batch_size = _batch_size(batch)
        prepared = _prepare_parallel_batch(batch, plan)
        if prepared is None:
            continue
        metrics = _host_metrics(step_fn(state, prepared), plan)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("validation split produced no evaluable batches")
    return {key: value / total_examples for key, value in totals.items()}


def _aggregate_eval_metrics_by_trajectory(
    state: TrainState,
    trajectory_ids: tuple[str, ...],
    *,
    batches_for_trajectory: Any,
    plan: ParallelPlan,
    step_fn: Any,
) -> dict[str, float]:
    """Average validation metrics with one equal contribution per trajectory."""

    if not trajectory_ids:
        raise ValueError("trajectory-balanced validation requires at least one trajectory")
    per_trajectory = [
        _aggregate_eval_metrics(
            state,
            batches_for_trajectory(trajectory_id),
            plan=plan,
            step_fn=step_fn,
        )
        for trajectory_id in trajectory_ids
    ]
    metric_names = set(per_trajectory[0])
    if any(set(metrics) != metric_names for metrics in per_trajectory[1:]):
        raise ValueError("validation trajectories produced inconsistent metric sets")
    return {
        name: float(np.mean([metrics[name] for metrics in per_trajectory]))
        for name in sorted(metric_names)
    }


def _scheduled_learning_rate(config: ExperimentConfig, completed_step: int) -> float:
    schedule_step = max(completed_step - 1, 0)
    return float(learning_rate_schedule(config.training)(schedule_step))


def _direct_diagnostic_loss(
    params: Any,
    state: TrainState,
    batch: Mapping[str, Any],
    *,
    train: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    rngs = {"dropout": state.rng} if train else None
    predictions = state.apply_fn(
        {"params": params},
        batch["x"],
        train=train,
        **({"rngs": rngs} if rngs is not None else {}),
    )
    flux_target = batch.get("flux")
    spectra_target = batch.get("spectra")
    loss, metrics = diagnostic_prediction_loss(
        predictions,
        flux_target=flux_target,
        spectra_target=spectra_target,
        flux_weight=float(state.model_config["flux_weight"]),
        spectra_weight=float(state.model_config["spectra_weight"]),
        log_spectra=bool(state.model_config["use_log_spectra"]),
        spectra_eps=float(state.model_config["spectra_epsilon"]),
    )
    flat = {key.replace("loss/", ""): value for key, value in metrics.items()}
    if predictions.flux is not None and flux_target is not None:
        flat["flux_mse"] = jnp.mean(jnp.square(predictions.flux - flux_target))
    for key, prediction in predictions.spectra.items():
        if isinstance(spectra_target, Mapping) and key in spectra_target:
            flat[f"spectra_{key}_mse"] = jnp.mean(jnp.square(prediction - spectra_target[key]))
    return loss, flat


@jax.jit
def _train_direct_diagnostic_step(
    state: TrainState,
    batch: Mapping[str, Any],
) -> tuple[TrainState, dict[str, jax.Array]]:
    rng, step_rng = jax.random.split(state.rng)
    step_state = state.replace_rng(step_rng)

    def loss_fn(params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        return _direct_diagnostic_loss(params, step_state, batch, train=True)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads).replace(rng=rng), {**metrics, "loss": loss}


@jax.jit
def _eval_direct_diagnostic_step(state: TrainState, batch: Mapping[str, Any]) -> dict[str, jax.Array]:
    _, metrics = _direct_diagnostic_loss(state.params, state, batch, train=False)
    return metrics


def _device_summary(plan: ParallelPlan) -> dict[str, Any]:
    return {
        "parallel_mode": plan.mode,
        "num_devices": plan.num_devices,
        "local_device_count": plan.local_device_count,
        "global_batch_size": plan.global_batch_size,
        "per_device_batch_size": plan.per_device_batch_size,
        "devices": list(plan.devices),
    }


def _system_metrics(plan: ParallelPlan) -> dict[str, Any]:
    return {
        "system/device_count": plan.num_devices,
        "system/local_device_count": plan.local_device_count,
        "system/global_batch_size": plan.global_batch_size,
        "system/per_device_batch_size": plan.per_device_batch_size,
    }


def _trajectory_manifest_sha256(trajectory_ids: tuple[str, ...]) -> str:
    """Return a stable identity for an ordered trajectory selection."""

    payload = json.dumps(list(trajectory_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _portable_trajectory_ids(config: ExperimentConfig, trajectory_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Remove machine-specific roots from trajectory provenance identifiers."""

    portable = _portable_trajectory_ids_from_values(trajectory_ids)
    if len(set(portable)) != len(portable):
        raise ValueError("portable trajectory IDs collide after removing machine-specific roots")
    return portable


def _portable_trajectory_ids_from_values(trajectory_ids: tuple[str, ...]) -> tuple[str, ...]:
    portable = []
    for value in trajectory_ids:
        identifier = Path(value).name if Path(value).is_absolute() else value
        for suffix in ("_ifft_realpotens", "_ifft"):
            if identifier.endswith(suffix):
                identifier = identifier[: -len(suffix)]
                break
        portable.append(identifier)
    return tuple(portable)


def _trajectory_splits(config: ExperimentConfig, universe_ids: tuple[str, ...]) -> TrajectorySplits:
    portable_ids = _portable_trajectory_ids(config, universe_ids)
    raw_by_portable = dict(zip(portable_ids, universe_ids, strict=True))
    if len(portable_ids) == 1 and config.data.split_manifest is None:
        return TrajectorySplits(
            train=universe_ids,
            val=(),
            test=(),
            strategy="single_trajectory_fallback",
        )
    if config.data.split_manifest is None:
        portable_splits = resolve_trajectory_splits(portable_ids, seed=config.data.seed)
    else:
        portable_splits = resolve_trajectory_splits(
            portable_ids,
            seed=config.data.seed,
            manifest_path=config.data.split_manifest,
        )
    return TrajectorySplits(
        train=tuple(raw_by_portable[value] for value in portable_splits.train),
        val=tuple(raw_by_portable[value] for value in portable_splits.val),
        test=tuple(raw_by_portable[value] for value in portable_splits.test),
        strategy=portable_splits.strategy,
        manifest_path=portable_splits.manifest_path,
        manifest_sha256=portable_splits.manifest_sha256,
        fold_id=portable_splits.fold_id,
    )


def _validate_explicit_manifest_against_cache(config: ExperimentConfig) -> None:
    if config.data.split_manifest is None:
        return
    if not config.latent_cache.path or not Path(config.latent_cache.path).is_file():
        raise FileNotFoundError("explicit split manifest validation requires an existing latent cache")
    cache = LatentCacheDataset(config.latent_cache.path)
    _trajectory_splits(config, tuple(cache.trajectory_ids()))


def _protocol_fields(
    config: ExperimentConfig,
    trajectory_ids: tuple[str, ...],
    *,
    aggregation: str,
    universe_trajectory_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    universe_ids = trajectory_ids if universe_trajectory_ids is None else universe_trajectory_ids
    splits = _trajectory_splits(config, universe_ids) if len(universe_ids) >= 2 else None
    portable_selected = _portable_trajectory_ids(config, trajectory_ids)
    portable_universe = _portable_trajectory_ids(config, universe_ids)
    return {
        "protocol_version": _PROTOCOL_VERSION,
        "data_backend": config.data.backend,
        "data_split": config.data.split,
        "data_split_seed": config.data.seed,
        "training_seed": config.training.seed,
        "selected_trajectory_ids": list(portable_selected),
        "num_selected_trajectories": len(trajectory_ids),
        "trajectory_manifest_sha256": _trajectory_manifest_sha256(portable_selected),
        "universe_trajectory_ids": list(portable_universe),
        "num_universe_trajectories": len(universe_ids),
        "universe_manifest_sha256": _trajectory_manifest_sha256(portable_universe),
        "split_strategy": splits.strategy if splits else "single_trajectory",
        "split_manifest_path": Path(splits.manifest_path).name if splits and splits.manifest_path else None,
        "split_manifest_sha256": splits.manifest_sha256 if splits else None,
        "split_fold_id": splits.fold_id if splits else None,
        "train_trajectory_ids": (
            list(_portable_trajectory_ids(config, splits.train)) if splits else list(portable_universe)
        ),
        "validation_trajectory_ids": list(_portable_trajectory_ids(config, splits.val)) if splits else [],
        "test_trajectory_ids": list(_portable_trajectory_ids(config, splits.test)) if splits else [],
        "aggregation": aggregation,
    }


def _metadata_data_seed(payload: Mapping[str, Any] | None) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, Mapping) or "seed" not in data:
        return None
    return int(data["seed"])


def _checkpoint_run_config(path: str | Path) -> Mapping[str, Any] | None:
    candidate = Path(path)
    search_root = candidate.parent if candidate.is_file() else candidate
    for depth, parent in enumerate((search_root, *search_root.parents)):
        if depth > 3:
            break
        config_path = parent / "config_resolved.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, Mapping) else None
    return None


def _checkpoint_run_artifacts(
    path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    candidate = Path(path)
    search_root = candidate.parent if candidate.is_file() else candidate
    for depth, parent in enumerate((search_root, *search_root.parents)):
        if depth > 3:
            break
        config_path = parent / "config_resolved.json"
        metrics_path = parent / "metrics.json"
        if config_path.is_file() and metrics_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                config_payload = json.load(handle)
            with metrics_path.open(encoding="utf-8") as handle:
                metrics_payload = json.load(handle)
            if isinstance(config_payload, Mapping) and isinstance(metrics_payload, Mapping):
                return config_payload, metrics_payload
    return None


def _validate_checkpoint_split_seed(
    checkpoint_path: str | Path,
    *,
    expected_seed: int,
    role: str,
) -> None:
    config_payload = _checkpoint_run_config(checkpoint_path)
    actual_seed = _metadata_data_seed(config_payload)
    if actual_seed is not None and actual_seed != expected_seed:
        raise ValueError(
            f"{role} split seed {actual_seed} does not match configured data.seed {expected_seed}; "
            "all stages in one scientific protocol must use the same trajectory split"
        )


def _protocol_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _validate_checkpoint_protocol(
    checkpoint_path: str | Path,
    *,
    expected_seed: int,
    expected_backend: str,
    expected_universe_ids: tuple[str, ...],
    expected_artifact_role: str,
    role: str,
    require_complete: bool,
    expected_cache_path: str | Path | None = None,
    expected_encoder_checkpoint: str | Path | None = None,
    expected_split_manifest: str | Path | None = None,
) -> None:
    """Validate a trained checkpoint against one complete trajectory protocol."""

    artifacts = _checkpoint_run_artifacts(checkpoint_path)
    if not require_complete and expected_split_manifest is None:
        _validate_checkpoint_split_seed(checkpoint_path, expected_seed=expected_seed, role=role)
        return
    if artifacts is None:
        raise ValueError(f"{role} is missing colocated config_resolved.json and metrics.json protocol metadata")
    config_payload, metrics = artifacts
    data = config_payload.get("data")
    training = config_payload.get("training")
    if not isinstance(data, Mapping) or not isinstance(training, Mapping):
        raise ValueError(f"{role} resolved config is missing data/training mappings")
    actual_seed = _metadata_data_seed(config_payload)
    if actual_seed != expected_seed:
        raise ValueError(f"{role} split seed {actual_seed} does not match configured data.seed {expected_seed}")
    if data.get("backend") != expected_backend:
        raise ValueError(
            f"{role} backend {data.get('backend')!r} does not match configured backend {expected_backend!r}"
        )
    if data.get("split") != "train":
        raise ValueError(f"{role} must be trained with data.split='train', got {data.get('split')!r}")
    expected_portable_universe = _portable_trajectory_ids_from_values(expected_universe_ids)
    expected_splits = resolve_trajectory_splits(
        expected_portable_universe,
        seed=expected_seed,
        manifest_path=expected_split_manifest,
    )
    if expected_split_manifest is not None and not data.get("split_manifest"):
        raise ValueError(f"{role} resolved config is missing its explicit split manifest")

    required = {
        "protocol_version",
        "artifact_role",
        "data_backend",
        "data_split",
        "data_split_seed",
        "training_seed",
        "selected_trajectory_ids",
        "trajectory_manifest_sha256",
        "universe_trajectory_ids",
        "universe_manifest_sha256",
        "checkpoint",
    }
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"{role} metrics are missing protocol fields: {', '.join(missing)}")
    if metrics.get("protocol_version") != _PROTOCOL_VERSION:
        raise ValueError(f"{role} has unsupported protocol version {metrics.get('protocol_version')!r}")
    if metrics.get("artifact_role") != expected_artifact_role:
        raise ValueError(
            f"{role} artifact role {metrics.get('artifact_role')!r} does not match {expected_artifact_role!r}"
        )
    if metrics.get("data_backend") != expected_backend:
        raise ValueError(f"{role} metrics backend does not match its resolved config")
    if metrics.get("data_split") != "train" or metrics.get("data_split_seed") != expected_seed:
        raise ValueError(f"{role} metrics do not record the required seed-{expected_seed} training split")
    if metrics.get("training_seed") != training.get("seed"):
        raise ValueError(f"{role} training seed differs between resolved config and metrics")

    universe_ids = _protocol_tuple(metrics, "universe_trajectory_ids")
    selected_ids = _protocol_tuple(metrics, "selected_trajectory_ids")
    if universe_ids != expected_portable_universe:
        raise ValueError(f"{role} trajectory universe does not match the consuming protocol")
    if metrics.get("universe_manifest_sha256") != _trajectory_manifest_sha256(expected_portable_universe):
        raise ValueError(f"{role} trajectory-universe manifest is invalid")
    expected_train_ids = expected_splits.train
    if selected_ids != expected_train_ids:
        raise ValueError(f"{role} selected trajectory IDs do not match the canonical training split")
    if metrics.get("trajectory_manifest_sha256") != _trajectory_manifest_sha256(expected_train_ids):
        raise ValueError(f"{role} training trajectory manifest is invalid")
    if metrics.get("split_manifest_sha256") != expected_splits.manifest_sha256:
        raise ValueError(f"{role} split-manifest lineage does not match the consuming fold")
    if Path(str(metrics["checkpoint"])).resolve() != Path(checkpoint_path).resolve():
        raise ValueError(f"{role} metrics checkpoint path does not identify the loaded checkpoint")
    if expected_artifact_role == "sequence_checkpoint":
        latent_cache_config = config_payload.get("latent_cache")
        if not isinstance(latent_cache_config, Mapping):
            raise ValueError(f"{role} resolved config is missing latent-cache lineage")
        if expected_cache_path is None or expected_encoder_checkpoint is None:
            raise ValueError(f"{role} validation requires cache and encoder lineage")
        configured_cache = latent_cache_config.get("path")
        configured_encoder = latent_cache_config.get("encoder_checkpoint_path")
        if (
            not configured_cache
            or not configured_encoder
            or Path(str(configured_cache)).resolve() != Path(expected_cache_path).resolve()
            or Path(str(configured_encoder)).resolve() != Path(expected_encoder_checkpoint).resolve()
        ):
            raise ValueError(f"{role} resolved config does not match the loaded cache/encoder lineage")
        if (
            Path(str(metrics.get("latent_cache", ""))).resolve() != Path(expected_cache_path).resolve()
            or Path(str(metrics.get("encoder_checkpoint", ""))).resolve()
            != Path(expected_encoder_checkpoint).resolve()
        ):
            raise ValueError(f"{role} metrics do not match the loaded cache/encoder lineage")
        if metrics.get("latent_cache_sha256") != _sha256_file(expected_cache_path):
            raise ValueError(f"{role} latent-cache content hash does not match the loaded cache")
        if metrics.get("encoder_checkpoint_sha256") != _sha256_file(
            _checkpoint_pickle(expected_encoder_checkpoint)
        ):
            raise ValueError(f"{role} encoder-checkpoint content hash does not match cache lineage")


def _latent_cache_run_config(path: str | Path) -> Mapping[str, Any] | None:
    with h5py.File(path, "r") as handle:
        raw = handle["metadata"].attrs.get("config_yaml", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return None
    payload = yaml.safe_load(str(raw))
    return payload if isinstance(payload, Mapping) else None


def _latent_cache_protocol(path: str | Path) -> Mapping[str, Any] | None:
    with h5py.File(path, "r") as handle:
        raw = handle["metadata"].attrs.get("protocol_json", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return None
    payload = json.loads(str(raw))
    return payload if isinstance(payload, Mapping) and payload else None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_pickle(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "checkpoint.pkl" if candidate.is_dir() else candidate


def _checkpoint_sidecar(path: str | Path, name: str) -> Path:
    candidate = Path(path)
    search_root = candidate.parent if candidate.is_file() else candidate
    for depth, parent in enumerate((search_root, *search_root.parents)):
        if depth > 3:
            break
        sidecar = parent / name
        if sidecar.is_file():
            return sidecar
    raise FileNotFoundError(f"checkpoint {path} is missing required {name} lineage artifact")


def _cache_encoder_checkpoint(path: str | Path) -> Path:
    """Resolve encoder lineage from cache metadata and verify its content hash.

    The consuming config is deliberately not a source of truth here. It may only
    agree with the cache lineage checked by ``_validate_cache_encoder_lineage``.
    """

    protocol = _latent_cache_protocol(path)
    config_payload = _latent_cache_run_config(path)
    with h5py.File(path, "r") as handle:
        raw_attr = handle["metadata"].attrs.get("encoder_checkpoint_path", "")
    if isinstance(raw_attr, bytes):
        raw_attr = raw_attr.decode("utf-8")
    config_cache = config_payload.get("latent_cache") if isinstance(config_payload, Mapping) else None
    configured = config_cache.get("encoder_checkpoint_path") if isinstance(config_cache, Mapping) else None
    protocol_path = protocol.get("encoder_checkpoint") if isinstance(protocol, Mapping) else None
    candidates = [str(value) for value in (raw_attr, configured, protocol_path) if value]
    if not candidates:
        raise ValueError("latent cache is missing authoritative encoder-checkpoint lineage")
    cache_parent = Path(path).resolve().parent
    resolved = {
        (Path(value) if Path(value).is_absolute() else cache_parent / value).resolve()
        for value in candidates
    }
    if len(resolved) != 1:
        raise ValueError("latent cache metadata disagrees about its encoder-checkpoint lineage")
    checkpoint = next(iter(resolved))
    checkpoint_file = _checkpoint_pickle(checkpoint)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"latent cache encoder checkpoint not found: {checkpoint}")
    expected_hash = protocol.get("encoder_checkpoint_sha256") if isinstance(protocol, Mapping) else None
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("latent cache protocol is missing encoder_checkpoint_sha256")
    if _sha256_file(checkpoint_file) != expected_hash:
        raise ValueError("latent cache encoder-checkpoint hash does not match the stored lineage")
    return checkpoint


def _validate_cache_encoder_lineage(config: ExperimentConfig, path: str | Path) -> Path:
    checkpoint = _cache_encoder_checkpoint(path)
    configured = config.latent_cache.encoder_checkpoint_path
    if configured and Path(configured).resolve() != checkpoint:
        raise ValueError("configured encoder checkpoint does not match the latent cache lineage")
    return checkpoint


def _optional_cache_encoder_lineage(config: ExperimentConfig, path: str | Path) -> Path | None:
    protocol = _latent_cache_protocol(path)
    with h5py.File(path, "r") as handle:
        raw = handle["metadata"].attrs.get("encoder_checkpoint_path", "")
    if not raw and not (isinstance(protocol, Mapping) and protocol.get("encoder_checkpoint")):
        return None
    return _validate_cache_encoder_lineage(config, path)


def _validate_latent_cache_split_seed(path: str | Path, *, expected_seed: int) -> None:
    cache_config = _latent_cache_run_config(path)
    actual_seed = _metadata_data_seed(cache_config)
    if actual_seed is not None and actual_seed != expected_seed:
        raise ValueError(
            f"latent cache split seed {actual_seed} does not match configured data.seed {expected_seed}; "
            "rebuild or select a cache from the same trajectory-split protocol"
        )
    latent_cache = cache_config.get("latent_cache") if isinstance(cache_config, Mapping) else None
    encoder_checkpoint = latent_cache.get("encoder_checkpoint_path") if isinstance(latent_cache, Mapping) else None
    if encoder_checkpoint:
        _validate_checkpoint_split_seed(
            str(encoder_checkpoint),
            expected_seed=expected_seed,
            role="latent cache encoder checkpoint",
        )


def _validate_latent_cache_protocol(
    path: str | Path,
    *,
    expected_seed: int,
    expected_backend: str,
    require_complete: bool,
    expected_split_manifest: str | Path | None = None,
) -> tuple[str, ...]:
    cache_config = _latent_cache_run_config(path)
    protocol = _latent_cache_protocol(path)
    cache = LatentCacheDataset(path)
    actual_ids = tuple(cache.trajectory_ids())
    if not require_complete and expected_split_manifest is None:
        _validate_latent_cache_split_seed(path, expected_seed=expected_seed)
        return actual_ids
    if cache_config is None or protocol is None:
        raise ValueError("real-data latent cache is missing resolved config or protocol metadata; rebuild it")
    data = cache_config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("real-data latent cache config is missing its data mapping")
    if data.get("backend") != expected_backend or _metadata_data_seed(cache_config) != expected_seed:
        raise ValueError("real-data latent cache backend/seed does not match the consuming protocol")
    if data.get("split") != "all":
        raise ValueError(f"real-data latent cache must be embedded with data.split='all', got {data.get('split')!r}")
    portable_actual_ids = _portable_trajectory_ids_from_values(actual_ids)
    expected_splits = resolve_trajectory_splits(
        portable_actual_ids,
        seed=expected_seed,
        manifest_path=expected_split_manifest,
    )
    if expected_split_manifest is not None and not data.get("split_manifest"):
        raise ValueError("latent cache resolved config is missing its explicit split manifest")
    required = {
        "protocol_version",
        "artifact_role",
        "data_backend",
        "data_split",
        "data_split_seed",
        "selected_trajectory_ids",
        "trajectory_manifest_sha256",
        "universe_trajectory_ids",
        "universe_manifest_sha256",
        "encoder_checkpoint",
    }
    missing = sorted(required.difference(protocol))
    if missing:
        raise ValueError(f"real-data latent cache protocol is missing fields: {', '.join(missing)}")
    if protocol.get("protocol_version") != _PROTOCOL_VERSION or protocol.get("artifact_role") != "latent_cache":
        raise ValueError("real-data latent cache has an unsupported protocol identity")
    if (
        protocol.get("data_backend") != expected_backend
        or protocol.get("data_split") != "all"
        or protocol.get("data_split_seed") != expected_seed
    ):
        raise ValueError("real-data latent cache protocol disagrees with its resolved config")
    selected_ids = _protocol_tuple(protocol, "selected_trajectory_ids")
    universe_ids = _protocol_tuple(protocol, "universe_trajectory_ids")
    actual_manifest = _trajectory_manifest_sha256(portable_actual_ids)
    if selected_ids != portable_actual_ids or universe_ids != portable_actual_ids:
        raise ValueError("real-data latent cache trajectory IDs differ from its stored protocol universe")
    if (
        protocol.get("trajectory_manifest_sha256") != actual_manifest
        or protocol.get("universe_manifest_sha256") != actual_manifest
    ):
        raise ValueError("real-data latent cache trajectory manifest is invalid")
    if protocol.get("split_manifest_sha256") != expected_splits.manifest_sha256:
        raise ValueError("latent cache split-manifest lineage does not match the consuming fold")
    latent_cache = cache_config.get("latent_cache")
    configured_encoder = latent_cache.get("encoder_checkpoint_path") if isinstance(latent_cache, Mapping) else None
    if not configured_encoder or protocol.get("encoder_checkpoint") != configured_encoder:
        raise ValueError("real-data latent cache does not identify one consistent encoder checkpoint")
    configured_encoder_path = Path(str(configured_encoder))
    if not configured_encoder_path.is_absolute():
        configured_encoder_path = Path(path).resolve().parent / configured_encoder_path
    _validate_checkpoint_protocol(
        configured_encoder_path,
        expected_seed=expected_seed,
        expected_backend=expected_backend,
        expected_universe_ids=actual_ids,
        expected_artifact_role="encoder_checkpoint",
        role="latent cache encoder checkpoint",
        require_complete=True,
        expected_split_manifest=expected_split_manifest,
    )
    return actual_ids


def _selected_trajectory_ids(config: ExperimentConfig) -> tuple[str, ...]:
    dataset = _build_universe_dataset(config)
    ids = tuple(dataset.trajectory_ids())
    return _selected_ids_from_universe(config, ids)


def _build_universe_dataset(config: ExperimentConfig) -> Any:
    data = config.data.model_copy(update={"split": "all"})
    dataset = build_dataset(data)
    requested = config.data.cyclone.trajectories if config.data.cyclone is not None else None
    if requested:
        actual = _portable_trajectory_ids(config, tuple(dataset.trajectory_ids()))
        expected = _portable_trajectory_ids(config, tuple(str(value) for value in requested))
        missing = tuple(value for value in expected if value not in set(actual))
        if missing:
            raise ValueError(f"explicit requested trajectories are missing from dataset: {', '.join(missing)}")
        if set(actual) - set(expected):
            raise ValueError("dataset returned trajectories outside the explicit requested trajectory list")
    return dataset


def _selected_ids_from_universe(config: ExperimentConfig, ids: tuple[str, ...]) -> tuple[str, ...]:
    if config.data.split == "all" or len(ids) < 2:
        return ids
    return _trajectory_splits(config, ids).as_dict()[config.data.split]


def _requires_complete_protocol(config: ExperimentConfig) -> bool:
    return config.data.backend == "cyclone_kvikio"


def _normalization_stats(config: ExperimentConfig) -> NormalizationStats | None:
    mode = config.data.normalization.mode
    if mode in {"none", "sample", "trajectory"}:
        return None
    if mode == "fixed":
        return NormalizationStats(
            mean=np.asarray(config.data.normalization.mean, dtype=np.float32),
            std=np.asarray(config.data.normalization.std, dtype=np.float32),
        )
    dataset = _build_universe_dataset(config)
    trajectory_ids = _selected_trajectory_ids(config)
    return estimate_dataset_stats(
        dataset,
        trajectory_ids=trajectory_ids,
        max_samples=config.data.normalization.max_samples,
    )


def _trajectory_normalization_stats(config: ExperimentConfig, ids: tuple[str, ...]) -> dict[str, NormalizationStats]:
    if config.data.normalization.mode != "trajectory":
        return {}
    dataset = _build_universe_dataset(config)
    return {trajectory_id: estimate_trajectory_stats(dataset, trajectory_id) for trajectory_id in ids}


def _snapshot_batches(
    config: ExperimentConfig,
    *,
    repeat: bool = True,
    trajectory_ids: tuple[str, ...] | None = None,
    normalization_stats: NormalizationStats | None | object = _UNSET,
) -> Iterator[dict[str, Any]]:
    dataset = _build_universe_dataset(config)
    ids = _selected_trajectory_ids(config) if trajectory_ids is None else trajectory_ids
    stats = _normalization_stats(config) if normalization_stats is _UNSET else normalization_stats
    if stats is not None and not isinstance(stats, NormalizationStats):
        raise TypeError("normalization_stats must be NormalizationStats or None")
    trajectory_stats = _trajectory_normalization_stats(config, ids)
    index = 0
    rng = np.random.default_rng(config.data.seed)
    order = [
        (trajectory_id, timestep) for trajectory_id in ids for timestep in range(dataset.num_timesteps(trajectory_id))
    ]
    while True:
        if config.data.shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), config.data.batch_size):
            selected = order[start : start + config.data.batch_size]
            if not selected:
                continue
            samples = [dataset.get_snapshot(trajectory_id, timestep) for trajectory_id, timestep in selected]
            if config.data.normalization.mode != "none":
                samples = [
                    sample.__class__(
                        x=normalize_snapshot(
                            sample.x,
                            mode=config.data.normalization.mode,
                            stats=trajectory_stats.get(sample.trajectory_id, stats),
                        ),
                        targets=sample.targets,
                        trajectory_id=sample.trajectory_id,
                        trajectory_index=sample.trajectory_index,
                        timestep_index=sample.timestep_index,
                        physical_time=sample.physical_time,
                        metadata=sample.metadata,
                    )
                    for sample in samples
                ]
            batch = collate_snapshots(samples)
            yield {
                "x": batch.x,
                "flux": batch.flux,
                "spectra": batch.spectra,
                "trajectory_index": batch.trajectory_index,
                "timestep_index": batch.timestep_index,
            }
            index += 1
        if not repeat:
            return


def _init_encoder_state(config: ExperimentConfig, *, simsiam: bool = False) -> tuple[TrainState, Any]:
    model = (
        build_simsiam_encoder_with_diagnostics(config.model)
        if simsiam
        else build_encoder_with_diagnostics(config.model)
    )
    dataset = _build_universe_dataset(config)
    sample = dataset.get_snapshot(dataset.trajectory_ids()[0], 0)
    x = jnp.asarray(sample.x[None, ...], dtype=jnp.float32)
    rng = jax.random.PRNGKey(config.training.seed)
    variables = model.init(rng, x, train=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=build_optimizer(config.training),
        rng=rng,
        model_config={
            **_loss_config(config),
            "simsiam_weight": config.loss.simsiam_weight if simsiam else 0.0,
            "augmentations": config.data.augmentations.model_dump(mode="json"),
        },
    )
    return state, model


def train_encoder(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    plan = _parallel_plan(config)
    train_config = _config_with_parallel_optimizer(config, plan)
    state, _model = _init_encoder_state(train_config, simsiam=config.loss.simsiam_weight > 0)
    first_batch = next(_snapshot_batches(config, repeat=False))
    if dry_run or config.training.max_steps == 0:
        return {
            "dry_run": True,
            "input_shape": list(first_batch["x"].shape),
            "selected_trajectories": list(_selected_trajectory_ids(config)),
            **_device_summary(plan),
        }

    out = _output_dir(config)
    write_run_metadata(out, config=train_config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(train_config), encoding="utf-8")
    if config.parallel.log_device_summary:
        write_device_report(out, config=config.parallel, plan=plan)
    training_stats = _normalization_stats(config)
    if config.data.normalization.mode in {"dataset", "trajectory", "fixed"} and training_stats is not None:
        training_stats.save_npz(out / "normalization_stats.npz")

    logger = _metrics_logger(out, train_config)
    metrics: Mapping[str, Any] = {}
    state = _replicated_if_needed(state, plan)
    step_fn = make_pmap_encoder_train_step(plan.axis_name) if plan.uses_pmap else train_encoder_step
    eval_fn = make_pmap_encoder_eval_step(plan.axis_name) if plan.uses_pmap else eval_encoder_step
    batches = _snapshot_batches(config, repeat=True, normalization_stats=training_stats)
    dataset = _build_universe_dataset(config)
    universe_ids = tuple(dataset.trajectory_ids())
    validation_ids = _trajectory_splits(config, universe_ids).val if len(universe_ids) > 1 else ()
    if validation_ids and config.data.normalization.mode == "trajectory":
        raise ValueError("trajectory normalization cannot be fit independently on held-out validation trajectories")
    train_totals: dict[str, float] = {}
    train_examples = 0
    best_metric = float("inf")
    best_step: int | None = None
    best_checkpoint: Path | None = None
    latest_validation: dict[str, float] = {}
    step_value = 0
    while step_value < config.training.max_steps:
        raw_batch = next(batches)
        raw_batch_size = _batch_size(raw_batch)
        batch = _prepare_parallel_batch(raw_batch, plan)
        if batch is None:
            continue
        state, metrics = step_fn(state, batch)
        host_state = _host_state(state, plan)
        host_metrics = _host_metrics(metrics, plan)
        step_value = int(host_state.step)
        for key, value in host_metrics.items():
            train_totals[key] = train_totals.get(key, 0.0) + float(value) * raw_batch_size
        train_examples += raw_batch_size
        row = {
            "step": step_value,
            "lr": _scheduled_learning_rate(train_config, step_value),
            **{k: float(v) for k, v in host_metrics.items()},
        }
        if step_value % config.training.log_every == 0 or step_value == config.training.max_steps:
            logger.log(row, prefix="train")
        if step_value % config.training.checkpoint_every == 0:
            save_checkpoint(host_state, out, step=step_value)
        should_validate = bool(validation_ids) and (
            step_value % config.training.eval_every == 0 or step_value == config.training.max_steps
        )
        if should_validate:
            latest_validation = _aggregate_eval_metrics_by_trajectory(
                state,
                validation_ids,
                batches_for_trajectory=lambda trajectory_id: _snapshot_batches(
                    config,
                    repeat=False,
                    trajectory_ids=(trajectory_id,),
                    normalization_stats=training_stats,
                ),
                plan=plan,
                step_fn=eval_fn,
            )
            if config.data.target_flux and config.loss.flux_weight > 0.0:
                primary_name = "flux_rmse"
                primary_value = float(np.sqrt(max(latest_validation["flux_loss"], 0.0)))
                latest_validation[primary_name] = primary_value
            else:
                primary_name = "loss"
                primary_value = latest_validation[primary_name]
            logger.log({"step": step_value, **latest_validation}, prefix="validation")
            if primary_value < best_metric:
                best_metric = primary_value
                best_step = step_value
                best_checkpoint = save_checkpoint(host_state, out, step=step_value)
    final_state = _host_state(state, plan)
    metrics = _host_metrics(metrics, plan)
    final_checkpoint = save_checkpoint(final_state, out, step=int(final_state.step))
    ckpt = best_checkpoint or final_checkpoint
    if best_step is None:
        best_step = int(final_state.step)
    selected_ids = _selected_ids_from_universe(config, universe_ids)
    aggregate_metrics = {
        key: value / train_examples for key, value in train_totals.items()
    }
    summary = {
        "artifact_role": "encoder_checkpoint",
        "step": int(final_state.step),
        "checkpoint": str(ckpt),
        "checkpoint_step": best_step,
        "checkpoint_sha256": _sha256_file(_checkpoint_pickle(ckpt)),
        "checkpoint_selection": "minimum_validation_flux_rmse" if validation_ids and config.data.target_flux else (
            "minimum_validation_loss" if validation_ids else "final_step_no_validation_split"
        ),
        "best_validation_metric": None if not np.isfinite(best_metric) else best_metric,
        "last_minibatch": {k: float(v) for k, v in metrics.items()},
        "train_aggregate": aggregate_metrics,
        "validation": latest_validation,
        "actual_learning_rate": _scheduled_learning_rate(train_config, int(final_state.step)),
        **_protocol_fields(
            config,
            selected_ids,
            aggregation="snapshot_minibatch_training",
            universe_trajectory_ids=universe_ids,
        ),
        **_device_summary(plan),
        **_system_metrics(plan),
        **{k: float(v) for k, v in metrics.items()},
    }
    if logger.wandb_status().get("requested"):
        summary["wandb"] = logger.wandb_status()
    logger.write_summary(summary)
    return summary


def train_direct_diagnostics(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Train the same-time snapshot-to-diagnostics control on train trajectories.

    Checkpoint selection and reported metrics use only the canonical validation
    split. This command deliberately does not inspect the test split.
    """

    dataset = _build_universe_dataset(config)
    universe_ids = tuple(dataset.trajectory_ids())
    if len(universe_ids) < 2:
        raise ValueError("direct diagnostic baseline requires at least two trajectories for held-out validation")
    splits = _trajectory_splits(config, universe_ids)
    train_ids = splits.train
    validation_ids = splits.val
    if not train_ids or not validation_ids:
        raise ValueError("direct diagnostic baseline requires non-empty train and validation trajectory splits")
    sample = dataset.get_snapshot(train_ids[0], 0)
    first_x = jnp.asarray(sample.x[None, ...], dtype=jnp.float32)
    model = build_direct_diagnostic_baseline(config.model.diagnostics)
    rng = jax.random.PRNGKey(config.training.seed)
    variables = model.init(rng, first_x, train=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=build_optimizer(config.training),
        rng=rng,
        model_config=_loss_config(config),
    )
    if dry_run or config.training.max_steps == 0:
        return {
            "dry_run": True,
            "baseline": "direct_snapshot_diagnostics",
            "input_shape": list(first_x.shape),
            "train_trajectory_ids": list(train_ids),
            "validation_trajectory_ids": list(validation_ids),
            "test_split_inspected": False,
        }
    mode = config.data.normalization.mode
    if mode == "trajectory":
        raise ValueError(
            "direct diagnostic baseline does not fit trajectory-specific stats on held-out validation data"
        )
    if mode == "dataset":
        normalization_stats = estimate_dataset_stats(
            dataset,
            trajectory_ids=train_ids,
            max_samples=config.data.normalization.max_samples,
        )
    else:
        normalization_stats = _normalization_stats(config)

    out = _output_dir(config)
    write_run_metadata(out, config=config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    if normalization_stats is not None:
        normalization_stats.save_npz(out / "normalization_stats.npz")
    logger = _metrics_logger(out, config)
    batches = _snapshot_batches(
        config,
        repeat=True,
        trajectory_ids=train_ids,
        normalization_stats=normalization_stats,
    )
    train_totals: dict[str, float] = {}
    train_examples = 0
    last_metrics: Mapping[str, Any] = {}
    best_metric = float("inf")
    best_checkpoint: Path | None = None
    best_step: int | None = None
    validation_metrics: dict[str, float] = {}
    while int(state.step) < config.training.max_steps:
        batch = next(batches)
        batch_size = _batch_size(batch)
        state, last_metrics = _train_direct_diagnostic_step(state, batch)
        step = int(state.step)
        for key, value in last_metrics.items():
            train_totals[key] = train_totals.get(key, 0.0) + float(value) * batch_size
        train_examples += batch_size
        if step % config.training.log_every == 0 or step == config.training.max_steps:
            logger.log(
                {
                    "step": step,
                    "lr": _scheduled_learning_rate(config, step),
                    **{key: float(value) for key, value in last_metrics.items()},
                },
                prefix="train",
            )
        if step % config.training.checkpoint_every == 0:
            save_checkpoint(state, out, step=step)
        if step % config.training.eval_every == 0 or step == config.training.max_steps:
            validation_metrics = _aggregate_eval_metrics_by_trajectory(
                state,
                validation_ids,
                batches_for_trajectory=lambda trajectory_id: _snapshot_batches(
                    config,
                    repeat=False,
                    trajectory_ids=(trajectory_id,),
                    normalization_stats=normalization_stats,
                ),
                plan=resolve_parallel_mode(
                    config.parallel.model_copy(update={"mode": "single"}),
                    batch_size=config.data.batch_size,
                ),
                step_fn=_eval_direct_diagnostic_step,
            )
            if "flux_mse" in validation_metrics:
                validation_metrics["flux_rmse"] = float(np.sqrt(max(validation_metrics["flux_mse"], 0.0)))
                selection_value = validation_metrics["flux_rmse"]
                selection_name = "validation_flux_rmse"
            else:
                selection_value = validation_metrics["diagnostics"]
                selection_name = "validation_diagnostic_loss"
            logger.log({"step": step, **validation_metrics}, prefix="validation")
            if selection_value < best_metric:
                best_metric = selection_value
                best_step = step
                best_checkpoint = save_checkpoint(state, out, step=step)

    final_checkpoint = save_checkpoint(state, out, step=int(state.step))
    selected_checkpoint = best_checkpoint or final_checkpoint
    summary = {
        "artifact_role": "direct_diagnostic_checkpoint",
        "baseline": "same_time_snapshot_to_diagnostics",
        "checkpoint": str(selected_checkpoint),
        "checkpoint_step": best_step or int(state.step),
        "checkpoint_sha256": _sha256_file(_checkpoint_pickle(selected_checkpoint)),
        "checkpoint_selection": selection_name,
        "best_validation_metric": best_metric,
        "train_trajectory_ids": list(train_ids),
        "validation_trajectory_ids": list(validation_ids),
        "test_split_inspected": False,
        "last_minibatch": {key: float(value) for key, value in last_metrics.items()},
        "train_aggregate": {key: value / train_examples for key, value in train_totals.items()},
        "validation": validation_metrics,
        "actual_learning_rate": _scheduled_learning_rate(config, int(state.step)),
        **_protocol_fields(
            config,
            train_ids,
            aggregation="trajectory_balanced_validation",
            universe_trajectory_ids=universe_ids,
        ),
    }
    if logger.wandb_status().get("requested"):
        summary["wandb"] = logger.wandb_status()
    logger.write_summary(summary)
    return summary


def _block_until_ready(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item, value
    )


def benchmark_step_time(
    config: ExperimentConfig,
    *,
    dry_run: bool = False,
    measured_steps: int = 3,
) -> dict[str, Any]:
    plan = _parallel_plan(config)
    train_config = _config_with_parallel_optimizer(config, plan)
    state, _model = _init_encoder_state(train_config, simsiam=config.loss.simsiam_weight > 0)
    batch = next(_snapshot_batches(config, repeat=True))
    devices = [str(device) for device in jax.devices()]
    if dry_run:
        return {
            "dry_run": True,
            "benchmark": "encoder_train_step",
            "devices": devices,
            "backend": jax.default_backend(),
            "input_shape": list(batch["x"].shape),
            "measured_steps": measured_steps,
            **_device_summary(plan),
        }

    state = _replicated_if_needed(state, plan)
    step_fn = make_pmap_encoder_train_step(plan.axis_name) if plan.uses_pmap else train_encoder_step
    prepared_batch = _prepare_parallel_batch(batch, plan)
    if prepared_batch is None:
        raise ValueError("first benchmark batch was dropped by parallel batch preparation")
    start = perf_counter()
    state, metrics = step_fn(state, prepared_batch)
    _block_until_ready((state.params, metrics))
    first_step_seconds = perf_counter() - start

    step_seconds = []
    batches = _snapshot_batches(config, repeat=True)
    for _ in range(max(1, measured_steps)):
        batch = next(batches)
        prepared_batch = _prepare_parallel_batch(batch, plan)
        if prepared_batch is None:
            continue
        start = perf_counter()
        state, metrics = step_fn(state, prepared_batch)
        _block_until_ready((state.params, metrics))
        step_seconds.append(perf_counter() - start)

    metrics = _host_metrics(metrics, plan)
    losses = {key: float(value) for key, value in metrics.items()}
    return {
        "benchmark": "encoder_train_step",
        "devices": devices,
        "backend": jax.default_backend(),
        "input_shape": list(batch["x"].shape),
        "first_step_seconds": first_step_seconds,
        "mean_step_seconds": float(np.mean(step_seconds)),
        "min_step_seconds": float(np.min(step_seconds)),
        "max_step_seconds": float(np.max(step_seconds)),
        "measured_steps": len(step_seconds),
        **_device_summary(plan),
        **losses,
    }


def _encoder_params_for_embedding(
    config: ExperimentConfig,
    *,
    expected_universe_ids: tuple[str, ...],
) -> tuple[Any, Any, Path]:
    state, model = _init_encoder_state(config)
    del state
    checkpoint_path = config.latent_cache.encoder_checkpoint_path
    ckpt = Path(checkpoint_path) if checkpoint_path else latest_checkpoint(config.output_dir)
    if ckpt is None:
        raise FileNotFoundError("embed-dataset requires latent_cache.encoder_checkpoint_path or an existing checkpoint")
    if not ckpt.exists():
        raise FileNotFoundError(f"encoder checkpoint not found: {ckpt}")
    _validate_checkpoint_protocol(
        ckpt,
        expected_seed=config.data.seed,
        expected_backend=config.data.backend,
        expected_universe_ids=expected_universe_ids,
        expected_artifact_role="encoder_checkpoint",
        role="encoder checkpoint",
        require_complete=_requires_complete_protocol(config),
        expected_split_manifest=config.data.split_manifest,
    )
    payload = load_checkpoint(ckpt)
    return model, _encoder_apply_params(payload["params"]), ckpt.resolve()


def _load_normalization_stats(path: str | Path) -> NormalizationStats:
    with np.load(path) as payload:
        if "mean" not in payload or "std" not in payload:
            raise ValueError(f"normalization artifact is missing mean/std arrays: {path}")
        return NormalizationStats(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def _embedding_normalization(
    config: ExperimentConfig,
    checkpoint: Path,
) -> tuple[NormalizationStats | None, dict[str, NormalizationStats], dict[str, Any]]:
    mode = config.data.normalization.mode
    metadata: dict[str, Any] = {"snapshot_normalization_mode": mode}
    if mode in {"none", "sample"}:
        return None, {}, metadata
    if mode == "trajectory":
        raise ValueError(
            "trajectory normalization cannot be re-fit while embedding held-out trajectories; "
            "use training-derived dataset/fixed stats or disable normalization"
        )
    if mode == "fixed":
        return _normalization_stats(config), {}, metadata
    artifact = _checkpoint_sidecar(checkpoint, "normalization_stats.npz")
    metadata.update(
        {
            "snapshot_normalization_stats": str(artifact.resolve()),
            "snapshot_normalization_sha256": _sha256_file(artifact),
        }
    )
    return _load_normalization_stats(artifact), {}, metadata


def _latent_cache_config_yaml(config: ExperimentConfig, *, encoder_checkpoint: str) -> str:
    """Serialize portable cache provenance without machine-specific roots."""

    payload = config.model_dump(mode="json")
    data = payload["data"]
    data["root"] = None
    cyclone = data.get("cyclone")
    if isinstance(cyclone, dict) and isinstance(cyclone.get("trajectories"), list):
        cyclone["trajectories"] = [Path(value).name for value in cyclone["trajectories"]]
    payload["output_dir"] = Path(str(payload["output_dir"])).name
    latent_cache = payload["latent_cache"]
    latent_cache["path"] = Path(str(latent_cache["path"])).name if latent_cache.get("path") else None
    latent_cache["encoder_checkpoint_path"] = encoder_checkpoint
    latent_cache["sequence_checkpoint_path"] = None
    if data.get("split_manifest"):
        data["split_manifest"] = Path(str(data["split_manifest"])).name
    return yaml.safe_dump(payload, sort_keys=False)


def _encoder_apply_params(params: Any) -> Any:
    """Return params compatible with ``EncoderWithDiagnostics`` apply.

    Supervised encoder checkpoints already use this tree. SimSiam checkpoints add
    projection and prediction heads; those are intentionally omitted for embedding.
    """

    if not isinstance(params, Mapping) or "encoder" not in params:
        return params
    apply_params: dict[str, Any] = {"encoder": params["encoder"]}
    if "diagnostic_heads" in params:
        apply_params["diagnostic_heads"] = params["diagnostic_heads"]
    return apply_params


def _diagnostic_params_from_encoder_params(params: Any) -> Any | None:
    if isinstance(params, Mapping):
        return params.get("diagnostic_heads")
    return None


def embed_dataset(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    dataset = _build_universe_dataset(config)
    trajectory_ids = tuple(dataset.trajectory_ids())
    _trajectory_splits(config, trajectory_ids)
    if dry_run:
        return {
            "dry_run": True,
            "planned_latent_cache": config.latent_cache.path or str(Path(config.output_dir) / "latent_cache.h5"),
            "trajectories": len(trajectory_ids),
        }

    out = _output_dir(config)
    write_run_metadata(out, config=config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    model, params, encoder_checkpoint = _encoder_params_for_embedding(
        config, expected_universe_ids=trajectory_ids
    )
    cache_path = Path(config.latent_cache.path) if config.latent_cache.path else out / "latent_cache.h5"
    encoder_checkpoint_path = os.path.relpath(encoder_checkpoint, start=cache_path.resolve().parent)
    stats, trajectory_stats, normalization_lineage = _embedding_normalization(config, encoder_checkpoint)
    if "snapshot_normalization_stats" in normalization_lineage:
        normalization_lineage["snapshot_normalization_stats"] = Path(
            str(normalization_lineage["snapshot_normalization_stats"])
        ).name
    cache_protocol = {
        "artifact_role": "latent_cache",
        "encoder_checkpoint": encoder_checkpoint_path,
        "encoder_checkpoint_sha256": _sha256_file(_checkpoint_pickle(encoder_checkpoint)),
        **normalization_lineage,
        **_protocol_fields(
            config,
            trajectory_ids,
            aggregation="per_snapshot_embedding",
            universe_trajectory_ids=trajectory_ids,
        ),
    }
    if cache_protocol.get("split_manifest_path"):
        cache_protocol["split_manifest_path"] = Path(str(cache_protocol["split_manifest_path"])).name
    writer = LatentCacheWriter(
        cache_path,
        latent_dim=config.model.encoder.latent_dim,
        config_yaml=_latent_cache_config_yaml(config, encoder_checkpoint=encoder_checkpoint_path),
        encoder_checkpoint_path=encoder_checkpoint_path,
        protocol_metadata=cache_protocol,
    )
    for trajectory_id in trajectory_ids:
        portable_trajectory_id = _portable_trajectory_ids(config, (trajectory_id,))[0]
        snapshots = []
        flux_rows = []
        spectra_rows: dict[str, list[np.ndarray]] = {key: [] for key in config.data.target_spectra}
        times = []
        for timestep in range(dataset.num_timesteps(trajectory_id)):
            sample = dataset.get_snapshot(trajectory_id, timestep)
            x = sample.x
            if config.data.normalization.mode != "none":
                x = normalize_snapshot(
                    x,
                    mode=config.data.normalization.mode,
                    stats=trajectory_stats.get(sample.trajectory_id, stats),
                )
            snapshots.append(x)
            if sample.targets.flux is not None:
                flux_rows.append(np.asarray(sample.targets.flux, dtype=np.float32))
            for key in spectra_rows:
                spectra_rows[key].append(np.asarray(sample.targets.spectra[key], dtype=np.float32))
            times.append(float(sample.physical_time if sample.physical_time is not None else timestep))
        z = encode_snapshots(
            model.apply,
            params,
            np.asarray(snapshots, dtype=np.float32),
            batch_size=config.data.batch_size,
        )
        writer.write_trajectory(
            portable_trajectory_id,
            z,
            physical_time=np.asarray(times, dtype=np.float32),
            flux=np.asarray(flux_rows, dtype=np.float32) if flux_rows else None,
            spectra={key: np.asarray(rows, dtype=np.float32) for key, rows in spectra_rows.items()},
        )
    summary = {
        "artifact_role": "latent_cache",
        "latent_cache": str(cache_path),
        "encoder_checkpoint": encoder_checkpoint_path,
        "trajectories": len(trajectory_ids),
        "embedding_batch_size": config.data.batch_size,
        **_protocol_fields(
            config,
            trajectory_ids,
            aggregation="per_snapshot_embedding",
            universe_trajectory_ids=trajectory_ids,
        ),
    }
    write_json(out / "metrics.json", summary)
    return summary


def _latent_cache_for(config: ExperimentConfig) -> Path:
    if not config.latent_cache.path:
        raise ValueError("latent_cache.path is required")
    candidate = Path(config.latent_cache.path)
    if not candidate.exists():
        raise FileNotFoundError(f"latent cache not found: {candidate}")
    _validate_latent_cache_protocol(
        candidate,
        expected_seed=config.data.seed,
        expected_backend=config.data.backend,
        require_complete=_requires_complete_protocol(config),
        expected_split_manifest=config.data.split_manifest,
    )
    if _requires_complete_protocol(config) and config.latent_cache.encoder_checkpoint_path:
        protocol = _latent_cache_protocol(candidate)
        protocol_encoder = protocol.get("encoder_checkpoint") if isinstance(protocol, Mapping) else None
        protocol_encoder_path = (
            (candidate.resolve().parent / str(protocol_encoder)).resolve() if protocol_encoder else None
        )
        configured_encoder_path = Path(config.latent_cache.encoder_checkpoint_path).resolve()
        if protocol_encoder_path != configured_encoder_path:
            raise ValueError("configured encoder checkpoint does not match the real-data latent cache lineage")
    return candidate


def _selected_cache_trajectory_ids(cache: LatentCacheDataset, config: ExperimentConfig) -> tuple[str, ...]:
    ids = _cache_trajectory_ids(cache, config)
    if config.data.split == "all" or len(ids) < 2:
        return ids
    return _trajectory_splits(config, ids).as_dict()[config.data.split]


def _cache_trajectory_ids(cache: LatentCacheDataset, config: ExperimentConfig) -> tuple[str, ...]:
    ids = tuple(cache.trajectory_ids())
    requested = _configured_cache_trajectory_ids(config)
    if requested is None:
        return ids
    available = set(ids)
    missing = tuple(trajectory_id for trajectory_id in requested if trajectory_id not in available)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        msg = f"requested trajectories are missing from latent cache: {preview}{suffix}"
        raise ValueError(msg)
    return requested


def _configured_cache_trajectory_ids(config: ExperimentConfig) -> tuple[str, ...] | None:
    if config.data.backend != "cyclone_kvikio" or config.data.cyclone is None:
        return None
    trajectories = config.data.cyclone.trajectories
    if not trajectories:
        return None
    selected = tuple(str(trajectory_id) for trajectory_id in trajectories)
    unresolved = tuple(trajectory_id for trajectory_id in selected if _looks_like_unresolved_env_var(trajectory_id))
    if unresolved:
        joined = ", ".join(unresolved)
        msg = f"configured trajectories contain unresolved environment variables: {joined}"
        raise ValueError(msg)
    duplicates = tuple(dict.fromkeys(trajectory_id for trajectory_id in selected if selected.count(trajectory_id) > 1))
    if duplicates:
        joined = ", ".join(duplicates)
        msg = f"configured trajectories must be distinct for held-out validation: {joined}"
        raise ValueError(msg)
    return selected


def _looks_like_unresolved_env_var(value: str) -> bool:
    return value.startswith("${") and value.endswith("}")


def _cache_train_eval_ids(
    cache: LatentCacheDataset,
    config: ExperimentConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    ids = _cache_trajectory_ids(cache, config)
    if len(ids) < 2:
        raise ValueError("flux-head validation requires at least two cached trajectories")
    splits = _trajectory_splits(config, ids).as_dict()
    eval_split = config.data.split
    eval_ids = ids if eval_split == "all" else splits[eval_split]
    if not splits["train"]:
        raise ValueError("flux-head validation split has no training trajectories")
    if not eval_ids:
        raise ValueError(f"flux-head evaluation split {eval_split!r} has no trajectories")
    return splits["train"], tuple(eval_ids), eval_split


def _latent_normalization_trajectory_ids(cache: LatentCacheDataset, config: ExperimentConfig) -> tuple[str, ...]:
    ids = _cache_trajectory_ids(cache, config)
    if len(ids) < 2:
        return ids
    if config.latent_cache.latent_normalization_split != "train":
        raise ValueError(
            "latent normalization must be fit on the canonical training split; "
            "latent_normalization_split='all'/'selected' can leak held-out trajectories"
        )
    return _trajectory_splits(config, ids).train


def _latent_normalization_stats(
    cache: LatentCacheDataset, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray] | None:
    if config.latent_cache.latent_normalization == "none":
        return None
    trajectory_ids = _latent_normalization_trajectory_ids(cache, config)
    if not trajectory_ids:
        return None
    z = np.concatenate([cache.get_trajectory_latents(trajectory_id) for trajectory_id in trajectory_ids], axis=0)
    mean = np.mean(z, axis=0, dtype=np.float32)
    std = np.std(z, axis=0, dtype=np.float32)
    std = np.maximum(std, config.latent_cache.latent_normalization_epsilon).astype(np.float32)
    return mean.astype(np.float32), std


def _save_latent_normalization_stats(
    stats: tuple[np.ndarray, np.ndarray] | None,
    output_dir: str | Path,
) -> Path | None:
    if stats is None:
        return None
    path = Path(output_dir) / "latent_normalization_stats.npz"
    np.savez(path, mean=stats[0].astype(np.float32), std=stats[1].astype(np.float32))
    return path


def _sequence_checkpoint_latent_stats(
    checkpoint: str | Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    artifacts = _checkpoint_run_artifacts(checkpoint)
    if artifacts is None:
        raise ValueError("sequence checkpoint is missing config/metrics required for normalization lineage")
    config_payload, metrics = artifacts
    latent_cache = config_payload.get("latent_cache")
    mode = latent_cache.get("latent_normalization") if isinstance(latent_cache, Mapping) else None
    if mode == "none":
        return None
    if mode != "cache":
        raise ValueError(f"sequence checkpoint has unsupported latent normalization mode {mode!r}")
    artifact = _checkpoint_sidecar(checkpoint, "latent_normalization_stats.npz")
    expected_hash = metrics.get("latent_normalization_sha256")
    if not isinstance(expected_hash, str) or _sha256_file(artifact) != expected_hash:
        raise ValueError("sequence checkpoint latent-normalization artifact hash mismatch")
    loaded = _load_normalization_stats(artifact)
    return loaded.mean, loaded.std


def _normalize_latents(values: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if stats is None:
        return values
    mean, std = stats
    return ((values - mean) / std).astype(np.float32)


def _denormalize_latents(values: jax.Array, stats: tuple[np.ndarray, np.ndarray] | None) -> jax.Array:
    if stats is None:
        return values
    mean, std = stats
    return values * jnp.asarray(std, dtype=values.dtype) + jnp.asarray(mean, dtype=values.dtype)


def _trajectory_values_by_step(values: jax.Array, trajectory_ids: list[str]) -> jax.Array:
    """Average window-level ``[N, T]`` values within each trajectory."""

    values = jnp.asarray(values)
    if values.ndim != 2 or values.shape[0] != len(trajectory_ids):
        raise ValueError("values must have shape [num_windows, rollout_steps] matching trajectory_ids")
    unique_ids = tuple(dict.fromkeys(trajectory_ids))
    trajectory_means = []
    for trajectory_id in unique_ids:
        indices = np.asarray([index for index, item in enumerate(trajectory_ids) if item == trajectory_id])
        trajectory_means.append(jnp.mean(values[indices], axis=0))
    return jnp.stack(trajectory_means)


def _trajectory_relative_l2_by_step(
    error: jax.Array,
    target: jax.Array,
    trajectory_ids: list[str],
    *,
    eps: float,
) -> jax.Array:
    """Return global relative-L2 curves for each trajectory."""

    values = []
    for trajectory_id in dict.fromkeys(trajectory_ids):
        indices = np.asarray([index for index, item in enumerate(trajectory_ids) if item == trajectory_id])
        values.append(
            jnp.linalg.norm(error[indices], axis=(0, 2)) / (jnp.linalg.norm(target[indices], axis=(0, 2)) + eps)
        )
    return jnp.stack(values)


def _trajectory_time_average_errors(pred: jax.Array, target: jax.Array, trajectory_ids: list[str]) -> jax.Array:
    """Return one absolute time-average error per trajectory."""

    unique_ids = tuple(dict.fromkeys(trajectory_ids))
    errors = []
    for trajectory_id in unique_ids:
        indices = np.asarray([index for index, item in enumerate(trajectory_ids) if item == trajectory_id])
        pred_mean = jnp.mean(pred[indices], axis=(0, 1))
        target_mean = jnp.mean(target[indices], axis=(0, 1))
        errors.append(jnp.mean(jnp.abs(pred_mean - target_mean)))
    return jnp.stack(errors)


def _shape_correlation(pred: jax.Array, target: jax.Array, *, eps: float) -> jax.Array:
    """Return a stable last-axis Pearson correlation for each leading index."""

    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    if pred.shape != target.shape or pred.ndim == 0:
        raise ValueError(f"pred and target must have the same non-scalar shape, got {pred.shape} and {target.shape}")
    pred_centered = pred - jnp.mean(pred, axis=-1, keepdims=True)
    target_centered = target - jnp.mean(target, axis=-1, keepdims=True)
    numerator = jnp.sum(pred_centered * target_centered, axis=-1)
    pred_norm = jnp.sqrt(jnp.sum(jnp.square(pred_centered), axis=-1) + eps**2)
    target_norm = jnp.sqrt(jnp.sum(jnp.square(target_centered), axis=-1) + eps**2)
    return numerator / (pred_norm * target_norm)


def _diagnostic_reference_error_summary(
    pred: jax.Array,
    target: jax.Array,
    trajectory_ids: list[str],
    *,
    prefix: str,
    eps: float,
) -> dict[str, np.ndarray]:
    """Trajectory-balanced errors for a named diagnostic reference.

    The explicit prefix prevents latent-state persistence decoded by a learned
    head from being confused with persistence of the observed diagnostic.
    """

    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"diagnostic arrays must share shape [N, T, D], got {pred.shape} and {target.shape}")
    error = pred - target
    mse_by_trajectory = _trajectory_values_by_step(jnp.mean(jnp.square(error), axis=2), trajectory_ids)
    mae_by_trajectory = _trajectory_values_by_step(jnp.mean(jnp.abs(error), axis=2), trajectory_ids)
    relative_l2_by_trajectory = _trajectory_relative_l2_by_step(error, target, trajectory_ids, eps=eps)
    mse_by_step = jnp.mean(mse_by_trajectory, axis=0)
    mae_by_step = jnp.mean(mae_by_trajectory, axis=0)
    relative_l2_by_step = jnp.mean(relative_l2_by_trajectory, axis=0)
    return {
        f"{prefix}_mse_by_step": np.asarray(jax.device_get(mse_by_step)),
        f"{prefix}_mse_std_by_step": np.asarray(jax.device_get(jnp.std(mse_by_trajectory, axis=0))),
        f"{prefix}_rmse_by_step": np.asarray(jax.device_get(jnp.sqrt(mse_by_step))),
        f"{prefix}_rmse_by_trajectory": np.asarray(
            jax.device_get(jnp.sqrt(jnp.mean(mse_by_trajectory, axis=1)))
        ),
        f"{prefix}_mae_by_step": np.asarray(jax.device_get(mae_by_step)),
        f"{prefix}_relative_l2_by_step": np.asarray(jax.device_get(relative_l2_by_step)),
        f"{prefix}_mse": np.asarray(jax.device_get(jnp.mean(mse_by_step))),
        f"{prefix}_rmse": np.asarray(jax.device_get(jnp.sqrt(jnp.mean(mse_by_step)))),
        f"{prefix}_mae": np.asarray(jax.device_get(jnp.mean(mae_by_step))),
        f"{prefix}_relative_l2": np.asarray(jax.device_get(jnp.mean(relative_l2_by_step))),
    }


def _diagnostic_reference_shape_summary(
    pred: jax.Array,
    target: jax.Array,
    trajectory_ids: list[str],
    *,
    prefix: str,
    eps: float,
) -> dict[str, np.ndarray]:
    shape_by_trajectory = _trajectory_values_by_step(_shape_correlation(pred, target, eps=eps), trajectory_ids)
    shape_by_step = jnp.mean(shape_by_trajectory, axis=0)
    return {
        f"{prefix}_shape_corr_by_step": np.asarray(jax.device_get(shape_by_step)),
        f"{prefix}_shape_corr_std_by_step": np.asarray(
            jax.device_get(jnp.std(shape_by_trajectory, axis=0))
        ),
        f"{prefix}_shape_corr": np.asarray(jax.device_get(jnp.mean(shape_by_step))),
    }


def _sequence_batches(
    config: ExperimentConfig,
    cache_path: Path,
    *,
    repeat: bool = True,
    trajectory_ids: tuple[str, ...] | None = None,
    latent_stats: tuple[np.ndarray, np.ndarray] | None | object = _UNSET,
) -> Iterator[dict[str, Any]]:
    cache = LatentCacheDataset(cache_path)
    if latent_stats is _UNSET:
        latent_stats = _latent_normalization_stats(cache, config)
    if latent_stats is not None and not isinstance(latent_stats, tuple):
        raise TypeError("latent_stats must be a (mean, std) tuple or None")
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    selected_ids = _selected_cache_trajectory_ids(cache, config) if trajectory_ids is None else trajectory_ids
    for trajectory_id in selected_ids:
        starts = valid_sequence_starts(
            cache,
            trajectory_id,
            context_length=config.data.context_length,
            prediction_length=config.data.prediction_length,
        )
        for start in starts:
            context, target, _targets = cache.get_sequence_window(
                trajectory_id,
                start,
                context_length=config.data.context_length,
                prediction_length=config.data.prediction_length,
            )
            windows.append((_normalize_latents(context, latent_stats), _normalize_latents(target, latent_stats)))
    if not windows:
        msg = "no latent sequence windows found for the configured cache, trajectories, and context length"
        raise ValueError(msg)
    rng = np.random.default_rng(config.training.seed)
    while True:
        if config.data.shuffle:
            rng.shuffle(windows)
        for start in range(0, len(windows), config.data.batch_size):
            chunk = windows[start : start + config.data.batch_size]
            if not chunk:
                continue
            yield {
                "z_context": jnp.asarray([item[0] for item in chunk], dtype=jnp.float32),
                "z_target": jnp.asarray([item[1] for item in chunk], dtype=jnp.float32),
            }
        if not repeat:
            return


def _init_sequence_state(config: ExperimentConfig) -> tuple[TrainState, Any]:
    if config.model.sequence is None:
        raise ValueError("sequence config is required")
    model = build_sequence_model(config.model.sequence)
    rng = jax.random.PRNGKey(config.training.seed)
    dummy = jnp.zeros(
        (config.data.batch_size, config.data.context_length, config.model.sequence.latent_dim),
        dtype=jnp.float32,
    )
    variables = model.init(rng, dummy, train=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=build_optimizer(config.training),
        rng=rng,
        model_config=_loss_config(config),
    )
    return state, model


def train_sequence(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    plan = _parallel_plan(config)
    train_config = _config_with_parallel_optimizer(config, plan)
    state, _model = _init_sequence_state(train_config)
    if dry_run:
        _validate_explicit_manifest_against_cache(config)
        return {
            "dry_run": True,
            "latent_cache": config.latent_cache.path,
            "context_shape": [
                config.data.batch_size,
                config.data.context_length,
                config.model.sequence.latent_dim if config.model.sequence else 0,
            ],
            **_device_summary(plan),
        }
    out = _output_dir(config)
    write_run_metadata(out, config=train_config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(train_config), encoding="utf-8")
    if config.parallel.log_device_summary:
        write_device_report(out, config=config.parallel, plan=plan)
    cache_path = _latent_cache_for(config)
    encoder_checkpoint = _validate_cache_encoder_lineage(config, cache_path)
    train_config = train_config.model_copy(
        update={
            "latent_cache": train_config.latent_cache.model_copy(
                update={"encoder_checkpoint_path": str(encoder_checkpoint)}
            )
        }
    )
    write_run_metadata(out, config=train_config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(train_config), encoding="utf-8")
    cache = LatentCacheDataset(cache_path)
    latent_stats = _latent_normalization_stats(cache, train_config)
    latent_stats_path = _save_latent_normalization_stats(latent_stats, out)
    first_batch = next(_sequence_batches(train_config, cache_path, repeat=False, latent_stats=latent_stats))
    if config.training.max_steps == 0:
        metrics = {
            "dry_run": True,
            "context_shape": list(first_batch["z_context"].shape),
            **_device_summary(plan),
        }
        write_json(out / "metrics.json", metrics)
        return metrics
    logger = _metrics_logger(out, train_config)
    metrics: Mapping[str, Any] = {}
    state = _replicated_if_needed(state, plan)
    step_fn = make_pmap_sequence_train_step(plan.axis_name) if plan.uses_pmap else train_sequence_step
    eval_fn = make_pmap_sequence_eval_step(plan.axis_name) if plan.uses_pmap else eval_sequence_step
    batches = _sequence_batches(train_config, cache_path, repeat=True, latent_stats=latent_stats)
    universe_ids = tuple(cache.trajectory_ids())
    validation_ids = _trajectory_splits(config, universe_ids).val if len(universe_ids) > 1 else ()
    train_totals: dict[str, float] = {}
    train_examples = 0
    best_metric = float("inf")
    best_step: int | None = None
    best_checkpoint: Path | None = None
    latest_validation: dict[str, float] = {}
    step_value = 0
    while step_value < config.training.max_steps:
        raw_batch = next(batches)
        raw_batch_size = _batch_size(raw_batch)
        batch = _prepare_parallel_batch(raw_batch, plan)
        if batch is None:
            continue
        state, metrics = step_fn(state, batch)
        host_state = _host_state(state, plan)
        host_metrics = _host_metrics(metrics, plan)
        step_value = int(host_state.step)
        for key, value in host_metrics.items():
            train_totals[key] = train_totals.get(key, 0.0) + float(value) * raw_batch_size
        train_examples += raw_batch_size
        row = {
            "step": step_value,
            "lr": _scheduled_learning_rate(train_config, step_value),
            **{k: float(v) for k, v in host_metrics.items()},
        }
        if step_value % config.training.log_every == 0 or step_value == config.training.max_steps:
            logger.log(row, prefix="train")
        if step_value % config.training.checkpoint_every == 0:
            save_checkpoint(host_state, out, step=step_value)
        should_validate = bool(validation_ids) and (
            step_value % config.training.eval_every == 0 or step_value == config.training.max_steps
        )
        if should_validate:
            latest_validation = _aggregate_eval_metrics_by_trajectory(
                state,
                validation_ids,
                batches_for_trajectory=lambda trajectory_id: _sequence_batches(
                    train_config,
                    cache_path,
                    repeat=False,
                    trajectory_ids=(trajectory_id,),
                    latent_stats=latent_stats,
                ),
                plan=plan,
                step_fn=eval_fn,
            )
            latest_validation["latent_rmse"] = float(
                np.sqrt(max(latest_validation["latent_mse"], 0.0))
            )
            logger.log({"step": step_value, **latest_validation}, prefix="validation")
            if latest_validation["latent_rmse"] < best_metric:
                best_metric = latest_validation["latent_rmse"]
                best_step = step_value
                best_checkpoint = save_checkpoint(host_state, out, step=step_value)
    final_state = _host_state(state, plan)
    metrics = _host_metrics(metrics, plan)
    final_checkpoint = save_checkpoint(final_state, out, step=int(final_state.step))
    ckpt = best_checkpoint or final_checkpoint
    if best_step is None:
        best_step = int(final_state.step)
    selected_ids = _selected_cache_trajectory_ids(cache, train_config)
    aggregate_metrics = {key: value / train_examples for key, value in train_totals.items()}
    summary = {
        "artifact_role": "sequence_checkpoint",
        "step": int(final_state.step),
        "checkpoint": str(ckpt),
        "checkpoint_step": best_step,
        "checkpoint_sha256": _sha256_file(_checkpoint_pickle(ckpt)),
        "checkpoint_selection": (
            "minimum_validation_latent_rmse" if validation_ids else "final_step_no_validation_split"
        ),
        "best_validation_metric": None if not np.isfinite(best_metric) else best_metric,
        "latent_cache": str(cache_path),
        "latent_cache_sha256": _sha256_file(cache_path),
        "encoder_checkpoint": _relative_artifact_path(encoder_checkpoint),
        "encoder_checkpoint_sha256": _sha256_file(_checkpoint_pickle(encoder_checkpoint)),
        "latent_normalization_sha256": _sha256_file(latent_stats_path) if latent_stats_path else None,
        "last_minibatch": {k: float(v) for k, v in metrics.items()},
        "train_aggregate": aggregate_metrics,
        "validation": latest_validation,
        "actual_learning_rate": _scheduled_learning_rate(train_config, int(final_state.step)),
        **_protocol_fields(
            config,
            selected_ids,
            aggregation="sequence_window_minibatch_training",
            universe_trajectory_ids=universe_ids,
        ),
        **_device_summary(plan),
        **_system_metrics(plan),
        **{k: float(v) for k, v in metrics.items()},
    }
    if logger.wandb_status().get("requested"):
        summary["wandb"] = logger.wandb_status()
    logger.write_summary(summary)
    return summary


def evaluate_flux_head(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        _validate_explicit_manifest_against_cache(config)
        return {
            "dry_run": True,
            "latent_cache": config.latent_cache.path,
            "train_split": "train",
            "eval_split": config.data.split,
            "flux_head": "ridge_linear",
            "ridge_alpha": config.evaluation.flux_head_ridge_alpha,
        }

    cache_path = _latent_cache_for(config)
    encoder_checkpoint = _optional_cache_encoder_lineage(config, cache_path)
    cache = LatentCacheDataset(cache_path)
    configured_ids = _cache_trajectory_ids(cache, config)
    train_ids, eval_ids, eval_split = _cache_train_eval_ids(cache, config)
    summary = evaluate_linear_flux_head(
        cache,
        train_ids=train_ids,
        eval_ids=eval_ids,
        eval_split=eval_split,
        alpha=config.evaluation.flux_head_ridge_alpha,
        eps=config.loss.spectra_epsilon,
    )
    summary["configured_trajectories"] = list(configured_ids)
    summary["num_configured_trajectories"] = len(configured_ids)
    summary["latent_cache"] = str(cache_path)
    summary["encoder_checkpoint"] = (
        _relative_artifact_path(encoder_checkpoint) if encoder_checkpoint else None
    )
    summary.update(
        _protocol_fields(
            config,
            tuple(eval_ids),
            aggregation="sample_weighted",
            universe_trajectory_ids=tuple(cache.trajectory_ids()),
        )
    )
    flux_pred = np.asarray(summary.pop("flux_pred"), dtype=np.float32)
    flux_target = np.asarray(summary.pop("flux_target"), dtype=np.float32)
    out = _output_dir(config)
    write_run_metadata(out, config=config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    predictions_path = out / "flux_head_predictions.npz"
    np.savez_compressed(predictions_path, flux_pred=flux_pred, flux_target=flux_target)
    serializable = {key: _result_value(value) for key, value in summary.items()}
    metrics_path = write_json(out / "metrics.json", serializable)
    result = dict(serializable)
    result["metrics_json"] = str(metrics_path)
    result["predictions_npz"] = str(predictions_path)
    logger = _metrics_logger(out, config)
    if logger.wandb_status().get("requested"):
        result["wandb"] = logger.wandb_status()
        logger.log(result, prefix="eval")
        logger.finish(result, artifact_paths=(metrics_path, predictions_path))
    return result


def plot_representation(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        _validate_explicit_manifest_against_cache(config)
        return {
            "dry_run": True,
            "latent_cache": config.latent_cache.path,
            "perplexities": list(config.evaluation.tsne_perplexities),
            "max_points": config.evaluation.representation_max_points,
        }

    cache_path = _latent_cache_for(config)
    encoder_checkpoint = _optional_cache_encoder_lineage(config, cache_path)
    cache = LatentCacheDataset(cache_path)
    selected_ids = _selected_cache_trajectory_ids(cache, config)
    out = _output_dir(config)
    write_run_metadata(out, config=config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    summary = evaluate_representation(
        cache,
        out,
        split_seed=config.data.seed,
        trajectory_ids=selected_ids,
        perplexities=config.evaluation.tsne_perplexities,
        tsne_max_iter=config.evaluation.tsne_max_iter,
        max_points=config.evaluation.representation_max_points,
    )
    summary["latent_cache"] = str(cache_path)
    summary["encoder_checkpoint"] = (
        _relative_artifact_path(encoder_checkpoint) if encoder_checkpoint else None
    )
    summary.update(
        _protocol_fields(
            config,
            selected_ids,
            aggregation="point_level_projection",
            universe_trajectory_ids=tuple(cache.trajectory_ids()),
        )
    )
    metrics_path = write_json(out / "metrics.json", summary)
    result = dict(summary)
    result["metrics_json"] = str(metrics_path)
    logger = _metrics_logger(out, config)
    if logger.wandb_status().get("requested"):
        result["wandb"] = logger.wandb_status()
        artifact_paths = [metrics_path, result["points_npz"], result["points_csv"], *result["plot_paths"].values()]
        logger.log(result, prefix="eval")
        logger.finish(result, artifact_paths=artifact_paths)
    return result


def evaluate_rollout(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    baseline_mode = config.evaluation.baseline_mode
    if dry_run:
        _validate_explicit_manifest_against_cache(config)
        return {
            "dry_run": True,
            "latent_cache": config.latent_cache.path,
            "sequence_checkpoint": config.latent_cache.sequence_checkpoint_path,
            "rollout_steps": config.evaluation.rollout_steps,
            "baseline_mode": baseline_mode,
        }

    cache_path = _latent_cache_for(config)
    encoder_checkpoint = _validate_cache_encoder_lineage(config, cache_path)
    cache = LatentCacheDataset(cache_path)
    configured_ids = _cache_trajectory_ids(cache, config)
    selected_ids = _selected_cache_trajectory_ids(cache, config)
    use_latent_persistence = baseline_mode == "latent_state_persistence_decoded"
    use_observed_persistence = baseline_mode == "observed_diagnostic_persistence"
    latent_stats = _latent_normalization_stats(cache, config) if use_latent_persistence else None
    model = None
    params = None
    requested_metrics = set(config.evaluation.metrics)
    flux_metrics_requested = config.data.target_flux and any("flux" in metric for metric in requested_metrics)
    spectra_metrics_requested = bool(config.data.target_spectra) and any(
        "spectra" in metric for metric in requested_metrics
    )
    diagnostic_metrics_requested = flux_metrics_requested or spectra_metrics_requested
    diagnostic_warnings: list[str] = []
    diagnostic_model = build_diagnostic_heads(config.model.diagnostics)
    diagnostic_params = None
    if diagnostic_model is None:
        if diagnostic_metrics_requested:
            diagnostic_warnings.append("diagnostic metrics requested but model diagnostics are disabled")
    elif diagnostic_metrics_requested:
        _validate_checkpoint_protocol(
            encoder_checkpoint,
            expected_seed=config.data.seed,
            expected_backend=config.data.backend,
            expected_universe_ids=tuple(cache.trajectory_ids()),
            expected_artifact_role="encoder_checkpoint",
            role="encoder checkpoint",
            require_complete=_requires_complete_protocol(config),
            expected_split_manifest=config.data.split_manifest,
        )
        encoder_params = load_checkpoint(encoder_checkpoint)["params"]
        diagnostic_params = _diagnostic_params_from_encoder_params(encoder_params)
        if diagnostic_params is None and diagnostic_metrics_requested:
            raise ValueError("diagnostic metrics requested but cache-lineage encoder has no diagnostic_heads params")
    diagnostic_heads_loaded = diagnostic_model is not None and diagnostic_params is not None
    if baseline_mode == "none":
        if config.model.sequence is None or config.latent_cache.sequence_checkpoint_path is None:
            raise ValueError("sequence model and checkpoint are required unless persistence baseline is enabled")
        _validate_checkpoint_protocol(
            config.latent_cache.sequence_checkpoint_path,
            expected_seed=config.data.seed,
            expected_backend=config.data.backend,
            expected_universe_ids=tuple(cache.trajectory_ids()),
            expected_artifact_role="sequence_checkpoint",
            role="sequence checkpoint",
            require_complete=_requires_complete_protocol(config),
            expected_cache_path=cache_path,
            expected_encoder_checkpoint=encoder_checkpoint,
            expected_split_manifest=config.data.split_manifest,
        )
        latent_stats = _sequence_checkpoint_latent_stats(config.latent_cache.sequence_checkpoint_path)
        model = build_sequence_model(config.model.sequence)
        params = load_checkpoint(config.latent_cache.sequence_checkpoint_path)["params"]
    preds = []
    targets = []
    rollout_trajectory_ids: list[str] = []
    flux_targets = []
    flux_last_observed = []
    spectra_targets: dict[str, list[np.ndarray]] = {key: [] for key in config.data.target_spectra}
    spectra_last_observed: dict[str, list[np.ndarray]] = {key: [] for key in config.data.target_spectra}
    for trajectory_id in selected_ids:
        total = cache.num_timesteps(trajectory_id)
        required = config.data.context_length + config.evaluation.rollout_steps
        if total < required:
            continue
        for start in range(0, total - required + 1):
            context, target, diagnostics = cache.get_sequence_window(
                trajectory_id,
                start,
                context_length=config.data.context_length,
                prediction_length=config.evaluation.rollout_steps,
            )
            context_model = _normalize_latents(context, latent_stats)
            context_b = jnp.asarray(context_model[None, ...], dtype=jnp.float32)
            if use_latent_persistence:
                pred = persistence_rollout(context_b, config.evaluation.rollout_steps)
            elif use_observed_persistence:
                # Observed persistence has no latent forecast. True future latents
                # are carried only to compute the diagnostic-head oracle reference;
                # latent rollout metrics are removed below.
                pred = jnp.asarray(target[None, ...], dtype=jnp.float32)
            else:
                if model is None or params is None:
                    raise AssertionError("sequence model state was not initialized")
                pred = autoregressive_rollout(
                    model.apply,
                    params,
                    context_b,
                    config.evaluation.rollout_steps,
                )
            pred = _denormalize_latents(pred, latent_stats)
            preds.append(np.asarray(jax.device_get(pred), dtype=np.float32))
            targets.append(target[None, ...].astype(np.float32))
            rollout_trajectory_ids.append(trajectory_id)
            last_observed = cache.get_latent(trajectory_id, start + config.data.context_length - 1).targets
            if diagnostics.flux is not None:
                flux_targets.append(np.asarray(diagnostics.flux[None, ...], dtype=np.float32))
                if last_observed.flux is not None:
                    flux_last_observed.append(np.asarray(last_observed.flux[None, ...], dtype=np.float32))
            for key in spectra_targets:
                if key in diagnostics.spectra:
                    spectra_targets[key].append(np.asarray(diagnostics.spectra[key][None, ...], dtype=np.float32))
                    if key in last_observed.spectra:
                        spectra_last_observed[key].append(
                            np.asarray(last_observed.spectra[key][None, ...], dtype=np.float32)
                        )
    if not preds:
        raise ValueError("no valid rollout windows found")
    pred_all = jnp.asarray(np.concatenate(preds, axis=0))
    target_all = jnp.asarray(np.concatenate(targets, axis=0))
    metrics = trajectory_balanced_rollout_metrics(pred_all, target_all, rollout_trajectory_ids)
    summary = {key: np.asarray(jax.device_get(value)) for key, value in metrics.items()}
    flux_metrics_computed = False
    spectra_metrics_computed = False
    diagnostic_samples: dict[str, np.ndarray] = {}
    sample_windows = min(16, int(pred_all.shape[0]))
    if len(flux_targets) == len(preds) and len(flux_last_observed) == len(preds):
        observed_flux_target = jnp.asarray(np.concatenate(flux_targets, axis=0))
        last_flux = jnp.asarray(np.concatenate(flux_last_observed, axis=0))
        observed_flux_pred = observed_diagnostic_persistence(last_flux, config.evaluation.rollout_steps)
        summary.update(
            _diagnostic_reference_error_summary(
                observed_flux_pred,
                observed_flux_target,
                rollout_trajectory_ids,
                prefix="observed_diagnostic_persistence_flux",
                eps=config.loss.spectra_epsilon,
            )
        )
        diagnostic_samples["observed_diagnostic_persistence_flux_pred"] = np.asarray(
            jax.device_get(observed_flux_pred[:sample_windows]), dtype=np.float32
        )
    for key, rows in spectra_targets.items():
        if len(rows) != len(preds) or len(spectra_last_observed[key]) != len(preds):
            continue
        observed_spectra_target = jnp.asarray(np.concatenate(rows, axis=0))
        last_spectra = jnp.asarray(np.concatenate(spectra_last_observed[key], axis=0))
        observed_spectra_pred = observed_diagnostic_persistence(last_spectra, config.evaluation.rollout_steps)
        observed_prefix = f"observed_diagnostic_persistence_spectra_{key}"
        summary.update(
            _diagnostic_reference_error_summary(
                observed_spectra_pred,
                observed_spectra_target,
                rollout_trajectory_ids,
                prefix=observed_prefix,
                eps=config.loss.spectra_epsilon,
            )
        )
        summary.update(
            _diagnostic_reference_shape_summary(
                observed_spectra_pred,
                observed_spectra_target,
                rollout_trajectory_ids,
                prefix=observed_prefix,
                eps=config.loss.spectra_epsilon,
            )
        )
    if diagnostic_model is not None and diagnostic_params is not None:
        flat_pred = pred_all.reshape((-1, pred_all.shape[-1]))
        diagnostics = cast(
            DiagnosticPredictions,
            diagnostic_model.apply({"params": diagnostic_params}, flat_pred, train=False),
        )
        flat_target = target_all.reshape((-1, target_all.shape[-1]))
        diagnostic_head_oracle = cast(
            DiagnosticPredictions,
            diagnostic_model.apply({"params": diagnostic_params}, flat_target, train=False),
        )
        if diagnostics.flux is not None and len(flux_targets) == len(preds):
            flux_target = jnp.asarray(np.concatenate(flux_targets, axis=0))
            flux_pred = diagnostics.flux.reshape(flux_target.shape)
            flux_error = flux_pred - flux_target
            flux_mse_by_trajectory = _trajectory_values_by_step(
                jnp.mean(jnp.square(flux_error), axis=2), rollout_trajectory_ids
            )
            flux_mae_by_trajectory = _trajectory_values_by_step(
                jnp.mean(jnp.abs(flux_error), axis=2), rollout_trajectory_ids
            )
            flux_relative_error_by_trajectory = _trajectory_values_by_step(
                jnp.mean(jnp.abs(flux_error) / (jnp.abs(flux_target) + config.loss.spectra_epsilon), axis=2),
                rollout_trajectory_ids,
            )
            flux_mse_by_step = jnp.mean(flux_mse_by_trajectory, axis=0)
            flux_mae_by_step = jnp.mean(flux_mae_by_trajectory, axis=0)
            flux_relative_error_by_step = jnp.mean(flux_relative_error_by_trajectory, axis=0)
            flux_mse_value = jnp.mean(flux_mse_by_step)
            flux_rmse_by_trajectory = jnp.sqrt(jnp.mean(flux_mse_by_trajectory, axis=1))
            flux_time_average_by_trajectory = _trajectory_time_average_errors(
                flux_pred, flux_target, rollout_trajectory_ids
            )
            summary["flux_rmse_by_trajectory"] = np.asarray(jax.device_get(flux_rmse_by_trajectory))
            summary["flux_trajectory_ids"] = np.asarray(tuple(dict.fromkeys(rollout_trajectory_ids)), dtype=str)
            summary["flux_mse_by_step"] = np.asarray(jax.device_get(flux_mse_by_step))
            summary["flux_mse_std_by_step"] = np.asarray(jax.device_get(jnp.std(flux_mse_by_trajectory, axis=0)))
            summary["flux_rmse_by_step"] = np.asarray(jax.device_get(jnp.sqrt(flux_mse_by_step)))
            summary["flux_rmse_std_by_step"] = np.asarray(
                jax.device_get(jnp.std(jnp.sqrt(flux_mse_by_trajectory), axis=0))
            )
            summary["flux_mae_by_step"] = np.asarray(jax.device_get(flux_mae_by_step))
            summary["flux_mae_std_by_step"] = np.asarray(jax.device_get(jnp.std(flux_mae_by_trajectory, axis=0)))
            summary["flux_relative_error_by_step"] = np.asarray(jax.device_get(flux_relative_error_by_step))
            summary["flux_relative_error_std_by_step"] = np.asarray(
                jax.device_get(jnp.std(flux_relative_error_by_trajectory, axis=0))
            )
            summary["flux_mse"] = np.asarray(jax.device_get(flux_mse_value))
            summary["flux_rmse"] = np.asarray(jax.device_get(jnp.sqrt(flux_mse_value)))
            summary["flux_mae"] = np.asarray(jax.device_get(jnp.mean(flux_mae_by_step)))
            summary["flux_relative_error"] = np.asarray(jax.device_get(jnp.mean(flux_relative_error_by_step)))
            summary["flux_time_average_error"] = np.asarray(jax.device_get(jnp.mean(flux_time_average_by_trajectory)))
            summary["flux_time_average_error_std"] = np.asarray(
                jax.device_get(jnp.std(flux_time_average_by_trajectory))
            )
            diagnostic_samples["flux_pred"] = np.asarray(jax.device_get(flux_pred[:sample_windows]), dtype=np.float32)
            diagnostic_samples["flux_target"] = np.asarray(
                jax.device_get(flux_target[:sample_windows]),
                dtype=np.float32,
            )
            if diagnostic_head_oracle.flux is not None:
                oracle_flux_pred = diagnostic_head_oracle.flux.reshape(flux_target.shape)
                summary.update(
                    _diagnostic_reference_error_summary(
                        oracle_flux_pred,
                        flux_target,
                        rollout_trajectory_ids,
                        prefix="diagnostic_head_oracle_flux",
                        eps=config.loss.spectra_epsilon,
                    )
                )
                diagnostic_samples["diagnostic_head_oracle_flux_pred"] = np.asarray(
                    jax.device_get(oracle_flux_pred[:sample_windows]), dtype=np.float32
                )
            flux_metrics_computed = True
        elif flux_metrics_requested:
            diagnostic_warnings.append("flux metrics requested but flux predictions or targets are unavailable")
        spectra_mse_by_trajectory = []
        spectra_log_mse_by_trajectory = []
        spectra_relative_l2_by_trajectory = []
        spectra_shape_corr_by_trajectory = []
        for key, rows in spectra_targets.items():
            if len(rows) == len(preds) and key in diagnostics.spectra:
                spectra_target = jnp.asarray(np.concatenate(rows, axis=0))
                spectra_pred = diagnostics.spectra[key].reshape(spectra_target.shape)
                spectra_error = spectra_pred - spectra_target
                by_trajectory = _trajectory_values_by_step(
                    jnp.mean(jnp.square(spectra_error), axis=2), rollout_trajectory_ids
                )
                by_step = jnp.mean(by_trajectory, axis=0)
                pred_log = jnp.log(jnp.maximum(spectra_pred, 0.0) + config.loss.spectra_epsilon)
                target_log = jnp.log(jnp.maximum(spectra_target, 0.0) + config.loss.spectra_epsilon)
                log_by_trajectory = _trajectory_values_by_step(
                    jnp.mean(jnp.square(pred_log - target_log), axis=2), rollout_trajectory_ids
                )
                relative_l2_by_trajectory = _trajectory_relative_l2_by_step(
                    spectra_error,
                    spectra_target,
                    rollout_trajectory_ids,
                    eps=config.loss.spectra_epsilon,
                )
                shape_corr_by_trajectory = _trajectory_values_by_step(
                    _shape_correlation(
                        spectra_pred,
                        spectra_target,
                        eps=config.loss.spectra_epsilon,
                    ),
                    rollout_trajectory_ids,
                )
                log_by_step = jnp.mean(log_by_trajectory, axis=0)
                relative_l2_by_step = jnp.mean(relative_l2_by_trajectory, axis=0)
                shape_corr_by_step = jnp.mean(shape_corr_by_trajectory, axis=0)
                summary[f"spectra_{key}_mse_by_step"] = np.asarray(jax.device_get(by_step))
                summary[f"spectra_{key}_mse_std_by_step"] = np.asarray(jax.device_get(jnp.std(by_trajectory, axis=0)))
                summary[f"spectra_{key}_mse"] = np.asarray(jax.device_get(jnp.mean(by_step)))
                summary[f"spectra_{key}_log_mse_by_step"] = np.asarray(jax.device_get(log_by_step))
                summary[f"spectra_{key}_log_mse_std_by_step"] = np.asarray(
                    jax.device_get(jnp.std(log_by_trajectory, axis=0))
                )
                summary[f"spectra_{key}_log_mse"] = np.asarray(jax.device_get(jnp.mean(log_by_step)))
                summary[f"spectra_{key}_relative_l2_by_step"] = np.asarray(jax.device_get(relative_l2_by_step))
                summary[f"spectra_{key}_relative_l2_std_by_step"] = np.asarray(
                    jax.device_get(jnp.std(relative_l2_by_trajectory, axis=0))
                )
                summary[f"spectra_{key}_relative_l2"] = np.asarray(jax.device_get(jnp.mean(relative_l2_by_step)))
                summary[f"spectra_{key}_shape_corr_by_step"] = np.asarray(jax.device_get(shape_corr_by_step))
                summary[f"spectra_{key}_shape_corr_std_by_step"] = np.asarray(
                    jax.device_get(jnp.std(shape_corr_by_trajectory, axis=0))
                )
                summary[f"spectra_{key}_shape_corr"] = np.asarray(jax.device_get(jnp.mean(shape_corr_by_step)))
                diagnostic_samples[f"{key}_pred"] = np.asarray(
                    jax.device_get(spectra_pred[:sample_windows]),
                    dtype=np.float32,
                )
                diagnostic_samples[f"{key}_target"] = np.asarray(
                    jax.device_get(spectra_target[:sample_windows]),
                    dtype=np.float32,
                )
                if key in diagnostic_head_oracle.spectra:
                    oracle_spectra_pred = diagnostic_head_oracle.spectra[key].reshape(spectra_target.shape)
                    oracle_prefix = f"diagnostic_head_oracle_spectra_{key}"
                    summary.update(
                        _diagnostic_reference_error_summary(
                            oracle_spectra_pred,
                            spectra_target,
                            rollout_trajectory_ids,
                            prefix=oracle_prefix,
                            eps=config.loss.spectra_epsilon,
                        )
                    )
                    summary.update(
                        _diagnostic_reference_shape_summary(
                            oracle_spectra_pred,
                            spectra_target,
                            rollout_trajectory_ids,
                            prefix=oracle_prefix,
                            eps=config.loss.spectra_epsilon,
                        )
                    )
                spectra_mse_by_trajectory.append(by_trajectory)
                spectra_log_mse_by_trajectory.append(log_by_trajectory)
                spectra_relative_l2_by_trajectory.append(relative_l2_by_trajectory)
                spectra_shape_corr_by_trajectory.append(shape_corr_by_trajectory)
            elif spectra_metrics_requested:
                diagnostic_warnings.append(
                    f"spectra metrics requested but spectra {key!r} predictions or targets are unavailable"
                )
        if spectra_mse_by_trajectory:
            aggregated_spectra = {
                "mse": jnp.mean(jnp.stack(spectra_mse_by_trajectory), axis=0),
                "log_mse": jnp.mean(jnp.stack(spectra_log_mse_by_trajectory), axis=0),
                "relative_l2": jnp.mean(jnp.stack(spectra_relative_l2_by_trajectory), axis=0),
                "shape_corr": jnp.mean(jnp.stack(spectra_shape_corr_by_trajectory), axis=0),
            }
            for metric_name, by_trajectory in aggregated_spectra.items():
                by_step = jnp.mean(by_trajectory, axis=0)
                summary[f"spectra_{metric_name}_by_step"] = np.asarray(jax.device_get(by_step))
                summary[f"spectra_{metric_name}_std_by_step"] = np.asarray(
                    jax.device_get(jnp.std(by_trajectory, axis=0))
                )
                summary[f"spectra_{metric_name}"] = np.asarray(jax.device_get(jnp.mean(by_step)))
            spectra_metrics_computed = True
    elif diagnostic_metrics_requested:
        diagnostic_warnings.append("diagnostic metrics were skipped because diagnostic heads were not loaded")
    summary["diagnostic_heads_loaded"] = np.asarray(diagnostic_heads_loaded)
    summary["diagnostic_metrics_requested"] = np.asarray(diagnostic_metrics_requested)
    summary["flux_metrics_computed"] = np.asarray(flux_metrics_computed)
    summary["spectra_metrics_computed"] = np.asarray(spectra_metrics_computed)
    summary["diagnostic_warnings"] = np.asarray(tuple(dict.fromkeys(diagnostic_warnings)), dtype=str)
    summary["configured_trajectory_ids"] = np.asarray(configured_ids, dtype=str)
    summary["selected_trajectory_ids"] = np.asarray(selected_ids, dtype=str)
    summary["num_configured_trajectories"] = np.asarray(len(configured_ids), dtype=np.int32)
    summary["num_selected_trajectories"] = np.asarray(len(selected_ids), dtype=np.int32)
    summary["num_rollout_windows"] = np.asarray(len(preds), dtype=np.int32)
    summary["latent_cache"] = str(cache_path)
    summary["encoder_checkpoint"] = _relative_artifact_path(encoder_checkpoint)
    if use_observed_persistence:
        for suffix in (
            "mse_by_step",
            "mse_std_by_step",
            "mae_by_step",
            "mae_std_by_step",
            "relative_l2_by_step",
            "relative_l2_std_by_step",
            "cosine_by_step",
            "cosine_std_by_step",
            "mse",
            "mae",
            "relative_l2",
            "cosine",
            "stable",
        ):
            summary.pop(suffix, None)
        for key in tuple(summary):
            if key.startswith(("flux_", "spectra_")):
                summary.pop(key)
        observed_flux_prefix = "observed_diagnostic_persistence_flux_"
        for key, value in tuple(summary.items()):
            if key.startswith(observed_flux_prefix):
                summary[key.removeprefix("observed_diagnostic_persistence_")] = value
        diagnostic_samples.pop("flux_pred", None)
        for key in tuple(diagnostic_samples):
            if key.endswith("_pred") and not key.startswith(
                ("observed_diagnostic_persistence_", "diagnostic_head_oracle_")
            ):
                diagnostic_samples.pop(key)
        # These flags describe the learned diagnostic forecast, which is
        # intentionally absent for an observed persistence baseline.  Keep
        # them explicit instead of deleting them with the ``flux_*`` and
        # ``spectra_*`` metric keys above.
        summary["flux_metrics_computed"] = np.asarray(False)
        summary["spectra_metrics_computed"] = np.asarray(False)
    summary["rollout_method"] = (
        baseline_mode if baseline_mode != "none" else "learned_sequence_model"
    )
    summary["sequence_checkpoint"] = (
        config.latent_cache.sequence_checkpoint_path or baseline_mode
    )
    summary["rollout_horizon"] = np.asarray(config.evaluation.rollout_steps, dtype=np.int32)
    summary.update(
        _protocol_fields(
            config,
            selected_ids,
            aggregation="trajectory_balanced_mean_with_between_trajectory_std",
            universe_trajectory_ids=tuple(cache.trajectory_ids()),
        )
    )
    out = _output_dir(config)
    write_run_metadata(out, config=config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    diagnostic_samples_path = _save_diagnostic_samples(diagnostic_samples, out)
    paths = save_rollout_report(summary, out)
    plots = save_rollout_plots(summary, out)
    result = {key: _result_value(value) for key, value in summary.items()}
    result["metrics_json"] = str(paths["metrics_json"])
    result["metrics_by_step_csv"] = str(paths["metrics_by_step_csv"])
    if diagnostic_samples_path is not None:
        result["diagnostic_samples_npz"] = str(diagnostic_samples_path)
    if plots:
        result["plots"] = {key: str(path) for key, path in plots.items()}
        # Persistence baselines intentionally remove latent metrics, so there
        # may be no latent-MSE plot to use as the historical compatibility alias.
        primary_plot = plots.get("latent_mse") or next(iter(plots.values()))
        result["plot"] = str(primary_plot)
    logger = _metrics_logger(out, config)
    if logger.wandb_status().get("requested"):
        result["wandb"] = logger.wandb_status()
        logger.log(result, prefix="eval")
        logger.finish(
            result,
            artifact_paths=(
                paths["metrics_json"],
                paths["metrics_by_step_csv"],
                diagnostic_samples_path,
                *plots.values(),
            ),
        )
    return result


def _save_diagnostic_samples(samples: Mapping[str, np.ndarray], output_dir: str | Path) -> Path | None:
    if not samples:
        return None
    path = ensure_dir(output_dir) / "diagnostic_samples.npz"
    np.savez_compressed(path, **samples)
    return path


def _result_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value
