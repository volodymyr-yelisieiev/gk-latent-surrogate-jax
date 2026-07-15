"""Small CPU-safe end-to-end pipeline helpers used by the CLI and tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
from gk_surrogate.data.split import split_trajectory_ids
from gk_surrogate.evaluation.flux_head import evaluate_flux_head as evaluate_linear_flux_head
from gk_surrogate.evaluation.reports import save_rollout_plots, save_rollout_report
from gk_surrogate.evaluation.representation import evaluate_representation
from gk_surrogate.evaluation.rollout import (
    autoregressive_rollout,
    persistence_rollout,
    trajectory_balanced_rollout_metrics,
)
from gk_surrogate.factory import (
    build_diagnostic_heads,
    build_encoder_with_diagnostics,
    build_sequence_model,
    build_simsiam_encoder_with_diagnostics,
)
from gk_surrogate.parallel.batch import drop_or_pad_to_multiple, shard_batch
from gk_surrogate.parallel.devices import ParallelPlan, get_local_devices, resolve_parallel_mode, write_device_report
from gk_surrogate.parallel.pmap_steps import make_pmap_encoder_train_step, make_pmap_sequence_train_step
from gk_surrogate.parallel.replicate import replicate_state, unreplicate_state, unreplicate_tree
from gk_surrogate.training.checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from gk_surrogate.training.embed_dataset import encode_snapshots
from gk_surrogate.training.logging import MetricsLogger, write_json, write_run_metadata
from gk_surrogate.training.optimizer import build_optimizer
from gk_surrogate.training.state import TrainState
from gk_surrogate.training.train_encoder import train_encoder_step
from gk_surrogate.training.train_sequence import train_sequence_step
from gk_surrogate.utils.paths import ensure_dir


def _output_dir(config: ExperimentConfig) -> Path:
    return ensure_dir(config.output_dir)


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


def _selected_trajectory_ids(config: ExperimentConfig) -> tuple[str, ...]:
    dataset = build_dataset(config.data)
    ids = tuple(dataset.trajectory_ids())
    if config.data.split == "all" or len(ids) < 2:
        return ids
    split = split_trajectory_ids(ids, seed=config.data.seed)
    return split[config.data.split]


def _normalization_stats(config: ExperimentConfig) -> NormalizationStats | None:
    mode = config.data.normalization.mode
    if mode in {"none", "sample", "trajectory"}:
        return None
    if mode == "fixed":
        return NormalizationStats(
            mean=np.asarray(config.data.normalization.mean, dtype=np.float32),
            std=np.asarray(config.data.normalization.std, dtype=np.float32),
        )
    dataset = build_dataset(config.data)
    return estimate_dataset_stats(dataset, max_samples=config.data.normalization.max_samples)


def _trajectory_normalization_stats(config: ExperimentConfig, ids: tuple[str, ...]) -> dict[str, NormalizationStats]:
    if config.data.normalization.mode != "trajectory":
        return {}
    dataset = build_dataset(config.data)
    return {trajectory_id: estimate_trajectory_stats(dataset, trajectory_id) for trajectory_id in ids}


def _snapshot_batches(config: ExperimentConfig, *, repeat: bool = True) -> Iterator[dict[str, Any]]:
    dataset = build_dataset(config.data)
    ids = _selected_trajectory_ids(config)
    stats = _normalization_stats(config)
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
    dataset = build_dataset(config.data)
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
    if config.data.normalization.mode in {"dataset", "trajectory", "fixed"}:
        stats = _normalization_stats(config)
        if stats is not None:
            stats.save_npz(out / "normalization_stats.npz")

    logger = _metrics_logger(out, train_config)
    metrics: Mapping[str, Any] = {}
    state = _replicated_if_needed(state, plan)
    step_fn = make_pmap_encoder_train_step(plan.axis_name) if plan.uses_pmap else train_encoder_step
    batches = _snapshot_batches(config, repeat=True)
    step_value = 0
    while step_value < config.training.max_steps:
        batch = _prepare_parallel_batch(next(batches), plan)
        if batch is None:
            continue
        state, metrics = step_fn(state, batch)
        host_state = _host_state(state, plan)
        host_metrics = _host_metrics(metrics, plan)
        step_value = int(host_state.step)
        row = {
            "step": step_value,
            "lr": train_config.training.learning_rate,
            **{k: float(v) for k, v in host_metrics.items()},
        }
        if step_value % config.training.log_every == 0 or step_value == config.training.max_steps:
            logger.log(row, prefix="train")
        if step_value % config.training.checkpoint_every == 0:
            save_checkpoint(host_state, out, step=step_value)
    final_state = _host_state(state, plan)
    metrics = _host_metrics(metrics, plan)
    ckpt = save_checkpoint(final_state, out, step=int(final_state.step))
    summary = {
        "step": int(final_state.step),
        "checkpoint": str(ckpt),
        **_device_summary(plan),
        **_system_metrics(plan),
        **{k: float(v) for k, v in metrics.items()},
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


def _encoder_params_for_embedding(config: ExperimentConfig) -> tuple[Any, Any]:
    state, model = _init_encoder_state(config)
    del state
    checkpoint_path = config.latent_cache.encoder_checkpoint_path
    ckpt = Path(checkpoint_path) if checkpoint_path else latest_checkpoint(config.output_dir)
    if ckpt is None:
        raise FileNotFoundError("embed-dataset requires latent_cache.encoder_checkpoint_path or an existing checkpoint")
    if not ckpt.exists():
        raise FileNotFoundError(f"encoder checkpoint not found: {ckpt}")
    payload = load_checkpoint(ckpt)
    return model, _encoder_apply_params(payload["params"])


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
    dataset = build_dataset(config.data)
    if dry_run:
        return {
            "dry_run": True,
            "planned_latent_cache": config.latent_cache.path or str(Path(config.output_dir) / "latent_cache.h5"),
            "trajectories": len(dataset.trajectory_ids()),
        }

    out = _output_dir(config)
    write_run_metadata(out, config=config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    model, params = _encoder_params_for_embedding(config)
    cache_path = Path(config.latent_cache.path) if config.latent_cache.path else out / "latent_cache.h5"

    encoder_checkpoint_path = str(
        config.latent_cache.encoder_checkpoint_path or latest_checkpoint(config.output_dir) or ""
    )
    writer = LatentCacheWriter(
        cache_path,
        latent_dim=config.model.encoder.latent_dim,
        config_yaml=config_to_yaml(config),
        encoder_checkpoint_path=encoder_checkpoint_path,
    )
    stats = _normalization_stats(config)
    trajectory_stats = _trajectory_normalization_stats(config, tuple(dataset.trajectory_ids()))
    for trajectory_id in dataset.trajectory_ids():
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
            trajectory_id,
            z,
            physical_time=np.asarray(times, dtype=np.float32),
            flux=np.asarray(flux_rows, dtype=np.float32) if flux_rows else None,
            spectra={key: np.asarray(rows, dtype=np.float32) for key, rows in spectra_rows.items()},
        )
    summary = {
        "latent_cache": str(cache_path),
        "trajectories": len(dataset.trajectory_ids()),
        "embedding_batch_size": config.data.batch_size,
    }
    write_json(out / "metrics.json", summary)
    return summary


def _latent_cache_for(config: ExperimentConfig) -> Path:
    if not config.latent_cache.path:
        raise ValueError("latent_cache.path is required")
    candidate = Path(config.latent_cache.path)
    if not candidate.exists():
        raise FileNotFoundError(f"latent cache not found: {candidate}")
    return candidate


def _selected_cache_trajectory_ids(cache: LatentCacheDataset, config: ExperimentConfig) -> tuple[str, ...]:
    ids = _cache_trajectory_ids(cache, config)
    if config.data.split == "all" or len(ids) < 2:
        return ids
    return split_trajectory_ids(ids, seed=config.data.seed)[config.data.split]


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
    splits = split_trajectory_ids(ids, seed=config.data.seed)
    eval_split = config.data.split
    eval_ids = ids if eval_split == "all" else splits[eval_split]
    if not splits["train"]:
        raise ValueError("flux-head validation split has no training trajectories")
    if not eval_ids:
        raise ValueError(f"flux-head evaluation split {eval_split!r} has no trajectories")
    return splits["train"], tuple(eval_ids), eval_split


def _latent_normalization_trajectory_ids(cache: LatentCacheDataset, config: ExperimentConfig) -> tuple[str, ...]:
    ids = _cache_trajectory_ids(cache, config)
    if config.latent_cache.latent_normalization_split == "all" or len(ids) < 2:
        return ids
    if config.latent_cache.latent_normalization_split == "selected":
        return _selected_cache_trajectory_ids(cache, config)
    return split_trajectory_ids(ids, seed=config.data.seed)["train"]


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
            jnp.linalg.norm(error[indices], axis=(0, 2))
            / (jnp.linalg.norm(target[indices], axis=(0, 2)) + eps)
        )
    return jnp.stack(values)


def _trajectory_time_average_errors(
    pred: jax.Array, target: jax.Array, trajectory_ids: list[str]
) -> jax.Array:
    """Return one absolute time-average error per trajectory."""

    unique_ids = tuple(dict.fromkeys(trajectory_ids))
    errors = []
    for trajectory_id in unique_ids:
        indices = np.asarray([index for index, item in enumerate(trajectory_ids) if item == trajectory_id])
        pred_mean = jnp.mean(pred[indices], axis=(0, 1))
        target_mean = jnp.mean(target[indices], axis=(0, 1))
        errors.append(jnp.mean(jnp.abs(pred_mean - target_mean)))
    return jnp.stack(errors)


def _sequence_batches(config: ExperimentConfig, cache_path: Path, *, repeat: bool = True) -> Iterator[dict[str, Any]]:
    cache = LatentCacheDataset(cache_path)
    latent_stats = _latent_normalization_stats(cache, config)
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for trajectory_id in _selected_cache_trajectory_ids(cache, config):
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
    first_batch = next(_sequence_batches(config, cache_path, repeat=False))
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
    batches = _sequence_batches(config, cache_path, repeat=True)
    step_value = 0
    while step_value < config.training.max_steps:
        batch = _prepare_parallel_batch(next(batches), plan)
        if batch is None:
            continue
        state, metrics = step_fn(state, batch)
        host_state = _host_state(state, plan)
        host_metrics = _host_metrics(metrics, plan)
        step_value = int(host_state.step)
        row = {
            "step": step_value,
            "lr": train_config.training.learning_rate,
            **{k: float(v) for k, v in host_metrics.items()},
        }
        if step_value % config.training.log_every == 0 or step_value == config.training.max_steps:
            logger.log(row, prefix="train")
        if step_value % config.training.checkpoint_every == 0:
            save_checkpoint(host_state, out, step=step_value)
    final_state = _host_state(state, plan)
    metrics = _host_metrics(metrics, plan)
    ckpt = save_checkpoint(final_state, out, step=int(final_state.step))
    summary = {
        "step": int(final_state.step),
        "checkpoint": str(ckpt),
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
        return {
            "dry_run": True,
            "latent_cache": config.latent_cache.path,
            "train_split": "train",
            "eval_split": config.data.split,
            "flux_head": "ridge_linear",
            "ridge_alpha": config.evaluation.flux_head_ridge_alpha,
        }

    cache_path = _latent_cache_for(config)
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
        return {
            "dry_run": True,
            "latent_cache": config.latent_cache.path,
            "perplexities": list(config.evaluation.tsne_perplexities),
            "max_points": config.evaluation.representation_max_points,
        }

    cache_path = _latent_cache_for(config)
    cache = LatentCacheDataset(cache_path)
    configured_ids = _cache_trajectory_ids(cache, config)
    out = _output_dir(config)
    write_run_metadata(out, config=config.model_dump(mode="json"))
    (out / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    summary = evaluate_representation(
        cache,
        out,
        split_seed=config.data.seed,
        trajectory_ids=configured_ids,
        perplexities=config.evaluation.tsne_perplexities,
        tsne_max_iter=config.evaluation.tsne_max_iter,
        max_points=config.evaluation.representation_max_points,
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
    if dry_run:
        return {
            "dry_run": True,
            "latent_cache": config.latent_cache.path,
            "sequence_checkpoint": config.latent_cache.sequence_checkpoint_path,
            "rollout_steps": config.evaluation.rollout_steps,
        }

    cache_path = _latent_cache_for(config)
    cache = LatentCacheDataset(cache_path)
    configured_ids = _cache_trajectory_ids(cache, config)
    selected_ids = _selected_cache_trajectory_ids(cache, config)
    latent_stats = _latent_normalization_stats(cache, config)
    use_persistence = config.latent_cache.use_persistence_baseline
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
    elif config.latent_cache.encoder_checkpoint_path:
        encoder_ckpt_path = Path(config.latent_cache.encoder_checkpoint_path)
        if encoder_ckpt_path.exists():
            encoder_params = load_checkpoint(encoder_ckpt_path)["params"]
            diagnostic_params = _diagnostic_params_from_encoder_params(encoder_params)
            if diagnostic_params is None and diagnostic_metrics_requested:
                diagnostic_warnings.append(
                    "diagnostic metrics requested but encoder checkpoint has no diagnostic_heads params"
                )
        elif diagnostic_metrics_requested:
            diagnostic_warnings.append(
                f"diagnostic metrics requested but encoder checkpoint not found: {encoder_ckpt_path}"
            )
    elif diagnostic_metrics_requested:
        diagnostic_warnings.append("diagnostic metrics requested but latent_cache.encoder_checkpoint_path is unset")
    diagnostic_heads_loaded = diagnostic_model is not None and diagnostic_params is not None
    if not use_persistence:
        if config.model.sequence is None or config.latent_cache.sequence_checkpoint_path is None:
            raise ValueError("sequence model and checkpoint are required unless persistence baseline is enabled")
        model = build_sequence_model(config.model.sequence)
        params = load_checkpoint(config.latent_cache.sequence_checkpoint_path)["params"]
    preds = []
    targets = []
    rollout_trajectory_ids: list[str] = []
    flux_targets = []
    spectra_targets: dict[str, list[np.ndarray]] = {key: [] for key in config.data.target_spectra}
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
            if use_persistence:
                pred = persistence_rollout(context_b, config.evaluation.rollout_steps)
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
            if diagnostics.flux is not None:
                flux_targets.append(np.asarray(diagnostics.flux[None, ...], dtype=np.float32))
            for key in spectra_targets:
                if key in diagnostics.spectra:
                    spectra_targets[key].append(np.asarray(diagnostics.spectra[key][None, ...], dtype=np.float32))
    if not preds:
        raise ValueError("no valid rollout windows found")
    pred_all = jnp.asarray(np.concatenate(preds, axis=0))
    target_all = jnp.asarray(np.concatenate(targets, axis=0))
    metrics = trajectory_balanced_rollout_metrics(pred_all, target_all, rollout_trajectory_ids)
    summary = {key: np.asarray(jax.device_get(value)) for key, value in metrics.items()}
    flux_metrics_computed = False
    spectra_metrics_computed = False
    diagnostic_samples: dict[str, np.ndarray] = {}
    if diagnostic_model is not None and diagnostic_params is not None:
        flat_pred = pred_all.reshape((-1, pred_all.shape[-1]))
        diagnostics = diagnostic_model.apply({"params": diagnostic_params}, flat_pred, train=False)
        sample_windows = min(16, int(pred_all.shape[0]))
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
                jnp.mean(
                    jnp.abs(flux_error) / (jnp.abs(flux_target) + config.loss.spectra_epsilon), axis=2
                ),
                rollout_trajectory_ids,
            )
            flux_mse_by_step = jnp.mean(flux_mse_by_trajectory, axis=0)
            flux_mae_by_step = jnp.mean(flux_mae_by_trajectory, axis=0)
            flux_relative_error_by_step = jnp.mean(flux_relative_error_by_trajectory, axis=0)
            flux_mse_value = jnp.mean(flux_mse_by_step)
            flux_time_average_by_trajectory = _trajectory_time_average_errors(
                flux_pred, flux_target, rollout_trajectory_ids
            )
            summary["flux_mse_by_step"] = np.asarray(jax.device_get(flux_mse_by_step))
            summary["flux_mse_std_by_step"] = np.asarray(
                jax.device_get(jnp.std(flux_mse_by_trajectory, axis=0))
            )
            summary["flux_rmse_by_step"] = np.asarray(jax.device_get(jnp.sqrt(flux_mse_by_step)))
            summary["flux_rmse_std_by_step"] = np.asarray(
                jax.device_get(jnp.std(jnp.sqrt(flux_mse_by_trajectory), axis=0))
            )
            summary["flux_mae_by_step"] = np.asarray(jax.device_get(flux_mae_by_step))
            summary["flux_mae_std_by_step"] = np.asarray(
                jax.device_get(jnp.std(flux_mae_by_trajectory, axis=0))
            )
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
                pred_centered = spectra_pred - jnp.mean(spectra_pred, axis=2, keepdims=True)
                target_centered = spectra_target - jnp.mean(spectra_target, axis=2, keepdims=True)
                shape_corr_by_trajectory = _trajectory_values_by_step(
                    jnp.sum(pred_centered * target_centered, axis=2)
                    / (
                        jnp.linalg.norm(pred_centered, axis=2)
                        * jnp.linalg.norm(target_centered, axis=2)
                        + config.loss.spectra_epsilon
                    ),
                    rollout_trajectory_ids,
                )
                log_by_step = jnp.mean(log_by_trajectory, axis=0)
                relative_l2_by_step = jnp.mean(relative_l2_by_trajectory, axis=0)
                shape_corr_by_step = jnp.mean(shape_corr_by_trajectory, axis=0)
                summary[f"spectra_{key}_mse_by_step"] = np.asarray(jax.device_get(by_step))
                summary[f"spectra_{key}_mse_std_by_step"] = np.asarray(
                    jax.device_get(jnp.std(by_trajectory, axis=0))
                )
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
        result["plot"] = str(plots["latent_mse"])
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
