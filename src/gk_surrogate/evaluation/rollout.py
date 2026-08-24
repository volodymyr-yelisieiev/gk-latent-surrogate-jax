"""Autoregressive latent rollout and metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def _call_model(model_apply: Any, params: Any, context: Array, *, train: bool) -> Any:
    variables = {"params": params}
    try:
        return model_apply(variables, context, train=train)
    except TypeError:
        try:
            return model_apply(variables, context)
        except TypeError:
            try:
                return model_apply(params, context, train=train)
            except TypeError:
                return model_apply(params, context)


def _prediction_array(output: Any) -> Array:
    if isinstance(output, Mapping):
        for key in ("z", "prediction", "pred", "latent"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError("rollout model output mapping has no prediction key")
    pred = jnp.asarray(output)
    if pred.ndim == 3:
        pred = pred[:, -1, :]
    if pred.ndim != 2:
        raise ValueError(f"Expected one-step prediction shape [B, Z], got {pred.shape}.")
    return pred


def autoregressive_rollout(
    model_apply: Any,
    params: Any,
    z_initial_context: Array,
    rollout_steps: int,
    *,
    train: bool = False,
) -> Array:
    """Return recursive latent predictions with shape ``[B, rollout_steps, Z]``."""

    if z_initial_context.ndim != 3:
        raise ValueError("z_initial_context must have shape [B, T_context, Z].")
    if rollout_steps < 0:
        raise ValueError("rollout_steps must be non-negative")
    if rollout_steps == 0:
        raise ValueError("rollout_steps must be positive")
    context = jnp.asarray(z_initial_context)

    def step(carry: Array, _: Array) -> tuple[Array, Array]:
        output = _call_model(model_apply, params, carry, train=train)
        pred = _prediction_array(output)
        next_context = jnp.concatenate([carry[:, 1:, :], pred[:, None, :]], axis=1)
        return next_context, pred

    _, preds = jax.lax.scan(step, context, jnp.arange(rollout_steps))
    return jnp.swapaxes(preds, 0, 1)


def persistence_rollout(z_initial_context: Array, rollout_steps: int) -> Array:
    """Repeat the last latent state.

    Diagnostic metrics computed from this rollout pass the repeated latent through
    the frozen diagnostic head.  This is therefore a latent-state persistence
    baseline, not persistence of an observed diagnostic such as flux.
    """
    last = jnp.asarray(z_initial_context)[:, -1:, :]
    return jnp.repeat(last, rollout_steps, axis=1)


def observed_diagnostic_persistence(last_observed: Array, rollout_steps: int) -> Array:
    """Repeat the last observed diagnostic over the forecast horizon.

    ``last_observed`` may have shape ``[B, D]`` or ``[B, 1, D]``.  Unlike
    :func:`persistence_rollout`, this reference does not use a latent encoder or
    diagnostic head.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    observed = jnp.asarray(last_observed)
    if observed.ndim == 2:
        observed = observed[:, None, :]
    if observed.ndim != 3 or observed.shape[1] != 1:
        raise ValueError(f"last_observed must have shape [B, D] or [B, 1, D], got {observed.shape}")
    return jnp.repeat(observed, rollout_steps, axis=1)


def latent_rollout_metrics(pred: Array, target: Array, eps: float = 1e-8) -> dict[str, Array]:
    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have same shape, got {pred.shape} and {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"pred and target must have shape [B, T, Z], got {pred.shape}")
    error = pred - target
    mse_per_rollout = jnp.mean(jnp.square(error), axis=-1)
    mae_per_rollout = jnp.mean(jnp.abs(error), axis=-1)
    relative_l2_by_step = jnp.linalg.norm(error, axis=(0, 2)) / (jnp.linalg.norm(target, axis=(0, 2)) + eps)
    numerator = jnp.sum(pred * target, axis=-1)
    pred_norm = jnp.sqrt(jnp.sum(jnp.square(pred), axis=-1) + eps**2)
    target_norm = jnp.sqrt(jnp.sum(jnp.square(target), axis=-1) + eps**2)
    denom = pred_norm * target_norm
    cosine_per_rollout = numerator / denom
    mse_by_step = jnp.mean(mse_per_rollout, axis=0)
    mae_by_step = jnp.mean(mae_per_rollout, axis=0)
    cosine_by_step = jnp.mean(cosine_per_rollout, axis=0)
    return {
        "mse_by_step": mse_by_step,
        "mse_std_by_step": jnp.std(mse_per_rollout, axis=0),
        "mae_by_step": mae_by_step,
        "mae_std_by_step": jnp.std(mae_per_rollout, axis=0),
        "relative_l2_by_step": relative_l2_by_step,
        "cosine_by_step": cosine_by_step,
        "cosine_std_by_step": jnp.std(cosine_per_rollout, axis=0),
        "mse": jnp.mean(mse_by_step),
        "mae": jnp.mean(mae_by_step),
        "relative_l2": jnp.mean(relative_l2_by_step),
        "cosine": jnp.mean(cosine_by_step),
        "stable": jnp.all(jnp.isfinite(pred)),
        "stable_fraction": jnp.mean(jnp.all(jnp.isfinite(pred), axis=(1, 2))),
        "num_rollouts": jnp.asarray(pred.shape[0], dtype=jnp.int32),
    }


def horizon_until_threshold(mse_by_step: Array, threshold: float) -> int:
    """Return first 1-based horizon whose error exceeds threshold, or T if none."""

    values = np.asarray(mse_by_step)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("mse_by_step must be a non-empty one-dimensional array")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    crossed = np.nonzero(~np.isfinite(values) | (values > threshold))[0]
    return int(crossed[0] + 1) if len(crossed) else int(values.shape[0])


def trajectory_balanced_rollout_metrics(
    pred: Array,
    target: Array,
    trajectory_ids: Iterable[str],
    eps: float = 1e-8,
) -> dict[str, Array]:
    """Aggregate rollout metrics with equal weight per trajectory.

    Overlapping windows from a long trajectory must not give that trajectory more
    weight than a shorter one. The reported standard deviation is therefore the
    between-trajectory dispersion of each horizon metric.
    """

    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    ids = tuple(str(item) for item in trajectory_ids)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"pred and target must have the same [B, T, Z] shape, got {pred.shape} and {target.shape}")
    if len(ids) != pred.shape[0]:
        raise ValueError(f"trajectory_ids length {len(ids)} does not match batch size {pred.shape[0]}")
    unique_ids = tuple(dict.fromkeys(ids))
    if not unique_ids:
        raise ValueError("trajectory_ids must not be empty")

    per_trajectory = []
    for trajectory_id in unique_ids:
        indices = np.asarray([index for index, item in enumerate(ids) if item == trajectory_id], dtype=np.int32)
        per_trajectory.append(latent_rollout_metrics(pred[indices], target[indices], eps=eps))

    result: dict[str, Array] = {}
    for key in ("mse", "mae", "relative_l2", "cosine"):
        values = jnp.stack([metrics[f"{key}_by_step"] for metrics in per_trajectory])
        result[f"{key}_by_trajectory"] = jnp.mean(values, axis=1)
        result[f"{key}_by_step"] = jnp.mean(values, axis=0)
        result[f"{key}_std_by_step"] = jnp.std(values, axis=0)
        result[key] = jnp.mean(result[f"{key}_by_step"])
    stable_by_trajectory = jnp.stack([metrics["stable"] for metrics in per_trajectory])
    result["stable"] = jnp.all(stable_by_trajectory)
    result["stable_fraction"] = jnp.mean(stable_by_trajectory.astype(jnp.float32))
    result["num_rollouts"] = jnp.asarray(pred.shape[0], dtype=jnp.int32)
    result["num_trajectories"] = jnp.asarray(len(unique_ids), dtype=jnp.int32)
    return result


def evaluate_rollout_batches(
    model_apply: Any,
    params: Any,
    batches: Iterable[Mapping[str, Array]],
    *,
    rollout_steps: int,
    use_persistence: bool = False,
) -> dict[str, Any]:
    preds = []
    targets = []
    for batch in batches:
        context = batch.get("z_context", batch.get("context"))
        target = batch.get("z_target", batch.get("target"))
        if context is None or target is None:
            raise KeyError("rollout batch must contain context/target arrays")
        target = jnp.asarray(target)
        if target.ndim != 3 or target.shape[1] < rollout_steps:
            raise ValueError(f"rollout target must have shape [B, T, Z] with T >= {rollout_steps}, got {target.shape}")
        pred = (
            persistence_rollout(context, rollout_steps)
            if use_persistence
            else autoregressive_rollout(model_apply, params, context, rollout_steps)
        )
        preds.append(np.asarray(jax.device_get(pred), dtype=np.float32))
        targets.append(np.asarray(target[:, :rollout_steps, :], dtype=np.float32))
    if not preds:
        raise ValueError("no rollout batches were provided")
    pred_all = jnp.asarray(np.concatenate(preds, axis=0))
    target_all = jnp.asarray(np.concatenate(targets, axis=0))
    return latent_rollout_metrics(pred_all, target_all)
