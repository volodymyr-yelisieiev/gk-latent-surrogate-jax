"""Rollout metric helpers.

Autoregressive rollout execution lives in :mod:`gk_surrogate.evaluation.rollout`.
``autoregressive_rollout`` is re-exported here for backward compatibility with
early smoke tests and user imports.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from gk_surrogate.evaluation.rollout import autoregressive_rollout as autoregressive_rollout
from gk_surrogate.metrics.latent import rollout_cosine_by_step, rollout_mse_by_step

Array = jnp.ndarray

__all__ = [
    "autoregressive_rollout",
    "horizon_until_threshold",
    "rollout_stability",
    "summarize_rollout",
]


def rollout_stability(pred: Array) -> Array:
    return jnp.all(jnp.isfinite(pred))


def horizon_until_threshold(error_by_step: Array, threshold: float) -> Array:
    """Return first 1-based horizon whose error exceeds threshold, or T if none."""

    if error_by_step.ndim != 1 or error_by_step.shape[0] == 0:
        raise ValueError("error_by_step must be a non-empty one-dimensional array.")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")
    exceeded = jnp.logical_or(~jnp.isfinite(error_by_step), error_by_step > threshold)
    first = jnp.argmax(exceeded)
    return jnp.where(jnp.any(exceeded), first + 1, error_by_step.shape[0])


def summarize_rollout(
    pred: Array,
    target: Array,
    *,
    error_threshold: float | None = None,
) -> dict[str, Array]:
    if pred.shape != target.shape:
        raise ValueError(f"Rollout prediction and target shapes must match, got {pred.shape} and {target.shape}.")
    mse_by_step = rollout_mse_by_step(pred, target)
    cosine_by_step = rollout_cosine_by_step(pred, target)
    metrics = {
        "latent/mse_by_step": mse_by_step,
        "latent/cosine_by_step": cosine_by_step,
        "latent/mse_mean": jnp.mean(mse_by_step),
        "latent/mse_std": jnp.std(jnp.mean(jnp.square(pred - target), axis=(1, 2))),
        "stability/finite": rollout_stability(pred),
    }
    if error_threshold is not None:
        metrics["horizon/error_threshold"] = horizon_until_threshold(mse_by_step, error_threshold)
    return metrics
