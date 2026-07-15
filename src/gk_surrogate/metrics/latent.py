"""Latent-space metrics."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def _require_same_shape(pred: Array, target: Array, *, rollout: bool = False) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes must match, got {pred.shape} and {target.shape}.")
    expected_rank = 3 if rollout else None
    if expected_rank is not None and pred.ndim != expected_rank:
        raise ValueError("Expected rollout arrays with shape [B, T, Z].")
    if not rollout and pred.ndim == 0:
        raise ValueError("Latent arrays must have at least one dimension.")


def latent_mse(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target)
    return jnp.mean(jnp.square(pred - target))


def latent_mae(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target)
    return jnp.mean(jnp.abs(pred - target))


def latent_relative_l2(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    _require_same_shape(pred, target)
    return jnp.linalg.norm(pred - target) / (jnp.linalg.norm(target) + eps)


def latent_cosine_similarity(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    _require_same_shape(pred, target)
    pred_norm = pred / jnp.sqrt(jnp.sum(pred**2, axis=-1, keepdims=True) + eps)
    target_norm = target / jnp.sqrt(jnp.sum(target**2, axis=-1, keepdims=True) + eps)
    return jnp.mean(jnp.sum(pred_norm * target_norm, axis=-1))


def rollout_mse_by_step(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target, rollout=True)
    return jnp.mean(jnp.square(pred - target), axis=(0, 2))


def rollout_cosine_by_step(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    _require_same_shape(pred, target, rollout=True)
    pred_norm = pred / jnp.sqrt(jnp.sum(pred**2, axis=-1, keepdims=True) + eps)
    target_norm = target / jnp.sqrt(jnp.sum(target**2, axis=-1, keepdims=True) + eps)
    return jnp.mean(jnp.sum(pred_norm * target_norm, axis=-1), axis=0)
