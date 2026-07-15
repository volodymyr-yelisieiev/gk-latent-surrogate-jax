"""Latent dynamics losses."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def _require_compatible_latents(pred: Array, target: Array) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Latent prediction and target shapes must match, got {pred.shape} and {target.shape}.")
    if pred.ndim == 0:
        raise ValueError("Latent prediction and target must have at least one dimension.")


def _cosine_loss(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    pred_norm = pred / jnp.sqrt(jnp.sum(pred**2, axis=-1, keepdims=True) + eps)
    target_norm = target / jnp.sqrt(jnp.sum(target**2, axis=-1, keepdims=True) + eps)
    return 1.0 - jnp.mean(jnp.sum(pred_norm * target_norm, axis=-1))


def latent_prediction_loss(pred: Array, target: Array, mode: str = "mse") -> Array:
    _require_compatible_latents(pred, target)
    match mode:
        case "mse":
            return jnp.mean(jnp.square(pred - target))
        case "huber":
            error = pred - target
            abs_error = jnp.abs(error)
            quadratic = jnp.minimum(abs_error, 1.0)
            linear = abs_error - quadratic
            return jnp.mean(0.5 * quadratic**2 + linear)
        case "cosine":
            return _cosine_loss(pred, target)
        case "mse_plus_cosine":
            return jnp.mean(jnp.square(pred - target)) + _cosine_loss(pred, target)
        case _:
            raise ValueError(f"Unsupported latent loss mode {mode!r}; expected mse, huber, cosine, or mse_plus_cosine.")
