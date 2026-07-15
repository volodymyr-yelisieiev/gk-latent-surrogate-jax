"""SimSiam negative-cosine losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _require_compatible_vectors(first: Array, second: Array) -> None:
    if first.shape != second.shape:
        raise ValueError(f"SimSiam vector shapes must match, got {first.shape} and {second.shape}.")
    if first.ndim < 2:
        raise ValueError("SimSiam vectors must have shape [B, D] or higher rank with features last.")


def _l2_normalize(x: Array, eps: float) -> Array:
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps**2)
    return x / norm


def negative_cosine_similarity(p: Array, z_stop: Array, eps: float = 1e-8) -> Array:
    """Return mean negative cosine similarity with stop-gradient on target."""

    _require_compatible_vectors(p, z_stop)
    z_stop = jax.lax.stop_gradient(z_stop)
    p_norm = _l2_normalize(p, eps)
    z_norm = _l2_normalize(z_stop, eps)
    return -jnp.mean(jnp.sum(p_norm * z_norm, axis=-1))


def simsiam_loss(p1: Array, z2: Array, p2: Array, z1: Array, eps: float = 1e-8) -> Array:
    """Symmetric SimSiam loss.

    ``p1`` and ``p2`` are predictor outputs. ``z1`` and ``z2`` are projection
    targets and are stopped internally.
    """

    return 0.5 * (negative_cosine_similarity(p1, z2, eps=eps) + negative_cosine_similarity(p2, z1, eps=eps))
