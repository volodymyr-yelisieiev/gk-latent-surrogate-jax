"""Array validation and conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array


def as_float32(x: Any) -> Array:
    """Return ``x`` as a JAX float32 array."""

    return jnp.asarray(x, dtype=jnp.float32)


def to_numpy(x: Any) -> np.ndarray:
    """Move an array-like value to a NumPy array."""

    return np.asarray(x)


def assert_rank(x: Array, rank: int, name: str = "array") -> None:
    if x.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {x.shape}")


def assert_shape_suffix(x: Array, suffix: Sequence[int], name: str = "array") -> None:
    suffix_tuple = tuple(suffix)
    if tuple(x.shape[-len(suffix_tuple) :]) != suffix_tuple:
        raise ValueError(f"{name} must end with shape {suffix_tuple}, got {x.shape}")


def ensure_finite_tree(tree: Any, name: str = "tree") -> None:
    """Raise if any array leaf contains NaN or Inf."""

    def _check(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                _check(child, f"{path}.{key}")
            return
        arr = np.asarray(value)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values at {path}")

    _check(tree, name)


def mean_squared_error(pred: Array, target: Array) -> Array:
    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    return jnp.mean(jnp.square(pred - target))


def cosine_similarity(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    numerator = jnp.sum(pred * target, axis=-1)
    pred_norm = jnp.linalg.norm(pred, axis=-1)
    target_norm = jnp.linalg.norm(target, axis=-1)
    return numerator / jnp.maximum(pred_norm * target_norm, eps)
