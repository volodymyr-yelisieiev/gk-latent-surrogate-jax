"""Encoder batching helper for latent-cache generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def _call_encoder(apply_fn: Any, params: Any, x: Array, *, train: bool = False) -> Any:
    variables = {"params": params}
    return apply_fn(variables, x, train=train)


def _latent_from_output(output: Any) -> Array:
    if isinstance(output, Mapping):
        for key in ("z", "latent", "embedding"):
            if key in output:
                return output[key]
        raise KeyError("encoder output mapping has no latent key")
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "z"):
        return output.z
    return output


def encode_snapshots(
    apply_fn: Any,
    params: Any,
    snapshots: Array,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode snapshots ``[T, ...]`` into NumPy latents ``[T, Z]``."""

    latents: list[np.ndarray] = []
    for start in range(0, int(snapshots.shape[0]), batch_size):
        batch = jnp.asarray(snapshots[start : start + batch_size])
        output = _call_encoder(apply_fn, params, batch, train=False)
        z = _latent_from_output(output)
        latents.append(np.asarray(jax.device_get(z), dtype=np.float32))
    if not latents:
        raise ValueError("cannot encode an empty snapshot array")
    return np.concatenate(latents, axis=0)
