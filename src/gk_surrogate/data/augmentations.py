"""Shape-preserving JAX augmentations for SimSiam positive views."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AugmentationConfig:
    gaussian_noise_sigma: float = 0.0
    element_mask_probability: float = 0.0
    channel_dropout_probability: float = 0.0
    max_periodic_shift: int = 0
    amplitude_jitter: float = 0.0


def augment_snapshot(x: jax.Array, key: jax.Array, config: AugmentationConfig) -> jax.Array:
    """Apply configured augmentations without changing shape or mutating input."""

    out = jnp.asarray(x)
    keys = jax.random.split(key, 5)
    if config.gaussian_noise_sigma > 0:
        scale = jnp.std(out)
        noise = jax.random.normal(keys[0], out.shape, dtype=out.dtype)
        out = out + config.gaussian_noise_sigma * scale * noise
    if config.element_mask_probability > 0:
        keep = jax.random.bernoulli(keys[1], 1.0 - config.element_mask_probability, out.shape)
        out = jnp.where(keep, out, jnp.zeros_like(out))
    if config.channel_dropout_probability > 0:
        keep = jax.random.bernoulli(
            keys[2],
            1.0 - config.channel_dropout_probability,
            (out.shape[0],) + (1,) * (out.ndim - 1),
        )
        out = jnp.where(keep, out, jnp.zeros_like(out))
    if config.max_periodic_shift > 0:
        shifts = jax.random.randint(
            keys[3],
            shape=(out.ndim - 1,),
            minval=-config.max_periodic_shift,
            maxval=config.max_periodic_shift + 1,
        )
        for axis, shift in zip(range(1, out.ndim), shifts, strict=True):
            size = out.shape[axis]
            indices = (jnp.arange(size) - shift) % size
            out = jnp.take(out, indices, axis=axis)
    if config.amplitude_jitter > 0:
        scale = jax.random.uniform(
            keys[4],
            (),
            minval=1.0 - config.amplitude_jitter,
            maxval=1.0 + config.amplitude_jitter,
        )
        out = scale * out
    return out


def make_positive_pair(
    x: jax.Array,
    key: jax.Array,
    config: AugmentationConfig,
) -> tuple[jax.Array, jax.Array]:
    key1, key2 = jax.random.split(key)
    return augment_snapshot(x, key1, config), augment_snapshot(x, key2, config)
