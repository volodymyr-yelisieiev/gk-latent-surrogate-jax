from __future__ import annotations

import jax
import jax.numpy as jnp

from gk_surrogate.data.augmentations import AugmentationConfig, augment_snapshot, make_positive_pair


def test_augmentations_preserve_shape_deterministic_and_finite():
    x = jnp.ones((2, 4, 4, 4, 4, 4), dtype=jnp.float32)
    cfg = AugmentationConfig(
        gaussian_noise_sigma=0.01,
        element_mask_probability=0.1,
        channel_dropout_probability=0.1,
        amplitude_jitter=0.02,
    )
    key = jax.random.PRNGKey(0)
    y1 = augment_snapshot(x, key, cfg)
    y2 = augment_snapshot(x, key, cfg)
    assert y1.shape == x.shape
    assert jnp.allclose(y1, y2)
    assert jnp.isfinite(y1).all()
    assert jnp.allclose(augment_snapshot(x, key, AugmentationConfig()), x)


def test_positive_pair_shape():
    x = jnp.ones((2, 4, 4, 4, 4, 4), dtype=jnp.float32)
    a, b = make_positive_pair(x, jax.random.PRNGKey(1), AugmentationConfig(gaussian_noise_sigma=0.01))
    assert a.shape == b.shape == x.shape


def test_periodic_shift_is_jit_compatible_and_preserves_values():
    x = jnp.arange(2 * 3 * 2 * 2 * 2 * 2, dtype=jnp.float32).reshape((2, 3, 2, 2, 2, 2))
    cfg = AugmentationConfig(max_periodic_shift=1)
    fn = jax.jit(lambda value, key: augment_snapshot(value, key, cfg))
    y = fn(x, jax.random.PRNGKey(2))
    assert y.shape == x.shape
    assert jnp.allclose(jnp.sort(y.reshape(-1)), jnp.sort(x.reshape(-1)))
