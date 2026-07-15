"""PyTree helpers used by training and tests."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def tree_l2_norm(tree: Any) -> jax.Array:
    leaves = [jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def tree_allclose(a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(bool(jnp.allclose(x, y, rtol=rtol, atol=atol)) for x, y in zip(leaves_a, leaves_b, strict=True))


def tree_has_changed(a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return not tree_allclose(a, b, rtol=rtol, atol=atol)
