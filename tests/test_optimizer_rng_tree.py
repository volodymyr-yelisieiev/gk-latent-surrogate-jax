from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gk_surrogate.training.optimizer import build_optimizer, learning_rate_schedule
from gk_surrogate.training.rng import PRNGSequence, fold_in_rng, make_rng, split_rng
from gk_surrogate.utils.tree import tree_allclose, tree_has_changed, tree_l2_norm


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    warmup_steps: int = 2
    max_steps: int = 6
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0


def test_optimizer_schedule_variants_and_dataclass_clip():
    warmup_constant = learning_rate_schedule({"learning_rate": 1e-3, "warmup_steps": 2})
    assert warmup_constant(0) == 0.0
    assert warmup_constant(2) == 1e-3

    cosine = learning_rate_schedule({"learning_rate": 1e-3, "warmup_steps": 0, "max_steps": 6})
    assert cosine(0) == 1e-3
    joined = learning_rate_schedule(OptimConfig())
    assert joined(0) == 0.0
    assert joined(3) <= 1e-3

    tx = build_optimizer(OptimConfig())
    params = {"w": jnp.ones((2,), dtype=jnp.float32)}
    updates, _ = tx.update({"w": jnp.ones((2,), dtype=jnp.float32) * 100.0}, tx.init(params), params)
    assert jnp.isfinite(tree_l2_norm(updates))


def test_prng_sequence_and_tree_helpers_cover_empty_and_changed_cases():
    rng = make_rng(7)
    a, b = split_rng(rng, 2)
    assert a.shape == b.shape == rng.shape
    assert not jnp.allclose(fold_in_rng(rng, 1), fold_in_rng(rng, 2))

    seq = PRNGSequence(seed=7)
    first = seq.next()
    second, third = seq.split(2)
    assert first.shape == second.shape == third.shape == seq.key.shape
    assert not jnp.allclose(first, second)

    assert tree_l2_norm({}) == 0.0
    assert not tree_allclose({"x": jnp.ones((1,))}, {"x": jnp.ones((1,)), "y": jnp.ones((1,))})
    assert tree_has_changed({"x": jnp.ones((1,))}, {"x": jnp.zeros((1,))})
    assert tree_allclose(jax.device_get({"x": jnp.ones((1,))}), {"x": jnp.ones((1,))})
