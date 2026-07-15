"""Reproducible PRNG utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array


def make_rng(seed: int) -> Array:
    return jax.random.PRNGKey(int(seed))


def split_rng(rng: Array, num: int = 2) -> tuple[Array, ...]:
    return tuple(jax.random.split(rng, num))


def fold_in_rng(rng: Array, data: int) -> Array:
    return jax.random.fold_in(rng, int(data))


@dataclass
class PRNGSequence:
    """Small Python-side key sequence for input pipelines and loops."""

    seed: int

    def __post_init__(self) -> None:
        self._rng = make_rng(self.seed)

    @property
    def key(self) -> Array:
        return self._rng

    def next(self) -> Array:
        self._rng, child = jax.random.split(self._rng)
        return child

    def split(self, num: int) -> tuple[Array, ...]:
        keys = jax.random.split(self._rng, num + 1)
        self._rng = keys[0]
        return tuple(keys[1:])
