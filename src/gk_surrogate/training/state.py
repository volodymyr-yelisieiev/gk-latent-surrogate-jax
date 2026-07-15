"""Shared train state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import optax
from flax import struct
from flax.core import freeze
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Flax train state with RNG and model metadata.

    ``apply_fn`` and ``model_config`` are static leaves, which keeps JIT-compiled
    train steps usable with normal Python callables and configuration mappings.
    """

    rng: jax.Array
    model_config: Mapping[str, Any] = struct.field(pytree_node=False)
    batch_stats: Any | None = None
    ema_params: Any | None = None

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable[..., Any],
        params: Any,
        tx: optax.GradientTransformation,
        rng: jax.Array,
        model_config: Mapping[str, Any] | None = None,
        batch_stats: Any | None = None,
        ema_params: Any | None = None,
        **kwargs: Any,
    ) -> TrainState:
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            rng=rng,
            model_config=freeze(dict(model_config or {})),
            batch_stats=batch_stats,
            ema_params=ema_params,
            **kwargs,
        )

    def replace_rng(self, rng: jax.Array) -> TrainState:
        return self.replace(rng=rng)
