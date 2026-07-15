"""Optax optimizer construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import optax


def _get(config: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if is_dataclass(config):
        config = asdict(config)
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def learning_rate_schedule(config: Mapping[str, Any] | Any) -> optax.Schedule:
    lr = float(_get(config, "learning_rate", _get(config, "lr", 1e-3)))
    warmup_steps = int(_get(config, "warmup_steps", 0) or 0)
    decay_steps = _get(config, "decay_steps", _get(config, "max_steps", None))
    min_lr = float(_get(config, "min_learning_rate", _get(config, "min_lr", 0.0)) or 0.0)

    if decay_steps is None or int(decay_steps) <= max(warmup_steps, 1):
        if warmup_steps <= 0:
            return optax.constant_schedule(lr)
        return optax.warmup_constant_schedule(init_value=0.0, peak_value=lr, warmup_steps=warmup_steps)

    cosine = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=max(int(decay_steps) - warmup_steps, 1),
        alpha=min_lr / lr if lr > 0 else 0.0,
    )
    if warmup_steps <= 0:
        return cosine
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps),
            cosine,
        ],
        boundaries=[warmup_steps],
    )


def build_optimizer(config: Mapping[str, Any] | Any) -> optax.GradientTransformation:
    """Build AdamW with optional global-norm clipping and LR scheduling."""

    weight_decay = float(_get(config, "weight_decay", 0.0) or 0.0)
    clip_norm = _get(config, "gradient_clip_norm", _get(config, "clip_norm", None))
    schedule = learning_rate_schedule(config)

    transforms: list[optax.GradientTransformation] = []
    if clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(clip_norm)))
    transforms.append(optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    return optax.chain(*transforms)
