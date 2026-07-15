"""Reusable small-loop helpers for CPU smoke training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any


def cycle_batches(batches: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Yield batches forever from a finite or infinite iterable."""

    cached = list(batches)
    if not cached:
        raise ValueError("cannot cycle an empty batch iterable")
    while True:
        yield from cached


def run_fixed_steps(
    state: Any,
    batches: Iterable[Mapping[str, Any]],
    step_fn: Callable[[Any, Mapping[str, Any]], tuple[Any, Mapping[str, Any]]],
    *,
    max_steps: int,
) -> tuple[Any, list[dict[str, float]]]:
    """Run ``step_fn`` for a fixed number of steps and collect scalar metrics."""

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    iterator = cycle_batches(batches)
    history: list[dict[str, float]] = []
    for _ in range(max_steps):
        state, metrics = step_fn(state, next(iterator))
        history.append({key: float(value) for key, value in metrics.items()})
    return state, history
