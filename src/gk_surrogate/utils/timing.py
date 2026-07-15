"""Timing helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    start = perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = perf_counter() - start
