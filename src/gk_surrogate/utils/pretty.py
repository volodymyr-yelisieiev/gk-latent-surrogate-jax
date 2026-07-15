"""Formatting helpers for metric dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def scalarize(value: Any) -> float | int | str | bool | None:
    if value is None or isinstance(value, (float, int, str, bool)):
        return value
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    if arr.dtype.kind in {"O", "S", "U"}:
        return ",".join(str(item) for item in arr.reshape(-1))
    return float(np.mean(arr))


def flatten_dict(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_dict(value, name))
        else:
            flat[name] = scalarize(value)
    return flat
