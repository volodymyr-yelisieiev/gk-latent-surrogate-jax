"""Metric aggregation and serialization helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _to_python(value: Any) -> Any:
    value = jax.device_get(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _to_python(v) for k, v in value.items()}
    return value


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, name))
        else:
            flat[name] = value
    return flat


def aggregate_metrics(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batches:
        return {}
    flat_batches = [flatten_metrics(batch) for batch in batches]
    keys = sorted(set().union(*(batch.keys() for batch in flat_batches)))
    aggregated = {}
    for key in keys:
        values = [jnp.asarray(batch[key]) for batch in flat_batches if key in batch]
        shapes = {value.shape for value in values}
        if len(shapes) > 1:
            raise ValueError(f"Cannot aggregate metric {key!r} with inconsistent shapes {sorted(shapes)}.")
        if values:
            aggregated[key] = jnp.mean(jnp.stack(values), axis=0)
    return aggregated


def save_metrics_json(metrics: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_python(flatten_metrics(metrics)), indent=2, sort_keys=True))


def save_metrics_csv(metrics: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = _to_python(flatten_metrics(metrics))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(flat.items()):
            writer.writerow([key, json.dumps(value) if isinstance(value, list | dict) else value])
