"""Batch shape helpers for optional data-parallel training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp


def shard_batch(batch: Mapping[str, Any], num_devices: int) -> dict[str, Any]:
    """Reshape a global batch into ``[num_devices, per_device_batch, ...]`` leaves."""

    if num_devices < 1:
        raise ValueError("num_devices must be positive")
    batch_size, per_device = infer_global_and_per_device_batch_size(batch, num_devices)
    if batch_size != per_device * num_devices:
        msg = f"global batch size {batch_size} is not divisible by {num_devices} devices"
        raise ValueError(msg)

    def shard_leaf(value: Any) -> Any:
        if value is None:
            return None
        array = jnp.asarray(value)
        if array.ndim == 0:
            msg = "batch leaves must have a leading batch axis"
            raise ValueError(msg)
        return array.reshape((num_devices, per_device, *array.shape[1:]))

    return jax.tree_util.tree_map(shard_leaf, dict(batch), is_leaf=lambda item: item is None)


def unshard_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a ``[num_devices, per_device_batch, ...]`` batch back to global form."""

    def unshard_leaf(value: Any) -> Any:
        if value is None:
            return None
        array = jnp.asarray(value)
        if array.ndim < 2:
            msg = "sharded batch leaves must have device and per-device batch axes"
            raise ValueError(msg)
        return array.reshape((array.shape[0] * array.shape[1], *array.shape[2:]))

    return jax.tree_util.tree_map(unshard_leaf, dict(batch), is_leaf=lambda item: item is None)


def drop_or_pad_to_multiple(
    batch: Mapping[str, Any],
    multiple: int,
    *,
    drop_remainder: bool = True,
) -> dict[str, Any] | None:
    """Return a batch whose leading dimension is divisible by ``multiple``.

    With ``drop_remainder=True`` an undersized tail is dropped; if that would leave
    zero samples, ``None`` is returned. With ``drop_remainder=False`` the final
    samples are repeated to pad the batch.
    """

    if multiple < 1:
        raise ValueError("multiple must be positive")
    batch_size = _infer_batch_size(batch)
    remainder = batch_size % multiple
    if remainder == 0:
        return dict(batch)
    if drop_remainder:
        keep = batch_size - remainder
        if keep <= 0:
            return None
        return _slice_batch(batch, keep)
    pad = multiple - remainder
    return _pad_batch(batch, pad)


def infer_global_and_per_device_batch_size(batch: Mapping[str, Any], num_devices: int) -> tuple[int, int]:
    """Infer global and per-device batch sizes for a sharded step."""

    if num_devices < 1:
        raise ValueError("num_devices must be positive")
    batch_size = _infer_batch_size(batch)
    if batch_size % num_devices != 0:
        msg = f"global batch size {batch_size} is not divisible by {num_devices} devices"
        raise ValueError(msg)
    return batch_size, batch_size // num_devices


def _infer_batch_size(batch: Mapping[str, Any]) -> int:
    leaves = jax.tree_util.tree_leaves(batch)
    for leaf in leaves:
        if leaf is None:
            continue
        array = jnp.asarray(leaf)
        if array.ndim > 0:
            batch_size = int(array.shape[0])
            if batch_size < 1:
                raise ValueError("batch size must be positive")
            return batch_size
    raise ValueError("batch has no array leaves with a leading batch axis")


def _slice_batch(batch: Mapping[str, Any], keep: int) -> dict[str, Any]:
    def slice_leaf(value: Any) -> Any:
        if value is None:
            return None
        return jnp.asarray(value)[:keep]

    return jax.tree_util.tree_map(slice_leaf, dict(batch), is_leaf=lambda item: item is None)


def _pad_batch(batch: Mapping[str, Any], pad: int) -> dict[str, Any]:
    def pad_leaf(value: Any) -> Any:
        if value is None:
            return None
        array = jnp.asarray(value)
        if array.ndim == 0:
            msg = "batch leaves must have a leading batch axis"
            raise ValueError(msg)
        if array.shape[0] == 0:
            msg = "cannot pad an empty batch"
            raise ValueError(msg)
        pad_source = array[-1:, ...]
        pad_rows = jnp.repeat(pad_source, pad, axis=0)
        return jnp.concatenate([array, pad_rows], axis=0)

    return jax.tree_util.tree_map(pad_leaf, dict(batch), is_leaf=lambda item: item is None)
