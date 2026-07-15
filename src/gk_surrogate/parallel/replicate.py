"""Replication helpers for pmapped train states and trees."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def replicate_state(state: Any, devices: tuple[Any, ...] | None = None) -> Any:
    return _replicate_tree(state, devices)


def unreplicate_state(state: Any) -> Any:
    return unreplicate_tree(state)


def replicate_params(params: Any, devices: tuple[Any, ...] | None = None) -> Any:
    return _replicate_tree(params, devices)


def unreplicate_params(params: Any) -> Any:
    return unreplicate_tree(params)


def unreplicate_tree(tree: Any) -> Any:
    """Take the first replica from every array leaf."""

    def first_replica(value: Any) -> Any:
        if value is None:
            return None
        array = jnp.asarray(value)
        if array.ndim == 0:
            return value
        return array[0]

    return jax.tree_util.tree_map(first_replica, tree, is_leaf=lambda item: item is None)


def _replicate_tree(tree: Any, devices: tuple[Any, ...] | None = None) -> Any:
    selected = devices or tuple(jax.local_devices())
    if not selected:
        raise ValueError("at least one device is required for replication")
    replica_count = len(selected)

    def replicate_leaf(value: Any) -> Any:
        if value is None:
            return None
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (replica_count, *array.shape))

    return jax.tree_util.tree_map(replicate_leaf, tree, is_leaf=lambda item: item is None)
