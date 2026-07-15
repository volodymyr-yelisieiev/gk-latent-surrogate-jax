"""Deterministic trajectory-level splitting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def split_trajectory_ids(
    trajectory_ids: Sequence[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, tuple[str, ...]]:
    if len(trajectory_ids) < 2:
        msg = "trajectory-level train/val/test split requires at least two trajectories"
        raise ValueError(msg)
    if len(set(trajectory_ids)) != len(trajectory_ids):
        msg = "trajectory IDs must be unique before splitting"
        raise ValueError(msg)
    if len(ratios) != 3:
        msg = "split ratios must contain train, validation, and test values"
        raise ValueError(msg)
    if any(not np.isfinite(ratio) or ratio < 0 for ratio in ratios):
        msg = "split ratios must be finite and non-negative"
        raise ValueError(msg)
    total = float(sum(ratios))
    if total <= 0:
        msg = "split ratios must sum to a positive value"
        raise ValueError(msg)
    normalized = tuple(ratio / total for ratio in ratios)
    rng = np.random.default_rng(seed)
    ids = np.asarray(tuple(trajectory_ids), dtype=object)
    shuffled = ids[rng.permutation(len(ids))]
    counts = _allocate_split_counts(len(ids), normalized)
    n_train, n_val, _n_test = counts
    train = tuple(str(value) for value in shuffled[:n_train])
    val = tuple(str(value) for value in shuffled[n_train : n_train + n_val])
    test = tuple(str(value) for value in shuffled[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def _allocate_split_counts(size: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    raw = np.asarray(ratios, dtype=np.float64) * size
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts), kind="stable")[: size - int(np.sum(counts))]:
        counts[index] += 1

    positive = np.flatnonzero(np.asarray(ratios) > 0)
    if size >= len(positive):
        for empty_index in positive[counts[positive] == 0]:
            donors = np.flatnonzero(counts > 1)
            if donors.size == 0:
                break
            donor = donors[np.argmax(counts[donors] - raw[donors])]
            counts[donor] -= 1
            counts[empty_index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])
