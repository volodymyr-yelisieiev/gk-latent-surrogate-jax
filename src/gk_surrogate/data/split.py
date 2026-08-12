"""Deterministic trajectory-level splitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class TrajectorySplits:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    strategy: str
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    fold_id: str | None = None

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def split_manifest_assigned_ids(manifest_path: str | Path) -> tuple[str, ...]:
    """Return all assigned IDs so config validation can check a manifest before dataset I/O."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"trajectory split manifest not found: {path}")
    raw = path.read_bytes()
    try:
        payload: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trajectory split manifest must contain a mapping")
    source = payload.get("splits", payload)
    if not isinstance(source, dict):
        raise ValueError("trajectory split manifest 'splits' must be a mapping")
    assigned: list[str] = []
    for role in ("train", "val", "test"):
        value = source.get(role, source.get(f"{role}_trajectory_ids"))
        if isinstance(value, list | tuple):
            assigned.extend(str(item) for item in value)
    return tuple(assigned)


def resolve_trajectory_splits(
    trajectory_ids: Sequence[str],
    *,
    seed: int = 42,
    manifest_path: str | Path | None = None,
) -> TrajectorySplits:
    """Resolve an exact split partition, preferring an explicit fold manifest."""

    universe = tuple(str(value) for value in trajectory_ids)
    if manifest_path is None:
        if len(universe) == 1:
            return TrajectorySplits(
                train=universe,
                val=(),
                test=(),
                strategy="single_trajectory_fallback",
            )
        seeded = split_trajectory_ids(universe, seed=seed)
        return TrajectorySplits(**seeded, strategy="seeded")
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"trajectory split manifest not found: {path}")
    raw = path.read_bytes()
    try:
        payload: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trajectory split manifest must contain a mapping")
    source = payload.get("splits", payload)
    if not isinstance(source, dict):
        raise ValueError("trajectory split manifest 'splits' must be a mapping")
    resolved: dict[str, tuple[str, ...]] = {}
    for role in ("train", "val", "test"):
        value = source.get(role, source.get(f"{role}_trajectory_ids"))
        invalid = not isinstance(value, list | tuple) or not value
        if not invalid and isinstance(value, list | tuple):
            invalid = any(not isinstance(item, str) or not item for item in value)
        if invalid:
            raise ValueError(f"trajectory split manifest {role!r} IDs must be a non-empty string list")
        ids = tuple(value)
        if len(ids) != len(set(ids)):
            raise ValueError(f"trajectory split manifest {role!r} contains duplicate IDs")
        resolved[role] = ids
    role_sets = {role: set(ids) for role, ids in resolved.items()}
    overlaps = (
        (role_sets["train"] & role_sets["val"])
        | (role_sets["train"] & role_sets["test"])
        | (role_sets["val"] & role_sets["test"])
    )
    if overlaps:
        raise ValueError(f"trajectory split manifest roles overlap: {', '.join(sorted(overlaps))}")
    assigned = set().union(*role_sets.values())
    universe_set = set(universe)
    unknown = assigned - universe_set
    missing = universe_set - assigned
    if unknown:
        raise ValueError(f"trajectory split manifest IDs are missing from dataset/cache: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"trajectory split manifest does not assign dataset/cache IDs: {', '.join(sorted(missing))}")
    fold_id = payload.get("fold_id")
    if fold_id is not None and (not isinstance(fold_id, str) or not fold_id.strip()):
        raise ValueError("trajectory split manifest fold_id must be a non-empty string")
    return TrajectorySplits(
        train=resolved["train"],
        val=resolved["val"],
        test=resolved["test"],
        strategy="explicit_manifest",
        manifest_path=str(path),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        fold_id=fold_id,
    )


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
