"""Data-layer sample and batch types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

ArrayLike = np.ndarray | jax.Array


@dataclass(frozen=True)
class DiagnosticTargets:
    flux: ArrayLike | None
    spectra: Mapping[str, ArrayLike]


@dataclass(frozen=True)
class SnapshotSample:
    x: np.ndarray
    targets: DiagnosticTargets
    trajectory_id: str
    trajectory_index: int
    timestep_index: int
    physical_time: float | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SnapshotBatch:
    x: jax.Array
    flux: jax.Array | None
    spectra: Mapping[str, jax.Array]
    trajectory_index: jax.Array
    timestep_index: jax.Array


@dataclass(frozen=True)
class LatentSample:
    z: np.ndarray
    targets: DiagnosticTargets
    trajectory_id: str
    timestep_index: int
    physical_time: float | None


@dataclass(frozen=True)
class LatentSequenceBatch:
    z_context: jax.Array
    z_target: jax.Array
    flux_target: jax.Array | None
    spectra_target: Mapping[str, jax.Array]
    trajectory_index: jax.Array
    start_timestep_index: jax.Array
