"""Protocol interfaces for data access."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from gk_surrogate.data.types import LatentSample, SnapshotBatch, SnapshotSample


class TrajectoryDataset(Protocol):
    def trajectory_ids(self) -> Sequence[str]: ...

    def num_trajectories(self) -> int: ...

    def num_timesteps(self, trajectory_id: str) -> int: ...

    def snapshot_shape(self) -> tuple[int, ...]: ...

    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> SnapshotSample: ...


class SnapshotBatchIterator(Protocol):
    def __iter__(self) -> Iterator[SnapshotBatch]: ...

    def __len__(self) -> int: ...


class LatentTrajectoryDataset(Protocol):
    def trajectory_ids(self) -> Sequence[str]: ...

    def get_latent(self, trajectory_id: str, timestep_index: int) -> LatentSample: ...
