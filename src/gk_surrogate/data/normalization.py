"""Snapshot normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gk_surrogate.data.base import TrajectoryDataset


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def save_npz(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, mean=self.mean.astype(np.float32), std=self.std.astype(np.float32))


def normalize_snapshot(
    x: np.ndarray,
    *,
    mode: str,
    stats: NormalizationStats | None = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if mode == "none":
        return x.astype(np.float32, copy=True)
    if mode == "sample":
        mean = np.mean(x, keepdims=True)
        std = np.std(x, keepdims=True)
        return ((x - mean) / np.maximum(std, epsilon)).astype(np.float32)
    if mode == "fixed":
        if stats is None:
            msg = "fixed normalization requires stats"
            raise ValueError(msg)
        mean, std = _broadcast_stats(stats, x)
        return ((x - mean) / np.maximum(std, epsilon)).astype(np.float32)
    if mode in {"trajectory", "dataset"}:
        if stats is None:
            msg = f"{mode} normalization requires precomputed stats"
            raise ValueError(msg)
        mean, std = _broadcast_stats(stats, x)
        return ((x - mean) / np.maximum(std, epsilon)).astype(np.float32)
    msg = f"unknown normalization mode: {mode}"
    raise ValueError(msg)


def _broadcast_stats(stats: NormalizationStats, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    if mean.ndim == std.ndim == 1:
        if mean.shape != std.shape or mean.shape[0] != x.shape[0]:
            msg = f"channel normalization stats shape {mean.shape}/{std.shape} does not match {x.shape[0]} channels"
            raise ValueError(msg)
        channel_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return mean.reshape(channel_shape), std.reshape(channel_shape)
    try:
        np.broadcast_shapes(x.shape, mean.shape, std.shape)
    except ValueError as exc:
        msg = f"normalization stats shapes {mean.shape}/{std.shape} do not broadcast to snapshot shape {x.shape}"
        raise ValueError(msg) from exc
    return mean, std


def estimate_dataset_stats(
    dataset: TrajectoryDataset,
    *,
    max_samples: int = 128,
    epsilon: float = 1e-6,
) -> NormalizationStats:
    if max_samples <= 0:
        msg = "max_samples must be positive"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    moments = _RunningMoments()
    sampled_snapshots = 0
    for trajectory_id in dataset.trajectory_ids():
        for timestep in range(dataset.num_timesteps(trajectory_id)):
            moments.update(dataset.get_snapshot(trajectory_id, timestep).x)
            sampled_snapshots += 1
            if sampled_snapshots >= max_samples:
                break
        if sampled_snapshots >= max_samples:
            break
    if sampled_snapshots == 0:
        msg = "cannot estimate normalization stats from an empty dataset"
        raise ValueError(msg)
    return moments.finalize(epsilon=epsilon)


def estimate_trajectory_stats(
    dataset: TrajectoryDataset,
    trajectory_id: str,
    *,
    epsilon: float = 1e-6,
) -> NormalizationStats:
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    moments = _RunningMoments()
    sampled_snapshots = 0
    for timestep in range(dataset.num_timesteps(trajectory_id)):
        moments.update(dataset.get_snapshot(trajectory_id, timestep).x)
        sampled_snapshots += 1
    if sampled_snapshots == 0:
        msg = f"cannot estimate normalization stats from empty trajectory {trajectory_id!r}"
        raise ValueError(msg)
    return moments.finalize(epsilon=epsilon)


@dataclass
class _RunningMoments:
    """Accumulate numerically stable population moments without retaining snapshots."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        batch_count = int(array.size)
        if batch_count == 0:
            return
        batch_mean = float(np.mean(array, dtype=np.float64))
        centered = array - batch_mean
        batch_m2 = float(np.sum(centered * centered, dtype=np.float64))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / combined_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / combined_count
        self.count = combined_count

    def finalize(self, *, epsilon: float) -> NormalizationStats:
        if self.count == 0:
            msg = "cannot finalize normalization stats without values"
            raise ValueError(msg)
        variance = max(self.m2 / self.count, 0.0)
        return NormalizationStats(
            mean=np.asarray(self.mean, dtype=np.float32),
            std=np.asarray(max(np.sqrt(variance), epsilon), dtype=np.float32),
        )
