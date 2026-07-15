"""Deterministic synthetic 5D trajectory data for smoke tests and development."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gk_surrogate.config.schema import SyntheticDataConfig
from gk_surrogate.data.types import DiagnosticTargets, SnapshotSample


@dataclass(frozen=True)
class SyntheticTrajectoryDataset:
    config: SyntheticDataConfig
    seed: int = 42
    target_spectra: tuple[str, ...] | None = None
    target_flux: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_trajectory_ids", tuple(f"synthetic_{i:04d}" for i in range(self.config.num_trajectories))
        )
        x, flux, spectra = _generate_arrays(self.config, self.seed)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_flux", flux)
        object.__setattr__(self, "_spectra", spectra)

    def trajectory_ids(self) -> Sequence[str]:
        return self._trajectory_ids

    def num_trajectories(self) -> int:
        return len(self._trajectory_ids)

    def num_timesteps(self, trajectory_id: str) -> int:
        self._trajectory_index(trajectory_id)
        return self.config.timesteps

    def snapshot_shape(self) -> tuple[int, ...]:
        return (self.config.channels, *self.config.spatial_shape)

    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> SnapshotSample:
        trajectory_index = self._trajectory_index(trajectory_id)
        if timestep_index < 0 or timestep_index >= self.config.timesteps:
            msg = f"timestep_index out of range: {timestep_index}"
            raise IndexError(msg)
        selected_spectra = (
            self._spectra
            if self.target_spectra is None
            else {name: self._spectra[name] for name in self.target_spectra if name in self._spectra}
        )
        spectra = {name: values[trajectory_index, timestep_index].copy() for name, values in selected_spectra.items()}
        return SnapshotSample(
            x=self._x[trajectory_index, timestep_index].copy(),
            targets=DiagnosticTargets(
                flux=self._flux[trajectory_index, timestep_index].copy() if self.target_flux else None,
                spectra=spectra,
            ),
            trajectory_id=trajectory_id,
            trajectory_index=trajectory_index,
            timestep_index=timestep_index,
            physical_time=float(timestep_index),
            metadata={"source": "synthetic", "seed": self.seed},
        )

    def _trajectory_index(self, trajectory_id: str) -> int:
        try:
            return self._trajectory_ids.index(trajectory_id)
        except ValueError as exc:
            msg = f"unknown trajectory_id: {trajectory_id}"
            raise KeyError(msg) from exc


def _generate_arrays(
    config: SyntheticDataConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    shape = (
        config.num_trajectories,
        config.timesteps,
        config.channels,
        *config.spatial_shape,
    )
    spatial_axes = tuple(range(2, 2 + len(config.spatial_shape)))
    grid = _spatial_grid(config.spatial_shape)
    x = np.empty(shape, dtype=np.float32)
    latent = rng.normal(size=(config.num_trajectories, 3)).astype(np.float32)
    for trajectory_index in range(config.num_trajectories):
        state = latent[trajectory_index]
        channel_weights = rng.normal(size=(config.channels, 3)).astype(np.float32)
        for timestep in range(config.timesteps):
            state = 0.92 * state + 0.15 * rng.normal(size=3).astype(np.float32)
            phase = 0.2 * timestep + 0.1 * trajectory_index
            base = (
                state[0] * np.sin(grid[0] + phase)
                + state[1] * np.cos(grid[1] - 0.5 * phase)
                + state[2] * np.sin(grid[2] + grid[3] - grid[4])
            )
            for channel in range(config.channels):
                mixed = base + float(channel_weights[channel] @ state) * 0.1
                noise = 0.03 * rng.normal(size=config.spatial_shape)
                x[trajectory_index, timestep, channel] = (mixed + noise).astype(np.float32)
    energy = np.mean(x * x, axis=spatial_axes)
    channel_mean = np.mean(x, axis=spatial_axes)
    flux_base = energy.mean(axis=-1, keepdims=True) + 0.1 * np.tanh(channel_mean.mean(axis=-1, keepdims=True))
    flux = np.repeat(flux_base, config.flux_dim, axis=-1).astype(np.float32)
    if config.flux_dim > 1:
        flux *= np.linspace(1.0, 1.0 + 0.1 * (config.flux_dim - 1), config.flux_dim, dtype=np.float32)

    spectra = {
        name: _synthetic_spectrum(x, dim, offset=index) for index, (name, dim) in enumerate(config.spectra_dims.items())
    }
    return x, flux, spectra


def _spatial_grid(spatial_shape: tuple[int, int, int, int, int]) -> tuple[np.ndarray, ...]:
    axes = [np.linspace(0.0, 2.0 * np.pi, num=dim, endpoint=False, dtype=np.float32) for dim in spatial_shape]
    return tuple(np.meshgrid(*axes, indexing="ij"))


def _synthetic_spectrum(x: np.ndarray, dim: int, *, offset: int) -> np.ndarray:
    reduced = np.mean(x * x, axis=(2, 4, 5, 6, 7))
    profile = np.abs(np.fft.rfft(reduced, axis=-1))[..., :dim]
    if profile.shape[-1] < dim:
        pad_width = [(0, 0)] * profile.ndim
        pad_width[-1] = (0, dim - profile.shape[-1])
        profile = np.pad(profile, pad_width)
    scale = np.linspace(1.0, 0.25, dim, dtype=np.float32)
    return (profile[..., :dim] * scale + 0.01 * (offset + 1)).astype(np.float32)
