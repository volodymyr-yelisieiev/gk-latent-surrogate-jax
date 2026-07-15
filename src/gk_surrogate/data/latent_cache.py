"""HDF5 latent cache writer and reader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np

from gk_surrogate.data.types import DiagnosticTargets, LatentSample


def _trajectory_group_name(trajectory_id: str) -> str:
    return quote(str(trajectory_id), safe="") or "_"


def _stored_trajectory_id(group_name: str, group: h5py.Group) -> str:
    return str(group.attrs.get("trajectory_id", group_name))


@dataclass(frozen=True)
class LatentCacheWriter:
    path: str | Path
    latent_dim: int
    config_yaml: str = ""
    encoder_checkpoint_path: str = ""

    def __post_init__(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            metadata = handle.create_group("metadata")
            metadata.attrs["latent_dim"] = self.latent_dim
            metadata.attrs["config_yaml"] = self.config_yaml
            metadata.attrs["encoder_checkpoint_path"] = self.encoder_checkpoint_path
            metadata.attrs["created_at"] = datetime.now(UTC).isoformat()
            handle.create_group("trajectories")

    def write_trajectory(
        self,
        trajectory_id: str,
        z: np.ndarray,
        *,
        timestep_index: np.ndarray | None = None,
        physical_time: np.ndarray | None = None,
        flux: np.ndarray | None = None,
        spectra: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        z = np.asarray(z, dtype=np.float32)
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            msg = f"expected z shape [T, {self.latent_dim}], got {z.shape}"
            raise ValueError(msg)
        timesteps = (
            np.arange(z.shape[0], dtype=np.int32)
            if timestep_index is None
            else np.asarray(timestep_index, dtype=np.int32)
        )
        _validate_time_length("timestep_index", timesteps, z.shape[0])
        if physical_time is not None:
            _validate_time_length("physical_time", np.asarray(physical_time), z.shape[0])
        if flux is not None:
            _validate_time_length("flux", np.asarray(flux), z.shape[0])
        if spectra:
            for name, values in spectra.items():
                _validate_time_length(f"spectra/{name}", np.asarray(values), z.shape[0])
        with h5py.File(self.path, "a") as handle:
            group = handle["trajectories"].create_group(_trajectory_group_name(trajectory_id))
            group.attrs["trajectory_id"] = trajectory_id
            group.create_dataset("z", data=z)
            group.create_dataset("timestep_index", data=timesteps)
            if physical_time is not None:
                group.create_dataset("physical_time", data=np.asarray(physical_time, dtype=np.float32))
            if flux is not None:
                group.create_dataset("flux", data=np.asarray(flux, dtype=np.float32))
            if spectra:
                spectra_group = group.create_group("spectra")
                for name, values in spectra.items():
                    spectra_group.create_dataset(name, data=np.asarray(values, dtype=np.float32))


@dataclass(frozen=True)
class LatentCacheDataset:
    path: str | Path

    def __post_init__(self) -> None:
        with h5py.File(self.path, "r") as handle:
            latent_dim = int(handle["metadata"].attrs["latent_dim"])
            group_names: dict[str, str] = {}
            trajectory_ids = []
            for group_name, group in handle["trajectories"].items():
                trajectory_id = _stored_trajectory_id(group_name, group)
                trajectory_ids.append(trajectory_id)
                group_names[trajectory_id] = group_name
        object.__setattr__(self, "latent_dim", latent_dim)
        object.__setattr__(self, "_trajectory_ids", tuple(trajectory_ids))
        object.__setattr__(self, "_group_names", group_names)

    def trajectory_ids(self) -> tuple[str, ...]:
        return self._trajectory_ids

    def _group_name(self, trajectory_id: str) -> str:
        return self._group_names.get(trajectory_id, _trajectory_group_name(trajectory_id))

    def num_timesteps(self, trajectory_id: str) -> int:
        with h5py.File(self.path, "r") as handle:
            return int(handle["trajectories"][self._group_name(trajectory_id)]["z"].shape[0])

    def get_latent(self, trajectory_id: str, timestep_index: int) -> LatentSample:
        with h5py.File(self.path, "r") as handle:
            group = handle["trajectories"][self._group_name(trajectory_id)]
            z = np.asarray(group["z"][timestep_index], dtype=np.float32)
            flux = np.asarray(group["flux"][timestep_index], dtype=np.float32) if "flux" in group else None
            spectra = {}
            if "spectra" in group:
                spectra = {
                    name: np.asarray(dataset[timestep_index], dtype=np.float32)
                    for name, dataset in group["spectra"].items()
                }
            physical_time = None
            if "physical_time" in group:
                physical_time = float(group["physical_time"][timestep_index])
        return LatentSample(
            z=z,
            targets=DiagnosticTargets(flux=flux, spectra=spectra),
            trajectory_id=trajectory_id,
            timestep_index=timestep_index,
            physical_time=physical_time,
        )

    def get_trajectory_latents(self, trajectory_id: str) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            group = handle["trajectories"][self._group_name(trajectory_id)]
            return np.asarray(group["z"], dtype=np.float32)

    def get_trajectory_flux(self, trajectory_id: str) -> np.ndarray | None:
        with h5py.File(self.path, "r") as handle:
            group = handle["trajectories"][self._group_name(trajectory_id)]
            if "flux" not in group:
                return None
            return np.asarray(group["flux"], dtype=np.float32)

    def get_sequence_window(
        self,
        trajectory_id: str,
        start: int,
        *,
        context_length: int,
        prediction_length: int,
    ) -> tuple[np.ndarray, np.ndarray, DiagnosticTargets]:
        stop_context = start + context_length
        stop_target = stop_context + prediction_length
        with h5py.File(self.path, "r") as handle:
            group = handle["trajectories"][self._group_name(trajectory_id)]
            z = np.asarray(group["z"][start:stop_target], dtype=np.float32)
            if z.shape[0] != context_length + prediction_length:
                msg = "latent sequence window exceeds trajectory length"
                raise IndexError(msg)
            flux = None
            if "flux" in group:
                flux = np.asarray(group["flux"][stop_context:stop_target], dtype=np.float32)
            spectra = {}
            if "spectra" in group:
                spectra = {
                    name: np.asarray(dataset[stop_context:stop_target], dtype=np.float32)
                    for name, dataset in group["spectra"].items()
                }
        return z[:context_length], z[context_length:], DiagnosticTargets(flux=flux, spectra=spectra)


def _validate_time_length(name: str, values: np.ndarray, expected: int) -> None:
    if values.shape[0] != expected:
        msg = f"{name} length {values.shape[0]} does not match latent length {expected}"
        raise ValueError(msg)
