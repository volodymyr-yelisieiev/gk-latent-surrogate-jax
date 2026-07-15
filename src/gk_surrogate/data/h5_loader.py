"""Config-driven HDF5 trajectory loading and synthetic fixture writing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from gk_surrogate.config.schema import H5SchemaConfig, SyntheticDataConfig
from gk_surrogate.data.synthetic import SyntheticTrajectoryDataset
from gk_surrogate.data.types import DiagnosticTargets, SnapshotSample


@dataclass(frozen=True)
class H5TrajectoryDataset:
    root: str | Path
    schema: H5SchemaConfig
    target_spectra: tuple[str, ...] = ()
    target_flux: bool = True
    input_fields: tuple[str, ...] = ("df",)

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser()
        files = tuple(sorted(root.glob(self.schema.trajectory_glob)))
        if not files:
            msg = f"no HDF5 trajectories found in {root} matching {self.schema.trajectory_glob}"
            raise FileNotFoundError(msg)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_files", files)
        object.__setattr__(self, "_trajectory_ids", tuple(path.stem for path in files))

    def trajectory_ids(self) -> Sequence[str]:
        return self._trajectory_ids

    def num_trajectories(self) -> int:
        return len(self._files)

    def num_timesteps(self, trajectory_id: str) -> int:
        path = self._path_for_id(trajectory_id)
        with h5py.File(path, "r") as handle:
            if self.schema.timestep_key:
                times = _read_dataset(handle, self.schema.timestep_key, self.schema.metadata_group)
                return int(len(times))
            data_group = handle[self.schema.data_group]
            prefix = self.schema.timestep_key_template.split("{", maxsplit=1)[0]
            return sum(1 for key in data_group if key.startswith(prefix))

    def snapshot_shape(self) -> tuple[int, ...]:
        first_id = self._trajectory_ids[0]
        return self.get_snapshot(first_id, 0).x.shape

    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> SnapshotSample:
        path = self._path_for_id(trajectory_id)
        trajectory_index = self._trajectory_ids.index(trajectory_id)
        with h5py.File(path, "r") as handle:
            x = _load_input_fields(handle, self.schema, self.input_fields, timestep_index, path)
            x = _select_channels(x, self.schema.channel_indices)
            flux = _load_optional_time_row(
                handle,
                self.schema.flux_key,
                self.schema.metadata_group,
                timestep_index,
                required=self.target_flux,
            )
            spectra = {
                name: _load_optional_time_row(
                    handle,
                    self.schema.spectra_keys.get(name),
                    self.schema.metadata_group,
                    timestep_index,
                    required=True,
                )
                for name in self.target_spectra
            }
            physical_time = _load_physical_time(handle, self.schema, timestep_index)
        return SnapshotSample(
            x=x.astype(np.float32, copy=False),
            targets=DiagnosticTargets(flux=flux, spectra=spectra),
            trajectory_id=trajectory_id,
            trajectory_index=trajectory_index,
            timestep_index=timestep_index,
            physical_time=physical_time,
            metadata={"path": str(path)},
        )

    def _path_for_id(self, trajectory_id: str) -> Path:
        try:
            index = self._trajectory_ids.index(trajectory_id)
        except ValueError as exc:
            msg = f"unknown trajectory_id: {trajectory_id}"
            raise KeyError(msg) from exc
        return self._files[index]


def write_synthetic_h5(
    output_dir: str | Path,
    synthetic_config: SyntheticDataConfig,
    *,
    seed: int = 42,
    schema: H5SchemaConfig | None = None,
) -> list[Path]:
    """Write one HDF5 trajectory file per synthetic trajectory."""

    schema = schema or H5SchemaConfig(
        spectra_keys={name: f"metadata/{name}_spectrum" for name in synthetic_config.spectra_dims}
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticTrajectoryDataset(synthetic_config, seed=seed)
    written: list[Path] = []
    for trajectory_id in dataset.trajectory_ids():
        path = output / f"{trajectory_id}.h5"
        with h5py.File(path, "w") as handle:
            data_group = handle.require_group(schema.data_group)
            handle.require_group(schema.metadata_group)
            timesteps = np.arange(dataset.num_timesteps(trajectory_id), dtype=np.float32)
            _create_diagnostic_dataset(
                handle,
                schema.timestep_key or "timesteps",
                schema.metadata_group,
                timesteps,
            )
            flux_rows = []
            spectra_rows = {name: [] for name in synthetic_config.spectra_dims}
            for timestep in range(dataset.num_timesteps(trajectory_id)):
                sample = dataset.get_snapshot(trajectory_id, timestep)
                data_group.create_dataset(
                    schema.timestep_key_template.format(t=timestep),
                    data=sample.x,
                    compression="gzip",
                )
                if sample.targets.flux is not None:
                    flux_rows.append(sample.targets.flux)
                for name in spectra_rows:
                    spectra_rows[name].append(sample.targets.spectra[name])
            if flux_rows:
                _create_diagnostic_dataset(
                    handle,
                    schema.flux_key or "fluxes",
                    schema.metadata_group,
                    np.asarray(flux_rows, dtype=np.float32),
                )
            for name, rows in spectra_rows.items():
                _create_diagnostic_dataset(
                    handle,
                    schema.spectra_keys.get(name, f"{name}_spectrum"),
                    schema.metadata_group,
                    np.asarray(rows, dtype=np.float32),
                )
        written.append(path)
    return written


def _select_channels(x: np.ndarray, channel_indices: tuple[int, ...] | None) -> np.ndarray:
    if channel_indices is None:
        return x
    return x[np.asarray(channel_indices, dtype=np.int64)]


def _create_diagnostic_dataset(
    handle: h5py.File,
    key: str,
    metadata_group: str,
    values: np.ndarray,
) -> None:
    path = key if "/" in key else f"{metadata_group}/{key}"
    parent_path, _, dataset_name = path.rpartition("/")
    parent = handle.require_group(parent_path) if parent_path else handle
    parent.create_dataset(dataset_name, data=values)


def _load_input_fields(
    handle: h5py.File,
    schema: H5SchemaConfig,
    input_fields: tuple[str, ...],
    timestep_index: int,
    path: Path,
) -> np.ndarray:
    arrays = []
    for field in input_fields:
        if field == "df":
            template = schema.timestep_key_template
        elif field == "phi" and schema.phi_key_template is not None:
            template = schema.phi_key_template
        else:
            msg = f"unsupported or unconfigured input field {field!r}"
            raise KeyError(msg)
        dataset_path = f"{schema.data_group}/{template.format(t=timestep_index)}"
        if dataset_path not in handle:
            msg = f"snapshot key not found in {path}: {dataset_path}"
            raise KeyError(msg)
        arr = np.asarray(handle[dataset_path], dtype=schema.dtype)
        if arr.ndim == 5:
            arr = arr[None, ...]
        if arr.ndim != 6:
            msg = f"expected channel-first snapshot rank 6, got {arr.shape} for {dataset_path}"
            raise ValueError(msg)
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def _load_optional_time_row(
    handle: h5py.File,
    key: str | None,
    metadata_group: str,
    timestep_index: int,
    *,
    required: bool,
) -> np.ndarray | None:
    if key is None:
        if required:
            msg = "required diagnostic key is not configured"
            raise KeyError(msg)
        return None
    try:
        values = _read_dataset(handle, key, metadata_group)
    except KeyError:
        if required:
            raise
        return None
    row = values[timestep_index] if values.ndim >= 1 else values
    return np.atleast_1d(np.asarray(row, dtype=np.float32))


def _load_physical_time(
    handle: h5py.File,
    schema: H5SchemaConfig,
    timestep_index: int,
) -> float | None:
    if schema.timestep_key is None:
        return None
    try:
        times = _read_dataset(handle, schema.timestep_key, schema.metadata_group)
    except KeyError:
        return None
    return float(times[timestep_index])


def _read_dataset(handle: h5py.File, key: str, metadata_group: str) -> np.ndarray:
    candidates = (key, f"{metadata_group}/{key}")
    for candidate in candidates:
        if candidate in handle:
            return np.asarray(handle[candidate])
    msg = f"HDF5 dataset not found; tried {candidates}"
    raise KeyError(msg)
