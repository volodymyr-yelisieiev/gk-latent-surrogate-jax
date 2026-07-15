"""Dataset inspection helpers used by the CLI and script wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from gk_surrogate.config.schema import DataConfig
from gk_surrogate.data.base import TrajectoryDataset
from gk_surrogate.data.cyclone_kvikio import (
    CycloneKvikIODatasetAdapter,
    DirectCycloneKvikIODataset,
    SpectraUnavailableError,
)
from gk_surrogate.data.factory import build_dataset
from gk_surrogate.data.h5_loader import H5TrajectoryDataset


@dataclass(frozen=True)
class DatasetInspection:
    backend: str
    num_trajectories: int
    trajectory_ids: tuple[str, ...]
    timesteps: dict[str, int]
    snapshot_shape: tuple[int, ...]
    snapshot_dtype: str
    flux_shape: tuple[int, ...] | None
    spectra_shapes: dict[str, tuple[int, ...]]
    flux_stats: dict[str, float | int] | None
    spectra_stats: dict[str, dict[str, float | int]]
    bytes_per_snapshot: int
    bytes_per_batch: int
    warnings: tuple[str, ...]
    root: str | None = None
    first_snapshot_key: str | None = None
    bytes_per_trajectory: int | None = None
    recommended_batch_size_512mb: int | None = None
    metadata_keys: tuple[str, ...] = ()
    geometry_keys: tuple[str, ...] = ()
    h5_tree: tuple[str, ...] = ()
    sample_count_estimate: int | None = None
    kvikio_enabled: bool | None = None
    preferred_dtype: str | None = None
    quantized_shards_available: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": self.backend,
            "num_trajectories": self.num_trajectories,
            "trajectory_ids": self.trajectory_ids,
            "timesteps": self.timesteps,
            "snapshot_shape": self.snapshot_shape,
            "snapshot_dtype": self.snapshot_dtype,
            "flux_shape": self.flux_shape,
            "spectra_shapes": self.spectra_shapes,
            "flux_stats": self.flux_stats,
            "spectra_stats": self.spectra_stats,
            "bytes_per_snapshot": self.bytes_per_snapshot,
            "bytes_per_batch": self.bytes_per_batch,
            "root": self.root,
            "first_snapshot_key": self.first_snapshot_key,
            "bytes_per_trajectory": self.bytes_per_trajectory,
            "recommended_batch_size_512mb": self.recommended_batch_size_512mb,
            "metadata_keys": self.metadata_keys,
            "geometry_keys": self.geometry_keys,
            "warnings": self.warnings,
            "h5_tree": self.h5_tree,
            "sample_count_estimate": self.sample_count_estimate,
            "kvikio_enabled": self.kvikio_enabled,
            "preferred_dtype": self.preferred_dtype,
            "quantized_shards_available": self.quantized_shards_available,
        }
        if self.flux_stats is not None:
            for key, value in self.flux_stats.items():
                result[f"flux_{key}"] = value
        for name, stats in self.spectra_stats.items():
            for key, value in stats.items():
                result[f"spectra_{name}_{key}"] = value
        return result


def inspect_dataset(
    config: DataConfig,
    *,
    max_trajectories: int = 2,
    max_depth: int = 3,
    max_target_samples: int = 256,
    log_spectra: bool = False,
) -> DatasetInspection:
    if max_target_samples < 1:
        msg = "max_target_samples must be positive"
        raise ValueError(msg)
    dataset = build_dataset(config)
    trajectory_ids = tuple(dataset.trajectory_ids())
    selected = trajectory_ids[:max_trajectories]
    warnings = []
    try:
        first = dataset.get_snapshot(selected[0], 0)
    except SpectraUnavailableError as exc:
        if config.backend != "cyclone_kvikio" or not config.target_spectra:
            raise
        warnings.append(str(exc).strip("'"))
        fallback_config = config.model_copy(update={"target_spectra": ()})
        dataset = build_dataset(fallback_config)
        first = dataset.get_snapshot(selected[0], 0)
    timesteps = {trajectory_id: dataset.num_timesteps(trajectory_id) for trajectory_id in selected}
    if config.target_flux and first.targets.flux is None:
        warnings.append("flux target requested but unavailable")
    missing_spectra = set(config.target_spectra) - set(first.targets.spectra)
    if missing_spectra:
        warnings.append(f"missing spectra targets: {', '.join(sorted(missing_spectra))}")
    flux_stats, spectra_stats, target_warnings = _target_statistics(
        dataset,
        selected,
        max_target_samples=max_target_samples,
        log_spectra=log_spectra,
    )
    warnings.extend(target_warnings)
    h5_tree: tuple[str, ...] = ()
    sample_count_estimate: int | None = None
    kvikio_enabled: bool | None = None
    preferred_dtype: str | None = None
    quantized_shards_available: bool | None = None
    if config.backend == "h5" and isinstance(dataset, H5TrajectoryDataset):
        h5_tree = _h5_tree(dataset._files[0], max_depth=max_depth)
        metadata_keys = _group_keys(dataset._files[0], config.h5_schema.metadata_group if config.h5_schema else None)
        geometry_keys = _group_keys(dataset._files[0], config.h5_schema.geometry_group if config.h5_schema else None)
        first_snapshot_key = (
            f"{config.h5_schema.data_group}/{config.h5_schema.timestep_key_template.format(t=0)}"
            if config.h5_schema is not None
            else None
        )
        warnings.extend(_h5_target_length_warnings(dataset, config, selected))
    elif config.backend == "cyclone_kvikio" and isinstance(
        dataset,
        CycloneKvikIODatasetAdapter | DirectCycloneKvikIODataset,
    ):
        metadata_keys = dataset.metadata_keys()
        geometry_keys = dataset.geometry_keys()
        first_snapshot_key = "data/timestep_*.bin"
        warnings.extend(dataset.inspection_warnings())
        sample_count_estimate = dataset.sample_count_estimate()
        kvikio_enabled = config.cyclone.use_kvikio if config.cyclone else None
        preferred_dtype = config.cyclone.prefer_dtype if config.cyclone else None
        quantized_shards_available = dataset.quantized_shards_available()
    else:
        metadata_keys = ()
        geometry_keys = ()
        first_snapshot_key = None
    bytes_per_snapshot = int(np.prod(first.x.shape) * first.x.dtype.itemsize)
    first_timesteps = dataset.num_timesteps(selected[0])
    return DatasetInspection(
        backend=config.backend,
        num_trajectories=dataset.num_trajectories(),
        trajectory_ids=trajectory_ids,
        timesteps=timesteps,
        snapshot_shape=first.x.shape,
        snapshot_dtype=str(first.x.dtype),
        flux_shape=None if first.targets.flux is None else first.targets.flux.shape,
        spectra_shapes={name: value.shape for name, value in first.targets.spectra.items()},
        flux_stats=flux_stats,
        spectra_stats=spectra_stats,
        bytes_per_snapshot=bytes_per_snapshot,
        bytes_per_batch=bytes_per_snapshot * config.batch_size,
        root=config.root,
        first_snapshot_key=first_snapshot_key,
        bytes_per_trajectory=bytes_per_snapshot * first_timesteps,
        recommended_batch_size_512mb=max(1, (512 * 1024 * 1024) // max(bytes_per_snapshot, 1)),
        metadata_keys=metadata_keys,
        geometry_keys=geometry_keys,
        warnings=tuple(warnings),
        h5_tree=h5_tree,
        sample_count_estimate=sample_count_estimate,
        kvikio_enabled=kvikio_enabled,
        preferred_dtype=preferred_dtype,
        quantized_shards_available=quantized_shards_available,
    )


def _h5_tree(path: str, *, max_depth: int) -> tuple[str, ...]:
    rows: list[str] = []
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            depth = name.count("/")
            if depth > max_depth:
                return
            kind = "dataset" if isinstance(obj, h5py.Dataset) else "group"
            suffix = ""
            if isinstance(obj, h5py.Dataset):
                suffix = f" shape={obj.shape} dtype={obj.dtype}"
            rows.append(f"{name} [{kind}]{suffix}")

        handle.visititems(visit)
    return tuple(rows)


def _group_keys(path: str, group_name: str | None) -> tuple[str, ...]:
    if not group_name:
        return ()
    with h5py.File(path, "r") as handle:
        if group_name not in handle:
            return ()
        return tuple(sorted(handle[group_name].keys()))


def _target_statistics(
    dataset: TrajectoryDataset,
    trajectory_ids: tuple[str, ...],
    *,
    max_target_samples: int,
    log_spectra: bool,
) -> tuple[dict[str, float | int] | None, dict[str, dict[str, float | int]], list[str]]:
    flux_rows: list[np.ndarray] = []
    spectra_rows: dict[str, list[np.ndarray]] = {}
    warnings: list[str] = []
    for trajectory_id in trajectory_ids:
        sample_count = min(dataset.num_timesteps(trajectory_id), max_target_samples)
        for timestep in range(sample_count):
            try:
                sample = dataset.get_snapshot(trajectory_id, timestep)
            except (IndexError, KeyError, ValueError) as exc:
                warnings.append(f"target sampling failed for {trajectory_id}[{timestep}]: {exc}")
                continue
            if sample.targets.flux is not None:
                flux_rows.append(np.asarray(sample.targets.flux, dtype=np.float32))
            for name, value in sample.targets.spectra.items():
                spectra_rows.setdefault(name, []).append(np.asarray(value, dtype=np.float32))

    flux_stats = _array_stats("flux", flux_rows, warnings, warn_constant=True)
    spectra_stats = {
        name: stats
        for name, rows in sorted(spectra_rows.items())
        if (stats := _array_stats(f"spectra {name}", rows, warnings, warn_negative=log_spectra)) is not None
    }
    return flux_stats, spectra_stats, warnings


def _array_stats(
    label: str,
    rows: list[np.ndarray],
    warnings: list[str],
    *,
    warn_constant: bool = False,
    warn_negative: bool = False,
) -> dict[str, float | int] | None:
    if not rows:
        return None
    values = np.concatenate([np.ravel(row) for row in rows]).astype(np.float32, copy=False)
    count = int(values.size)
    if count == 0:
        warnings.append(f"{label} targets are empty")
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    finite = np.isfinite(values)
    if not bool(np.all(finite)):
        warnings.append(f"{label} targets contain NaN or Inf")
    finite_values = values[finite]
    if finite_values.size == 0:
        return {"count": count, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    if warn_constant and bool(np.allclose(finite_values, finite_values[0])):
        warnings.append(f"{label} targets are constant")
    if warn_negative and bool(np.any(finite_values < 0.0)):
        warnings.append(f"{label} targets contain negative values while log spectra are enabled")
    return {
        "count": count,
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


def _h5_target_length_warnings(
    dataset: H5TrajectoryDataset,
    config: DataConfig,
    trajectory_ids: tuple[str, ...],
) -> list[str]:
    if config.h5_schema is None:
        return []
    warnings: list[str] = []
    for trajectory_id in trajectory_ids:
        path = dataset._path_for_id(trajectory_id)
        expected = dataset.num_timesteps(trajectory_id)
        with h5py.File(path, "r") as handle:
            if config.target_flux and config.h5_schema.flux_key is not None:
                _append_h5_length_warning(
                    warnings,
                    handle,
                    key=config.h5_schema.flux_key,
                    metadata_group=config.h5_schema.metadata_group,
                    expected=expected,
                    label=f"flux target length mismatch for {trajectory_id}",
                )
            for name in config.target_spectra:
                key = config.h5_schema.spectra_keys.get(name)
                if key is None:
                    continue
                _append_h5_length_warning(
                    warnings,
                    handle,
                    key=key,
                    metadata_group=config.h5_schema.metadata_group,
                    expected=expected,
                    label=f"spectra {name} target length mismatch for {trajectory_id}",
                )
    return warnings


def _append_h5_length_warning(
    warnings: list[str],
    handle: h5py.File,
    *,
    key: str,
    metadata_group: str,
    expected: int,
    label: str,
) -> None:
    try:
        values = _read_h5_dataset(handle, key, metadata_group)
    except KeyError:
        return
    actual = int(values.shape[0]) if values.ndim >= 1 else 1
    if actual != expected:
        warnings.append(f"{label}: got {actual}, expected {expected}")


def _read_h5_dataset(handle: h5py.File, key: str, metadata_group: str) -> np.ndarray:
    candidates = (key, f"{metadata_group}/{key}")
    for candidate in candidates:
        if candidate in handle:
            return np.asarray(handle[candidate])
    msg = f"HDF5 dataset not found; tried {candidates}"
    raise KeyError(msg)
