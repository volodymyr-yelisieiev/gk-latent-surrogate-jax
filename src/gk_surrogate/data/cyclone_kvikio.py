"""Cyclone/KvikIO adapter for the internal trajectory dataset contract."""

from __future__ import annotations

import importlib
import inspect
import pickle
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gk_surrogate.config.schema import CycloneKvikIOConfig
from gk_surrogate.data.types import DiagnosticTargets, SnapshotSample


class MissingCycloneDependencyError(ImportError):
    """Raised when the optional upstream Cyclone dataset package is unavailable."""


class SpectraUnavailableError(KeyError):
    """Raised when spectra targets are requested but absent from an upstream sample."""


def direct_cyclone_layout_available(root: str | Path, config: CycloneKvikIOConfig) -> bool:
    """Return whether ``root`` exposes the public preprocessed KvikIO directory layout."""

    return bool(_direct_trajectory_paths(root, config))


@dataclass(frozen=True)
class NoSpectraProvider:
    requested: tuple[str, ...] = ()

    def spectra_for(self, sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
        del sample
        if self.requested:
            requested = ", ".join(self.requested)
            msg = f"spectra targets requested but no Cyclone spectra provider is configured: {requested}"
            raise SpectraUnavailableError(msg)
        return {}


@dataclass(frozen=True)
class StoredSpectraProvider:
    requested: tuple[str, ...]
    metadata: Mapping[str | int, Any]
    offsets: tuple[int, ...]
    bundle_seq_length: int

    def spectra_for(self, sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
        spectra: dict[str, np.ndarray] = {}
        nested = _as_mapping(sample.get("spectra"))
        for name in self.requested:
            value = _first_present(
                sample,
                _spectra_aliases(name),
                nested=nested,
            )
            if value is None:
                value = self._from_metadata(name, sample)
            if value is None:
                msg = f"spectra target {name!r} requested but not found in Cyclone sample"
                raise SpectraUnavailableError(msg)
            spectra[name] = np.atleast_1d(_to_numpy(value).astype(np.float32, copy=False))
        return spectra

    def _from_metadata(self, name: str, sample: Mapping[str, Any]) -> Any | None:
        file_index = _optional_int(_first_present(sample, ("file_index", "trajectory_index", "file_idx")))
        timestep_index = _optional_int(_first_present(sample, ("timestep_index", "time_index", "step_index")))
        if file_index is None or timestep_index is None:
            return None
        metadata = _metadata_for_file(self.metadata, file_index)
        if not metadata:
            return None
        value = _first_present(metadata, _spectra_aliases(name))
        if value is None:
            return None
        offset = self.offsets[file_index] if file_index < len(self.offsets) else 0
        target_index = timestep_index + offset + self.bundle_seq_length
        return _index_time_aligned_target(value, target_index, name)


@dataclass(frozen=True)
class ComputedSpectraProvider:
    requested: tuple[str, ...]

    def spectra_for(self, sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
        del sample
        requested = ", ".join(self.requested)
        msg = f"computed Cyclone spectra are not implemented; requested targets: {requested}"
        raise SpectraUnavailableError(msg)


@dataclass(frozen=True)
class DirectCycloneKvikIODataset:
    """Read public Cyclone/KvikIO directories without dispatching through upstream tensors."""

    root: str | Path
    cyclone: CycloneKvikIOConfig
    target_spectra: tuple[str, ...] = ()
    target_flux: bool = True
    input_fields: tuple[str, ...] = ("df",)
    split: str = "train"
    seed: int = 42

    def __post_init__(self) -> None:
        paths = _direct_trajectory_paths(self.root, self.cyclone)
        if not paths:
            msg = f"no direct Cyclone/KvikIO trajectories found under {self.root}"
            raise FileNotFoundError(msg)
        metadata = tuple(_read_direct_metadata(path) for path in paths)
        ids = tuple(str(path) for path in paths)
        object.__setattr__(self, "_root", str(Path(self.root).expanduser()))
        object.__setattr__(self, "_paths", paths)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_trajectory_ids", ids)
        object.__setattr__(self, "_index_by_id", {trajectory_id: index for index, trajectory_id in enumerate(ids)})
        object.__setattr__(
            self,
            "_sample_indices",
            tuple(_direct_sample_indices(item, self.cyclone) for item in metadata),
        )

    def trajectory_ids(self) -> Sequence[str]:
        return self._trajectory_ids

    def num_trajectories(self) -> int:
        return len(self._trajectory_ids)

    def num_timesteps(self, trajectory_id: str) -> int:
        return len(self._sample_indices[self._trajectory_index(trajectory_id)])

    def snapshot_shape(self) -> tuple[int, ...]:
        return self.get_snapshot(self._trajectory_ids[0], 0).x.shape

    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> SnapshotSample:
        trajectory_index = self._trajectory_index(trajectory_id)
        indices = self._sample_indices[trajectory_index]
        if timestep_index < 0 or timestep_index >= len(indices):
            msg = f"timestep_index out of range: {timestep_index}"
            raise IndexError(msg)
        raw_index = indices[timestep_index]
        path = self._paths[trajectory_index]
        metadata = self._metadata[trajectory_index]
        x = self._load_direct_input(path, metadata, raw_index)
        target_index = raw_index + self.cyclone.bundle_seq_length
        flux = (
            _direct_target(metadata, ("fluxes", "flux", "avg_flux", "heat_flux"), target_index)
            if self.target_flux
            else None
        )
        spectra = {name: _direct_target(metadata, _spectra_aliases(name), target_index) for name in self.target_spectra}
        physical_time = _direct_time(metadata, raw_index)
        sample_metadata = {
            key: _small_value(value)
            for key in self.cyclone.conditions
            if (value := _first_present(metadata, _condition_aliases(key))) is not None
        }
        return SnapshotSample(
            x=x,
            targets=DiagnosticTargets(flux=flux, spectra=spectra),
            trajectory_id=trajectory_id,
            trajectory_index=trajectory_index,
            timestep_index=timestep_index,
            physical_time=physical_time,
            metadata=sample_metadata,
        )

    def metadata_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for metadata in self._metadata:
            keys.update(str(key) for key in metadata)
        if "flux" in keys:
            keys.add("fluxes")
        return tuple(sorted(keys))

    def geometry_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for metadata in self._metadata:
            geometry = _as_mapping(metadata.get("geometry"))
            keys.update(str(key) for key in geometry)
        return tuple(sorted(keys))

    def inspection_warnings(self) -> tuple[str, ...]:
        warnings: list[str] = ["using direct Cyclone/KvikIO directory reader; upstream tensor dataset is bypassed"]
        if not self.target_spectra:
            available = self.spectra_keys()
            if available:
                warnings.append(
                    "spectra targets are not requested; stored Cyclone spectra keys are available: "
                    + ", ".join(available)
                )
            else:
                warnings.append("spectra targets are not requested; no stored Cyclone spectra keys were found")
        if self.cyclone.bundle_seq_length != 1:
            warnings.append("SnapshotSample uses one timestep; set cyclone.bundle_seq_length to 1 for training")
        return tuple(warnings)

    def spectra_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for metadata in self._metadata:
            for key in metadata:
                name = str(key)
                if name.endswith("spec") or name.startswith("spectra_") or name.endswith("_spectrum"):
                    keys.add(name)
        return tuple(sorted(keys))

    def sample_count_estimate(self) -> int:
        return sum(len(indices) for indices in self._sample_indices)

    def quantized_shards_available(self) -> bool | None:
        root = Path(self._root)
        if not root.exists():
            return None
        for pattern in ("*.bf16.bin", "data/timestep_*.bf16.bin", "*/data/timestep_*.bf16.bin", "*/*.bf16.bin"):
            if any(root.glob(pattern)):
                return True
        return False

    def _trajectory_index(self, trajectory_id: str) -> int:
        try:
            return self._index_by_id[trajectory_id]
        except KeyError as exc:
            msg = f"unknown trajectory_id: {trajectory_id}"
            raise KeyError(msg) from exc

    def _load_direct_input(self, path: Path, metadata: Mapping[str, Any], raw_index: int) -> np.ndarray:
        arrays = []
        for field in self.input_fields or self.cyclone.fields_to_load:
            if field == "df":
                arr = _read_direct_df(path, metadata, raw_index, self.cyclone.use_kvikio)
                if self.cyclone.separate_zf:
                    arr = _separate_zf_numpy(arr, axis=0)
                arrays.append(arr)
            elif field == "phi":
                arrays.append(_read_direct_phi(path, metadata, raw_index, self.cyclone.use_kvikio)[None, ...])
            else:
                msg = f"direct Cyclone/KvikIO reader does not support input field {field!r}"
                raise KeyError(msg)
        return np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]


@dataclass(frozen=True)
class CycloneKvikIODatasetAdapter:
    """Map upstream Cyclone samples into ``SnapshotSample`` objects.

    Internal snapshots are channel-first ``[C, S1, S2, S3, S4, S5]`` arrays.
    """

    root: str | Path
    cyclone: CycloneKvikIOConfig
    target_spectra: tuple[str, ...] = ()
    target_flux: bool = True
    input_fields: tuple[str, ...] = ("df",)
    split: str = "train"
    seed: int = 42

    def __post_init__(self) -> None:
        root = str(Path(self.root).expanduser())
        dataset_cls = _load_cyclone_dataset_class()
        upstream = _instantiate_upstream_dataset(
            dataset_cls,
            root=root,
            config=self.cyclone,
            split=_upstream_split(self.split),
        )
        _normalize_upstream_metadata(upstream)
        trajectory_ids = _discover_trajectory_ids(upstream, self.cyclone)
        spectra_provider: NoSpectraProvider | StoredSpectraProvider
        spectra_provider = (
            StoredSpectraProvider(
                self.target_spectra,
                metadata=_as_mapping(_call_or_value(upstream, "metadata")),
                offsets=_offsets_from_upstream(upstream, self.cyclone),
                bundle_seq_length=self.cyclone.bundle_seq_length,
            )
            if self.target_spectra
            else NoSpectraProvider()
        )
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_upstream", upstream)
        object.__setattr__(self, "_trajectory_ids", trajectory_ids)
        object.__setattr__(self, "_spectra_provider", spectra_provider)

    def trajectory_ids(self) -> Sequence[str]:
        return self._trajectory_ids

    def num_trajectories(self) -> int:
        return len(self._trajectory_ids)

    def num_timesteps(self, trajectory_id: str) -> int:
        self._trajectory_index(trajectory_id)
        value = _call_named(self._upstream, "num_timesteps", trajectory_id)
        if value is None:
            value = _call_named(self._upstream, "num_time_steps", trajectory_id)
        if value is not None:
            return int(value)
        for name in (
            "timesteps_per_trajectory",
            "n_timesteps",
            "num_timesteps",
            "time_steps",
            "file_num_samples",
            "file_num_timesteps",
        ):
            value = _call_or_value(self._upstream, name)
            result = _timesteps_from_value(value, trajectory_id, self._trajectory_ids)
            if result is not None:
                return result
        total = _safe_len(self._upstream)
        if total is not None:
            if len(self._trajectory_ids) == 1:
                return total
            if total % len(self._trajectory_ids) == 0:
                return total // len(self._trajectory_ids)
        msg = f"could not determine Cyclone timestep count for trajectory {trajectory_id!r}"
        raise ValueError(msg)

    def snapshot_shape(self) -> tuple[int, ...]:
        return self.get_snapshot(self._trajectory_ids[0], 0).x.shape

    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> SnapshotSample:
        trajectory_index = self._trajectory_index(trajectory_id)
        if timestep_index < 0 or timestep_index >= self.num_timesteps(trajectory_id):
            msg = f"timestep_index out of range: {timestep_index}"
            raise IndexError(msg)
        raw_sample = _load_upstream_sample(self._upstream, trajectory_id, trajectory_index, timestep_index, self)
        sample = _sample_mapping(raw_sample)
        x = _load_input_fields(sample, self.input_fields or self.cyclone.fields_to_load)
        flux = _load_flux(sample) if self.target_flux else None
        spectra = self._spectra_provider.spectra_for(sample)
        physical_time = _optional_float(_first_present(sample, ("physical_time", "timestep", "time", "t")))
        sample_trajectory_index = _optional_int(
            _first_present(sample, ("trajectory_index", "file_index", "traj_index", "file_idx"))
        )
        sample_timestep_index = _optional_int(_first_present(sample, ("timestep_index", "time_index", "step_index")))
        metadata = _sample_metadata(sample, self.cyclone.conditions)
        return SnapshotSample(
            x=x,
            targets=DiagnosticTargets(flux=flux, spectra=spectra),
            trajectory_id=trajectory_id,
            trajectory_index=trajectory_index if sample_trajectory_index is None else sample_trajectory_index,
            timestep_index=timestep_index if sample_timestep_index is None else sample_timestep_index,
            physical_time=physical_time,
            metadata=metadata,
        )

    def metadata_keys(self) -> tuple[str, ...]:
        metadata = _as_mapping(_call_or_value(self._upstream, "metadata"))
        if not metadata:
            return ()
        if not all(_is_int_like(key) for key in metadata):
            return tuple(sorted(str(key) for key in metadata))
        keys: set[str] = set()
        for key, value in metadata.items():
            nested = _as_mapping(value)
            if nested:
                keys.update(str(item) for item in nested)
            else:
                keys.add(str(key))
        return tuple(sorted(keys))

    def geometry_keys(self) -> tuple[str, ...]:
        for candidate in ("geometry", "geom"):
            geometry = _as_mapping(_call_or_value(self._upstream, candidate))
            if geometry:
                return tuple(sorted(geometry.keys()))
        metadata = _as_mapping(_call_or_value(self._upstream, "metadata"))
        geometry = _as_mapping(metadata.get("geometry"))
        if not geometry:
            for value in metadata.values():
                nested_geometry = _as_mapping(_as_mapping(value).get("geometry"))
                if nested_geometry:
                    geometry = nested_geometry
                    break
        return tuple(sorted(geometry.keys()))

    def inspection_warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if not self.target_spectra:
            available = self.spectra_keys()
            if available:
                warnings.append(
                    "spectra targets are not requested; stored Cyclone spectra keys are available: "
                    + ", ".join(available)
                )
            else:
                warnings.append("spectra targets are not requested; no stored Cyclone spectra keys were found")
        if self.cyclone.bundle_seq_length != 1:
            warnings.append("SnapshotSample uses one timestep; set cyclone.bundle_seq_length to 1 for training")
        return tuple(warnings)

    def spectra_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        metadata = _as_mapping(_call_or_value(self._upstream, "metadata"))
        if not metadata:
            return ()
        items = metadata.values() if all(_is_int_like(key) for key in metadata) else (metadata,)
        for item in items:
            nested = _as_mapping(item)
            for key in nested:
                name = str(key)
                if name.endswith("spec") or name.startswith("spectra_") or name.endswith("_spectrum"):
                    keys.add(name)
        return tuple(sorted(keys))

    def sample_count_estimate(self) -> int | None:
        total = _safe_len(self._upstream)
        if total is not None:
            return total
        try:
            return sum(self.num_timesteps(trajectory_id) for trajectory_id in self._trajectory_ids)
        except ValueError:
            return None

    def quantized_shards_available(self) -> bool | None:
        root = Path(self._root)
        if not root.exists():
            return None
        for pattern in ("*.bf16.bin", "data/timestep_*.bf16.bin", "*/data/timestep_*.bf16.bin", "*/*.bf16.bin"):
            if any(root.glob(pattern)):
                return True
        return False

    def _trajectory_index(self, trajectory_id: str) -> int:
        try:
            return self._trajectory_ids.index(trajectory_id)
        except ValueError as exc:
            msg = f"unknown trajectory_id: {trajectory_id}"
            raise KeyError(msg) from exc


def cyclone_dependency_available() -> bool:
    try:
        _load_cyclone_dataset_class()
    except MissingCycloneDependencyError:
        return False
    return True


def _load_cyclone_dataset_class() -> type[Any]:
    candidates = (
        ("neugk_jax.dataset.cyclone", "CycloneDataset"),
        ("neugk_jax.dataset", "CycloneDataset"),
        ("neugk.dataset.cyclone", "CycloneDataset"),
        ("neugk.dataset", "CycloneDataset"),
    )
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        dataset_cls = getattr(module, attr, None)
        if dataset_cls is not None:
            return dataset_cls
    msg = (
        "cyclone_kvikio backend requires the optional upstream package `neugk_jax` "
        "or `neugk` with `CycloneDataset`; install it in the environment or add it to PYTHONPATH."
    )
    raise MissingCycloneDependencyError(msg)


def _instantiate_upstream_dataset(
    dataset_cls: type[Any],
    *,
    root: str,
    config: CycloneKvikIOConfig,
    split: str,
) -> Any:
    kwargs = _constructor_kwargs(dataset_cls, root=root, config=config, split=split)
    try:
        return dataset_cls(**kwargs)
    except TypeError as keyword_exc:
        root_keys = {"root", "data_root", "dataset_root", "path"}
        positional_kwargs = {key: value for key, value in kwargs.items() if key not in root_keys}
        try:
            return dataset_cls(root, **positional_kwargs)
        except TypeError as positional_exc:
            keys = ", ".join(sorted(kwargs))
            msg = f"failed to construct upstream CycloneDataset with supported kwargs: {keys}"
            raise TypeError(msg) from positional_exc
        finally:
            del keyword_exc


def _upstream_split(split: str) -> str:
    return "val" if split in {"val", "test"} else "train"


def _constructor_kwargs(
    dataset_cls: type[Any],
    *,
    root: str,
    config: CycloneKvikIOConfig,
    split: str,
) -> dict[str, Any]:
    general: dict[str, Any] = {
        "split": split,
        "trajectories": config.trajectories,
        "fields_to_load": config.fields_to_load,
        "conditions": config.conditions,
        "normalization": config.normalization,
        "normalization_scope": config.normalization_scope,
        "normalization_stats": config.normalization_stats,
        "spatial_ifft": config.spatial_ifft,
        "real_potens": config.real_potens,
        "bundle_seq_length": config.bundle_seq_length,
        "offset": config.offset,
        "tail_offset": config.tail_offset,
        "subsample": config.subsample,
        "separate_zf": config.separate_zf,
        "decouple_mu": config.decouple_mu,
        "prefer_dtype": config.prefer_dtype,
        "use_kvikio": config.use_kvikio,
        "return_jax": config.return_jax,
    }
    general = {key: value for key, value in general.items() if value is not None}
    try:
        sig = inspect.signature(dataset_cls)
    except (TypeError, ValueError):
        return {"root": root, **general}
    params = sig.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    root_name = next((name for name in ("root", "data_root", "dataset_root", "path") if name in params), "root")
    backend = _backend_for_dataset(dataset_cls, config)
    if backend is not None:
        for name in ("backend", "data_backend"):
            if name in params or accepts_kwargs:
                general[name] = backend
                break
    if accepts_kwargs:
        return {root_name: root, **general}
    return {root_name: root, **{key: value for key, value in general.items() if key in params}}


def _backend_for_dataset(dataset_cls: type[Any], config: CycloneKvikIOConfig) -> Any | None:
    try:
        sig = inspect.signature(dataset_cls)
    except (TypeError, ValueError):
        return None
    params = sig.parameters
    if "backend" not in params and "data_backend" not in params:
        return None

    module_name = getattr(dataset_cls, "__module__", "")
    candidates = []
    if module_name.startswith("neugk_jax."):
        candidates.append("neugk_jax.dataset.backend")
    if module_name.startswith("neugk."):
        candidates.append("neugk.dataset.backend")
    candidates.extend(("neugk_jax.dataset.backend", "neugk.dataset.backend"))

    for candidate in dict.fromkeys(candidates):
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue
        backend_cls = getattr(module, "KvikIOBackend", None) or getattr(module, "CycloneBackend", None)
        if backend_cls is None:
            continue
        return _instantiate_backend(backend_cls, config)

    msg = (
        "CycloneDataset requires a backend argument, but no upstream KvikIO backend class "
        "was found in `neugk_jax.dataset.backend` or `neugk.dataset.backend`."
    )
    raise MissingCycloneDependencyError(msg)


def _instantiate_backend(backend_cls: type[Any], config: CycloneKvikIOConfig) -> Any:
    try:
        sig = inspect.signature(backend_cls)
    except (TypeError, ValueError):
        try:
            return backend_cls(use_kvikio=config.use_kvikio)
        except TypeError:
            return backend_cls()
    kwargs: dict[str, Any] = {}
    if "use_kvikio" in sig.parameters:
        kwargs["use_kvikio"] = config.use_kvikio
    if "rank" in sig.parameters:
        kwargs["rank"] = 0
    return backend_cls(**kwargs)


def _normalize_upstream_metadata(upstream: Any) -> None:
    metadata = _as_mapping(_call_or_value(upstream, "metadata"))
    if not metadata:
        return
    for item in metadata.values() if all(_is_int_like(key) for key in metadata) else (metadata,):
        if isinstance(item, MutableMapping) and "fluxes" not in item and "flux" in item:
            item["fluxes"] = item["flux"]


def _offsets_from_upstream(upstream: Any, config: CycloneKvikIOConfig) -> tuple[int, ...]:
    offsets = _call_or_value(upstream, "offsets")
    if isinstance(offsets, Sequence) and not isinstance(offsets, str):
        return tuple(int(item) for item in offsets)
    metadata = _as_mapping(_call_or_value(upstream, "metadata"))
    file_count = len(metadata) if metadata and all(_is_int_like(key) for key in metadata) else 1
    return (config.offset,) * int(file_count)


def _discover_trajectory_ids(upstream: Any, config: CycloneKvikIOConfig) -> tuple[str, ...]:
    if config.trajectories is not None:
        return tuple(str(item) for item in config.trajectories)
    for name in ("trajectory_ids", "trajectories", "trajectory_names", "file_indices", "files"):
        ids = _ids_from_value(_call_or_value(upstream, name))
        if ids:
            return ids
    for name in ("num_trajectories", "n_trajectories", "num_files"):
        value = _call_or_value(upstream, name)
        if value is not None:
            return tuple(f"cyclone_{index:04d}" for index in range(int(value)))
    if _safe_len(upstream) is not None:
        return ("cyclone_0000",)
    msg = "could not discover Cyclone trajectories from upstream dataset"
    raise ValueError(msg)


def _ids_from_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    array = np.asarray(value) if hasattr(value, "__array__") else None
    if array is not None and array.ndim > 0:
        return tuple(str(item) for item in array.tolist())
    return ()


def _timesteps_from_value(value: Any, trajectory_id: str, trajectory_ids: tuple[str, ...]) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if trajectory_id in value:
            return int(value[trajectory_id])
        index = trajectory_ids.index(trajectory_id)
        return int(value.get(index)) if index in value else None
    if isinstance(value, Sequence) and not isinstance(value, str):
        index = trajectory_ids.index(trajectory_id)
        return int(value[index]) if index < len(value) else None
    return int(value)


def _load_upstream_sample(
    upstream: Any,
    trajectory_id: str,
    trajectory_index: int,
    timestep_index: int,
    adapter: CycloneKvikIODatasetAdapter,
) -> Any:
    for name in ("get_snapshot", "get_sample", "sample_at"):
        method = getattr(upstream, name, None)
        if method is None:
            continue
        for args in ((trajectory_id, timestep_index), (trajectory_index, timestep_index)):
            try:
                return method(*args)
            except TypeError:
                continue
    getitem = getattr(upstream, "__getitem__", None)
    if getitem is not None:
        return getitem(_flat_index(adapter, trajectory_id, trajectory_index, timestep_index))
    msg = "upstream CycloneDataset exposes no supported sample access method"
    raise AttributeError(msg)


def _flat_index(
    adapter: CycloneKvikIODatasetAdapter,
    trajectory_id: str,
    trajectory_index: int,
    timestep_index: int,
) -> int:
    if adapter.num_trajectories() == 1:
        return timestep_index
    return sum(adapter.num_timesteps(item) for item in adapter.trajectory_ids()[:trajectory_index]) + timestep_index


def _sample_mapping(sample: Any) -> dict[str, Any]:
    if isinstance(sample, Mapping):
        return dict(sample)
    if hasattr(sample, "_asdict"):
        return dict(sample._asdict())
    if hasattr(sample, "__dict__"):
        return dict(vars(sample))
    if isinstance(sample, tuple | list):
        if len(sample) == 2:
            return {"df": sample[0], "flux": sample[1]}
        keys = ("df", "phi", "flux", "timestep", "file_index", "timestep_index", "conditioning")
        return {key: value for key, value in zip(keys, sample, strict=False)}
    return {"df": sample}


def _load_input_fields(sample: Mapping[str, Any], input_fields: tuple[str, ...]) -> np.ndarray:
    arrays = []
    for field in input_fields:
        value = _first_present(sample, _field_aliases(field))
        if value is None:
            msg = f"Cyclone sample is missing requested input field {field!r}"
            raise KeyError(msg)
        arrays.append(_snapshot_array(value))
    return np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]


def _field_aliases(field: str) -> tuple[str, ...]:
    if field == "df":
        return ("df", "x", "input", "field")
    if field == "phi":
        return ("phi", "poten", "potential")
    return (field,)


def _condition_aliases(name: str) -> tuple[str, ...]:
    aliases = {
        "itg": ("itg", "ion_temp_grad"),
        "dg": ("dg", "density_grad"),
        "s_hat": ("s_hat", "shat"),
        "q": ("q",),
    }
    return aliases.get(name, (name,))


def _snapshot_array(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 7 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 5:
        arr = arr[None, ...]
    if arr.ndim != 6:
        msg = f"expected channel-first Cyclone snapshot rank 6 [C, S1, S2, S3, S4, S5], got {arr.shape}"
        raise ValueError(msg)
    return arr.astype(np.float32, copy=False)


def _load_flux(sample: Mapping[str, Any]) -> np.ndarray:
    value = _first_present(sample, ("flux", "fluxes", "avg_flux", "heat_flux", "y_flux", "gt_flux", "y_fluxavg"))
    if value is None:
        msg = "flux target requested but Cyclone sample has no flux, fluxes, avg_flux, or heat_flux field"
        raise KeyError(msg)
    return np.atleast_1d(_to_numpy(value).astype(np.float32, copy=False))


def _metadata_for_file(metadata: Mapping[str | int, Any], file_index: int) -> Mapping[str, Any]:
    for key in (file_index, str(file_index)):
        if key in metadata:
            return _as_mapping(metadata[key])
    if not all(_is_int_like(key) for key in metadata):
        return _as_mapping(metadata)
    return {}


def _spectra_aliases(name: str) -> tuple[str, ...]:
    aliases = [name, f"{name}_spectrum", f"spectra_{name}"]
    if name == "ky":
        aliases.append("kyspec")
    if name == "flux":
        aliases.append("fluxspec")
    if not name.endswith("spec"):
        aliases.append(f"{name}spec")
    return tuple(dict.fromkeys(aliases))


def _index_time_aligned_target(value: Any, target_index: int, name: str) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim >= 2:
        if target_index >= arr.shape[0]:
            msg = f"spectra target {name!r} has {arr.shape[0]} timesteps, cannot read target index {target_index}"
            raise SpectraUnavailableError(msg)
        return arr[target_index]
    return arr


def _direct_trajectory_paths(root: str | Path, config: CycloneKvikIOConfig) -> tuple[Path, ...]:
    root_path = Path(root).expanduser()
    if config.trajectories is not None:
        paths = tuple(_format_direct_trajectory_path(root_path, item, config) for item in config.trajectories)
    else:
        try:
            # The protocol manifests use portable trajectory IDs (the directory
            # name with the IFFT suffix removed) as their canonical order.  A
            # plain ``sorted(Path)`` orders ``iteration_148`` before
            # ``iteration_14`` because of the suffix, which silently changes
            # the ordered universe hash and invalidates fold pairing.
            paths = tuple(
                sorted(
                    (path for path in root_path.iterdir() if path.is_dir()),
                    key=lambda path: (_direct_portable_id(path.name), path.name),
                )
            )
        except OSError:
            return ()
    return tuple(
        path
        for path in paths
        if (path / "metadata.pkl").exists() and (path / "data").is_dir() and _has_direct_float32_shards(path)
    )


def _direct_portable_id(name: str) -> str:
    """Return the manifest identifier for a direct Cyclone directory name."""

    for suffix in ("_ifft_realpotens", "_ifft"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _has_direct_float32_shards(path: Path) -> bool:
    return any(not item.name.endswith(".bf16.bin") for item in (path / "data").glob("timestep_*.bin"))


def _format_direct_trajectory_path(root: Path, trajectory: str, config: CycloneKvikIOConfig) -> Path:
    path = Path(trajectory).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.suffix == ".h5":
        path = path.with_suffix("")
    if config.spatial_ifft:
        tag = "_ifft_realpotens" if config.real_potens else "_ifft"
        if tag not in path.name:
            path = path.with_name(path.name + tag)
    return path


def _read_direct_metadata(path: Path) -> Mapping[str, Any]:
    light_path = path / "metadata_light.pkl"
    metadata_path = light_path if light_path.is_file() else path / "metadata.pkl"
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, Mapping):
        msg = f"Cyclone metadata must be a mapping: {metadata_path}"
        raise TypeError(msg)
    if "fluxes" not in metadata and "flux" in metadata and isinstance(metadata, MutableMapping):
        metadata["fluxes"] = metadata["flux"]
    return metadata


def _direct_sample_indices(metadata: Mapping[str, Any], config: CycloneKvikIOConfig) -> tuple[int, ...]:
    timesteps = np.asarray(metadata.get("timesteps", ()))
    if timesteps.ndim == 0 or timesteps.shape[0] == 0:
        return ()
    stop = timesteps.shape[0] - config.tail_offset if config.tail_offset else timesteps.shape[0]
    available = np.arange(config.offset, max(config.offset, stop), dtype=np.int64)[:: config.subsample]
    sample_count = max(0, available.shape[0] - config.bundle_seq_length * 2 + 1)
    return tuple(int(index) for index in available[:sample_count])


def _read_direct_df(path: Path, metadata: Mapping[str, Any], raw_index: int, use_kvikio: bool) -> np.ndarray:
    resolution = tuple(int(value) for value in np.asarray(metadata["resolution"]).tolist())
    shape = (2, *resolution)
    file_path = path / "data" / f"timestep_{raw_index:05d}.bin"
    return _read_cyclone_bin(file_path, shape, use_kvikio=use_kvikio)


def _read_direct_phi(path: Path, metadata: Mapping[str, Any], raw_index: int, use_kvikio: bool) -> np.ndarray:
    resolution = tuple(int(value) for value in np.asarray(metadata["resolution"]).tolist())
    shape = (resolution[3], resolution[2], resolution[4])
    file_path = path / "data" / f"poten_{raw_index:05d}.bin"
    return _read_cyclone_bin(file_path, shape, use_kvikio=use_kvikio)


def _read_cyclone_bin(path: Path, shape: tuple[int, ...], *, use_kvikio: bool) -> np.ndarray:
    if use_kvikio:
        try:
            import cupy as cp
            import kvikio
        except ImportError as exc:
            msg = "data.cyclone.use_kvikio=true requires `cupy` and `kvikio` in the environment"
            raise MissingCycloneDependencyError(msg) from exc
        gpu_array = cp.empty(int(np.prod(shape)), dtype=cp.float32)
        with kvikio.CuFile(str(path), "r") as handle:
            handle.read(gpu_array)
        return cp.asnumpy(gpu_array.reshape(shape)).astype(np.float32, copy=False)
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def _separate_zf_numpy(value: np.ndarray, *, axis: int = 0) -> np.ndarray:
    nky = value.shape[-1]
    zf = np.repeat(np.mean(value, axis=-1, keepdims=True), repeats=nky, axis=-1)
    return np.concatenate([zf, value - zf], axis=axis).astype(np.float32, copy=False)


def _direct_target(metadata: Mapping[str, Any], aliases: tuple[str, ...], target_index: int) -> np.ndarray:
    value = _first_present(metadata, aliases)
    if value is None:
        msg = f"target not found in Cyclone metadata; tried aliases: {', '.join(aliases)}"
        raise KeyError(msg)
    arr = _to_numpy(value)
    if arr.ndim >= 1 and arr.shape[0] > target_index:
        arr = arr[target_index]
    elif arr.ndim >= 1 and arr.shape[0] <= target_index:
        msg = f"Cyclone metadata target has {arr.shape[0]} timesteps, cannot read index {target_index}"
        raise IndexError(msg)
    return np.atleast_1d(arr).astype(np.float32, copy=False)


def _direct_time(metadata: Mapping[str, Any], raw_index: int) -> float | None:
    timesteps = _to_numpy(metadata.get("timesteps", ()))
    if timesteps.ndim == 0 or raw_index >= timesteps.shape[0]:
        return None
    return float(timesteps[raw_index])


def _sample_metadata(sample: Mapping[str, Any], conditions: tuple[str, ...]) -> dict[str, Any]:
    metadata = dict(_as_mapping(sample.get("metadata")))
    conditioning = _first_present(sample, ("conditioning", "conditions", "cond"))
    if conditioning is not None:
        metadata["conditioning"] = _to_numpy(conditioning).astype(np.float32, copy=False)
    for key in conditions:
        value = _first_present(sample, (key,))
        if value is not None:
            metadata[key] = _small_value(value)
    for key in ("file_index", "trajectory_index", "timestep_index"):
        value = _first_present(sample, (key,))
        if value is not None:
            metadata[key] = _small_value(value)
    return metadata


def _small_value(value: Any) -> Any:
    arr = _to_numpy(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size <= 16:
        return arr.tolist()
    return f"array(shape={tuple(arr.shape)}, dtype={arr.dtype})"


def _is_int_like(value: Any) -> bool:
    if isinstance(value, int | np.integer):
        return True
    if isinstance(value, str):
        return value.isdigit()
    return False


def _first_present(sample: Mapping[str, Any], keys: tuple[str, ...], *, nested: Mapping[str, Any] | None = None) -> Any:
    for key in keys:
        if nested is not None and key in nested and nested[key] is not None:
            return nested[key]
        if key in sample and sample[key] is not None:
            return sample[key]
        metadata = _as_mapping(sample.get("metadata"))
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "block_until_ready"):
        value = value.block_until_ready()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            if hasattr(value, "float"):
                return value.float().numpy()
    return np.asarray(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = _to_numpy(value)
    return float(arr.reshape(-1)[0]) if arr.size else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    arr = _to_numpy(value)
    return int(arr.reshape(-1)[0]) if arr.size else None


def _call_named(obj: Any, name: str, *args: Any) -> Any:
    attr = getattr(obj, name, None)
    if attr is None:
        return None
    if callable(attr):
        try:
            return attr(*args)
        except TypeError:
            return None
    return attr


def _call_or_value(obj: Any, name: str) -> Any:
    attr = getattr(obj, name, None)
    if attr is None:
        return None
    if callable(attr):
        try:
            return attr()
        except TypeError:
            return None
    return attr


def _safe_len(obj: Any) -> int | None:
    try:
        return len(obj)
    except TypeError:
        return None
