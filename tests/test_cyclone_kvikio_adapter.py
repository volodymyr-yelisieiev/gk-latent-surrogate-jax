from __future__ import annotations

import importlib
import pickle
import sys
import types
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from gk_surrogate.cli import main
from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import DataConfig
from gk_surrogate.data.cyclone_kvikio import (
    ComputedSpectraProvider,
    DirectCycloneKvikIODataset,
    MissingCycloneDependencyError,
    NoSpectraProvider,
    SpectraUnavailableError,
    _backend_for_dataset,
    _call_named,
    _call_or_value,
    _direct_sample_indices,
    _direct_target,
    _direct_time,
    _direct_trajectory_paths,
    _field_aliases,
    _first_present,
    _flat_index,
    _format_direct_trajectory_path,
    _ids_from_value,
    _instantiate_backend,
    _load_flux,
    _load_input_fields,
    _load_upstream_sample,
    _optional_float,
    _optional_int,
    _read_cyclone_bin,
    _read_direct_metadata,
    _read_direct_phi,
    _safe_len,
    _sample_mapping,
    _separate_zf_numpy,
    _small_value,
    _snapshot_array,
    _timesteps_from_value,
    _to_numpy,
    cyclone_dependency_available,
    direct_cyclone_layout_available,
)
from gk_surrogate.data.factory import build_dataset
from gk_surrogate.data.inspect import inspect_dataset

REAL_IMPORT_MODULE = importlib.import_module


class FakeCycloneDataset:
    instances: list[FakeCycloneDataset] = []

    def __init__(self, root: str, split: str = "train", fields_to_load: tuple[str, ...] = ("df",), **kwargs: Any):
        self.root = root
        self.split = split
        self.fields_to_load = fields_to_load
        self.kwargs = kwargs
        self.metadata = {"normalization": {"mode": "none"}, "geometry": {"kx": [0.0], "ky": [0.0]}}
        self.geometry = {"kx": np.arange(2), "ky": np.arange(3)}
        FakeCycloneDataset.instances.append(self)

    def trajectory_ids(self) -> tuple[str, str]:
        return ("traj_a", "traj_b")

    def num_timesteps(self, trajectory_id: str) -> int:
        if trajectory_id not in {"traj_a", "traj_b"}:
            raise KeyError(trajectory_id)
        return 3

    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> dict[str, Any]:
        trajectory_index = 0 if trajectory_id == "traj_a" else 1
        value = float(trajectory_index * 10 + timestep_index)
        return {
            "df": np.full((2, 2, 2, 2, 2, 2), value, dtype=np.float32),
            "flux": np.asarray([value + 0.5], dtype=np.float32),
            "timestep": np.asarray(value * 0.25, dtype=np.float32),
            "file_index": trajectory_index,
            "timestep_index": timestep_index,
            "conditioning": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "itg": np.asarray(1.0, dtype=np.float32),
            "spectra": {"ky": np.asarray([value, value + 1.0], dtype=np.float32)},
        }

    def __len__(self) -> int:
        return 6


class FlatCycloneDataset:
    def __init__(self, path: str, **kwargs: Any):
        self.path = path
        self.kwargs = kwargs
        self.n_timesteps = 4

    def __len__(self) -> int:
        return self.n_timesteps

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.full((2, 2, 2, 2, 2), index, dtype=np.float32)
        flux = np.asarray([index], dtype=np.float32)
        return x, flux


class PositionalCycloneDataset:
    def __init__(self, *args: Any, **kwargs: Any):
        if not args:
            raise TypeError("root must be positional")
        self.root = args[0]
        self.kwargs = kwargs
        self.trajectories = ("positional_a",)
        self.timesteps_per_trajectory = {"positional_a": 2}

    def get_sample(self, trajectory_id: str, timestep_index: int) -> SimpleNamespace:
        return SimpleNamespace(
            input=np.ones((1, 2, 2, 2, 2, 2), dtype=np.float32) * timestep_index,
            avg_flux=np.asarray([timestep_index + 1.0], dtype=np.float32),
            time=np.asarray([timestep_index], dtype=np.float32),
        )


class SequenceTimestepsDataset:
    def __init__(self, root: str, **kwargs: Any):
        self.root = root
        self.trajectory_names = ("seq_a", "seq_b")
        self.timesteps_per_trajectory = [2, 3]
        self.metadata = {"geometry": {"theta": [0.0]}}

    def sample_at(self, trajectory: int | str, timestep_index: int) -> dict[str, Any]:
        if not isinstance(trajectory, int):
            raise TypeError("integer trajectory index required")
        return {
            "x": np.ones((1, 1, 2, 2, 2, 2, 2), dtype=np.float32) * (trajectory + timestep_index),
            "heat_flux": np.asarray(2.0, dtype=np.float32),
            "metadata": {"q": np.asarray([1.4, 1.5], dtype=np.float32)},
        }


class RequiredBackendCycloneDataset:
    def __init__(
        self,
        backend: Any,
        path: str,
        split: str = "train",
        fields_to_load: tuple[str, ...] = ("df",),
        **kwargs: Any,
    ):
        self.backend = backend
        self.path = path
        self.split = split
        self.fields_to_load = fields_to_load
        self.kwargs = kwargs
        self.files = [f"{path}/iteration_13_ifft_realpotens"]
        self.file_num_samples = [2]
        self.metadata = {
            0: {
                "timesteps": np.asarray([80.0, 81.0, 82.0], dtype=np.float32),
                "flux": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "kyspec": np.asarray([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
                "fluxspec": np.asarray([[0.0, 1.0], [100.0, 101.0], [200.0, 201.0]], dtype=np.float32),
                "geometry": {"kx": np.asarray([0.0], dtype=np.float32)},
            }
        }
        self.offsets = [0]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(
            df=np.full((2, 2, 2, 2, 2, 2), index, dtype=np.float32),
            y_flux=np.asarray(index + 0.25, dtype=np.float32),
            timestep=np.asarray(index + 80.0, dtype=np.float32),
            file_index=np.asarray(0, dtype=np.int64),
            timestep_index=np.asarray(index, dtype=np.int64),
            itg=np.asarray(1.0, dtype=np.float32),
            dg=np.asarray(2.0, dtype=np.float32),
            s_hat=np.asarray(3.0, dtype=np.float32),
            q=np.asarray(4.0, dtype=np.float32),
        )


class FakeKvikIOBackend:
    def __init__(self, rank: int = 0, use_kvikio: bool = True):
        self.rank = rank
        self.use_kvikio = use_kvikio


class CountOnlyDataset:
    def __init__(self, root: str, **kwargs: Any):
        self.root = root
        self.num_trajectories = 2


class NoTrajectoryDataset:
    def __init__(self, root: str, **kwargs: Any):
        self.root = root


class LenOnlyCycloneDataset:
    def __init__(self, root: str, **kwargs: Any):
        self.root = root

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "df": np.ones((1, 2, 2, 2, 2, 2), dtype=np.float32) * index,
            "flux": np.asarray([index], dtype=np.float32),
        }


class TwoTrajectoryLenOnlyDataset:
    def __init__(self, root: str, **kwargs: Any):
        self.root = root
        self.num_trajectories = 2

    def __len__(self) -> int:
        return 6


class AlwaysFailDataset:
    def __init__(self, *args: Any, **kwargs: Any):
        raise TypeError("constructor failed")


class BlockingArray:
    def block_until_ready(self) -> np.ndarray:
        return np.asarray([1.0], dtype=np.float32)


class TorchLikeArray:
    def __init__(self):
        self.float_called = False

    def detach(self) -> TorchLikeArray:
        return self

    def cpu(self) -> TorchLikeArray:
        return self

    def numpy(self) -> np.ndarray:
        if not self.float_called:
            raise TypeError("unsupported dtype")
        return np.asarray([1.0], dtype=np.float32)

    def float(self) -> TorchLikeArray:
        self.float_called = True
        return self


def test_cyclone_adapter_maps_upstream_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_neugk(monkeypatch, FakeCycloneDataset)
    data = _cyclone_data(tmp_path)

    dataset = build_dataset(data)
    sample = dataset.get_snapshot("traj_b", 2)

    assert sample.x.shape == (2, 2, 2, 2, 2, 2)
    assert sample.x.dtype == np.float32
    assert sample.targets.flux is not None
    assert sample.targets.flux.shape == (1,)
    assert sample.targets.spectra == {}
    assert sample.trajectory_index == 1
    assert sample.timestep_index == 2
    assert sample.physical_time == pytest.approx(3.0)
    assert sample.metadata["conditioning"].shape == (4,)
    assert sample.metadata["itg"] == pytest.approx(1.0)
    assert FakeCycloneDataset.instances[-1].kwargs["prefer_dtype"] == "float32"
    assert FakeCycloneDataset.instances[-1].kwargs["use_kvikio"] is False

    build_dataset(data.model_copy(update={"split": "all"}))
    assert FakeCycloneDataset.instances[-1].split == "train"
    build_dataset(data.model_copy(update={"split": "test"}))
    assert FakeCycloneDataset.instances[-1].split == "val"


def test_cyclone_adapter_maps_stored_spectra_and_missing_spectra_is_clear(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_neugk(monkeypatch, FakeCycloneDataset)
    with_spectra = _cyclone_data(tmp_path, target_spectra=("ky",))
    sample = build_dataset(with_spectra).get_snapshot("traj_a", 1)
    assert sample.targets.spectra["ky"].shape == (2,)

    missing = _cyclone_data(tmp_path, target_spectra=("q",))
    with pytest.raises(SpectraUnavailableError, match="q"):
        build_dataset(missing).get_snapshot("traj_a", 1)


def test_cyclone_adapter_flat_getitem_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_neugk(monkeypatch, FlatCycloneDataset)
    data = _cyclone_data(tmp_path).model_copy(
        update={"cyclone": _cyclone_data(tmp_path).cyclone.model_copy(update={"prefer_dtype": "float32"})}
    )

    dataset = build_dataset(data)
    assert tuple(dataset.trajectory_ids()) == ("cyclone_0000",)
    assert dataset.num_timesteps("cyclone_0000") == 4
    sample = dataset.get_snapshot("cyclone_0000", 3)
    assert sample.x.shape == (1, 2, 2, 2, 2, 2)
    assert sample.targets.flux is not None
    assert sample.targets.flux.tolist() == [3.0]


def test_cyclone_adapter_positional_constructor_and_config_trajectories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_neugk(monkeypatch, PositionalCycloneDataset)
    base = _cyclone_data(tmp_path)
    assert base.cyclone is not None
    data = base.model_copy(update={"cyclone": base.cyclone.model_copy(update={"trajectories": ("positional_a",)})})

    dataset = build_dataset(data)
    assert tuple(dataset.trajectory_ids()) == ("positional_a",)
    assert dataset.num_timesteps("positional_a") == 2
    assert dataset.snapshot_shape() == (1, 2, 2, 2, 2, 2)
    sample = dataset.get_snapshot("positional_a", 1)
    assert sample.targets.flux is not None
    assert sample.targets.flux.tolist() == [2.0]


def test_cyclone_adapter_sequence_timesteps_and_sample_at_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_neugk(monkeypatch, SequenceTimestepsDataset)
    data = _cyclone_data(tmp_path)

    dataset = build_dataset(data)
    assert tuple(dataset.trajectory_ids()) == ("seq_a", "seq_b")
    assert dataset.num_timesteps("seq_b") == 3
    assert dataset.geometry_keys() == ("theta",)
    assert dataset.quantized_shards_available() is False
    sample = dataset.get_snapshot("seq_b", 2)
    assert sample.x.shape == (1, 2, 2, 2, 2, 2)
    assert sample.targets.flux is not None
    assert sample.targets.flux.tolist() == [2.0]
    assert sample.metadata["q"] == pytest.approx([1.4, 1.5])


def test_cyclone_adapter_supports_public_neugk_backend_constructor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_neugk(monkeypatch, RequiredBackendCycloneDataset, package_name="neugk")
    data = _cyclone_data(tmp_path)

    dataset = build_dataset(data)
    assert tuple(dataset.trajectory_ids()) == (f"{tmp_path}/iteration_13_ifft_realpotens",)
    assert dataset.num_timesteps(f"{tmp_path}/iteration_13_ifft_realpotens") == 2
    assert dataset.metadata_keys() == ("flux", "fluxes", "fluxspec", "geometry", "kyspec", "timesteps")
    assert dataset.spectra_keys() == ("fluxspec", "kyspec")
    assert "fluxes" in dataset._upstream.metadata[0]
    assert dataset.geometry_keys() == ("kx",)
    assert dataset._upstream.backend.use_kvikio is False

    sample = dataset.get_snapshot(f"{tmp_path}/iteration_13_ifft_realpotens", 1)
    assert sample.targets.flux is not None
    assert sample.targets.flux.tolist() == [1.25]
    assert sample.metadata["q"] == pytest.approx(4.0)


def test_cyclone_direct_kvikio_reader_bypasses_upstream_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cyclone"
    trajectory = root / "iteration_0_ifft_realpotens"
    data_dir = trajectory / "data"
    data_dir.mkdir(parents=True)
    resolution = np.asarray([2, 1, 1, 2, 2], dtype=np.int64)
    shape = (2, *resolution.tolist())
    for index in range(4):
        (np.ones(shape, dtype=np.float32) * index).tofile(data_dir / f"timestep_{index:05d}.bin")
    metadata = {
        "timesteps": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "resolution": resolution,
        "flux": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "kyspec": np.arange(8, dtype=np.float32).reshape(4, 2),
        "fluxspec": np.arange(8, 16, dtype=np.float32).reshape(4, 2),
        "ion_temp_grad": np.asarray([1.0], dtype=np.float32),
        "density_grad": np.asarray([2.0], dtype=np.float32),
        "s_hat": np.asarray([3.0], dtype=np.float32),
        "q": np.asarray([4.0], dtype=np.float32),
        "geometry": {"kx": np.asarray([0.0], dtype=np.float32)},
    }
    import pickle

    with (trajectory / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)

    def missing_neugk(name: str) -> types.ModuleType:
        if name.startswith(("neugk_jax", "neugk")):
            raise AssertionError("direct KvikIO reader should not import upstream neugk")
        return REAL_IMPORT_MODULE(name)

    def read_bin(path: Path, shape: tuple[int, ...], *, use_kvikio: bool) -> np.ndarray:
        assert use_kvikio is True
        return np.fromfile(path, dtype=np.float32).reshape(shape)

    monkeypatch.setattr("gk_surrogate.data.cyclone_kvikio.importlib.import_module", missing_neugk)
    monkeypatch.setattr("gk_surrogate.data.cyclone_kvikio._read_cyclone_bin", read_bin)
    data = _cyclone_data(root, target_spectra=("kyspec", "fluxspec")).model_copy(
        update={
            "cyclone": _cyclone_data(root).cyclone.model_copy(
                update={"trajectories": ("iteration_0",), "use_kvikio": True}
            )
        }
    )

    assert data.cyclone is not None
    assert direct_cyclone_layout_available(root, data.cyclone) is True
    dataset = build_dataset(data)
    assert isinstance(dataset, DirectCycloneKvikIODataset)
    assert dataset.trajectory_ids() == (str(trajectory),)
    assert dataset.num_timesteps(str(trajectory)) == 3
    sample = dataset.get_snapshot(str(trajectory), 0)
    assert sample.x.shape == (4, 2, 1, 1, 2, 2)
    assert sample.targets.flux is not None
    assert sample.targets.flux.tolist() == [1.0]
    assert sample.targets.spectra["kyspec"].tolist() == [2.0, 3.0]
    assert sample.metadata["itg"] == [1.0]


def test_cyclone_direct_reader_bypasses_upstream_without_kvikio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cyclone"
    trajectory = root / "iteration_0_ifft_realpotens"
    data_dir = trajectory / "data"
    data_dir.mkdir(parents=True)
    resolution = np.asarray([2, 1, 1, 2, 2], dtype=np.int64)
    shape = (2, *resolution.tolist())
    for index in range(4):
        (np.ones(shape, dtype=np.float32) * index).tofile(data_dir / f"timestep_{index:05d}.bin")
    metadata = {
        "timesteps": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "resolution": resolution,
        "flux": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "kyspec": np.arange(8, dtype=np.float32).reshape(4, 2),
        "fluxspec": np.arange(8, 16, dtype=np.float32).reshape(4, 2),
    }
    import pickle

    with (trajectory / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)

    def missing_neugk(name: str) -> types.ModuleType:
        if name.startswith(("neugk_jax", "neugk")):
            raise AssertionError("direct reader should not import upstream neugk")
        return REAL_IMPORT_MODULE(name)

    monkeypatch.setattr("gk_surrogate.data.cyclone_kvikio.importlib.import_module", missing_neugk)
    data = _cyclone_data(root, target_spectra=("kyspec", "fluxspec")).model_copy(
        update={
            "cyclone": _cyclone_data(root).cyclone.model_copy(
                update={"trajectories": ("iteration_0",), "use_kvikio": False}
            )
        }
    )

    dataset = build_dataset(data)

    assert isinstance(dataset, DirectCycloneKvikIODataset)
    assert dataset.trajectory_ids() == (str(trajectory),)
    sample = dataset.get_snapshot(str(trajectory), 0)
    assert sample.x.shape == (4, 2, 1, 1, 2, 2)
    assert sample.targets.flux is not None
    assert sample.targets.flux.tolist() == [1.0]
    assert sample.targets.spectra["fluxspec"].tolist() == [10.0, 11.0]


def test_direct_paths_follow_portable_manifest_order(tmp_path: Path) -> None:
    root = tmp_path / "cyclone"
    for name in ("iteration_148_ifft_realpotens", "iteration_14_ifft_realpotens", "iteration_2_ifft_realpotens"):
        path = root / name
        (path / "data").mkdir(parents=True)
        metadata = {"timesteps": np.asarray([0.0]), "resolution": np.asarray([1, 1, 1, 1, 1])}
        (path / "metadata.pkl").write_bytes(pickle.dumps(metadata))
        np.zeros((1,), dtype=np.float32).tofile(path / "data" / "timestep_00000.bin")
    config = _cyclone_data(root).cyclone
    assert config is not None
    assert [path.name for path in _direct_trajectory_paths(root, config)] == [
        "iteration_14_ifft_realpotens",
        "iteration_148_ifft_realpotens",
        "iteration_2_ifft_realpotens",
    ]


def test_cyclone_direct_reader_helpers_and_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "cyclone"
    trajectory = root / "iteration_1_ifft_realpotens"
    data_dir = trajectory / "data"
    data_dir.mkdir(parents=True)
    resolution = np.asarray([2, 1, 1, 2, 2], dtype=np.int64)
    df_shape = (2, *resolution.tolist())
    phi_shape = (2, 1, 2)
    for index in (1, 2):
        values = np.arange(np.prod(df_shape), dtype=np.float32).reshape(df_shape) + index
        values.tofile(data_dir / f"timestep_{index:05d}.bin")
    np.arange(np.prod(phi_shape), dtype=np.float32).reshape(phi_shape).tofile(data_dir / "poten_00001.bin")
    (data_dir / "timestep_00001.bf16.bin").write_bytes(b"quantized")
    metadata = {
        "timesteps": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "resolution": resolution,
        "flux": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "kyspec": np.arange(10, dtype=np.float32).reshape(5, 2),
        "geometry": {"kx": np.asarray([0.0], dtype=np.float32)},
    }
    import pickle

    with (trajectory / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    with (trajectory / "metadata_light.pkl").open("wb") as handle:
        pickle.dump({**metadata, "source": "light"}, handle)

    base = _cyclone_data(root, target_spectra=("kyspec",))
    assert base.cyclone is not None
    config = base.cyclone.model_copy(
        update={"trajectories": ("iteration_1",), "offset": 1, "tail_offset": 1, "use_kvikio": True}
    )
    assert _format_direct_trajectory_path(root, "iteration_1.h5", config) == trajectory
    assert _direct_trajectory_paths(root, config) == (trajectory,)
    bf16_only = root / "iteration_2_ifft_realpotens"
    (bf16_only / "data").mkdir(parents=True)
    (bf16_only / "metadata.pkl").write_bytes(b"fixture")
    (bf16_only / "data" / "timestep_00001.bf16.bin").write_bytes(b"quantized")
    no_float32 = config.model_copy(update={"trajectories": ("iteration_2",)})
    assert direct_cyclone_layout_available(root, no_float32) is False
    assert _direct_trajectory_paths(root, no_float32) == ()
    loaded_metadata = _read_direct_metadata(trajectory)
    assert loaded_metadata["source"] == "light"
    assert "fluxes" in loaded_metadata
    assert _direct_sample_indices(loaded_metadata, config) == (1, 2)
    assert _direct_target(loaded_metadata, ("kyspec",), 2).tolist() == [4.0, 5.0]
    assert _direct_time(loaded_metadata, 1) == pytest.approx(1.0)
    assert _direct_time(loaded_metadata, 99) is None
    with pytest.raises(KeyError, match="target not found"):
        _direct_target(loaded_metadata, ("missing",), 0)
    with pytest.raises(IndexError, match="cannot read index"):
        _direct_target(loaded_metadata, ("kyspec",), 99)

    assert _read_cyclone_bin(data_dir / "timestep_00001.bin", df_shape, use_kvikio=False).shape == df_shape
    assert _read_direct_phi(trajectory, loaded_metadata, 1, use_kvikio=False).shape == phi_shape
    separated = _separate_zf_numpy(np.ones(df_shape, dtype=np.float32), axis=0)
    assert separated.shape[0] == 4

    def missing_optional(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
        del args, kwargs
        if name in {"cupy", "kvikio"}:
            raise ImportError(name)
        return REAL_IMPORT_MODULE(name)

    monkeypatch.setattr("builtins.__import__", missing_optional)
    with pytest.raises(MissingCycloneDependencyError, match="cupy"):
        _read_cyclone_bin(data_dir / "timestep_00001.bin", df_shape, use_kvikio=True)

    data = base.model_copy(update={"cyclone": config, "target_spectra": ()})
    monkeypatch.setattr(
        "gk_surrogate.data.cyclone_kvikio._read_cyclone_bin",
        lambda path, shape, *, use_kvikio: np.fromfile(path, dtype=np.float32).reshape(shape),
    )
    dataset = DirectCycloneKvikIODataset(
        data.root,
        config,
        target_spectra=data.target_spectra,
        target_flux=False,
        input_fields=("df",),
    )
    assert dataset.num_trajectories() == 1
    assert dataset.snapshot_shape() == (4, 2, 1, 1, 2, 2)
    assert dataset.metadata_keys()[:2] == ("flux", "fluxes")
    assert dataset.geometry_keys() == ("kx",)
    assert dataset.spectra_keys() == ("kyspec",)
    assert dataset.sample_count_estimate() == 2
    assert dataset.quantized_shards_available() is True
    assert "spectra targets are not requested" in "\n".join(dataset.inspection_warnings())
    with pytest.raises(KeyError, match="unknown trajectory_id"):
        dataset.num_timesteps("missing")
    with pytest.raises(IndexError, match="out of range"):
        dataset.get_snapshot(str(trajectory), 2)
    bad_field = DirectCycloneKvikIODataset(data.root, config, input_fields=("bad",))
    with pytest.raises(KeyError, match="does not support"):
        bad_field.get_snapshot(str(trajectory), 0)
    with pytest.raises(FileNotFoundError):
        DirectCycloneKvikIODataset(tmp_path / "missing", config)


def test_cyclone_adapter_maps_stored_metadata_spectra_to_next_timestep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_neugk(monkeypatch, RequiredBackendCycloneDataset, package_name="neugk")
    data = _cyclone_data(tmp_path, target_spectra=("ky", "fluxspec"))

    dataset = build_dataset(data)
    sample = dataset.get_snapshot(f"{tmp_path}/iteration_13_ifft_realpotens", 1)

    assert sample.targets.spectra["ky"].tolist() == [20.0, 21.0]
    assert sample.targets.spectra["fluxspec"].tolist() == [200.0, 201.0]


def test_cyclone_adapter_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_neugk(monkeypatch, CountOnlyDataset)
    dataset = build_dataset(_cyclone_data(tmp_path))
    assert tuple(dataset.trajectory_ids()) == ("cyclone_0000", "cyclone_0001")
    assert dataset.sample_count_estimate() is None
    with pytest.raises(ValueError, match="timestep count"):
        dataset.num_timesteps("cyclone_0000")
    with pytest.raises(KeyError, match="unknown trajectory_id"):
        dataset.num_timesteps("missing")

    _install_fake_neugk(monkeypatch, NoTrajectoryDataset)
    with pytest.raises(ValueError, match="discover Cyclone trajectories"):
        build_dataset(_cyclone_data(tmp_path))

    _install_fake_neugk(monkeypatch, AlwaysFailDataset)
    with pytest.raises(TypeError, match="failed to construct"):
        build_dataset(_cyclone_data(tmp_path))


def test_cyclone_adapter_len_only_fallbacks_and_bounds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_neugk(monkeypatch, LenOnlyCycloneDataset)
    dataset = build_dataset(_cyclone_data(tmp_path))
    assert tuple(dataset.trajectory_ids()) == ("cyclone_0000",)
    assert dataset.num_timesteps("cyclone_0000") == 2
    assert dataset.metadata_keys() == ()
    with pytest.raises(IndexError, match="out of range"):
        dataset.get_snapshot("cyclone_0000", 2)

    base = _cyclone_data(tmp_path)
    assert base.cyclone is not None
    bundled = base.model_copy(update={"cyclone": base.cyclone.model_copy(update={"bundle_seq_length": 2})})
    bundled_dataset = build_dataset(bundled)
    assert "bundle_seq_length" in "\n".join(bundled_dataset.inspection_warnings())

    _install_fake_neugk(monkeypatch, TwoTrajectoryLenOnlyDataset)
    two = build_dataset(_cyclone_data(tmp_path))
    assert tuple(two.trajectory_ids()) == ("cyclone_0000", "cyclone_0001")
    assert two.num_timesteps("cyclone_0001") == 3


def test_cyclone_inspection_reports_backend_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_neugk(monkeypatch, FakeCycloneDataset)
    data_dir = tmp_path / "cyclone"
    (data_dir / "data").mkdir(parents=True)
    (data_dir / "data" / "timestep_00000.bf16.bin").write_bytes(b"fixture")
    data = _cyclone_data(data_dir)

    inspection = inspect_dataset(data, max_trajectories=1, max_target_samples=2)
    payload = inspection.as_dict()

    assert payload["backend"] == "cyclone_kvikio"
    assert payload["sample_count_estimate"] == 6
    assert payload["kvikio_enabled"] is False
    assert payload["preferred_dtype"] == "float32"
    assert payload["quantized_shards_available"] is True
    assert "normalization" in inspection.metadata_keys
    assert inspection.geometry_keys == ("kx", "ky")
    assert "spectra targets are not requested" in "\n".join(inspection.warnings)


def test_cyclone_dependency_error_is_optional_and_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing_neugk(name: str) -> types.ModuleType:
        if name.startswith(("neugk_jax", "neugk")):
            raise ImportError(name)
        return REAL_IMPORT_MODULE(name)

    monkeypatch.setattr("gk_surrogate.data.cyclone_kvikio.importlib.import_module", missing_neugk)
    assert cyclone_dependency_available() is False
    with pytest.raises(MissingCycloneDependencyError, match="neugk_jax.*neugk"):
        build_dataset(_cyclone_data(tmp_path))

    _install_fake_neugk(monkeypatch, FakeCycloneDataset)
    monkeypatch.setattr("gk_surrogate.data.cyclone_kvikio.importlib.import_module", REAL_IMPORT_MODULE)
    assert cyclone_dependency_available() is True


def test_cyclone_template_loads_and_cli_dry_run_validates_without_upstream(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path
) -> None:
    template = repo_root / "configs" / "data" / "cyclone_kvikio_template.yaml"
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", str(tmp_path))
    config = load_config(template, command="inspect-data")
    assert config.data.backend == "cyclone_kvikio"
    assert config.data.cyclone is not None
    assert config.data.cyclone.separate_zf is True
    assert config.data.target_spectra == ()

    def missing_neugk(name: str) -> types.ModuleType:
        if name.startswith(("neugk_jax", "neugk")):
            raise ImportError(name)
        return REAL_IMPORT_MODULE(name)

    monkeypatch.setattr("gk_surrogate.data.cyclone_kvikio.importlib.import_module", missing_neugk)
    assert main(["inspect-data", "--config", str(template), "--dry-run"]) == 0


def test_cyclone_config_rejects_invalid_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cyclone_kvikio backend"):
        DataConfig.model_validate({"backend": "cyclone_kvikio", "root": str(tmp_path)})
    with pytest.raises(ValueError, match="non-empty strings"):
        DataConfig.model_validate(
            {
                "backend": "cyclone_kvikio",
                "root": str(tmp_path),
                "cyclone": {"fields_to_load": [], "conditions": ["q"]},
            }
        )


def test_cyclone_private_mapping_helpers_cover_edge_contracts() -> None:
    with pytest.raises(SpectraUnavailableError, match="ky"):
        NoSpectraProvider(("ky",)).spectra_for({})
    with pytest.raises(SpectraUnavailableError, match="computed Cyclone spectra"):
        ComputedSpectraProvider(("ky",)).spectra_for({})
    assert _ids_from_value({"a": 1, "b": 2}) == ("a", "b")
    assert _ids_from_value("only") == ("only",)
    assert _ids_from_value(np.asarray([1, 2])) == ("1", "2")
    assert _ids_from_value(5) == ()
    assert _timesteps_from_value({"a": 2}, "a", ("a",)) == 2
    assert _timesteps_from_value({0: 3}, "a", ("a",)) == 3
    assert _timesteps_from_value([4], "a", ("a",)) == 4
    assert _timesteps_from_value(None, "a", ("a",)) is None
    assert _field_aliases("phi") == ("phi", "poten", "potential")
    assert _field_aliases("custom") == ("custom",)

    Record = namedtuple("Record", ["df", "flux"])
    x = np.ones((1, 2, 2, 2, 2, 2), dtype=np.float32)
    flux = np.asarray([1.0], dtype=np.float32)
    assert _sample_mapping(Record(x, flux))["flux"].tolist() == [1.0]
    assert _sample_mapping(SimpleNamespace(df=x, flux=flux))["df"].shape == x.shape
    assert _sample_mapping((x, None, flux, 0.0))["flux"].tolist() == [1.0]
    assert _sample_mapping(x)["df"].shape == x.shape
    assert _load_input_fields({"poten": x}, ("phi",)).shape == x.shape
    with pytest.raises(KeyError, match="input field"):
        _load_input_fields({}, ("df",))
    with pytest.raises(ValueError, match="rank 6"):
        _snapshot_array(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(KeyError, match="flux target"):
        _load_flux({})

    assert _first_present({"metadata": {"q": 1.0}}, ("q",)) == 1.0
    assert _to_numpy(BlockingArray()).tolist() == [1.0]
    assert _optional_float(None) is None
    assert _optional_float([]) is None
    assert _optional_int(None) is None
    assert _safe_len(object()) is None
    assert _call_named(SimpleNamespace(value=3), "value") == 3
    assert _call_named(SimpleNamespace(), "missing") is None
    assert _call_named(SimpleNamespace(f=lambda _x: (_ for _ in ()).throw(TypeError())), "f", 1) is None
    assert _call_or_value(SimpleNamespace(value=4), "value") == 4
    assert _call_or_value(SimpleNamespace(), "missing") is None
    assert _call_or_value(SimpleNamespace(f=lambda: (_ for _ in ()).throw(TypeError())), "f") is None
    assert _small_value(np.arange(20)) == "array(shape=(20,), dtype=int64)"
    assert _to_numpy(TorchLikeArray()).tolist() == [1.0]
    adapter = SimpleNamespace(
        num_trajectories=lambda: 2,
        num_timesteps=lambda _trajectory_id: 3,
        trajectory_ids=lambda: ("a", "b"),
    )
    assert _flat_index(adapter, "b", 1, 2) == 5
    with pytest.raises(AttributeError, match="no supported sample access"):
        _load_upstream_sample(SimpleNamespace(), "a", 0, 0, SimpleNamespace())


def test_cyclone_backend_helper_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class NeedsBackend:
        def __init__(self, backend, path: str):
            self.backend = backend
            self.path = path

    NeedsBackend.__module__ = "missing.dataset.cyclone"
    data = _cyclone_data(tmp_path)
    assert data.cyclone is not None

    def missing_backend_module(name: str) -> types.ModuleType:
        if name.endswith(".dataset.backend"):
            raise ImportError(name)
        return REAL_IMPORT_MODULE(name)

    monkeypatch.setattr("gk_surrogate.data.cyclone_kvikio.importlib.import_module", missing_backend_module)
    with pytest.raises(MissingCycloneDependencyError, match="requires a backend"):
        _backend_for_dataset(NeedsBackend, data.cyclone)

    assert _instantiate_backend(dict, data.cyclone) == {"use_kvikio": False}


def _install_fake_neugk(
    monkeypatch: pytest.MonkeyPatch, dataset_cls: type[Any], *, package_name: str = "neugk_jax"
) -> None:
    package = types.ModuleType(package_name)
    package.__path__ = []
    dataset_package = types.ModuleType(f"{package_name}.dataset")
    dataset_package.__path__ = []
    cyclone_module = types.ModuleType(f"{package_name}.dataset.cyclone")
    backend_module = types.ModuleType(f"{package_name}.dataset.backend")
    cyclone_module.CycloneDataset = dataset_cls
    cyclone_module.CycloneDataset.__module__ = f"{package_name}.dataset.cyclone"
    backend_module.KvikIOBackend = FakeKvikIOBackend
    dataset_package.CycloneDataset = dataset_cls
    dataset_package.cyclone = cyclone_module
    dataset_package.backend = backend_module
    package.dataset = dataset_package
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.dataset", dataset_package)
    monkeypatch.setitem(sys.modules, f"{package_name}.dataset.cyclone", cyclone_module)
    monkeypatch.setitem(sys.modules, f"{package_name}.dataset.backend", backend_module)


def _cyclone_data(tmp_path: Path, *, target_spectra: tuple[str, ...] = ()) -> DataConfig:
    return DataConfig.model_validate(
        {
            "backend": "cyclone_kvikio",
            "root": str(tmp_path),
            "split": "train",
            "input_fields": ["df"],
            "target_flux": True,
            "target_spectra": list(target_spectra),
            "batch_size": 1,
            "shuffle": False,
            "seed": 42,
            "cyclone": {
                "trajectories": None,
                "fields_to_load": ["df"],
                "conditions": ["itg", "dg", "s_hat", "q"],
                "prefer_dtype": "float32",
                "use_kvikio": False,
                "return_jax": False,
            },
        }
    )
