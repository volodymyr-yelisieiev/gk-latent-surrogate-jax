from __future__ import annotations

import h5py
import numpy as np
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import H5SchemaConfig
from gk_surrogate.data.h5_loader import H5TrajectoryDataset, write_synthetic_h5


def test_h5_fixture_loader_reads_snapshot_flux_spectra_and_channels(tmp_path, tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.synthetic is not None
    write_synthetic_h5(tmp_path, config.data.synthetic, seed=3)
    schema = H5SchemaConfig(
        flux_key="fluxes",
        timestep_key="timesteps",
        spectra_keys={"ky": "metadata/ky_spectrum", "q": "metadata/q_spectrum"},
        channel_indices=(0,),
    )
    dataset = H5TrajectoryDataset(tmp_path, schema, target_spectra=("ky", "q"), target_flux=True)
    assert dataset.num_trajectories() == 4
    sample = dataset.get_snapshot(dataset.trajectory_ids()[0], 0)
    assert sample.x.shape[0] == 1
    assert sample.targets.flux.shape == (1,)
    assert sample.targets.spectra["ky"].shape == (8,)
    assert sample.x.dtype.name == "float32"


def test_h5_missing_spectra_error_is_clear(tmp_path, tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.synthetic is not None
    write_synthetic_h5(tmp_path, config.data.synthetic, seed=3)
    schema = H5SchemaConfig(flux_key="fluxes", timestep_key="timesteps", spectra_keys={})
    dataset = H5TrajectoryDataset(tmp_path, schema, target_spectra=("ky",), target_flux=True)
    with pytest.raises(KeyError, match="diagnostic key"):
        dataset.get_snapshot(dataset.trajectory_ids()[0], 0)


def test_h5_time_series_flux_and_phi_field(tmp_path):
    path = tmp_path / "traj_000.h5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        metadata = handle.create_group("metadata")
        for t in range(3):
            data.create_dataset(f"timestep_{t:05d}", data=np.ones((1, 2, 2, 2, 2, 2), dtype=np.float64) * t)
            data.create_dataset(f"poten_{t:05d}", data=np.ones((1, 2, 2, 2, 2, 2), dtype=np.float64) * (t + 10))
        metadata.create_dataset("timesteps", data=np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
        metadata.create_dataset("fluxes", data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    schema = H5SchemaConfig(
        timestep_key_template="timestep_{t:05d}",
        phi_key_template="poten_{t:05d}",
        flux_key="fluxes",
        timestep_key="timesteps",
        dtype="float32",
    )
    dataset = H5TrajectoryDataset(tmp_path, schema, target_flux=True, input_fields=("df", "phi"))
    sample = dataset.get_snapshot("traj_000", 1)
    assert sample.x.shape[0] == 2
    assert sample.x.dtype.name == "float32"
    assert sample.targets.flux.shape == (1,)
    assert float(sample.targets.flux[0]) == 2.0
    assert sample.physical_time == 0.5
    with pytest.raises(KeyError, match="snapshot key"):
        dataset.get_snapshot("traj_000", 99)


def test_h5_rank_optional_and_unconfigured_field_edges(tmp_path):
    path = tmp_path / "traj_000.h5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.create_dataset("timestep_00000", data=np.ones((2, 2, 2, 2, 2), dtype=np.float64))
        data.create_dataset("bad_00000", data=np.ones((2, 2), dtype=np.float32))
    schema = H5SchemaConfig(timestep_key_template="timestep_{t:05d}", dtype="float32")
    dataset = H5TrajectoryDataset(tmp_path, schema, target_flux=False, input_fields=("df",))
    sample = dataset.get_snapshot("traj_000", 0)
    assert sample.x.shape == (1, 2, 2, 2, 2, 2)
    assert sample.targets.flux is None
    assert sample.physical_time is None
    with pytest.raises(KeyError, match="input field"):
        H5TrajectoryDataset(tmp_path, schema, target_flux=False, input_fields=("phi",)).get_snapshot("traj_000", 0)
    bad_schema = H5SchemaConfig(timestep_key_template="bad_{t:05d}")
    with pytest.raises(ValueError, match="rank 6"):
        H5TrajectoryDataset(tmp_path, bad_schema, target_flux=False).get_snapshot("traj_000", 0)
