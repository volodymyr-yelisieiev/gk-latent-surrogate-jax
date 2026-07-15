from __future__ import annotations

import h5py
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import DataConfig, H5SchemaConfig
from gk_surrogate.data.factory import build_dataset
from gk_surrogate.data.h5_loader import write_synthetic_h5
from gk_surrogate.data.inspect import inspect_dataset


def test_inspect_synthetic_returns_shapes_and_memory(tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    inspection = inspect_dataset(config.data)
    assert inspection.snapshot_shape == (2, 4, 4, 4, 4, 4)
    assert inspection.flux_stats is not None
    assert inspection.flux_stats["count"] > 0
    assert "flux_mean" in inspection.as_dict()
    assert "spectra_ky_mean" in inspection.as_dict()
    assert inspection.bytes_per_batch > 0
    assert inspection.bytes_per_trajectory > inspection.bytes_per_snapshot
    assert inspection.recommended_batch_size_512mb >= 1
    assert inspection.warnings == ()


def test_inspect_h5_returns_tree(tmp_path, tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.synthetic is not None
    write_synthetic_h5(tmp_path, config.data.synthetic)
    data = config.data.model_copy(
        update={
            "backend": "h5",
            "root": str(tmp_path),
            "h5_schema": H5SchemaConfig(
                flux_key="fluxes",
                timestep_key="timesteps",
                spectra_keys={"ky": "metadata/ky_spectrum", "q": "metadata/q_spectrum"},
            ),
            "synthetic": None,
        }
    )
    inspection = inspect_dataset(data)
    assert inspection.backend == "h5"
    assert inspection.first_snapshot_key == "data/timestep_00000"
    assert "fluxes" in inspection.metadata_keys
    assert inspection.h5_tree
    assert inspection.spectra_stats["ky"]["count"] > 0


def test_inspection_warnings_and_build_dataset_guards(tmp_path, tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.synthetic is not None
    data = config.data.model_copy(update={"target_spectra": ("missing",)})
    inspection = inspect_dataset(data, max_depth=0)
    assert "missing spectra targets: missing" in inspection.warnings

    with pytest.raises(ValueError, match="synthetic backend"):
        build_dataset(DataConfig.model_construct(backend="synthetic", synthetic=None))
    with pytest.raises(ValueError, match="h5 backend"):
        build_dataset(DataConfig.model_construct(backend="h5", root=None, h5_schema=None))


def test_inspection_target_statistics_warn_on_real_data_edges(tmp_path, tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.synthetic is not None
    written = write_synthetic_h5(tmp_path, config.data.synthetic)
    with h5py.File(written[0], "a") as handle:
        fluxes = handle["metadata/fluxes"][:]
        del handle["metadata/fluxes"]
        handle["metadata"].create_dataset("fluxes", data=fluxes[:-1] * 0.0 + 1.0)
        spectra = handle["metadata/ky_spectrum"][:]
        del handle["metadata/ky_spectrum"]
        handle["metadata"].create_dataset("ky_spectrum", data=-spectra)

    data = config.data.model_copy(
        update={
            "backend": "h5",
            "root": str(tmp_path),
            "h5_schema": H5SchemaConfig(
                flux_key="fluxes",
                timestep_key="timesteps",
                spectra_keys={"ky": "metadata/ky_spectrum", "q": "metadata/q_spectrum"},
            ),
            "synthetic": None,
        }
    )
    inspection = inspect_dataset(data, max_trajectories=1, max_target_samples=16, log_spectra=True)
    warnings = "\n".join(inspection.warnings)
    assert "flux target length mismatch" in warnings
    assert "flux targets are constant" in warnings
    assert "spectra ky targets contain negative values" in warnings


def test_inspection_rejects_nonpositive_target_sample_limit(tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    with pytest.raises(ValueError, match="max_target_samples"):
        inspect_dataset(config.data, max_target_samples=0)
