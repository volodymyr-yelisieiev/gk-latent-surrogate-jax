"""Dataset factory for validated data configs."""

from __future__ import annotations

from gk_surrogate.config.schema import DataConfig
from gk_surrogate.data.base import TrajectoryDataset
from gk_surrogate.data.cyclone_kvikio import (
    CycloneKvikIODatasetAdapter,
    DirectCycloneKvikIODataset,
    direct_cyclone_layout_available,
)
from gk_surrogate.data.h5_loader import H5TrajectoryDataset
from gk_surrogate.data.synthetic import SyntheticTrajectoryDataset


def build_dataset(config: DataConfig) -> TrajectoryDataset:
    if config.backend == "synthetic":
        if config.synthetic is None:
            msg = "synthetic backend requires config.synthetic"
            raise ValueError(msg)
        return SyntheticTrajectoryDataset(
            config.synthetic,
            seed=config.seed,
            target_spectra=config.target_spectra,
            target_flux=config.target_flux,
        )
    if config.backend == "h5":
        if config.h5_schema is None or config.root is None:
            msg = "h5 backend requires root and h5_schema"
            raise ValueError(msg)
        return H5TrajectoryDataset(
            config.root,
            config.h5_schema,
            target_spectra=config.target_spectra,
            target_flux=config.target_flux,
            input_fields=config.input_fields,
        )
    if config.cyclone is None or config.root is None:
        msg = "cyclone_kvikio backend requires root and cyclone config"
        raise ValueError(msg)
    if direct_cyclone_layout_available(config.root, config.cyclone):
        return DirectCycloneKvikIODataset(
            config.root,
            config.cyclone,
            target_spectra=config.target_spectra,
            target_flux=config.target_flux,
            input_fields=config.input_fields,
            split=config.split,
            seed=config.seed,
        )
    return CycloneKvikIODatasetAdapter(
        config.root,
        config.cyclone,
        target_spectra=config.target_spectra,
        target_flux=config.target_flux,
        input_fields=config.input_fields,
        split=config.split,
        seed=config.seed,
    )
