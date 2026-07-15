"""Batch collation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from gk_surrogate.data.types import SnapshotBatch, SnapshotSample


def collate_snapshots(samples: Sequence[SnapshotSample]) -> SnapshotBatch:
    if not samples:
        msg = "cannot collate an empty snapshot batch"
        raise ValueError(msg)

    x = jnp.asarray([sample.x for sample in samples], dtype=jnp.float32)
    flux_availability = {sample.targets.flux is not None for sample in samples}
    if len(flux_availability) > 1:
        msg = "snapshot batch mixes samples with and without flux targets"
        raise ValueError(msg)
    flux = None
    if flux_availability == {True}:
        flux = jnp.asarray([sample.targets.flux for sample in samples], dtype=jnp.float32)

    spectra_key_sets = [set(sample.targets.spectra) for sample in samples]
    if any(keys != spectra_key_sets[0] for keys in spectra_key_sets[1:]):
        msg = "snapshot batch contains inconsistent spectra target keys"
        raise ValueError(msg)
    spectra_keys = spectra_key_sets[0]
    spectra = {
        key: jnp.asarray([sample.targets.spectra[key] for sample in samples], dtype=jnp.float32)
        for key in sorted(spectra_keys)
    }
    return SnapshotBatch(
        x=x,
        flux=flux,
        spectra=spectra,
        trajectory_index=jnp.asarray([sample.trajectory_index for sample in samples], dtype=jnp.int32),
        timestep_index=jnp.asarray([sample.timestep_index for sample in samples], dtype=jnp.int32),
    )
