from __future__ import annotations

import numpy as np
import pytest

from gk_surrogate.data.collate import collate_snapshots
from gk_surrogate.data.types import DiagnosticTargets, SnapshotSample


def _sample(*, flux: bool = True, spectra: tuple[str, ...] = ("ky",)) -> SnapshotSample:
    return SnapshotSample(
        x=np.ones((1, 2, 2, 2, 2, 2), dtype=np.float32),
        targets=DiagnosticTargets(
            flux=np.ones((1,), dtype=np.float32) if flux else None,
            spectra={name: np.ones((2,), dtype=np.float32) for name in spectra},
        ),
        trajectory_id="trajectory",
        trajectory_index=0,
        timestep_index=0,
        physical_time=0.0,
        metadata={},
    )


def test_collate_rejects_empty_and_inconsistent_targets():
    with pytest.raises(ValueError, match="empty snapshot batch"):
        collate_snapshots([])
    with pytest.raises(ValueError, match="with and without flux"):
        collate_snapshots([_sample(flux=True), _sample(flux=False)])
    with pytest.raises(ValueError, match="inconsistent spectra"):
        collate_snapshots([_sample(spectra=("ky",)), _sample(spectra=("q",))])


def test_collate_preserves_complete_target_contract():
    batch = collate_snapshots([_sample(), _sample()])
    assert batch.x.shape == (2, 1, 2, 2, 2, 2, 2)
    assert batch.flux is not None and batch.flux.shape == (2, 1)
    assert batch.spectra["ky"].shape == (2, 2)
