from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gk_surrogate.config.schema import CycloneKvikIOConfig
from gk_surrogate.data.universe_manifest import (
    _input_paths,
    _sample_indices,
    _trajectory_id,
    build_cyclone_universe_manifest,
    verify_cyclone_universe_manifest,
)


def _write_trajectory(root: Path, name: str, value: float) -> None:
    trajectory = root / f"{name}_ifft_realpotens"
    data = trajectory / "data"
    data.mkdir(parents=True)
    metadata = {
        "timesteps": np.arange(6, dtype=np.float32),
        "resolution": np.asarray([1, 1, 1, 1, 1]),
    }
    with (trajectory / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    with (trajectory / "metadata_light.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    for index in range(6):
        np.full((2,), value + index, dtype=np.float32).tofile(data / f"timestep_{index:05d}.bin")


def test_universe_manifest_hashes_consumed_bytes_and_is_portable(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, "iteration_0", 1.0)
    _write_trajectory(tmp_path, "iteration_1", 2.0)
    config = CycloneKvikIOConfig(offset=1, subsample=2, bundle_seq_length=1)

    manifest = build_cyclone_universe_manifest(tmp_path, config, workers=2)
    assert manifest["trajectory_ids"] == ["iteration_0", "iteration_1"]
    assert str(tmp_path) not in json.dumps(manifest)
    assert all(item["input_file_count"] == 2 for item in manifest["trajectories"])
    assert verify_cyclone_universe_manifest(manifest, tmp_path, config, workers=2) == manifest

    changed = tmp_path / "iteration_0_ifft_realpotens" / "data" / "timestep_00001.bin"
    np.asarray([99.0, 99.0], dtype=np.float32).tofile(changed)
    with pytest.raises(ValueError, match="dataset bytes"):
        verify_cyclone_universe_manifest(manifest, tmp_path, config, workers=2)


def test_universe_manifest_handles_edge_contracts_and_missing_inputs(tmp_path: Path) -> None:
    assert _trajectory_id(Path("plain")) == "plain"
    config = CycloneKvikIOConfig(offset=1, tail_offset=1, subsample=2, bundle_seq_length=2)
    assert _sample_indices({}, config) == ()
    assert _sample_indices({"timesteps": np.asarray([], dtype=np.float32)}, config) == ()

    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(FileNotFoundError, match="poten_00001.bin"):
        _input_paths(tmp_path, (1,), ("df", "phi"))
    (data / "timestep_00001.bin").write_bytes(b"df")
    (data / "poten_00001.bin").write_bytes(b"phi")
    assert len(_input_paths(tmp_path, (1,), ("df", "phi"))) == 2


def test_universe_manifest_rejects_empty_and_non_mapping_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no Cyclone trajectories"):
        build_cyclone_universe_manifest(tmp_path, CycloneKvikIOConfig())

    trajectory = tmp_path / "bad_ifft"
    (trajectory / "data").mkdir(parents=True)
    with (trajectory / "metadata.pkl").open("wb") as handle:
        pickle.dump(["not", "a", "mapping"], handle)
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        build_cyclone_universe_manifest(tmp_path, CycloneKvikIOConfig())
