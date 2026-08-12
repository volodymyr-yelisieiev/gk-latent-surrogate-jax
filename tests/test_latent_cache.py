from __future__ import annotations

import h5py
import numpy as np
import pytest

from gk_surrogate.data.latent_cache import LatentCacheDataset, LatentCacheWriter
from gk_surrogate.data.latent_cache_report import validate_latent_cache, write_latent_cache_report
from gk_surrogate.data.sequence_dataset import valid_sequence_starts


def test_latent_cache_roundtrip_metadata_and_windows(tmp_path):
    path = tmp_path / "latent_cache.h5"
    writer = LatentCacheWriter(path, latent_dim=3, config_yaml="name: test")
    writer.write_trajectory(
        "traj",
        np.ones((6, 3), dtype=np.float32),
        physical_time=np.linspace(0.0, 1.0, 6, dtype=np.float32),
        flux=np.ones((6, 1), dtype=np.float32),
        spectra={"ky": np.ones((6, 4), dtype=np.float32)},
    )
    with h5py.File(path, "r") as handle:
        assert "created_at" in handle["metadata"].attrs
    dataset = LatentCacheDataset(path)
    assert dataset.trajectory_ids() == ("traj",)
    assert dataset.get_trajectory_latents("traj").shape == (6, 3)
    assert dataset.get_trajectory_flux("traj").shape == (6, 1)
    sample = dataset.get_latent("traj", 0)
    assert sample.z.shape == (3,)
    assert sample.physical_time == 0.0
    context, target, diagnostics = dataset.get_sequence_window(
        "traj",
        0,
        context_length=4,
        prediction_length=1,
    )
    assert context.shape == (4, 3)
    assert target.shape == (1, 3)
    assert diagnostics.flux.shape == (1, 1)
    assert valid_sequence_starts(dataset, "traj", context_length=4, prediction_length=1) == (0, 1)
    with pytest.raises(IndexError, match="exceeds"):
        dataset.get_sequence_window("traj", 3, context_length=4, prediction_length=1)
    with pytest.raises(ValueError, match="start"):
        dataset.get_sequence_window("traj", -1, context_length=4, prediction_length=1)
    with pytest.raises(ValueError, match="must be positive"):
        dataset.get_sequence_window("traj", 0, context_length=0, prediction_length=1)
    with pytest.raises(ValueError, match="must be positive"):
        valid_sequence_starts(dataset, "traj", context_length=4, prediction_length=0)


def test_latent_cache_roundtrips_path_like_trajectory_ids(tmp_path):
    path = tmp_path / "latent_cache_paths.h5"
    trajectory_id = "nested/path/trajectory_0"
    writer = LatentCacheWriter(path, latent_dim=2)
    writer.write_trajectory(trajectory_id, np.ones((5, 2), dtype=np.float32))
    with h5py.File(path, "r") as handle:
        assert "nested" not in handle["trajectories"]
        assert len(handle["trajectories"]) == 1

    dataset = LatentCacheDataset(path)
    assert dataset.trajectory_ids() == (trajectory_id,)
    assert dataset.num_timesteps(trajectory_id) == 5
    context, target, _diagnostics = dataset.get_sequence_window(
        trajectory_id,
        0,
        context_length=3,
        prediction_length=1,
    )
    assert context.shape == (3, 2)
    assert target.shape == (1, 2)


def test_latent_cache_requires_explicit_trajectory_ids(tmp_path):
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata").attrs["latent_dim"] = 2
        root = handle.create_group("trajectories")
        direct = root.create_group("missing_attribute")
        direct.create_dataset("z", data=np.ones((2, 2), dtype=np.float32))
        direct.create_dataset("timestep_index", data=np.arange(2, dtype=np.int32))

    with pytest.raises(ValueError, match="missing trajectory_id"):
        LatentCacheDataset(path)


def test_latent_cache_rejects_inconsistent_dim(tmp_path):
    writer = LatentCacheWriter(tmp_path / "cache.h5", latent_dim=3)
    with pytest.raises(ValueError, match="trajectory_id"):
        writer.write_trajectory("", np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected z shape"):
        writer.write_trajectory("bad", np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="flux length"):
        writer.write_trajectory(
            "bad_len",
            np.ones((2, 3), dtype=np.float32),
            flux=np.ones((3, 1), dtype=np.float32),
        )


def test_latent_cache_validation_report(tmp_path):
    path = tmp_path / "report_cache.h5"
    writer = LatentCacheWriter(path, latent_dim=2, config_yaml="name: report", encoder_checkpoint_path="ckpt")
    writer.write_trajectory(
        "traj-a",
        np.ones((6, 2), dtype=np.float32),
        flux=np.ones((6, 1), dtype=np.float32),
        spectra={"ky": np.ones((6, 3), dtype=np.float32)},
    )
    writer.write_trajectory(
        "traj-b",
        np.full((5, 2), 2.0, dtype=np.float32),
        flux=np.ones((5, 1), dtype=np.float32),
        spectra={"ky": np.ones((5, 3), dtype=np.float32)},
    )
    report = validate_latent_cache(path, context_length=4, prediction_length=1, split_seed=0)
    assert report["latent_dim"] == 2
    assert report["num_trajectories"] == 2
    assert report["finite_latents"]
    assert report["flux_available"]
    assert report["spectra_keys"] == ["ky"]
    assert report["sequence_windows_total"] == 3
    out = write_latent_cache_report(report, tmp_path / "report.json")
    assert out.exists()
