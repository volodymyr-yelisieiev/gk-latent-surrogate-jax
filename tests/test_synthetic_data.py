from __future__ import annotations

import numpy as np
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.data.factory import build_dataset
from gk_surrogate.data.split import split_trajectory_ids
from gk_surrogate.data.synthetic import SyntheticTrajectoryDataset


def test_synthetic_dataset_shapes_determinism_and_finite_targets(tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.synthetic is not None
    ds1 = SyntheticTrajectoryDataset(config.data.synthetic, seed=7)
    ds2 = SyntheticTrajectoryDataset(config.data.synthetic, seed=7)
    ds3 = SyntheticTrajectoryDataset(config.data.synthetic, seed=8)
    sample1 = ds1.get_snapshot(ds1.trajectory_ids()[0], 0)
    sample2 = ds2.get_snapshot(ds2.trajectory_ids()[0], 0)
    sample3 = ds3.get_snapshot(ds3.trajectory_ids()[0], 0)
    assert sample1.x.shape == (2, 4, 4, 4, 4, 4)
    assert np.allclose(sample1.x, sample2.x)
    assert not np.allclose(sample1.x, sample3.x)
    assert np.isfinite(sample1.targets.flux).all()
    assert np.std([ds1.get_snapshot(ds1.trajectory_ids()[0], t).targets.flux[0] for t in range(8)]) > 0


def test_build_dataset_and_split_no_overlap(tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    dataset = build_dataset(config.data)
    split = split_trajectory_ids(dataset.trajectory_ids(), seed=42)
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    assert split["val"]
    assert split["test"]


def test_split_rejects_invalid_ratios_and_preserves_small_dataset():
    with pytest.raises(ValueError, match="finite and non-negative"):
        split_trajectory_ids(["a", "b"], ratios=(0.8, -0.1, 0.3))
    with pytest.raises(ValueError, match="train, validation, and test"):
        split_trajectory_ids(["a", "b"], ratios=(0.8, 0.2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be unique"):
        split_trajectory_ids(["a", "a"])

    split = split_trajectory_ids(["a", "b"], ratios=(0.8, 0.0, 0.2))
    assert split["train"]
    assert split["test"]
    assert set().union(*map(set, split.values())) == {"a", "b"}
