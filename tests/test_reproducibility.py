from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gk_surrogate.config.load import load_config
from gk_surrogate.factory import build_encoder_with_diagnostics
from gk_surrogate.pipeline import _snapshot_batches


def test_same_seed_same_first_batch_and_initial_outputs(tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    batch1 = next(_snapshot_batches(config, repeat=False))
    batch2 = next(_snapshot_batches(config, repeat=False))
    assert np.allclose(np.asarray(batch1["x"]), np.asarray(batch2["x"]))

    model = build_encoder_with_diagnostics(config.model)
    variables1 = model.init(jax.random.PRNGKey(0), batch1["x"], train=False)
    variables2 = model.init(jax.random.PRNGKey(0), batch1["x"], train=False)
    out1 = model.apply(variables1, batch1["x"], train=False).z
    out2 = model.apply(variables2, batch1["x"], train=False).z
    assert jnp.allclose(out1, out2)


def test_different_seed_changes_data(tiny_config_path):
    config = load_config(tiny_config_path, overrides=["data.seed=99"], command="train-encoder")
    other = load_config(tiny_config_path, overrides=["data.seed=100"], command="train-encoder")
    assert not np.allclose(
        np.asarray(next(_snapshot_batches(config, repeat=False))["x"]),
        np.asarray(next(_snapshot_batches(other, repeat=False))["x"]),
    )
