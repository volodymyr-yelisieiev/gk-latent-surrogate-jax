from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.data.latent_cache import LatentCacheDataset, LatentCacheWriter
from gk_surrogate.models.sequence import GuppyLatentTransformer, MLPDeltaSequenceModel
from gk_surrogate.pipeline import (
    _denormalize_latents,
    _init_sequence_state,
    _latent_cache_for,
    _latent_normalization_stats,
    _normalize_latents,
    _sequence_batches,
)
from gk_surrogate.training.optimizer import build_optimizer
from gk_surrogate.training.state import TrainState
from gk_surrogate.training.train_sequence import eval_sequence_step, train_sequence_step


def test_train_sequence_step_changes_params_and_loss_finite(repo_root, tmp_path, params_changed):
    config = load_config(repo_root / "configs/experiment/smoke_sequence.yaml", command="train-sequence")
    cache_path = tmp_path / "latent_cache.h5"
    writer = LatentCacheWriter(cache_path, latent_dim=32)
    writer.write_trajectory("traj", np.ones((12, 32), dtype=np.float32))
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "seq"),
            "latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)}),
        }
    )
    cache_path = _latent_cache_for(config)
    state, _ = _init_sequence_state(config)
    batch = next(_sequence_batches(config, cache_path, repeat=False))
    before = state.params
    losses = []
    for _ in range(40):
        state, metrics = train_sequence_step(state, batch)
        losses.append(float(metrics["loss"]))
    assert params_changed(before, state.params)
    assert all(loss == loss for loss in losses)
    assert min(losses[-10:]) < losses[0] * 0.9
    assert float(eval_sequence_step(state, batch)["loss"]) == float(eval_sequence_step(state, batch)["loss"])


def test_sequence_batches_can_normalize_latents_from_cache(repo_root, tmp_path):
    config = load_config(repo_root / "configs/experiment/smoke_sequence.yaml", command="train-sequence")
    cache_path = tmp_path / "latent_cache.h5"
    z = np.repeat(np.arange(12, dtype=np.float32)[:, None], 32, axis=1)
    writer = LatentCacheWriter(cache_path, latent_dim=32)
    writer.write_trajectory("traj", z)
    config = config.model_copy(
        update={
            "latent_cache": config.latent_cache.model_copy(
                update={"path": str(cache_path), "latent_normalization": "cache"}
            ),
            "data": config.data.model_copy(update={"shuffle": False}),
        }
    )
    stats = _latent_normalization_stats(LatentCacheDataset(cache_path), config)
    assert stats is not None
    batch = next(_sequence_batches(config, cache_path, repeat=False))
    expected_first = (z[0] - z.mean(axis=0)) / z.std(axis=0)
    np.testing.assert_allclose(np.asarray(batch["z_context"])[0, 0], expected_first, rtol=1e-6)
    restored = _denormalize_latents(jnp.asarray(batch["z_context"]), stats)
    np.testing.assert_allclose(np.asarray(restored)[0, 0], z[0], rtol=1e-6)
    np.testing.assert_allclose(_normalize_latents(z[:1], None), z[:1])


def test_sequence_batches_reject_empty_window_set(repo_root, tmp_path):
    config = load_config(repo_root / "configs/experiment/smoke_sequence.yaml", command="train-sequence")
    cache_path = tmp_path / "latent_cache.h5"
    writer = LatentCacheWriter(cache_path, latent_dim=32)
    writer.write_trajectory("short", np.ones((2, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="no latent sequence windows"):
        next(_sequence_batches(config, cache_path, repeat=False))


def test_train_sequence_step_supports_multistep_targets_and_dropout():
    model = MLPDeltaSequenceModel(latent_dim=6, context_length=4, hidden_dims=(12,), dropout_rate=0.2)
    context = jnp.ones((3, 4, 6), dtype=jnp.float32)
    target = jnp.ones((3, 2, 6), dtype=jnp.float32) * 1.5
    rng = jax.random.PRNGKey(123)
    variables = model.init({"params": rng, "dropout": rng}, context, train=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=build_optimizer({"learning_rate": 1e-3}),
        rng=rng,
        model_config={"latent_loss": "mse", "latent_weight": 1.0},
    )
    new_state, metrics = train_sequence_step(state, {"z_context": context, "z_target": target})
    assert int(new_state.step) == 1
    assert not jnp.allclose(new_state.rng, state.rng)
    assert jnp.isfinite(metrics["loss"])


def test_train_sequence_step_supports_guppy_latent_transformer():
    model = GuppyLatentTransformer(
        latent_dim=6,
        context_length=4,
        model_dim=12,
        depth=1,
        num_heads=3,
        dropout_rate=0.0,
    )
    context = jnp.arange(3 * 4 * 6, dtype=jnp.float32).reshape(3, 4, 6) / 50.0
    target = context[:, -1:, :] + 0.1
    rng = jax.random.PRNGKey(456)
    variables = model.init(rng, context, train=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=build_optimizer({"learning_rate": 1e-3}),
        rng=rng,
        model_config={"latent_loss": "mse", "latent_weight": 1.0},
    )

    new_state, metrics = train_sequence_step(state, {"z_context": context, "z_target": target})

    assert int(new_state.step) == 1
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["latent_mse"])
