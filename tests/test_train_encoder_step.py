from __future__ import annotations

import jax.numpy as jnp

from gk_surrogate.config.load import load_config
from gk_surrogate.losses.diagnostics import spectra_mse
from gk_surrogate.losses.simsiam import simsiam_loss
from gk_surrogate.pipeline import _init_encoder_state, _snapshot_batches
from gk_surrogate.training.train_encoder import _simsiam_loss, _spectra_loss, eval_encoder_step, train_encoder_step


def test_train_encoder_step_changes_params_and_eval_is_finite(tiny_config_path, params_changed):
    config = load_config(tiny_config_path, command="train-encoder")
    state, _ = _init_encoder_state(config)
    batch = next(_snapshot_batches(config, repeat=False))
    before = state.params
    new_state, metrics = train_encoder_step(state, batch)
    assert params_changed(before, new_state.params)
    assert float(metrics["loss"]) == float(metrics["loss"])
    eval_metrics = eval_encoder_step(new_state, batch)
    assert float(eval_metrics["loss"]) == float(eval_metrics["loss"])


def test_simsiam_train_step_runs(repo_root):
    config = load_config(repo_root / "configs/experiment/smoke_encoder_simsiam.yaml", command="train-encoder")
    state, _ = _init_encoder_state(config, simsiam=True)
    batch = next(_snapshot_batches(config, repeat=False))
    _, metrics = train_encoder_step(state, batch)
    assert "simsiam_loss" in metrics


def test_encoder_training_loss_helpers_match_shared_losses():
    z = jnp.ones((2, 4), dtype=jnp.float32)
    spectra_pred = {"ky": z + 1.0}
    spectra_target = {"ky": z}
    assert jnp.allclose(
        _spectra_loss(spectra_pred, spectra_target, log_space=False, eps=1e-6),
        spectra_mse(spectra_pred, spectra_target, log_space=False, eps=1e-6),
    )
    out1 = {"prediction": z + 0.5, "projection": z + 1.0}
    out2 = {"prediction": z + 0.25, "projection": z + 2.0}
    assert jnp.allclose(
        _simsiam_loss(out1, out2),
        simsiam_loss(out1["prediction"], out2["projection"], out2["prediction"], out1["projection"]),
    )
