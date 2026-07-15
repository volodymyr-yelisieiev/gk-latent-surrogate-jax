from __future__ import annotations

import jax.numpy as jnp
import pytest

from gk_surrogate.losses.diagnostics import (
    diagnostic_prediction_loss,
    flux_huber,
    flux_mae,
    flux_mse,
    flux_relative_l2,
    spectra_mae,
    spectra_mse,
    spectra_relative_l2,
)
from gk_surrogate.losses.latent import latent_prediction_loss
from gk_surrogate.losses.total import encoder_total_loss, sequence_total_loss
from gk_surrogate.models.diagnostics import DiagnosticPredictions


def test_diagnostic_loss_functions_and_errors():
    pred = jnp.asarray([[1.0, 2.0]])
    target = jnp.asarray([[1.5, 1.5]])
    assert flux_mse(pred, target) >= 0
    assert flux_mae(pred, target) >= 0
    assert flux_huber(pred, target) >= 0
    assert flux_relative_l2(pred, target) >= 0
    spectra_pred = {"ky": pred}
    spectra_target = {"ky": target}
    assert spectra_mse(spectra_pred, spectra_target, log_space=True) >= 0
    assert spectra_mae(spectra_pred, spectra_target) >= 0
    assert spectra_relative_l2(spectra_pred, spectra_target) >= 0
    with pytest.raises(KeyError):
        spectra_mse({"q": pred}, spectra_target)
    with pytest.raises(KeyError, match="unexpected targets"):
        spectra_mse(spectra_pred, {"ky": target, "q": target})
    with pytest.raises(ValueError, match="shapes must match"):
        flux_mse(jnp.ones((2, 1)), jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="positive"):
        flux_huber(pred, target, delta=0.0)


def test_total_loss_composition_and_latent_modes():
    diagnostics = DiagnosticPredictions(flux=jnp.ones((1, 1)), spectra={"ky": jnp.ones((1, 2))})
    total, metrics = encoder_total_loss(
        diagnostics=diagnostics,
        flux_target=jnp.zeros((1, 1)),
        spectra_target={"ky": jnp.zeros((1, 2))},
        flux_weight=1.0,
        spectra_weight=1.0,
    )
    assert total > 0
    assert "loss/diagnostics" in metrics
    seq_total, seq_metrics = sequence_total_loss(jnp.ones((1, 2)), jnp.zeros((1, 2)), latent_loss_mode="mse")
    assert seq_total == seq_metrics["loss/total"]
    for mode in ("mse", "huber", "cosine", "mse_plus_cosine"):
        assert latent_prediction_loss(jnp.ones((1, 2)), jnp.zeros((1, 2)), mode=mode) >= 0
    with pytest.raises(ValueError):
        diagnostic_prediction_loss(diagnostics, flux_target=None, flux_weight=1.0)
