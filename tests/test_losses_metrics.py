import tempfile

import jax.numpy as jnp
import pytest

from gk_surrogate.losses.diagnostics import spectra_mse
from gk_surrogate.losses.latent import latent_prediction_loss
from gk_surrogate.metrics.aggregate import aggregate_metrics, save_metrics_csv, save_metrics_json
from gk_surrogate.metrics.diagnostics import flux_rmse, spectra_pearson_corr, spectra_relative_l2
from gk_surrogate.metrics.latent import latent_mse, latent_relative_l2, rollout_mse_by_step
from gk_surrogate.metrics.rollout import (
    horizon_until_threshold,
    rollout_stability,
    summarize_rollout,
)


def test_losses_and_metrics_are_finite():
    pred = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.asarray([[1.5, 2.0], [2.5, 5.0]])

    assert jnp.isfinite(latent_prediction_loss(pred, target, mode="mse_plus_cosine"))
    assert jnp.isfinite(spectra_mse({"ky": pred}, {"ky": target}, log_space=True, eps=1e-6))
    assert jnp.isfinite(latent_mse(pred, target))
    assert jnp.isfinite(flux_rmse(pred, target))
    assert jnp.isfinite(spectra_pearson_corr({"ky": pred}, {"ky": target})["ky"])


def test_rollout_metrics_and_aggregation_outputs():
    pred = jnp.ones((2, 3, 4), dtype=jnp.float32)
    target = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    mse_by_step = rollout_mse_by_step(pred, target)

    assert mse_by_step.shape == (3,)
    assert rollout_stability(pred)
    assert horizon_until_threshold(mse_by_step, 0.5) == 1
    assert summarize_rollout(pred, target, error_threshold=0.5)["latent/mse_by_step"].shape == (3,)

    aggregated = aggregate_metrics([{"loss": jnp.asarray(1.0)}, {"loss": jnp.asarray(3.0)}])
    assert aggregated["loss"] == 2.0
    vector_aggregated = aggregate_metrics(
        [{"error": jnp.asarray([1.0, 3.0])}, {"error": jnp.asarray([3.0, 5.0])}]
    )
    assert jnp.allclose(vector_aggregated["error"], jnp.asarray([2.0, 4.0]))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_metrics_json({"loss": jnp.asarray(1.0)}, f"{tmpdir}/metrics.json")
        save_metrics_csv({"loss": jnp.asarray(1.0)}, f"{tmpdir}/metrics.csv")


def test_public_rollout_helper_fallbacks_and_errors():
    context = jnp.ones((2, 3, 4), dtype=jnp.float32)

    def apply_without_train(variables, value):
        del variables
        return value[:, -1, :]

    def apply_raw_params(params, value, *, train):
        if isinstance(params, dict) and "params" in params:
            raise TypeError("raw params only")
        del train
        return value[:, -1, :]

    assert summarize_rollout(context, context)["latent/mse_by_step"].shape == (3,)
    assert rollout_stability(context)
    from gk_surrogate.metrics.rollout import autoregressive_rollout as metrics_rollout

    assert metrics_rollout(apply_without_train, {}, context, 1).shape == (2, 1, 4)
    assert metrics_rollout(apply_raw_params, {"w": 1}, context, 1).shape == (2, 1, 4)
    with pytest.raises(ValueError, match="z_initial_context"):
        metrics_rollout(apply_without_train, {}, jnp.ones((2, 4)), 1)
    with pytest.raises(ValueError, match="non-negative"):
        metrics_rollout(apply_without_train, {}, context, -1)
    with pytest.raises(ValueError, match="one-step"):
        metrics_rollout(lambda params, value, train=False: value[:, 0, 0], {}, context, 1)


def test_metrics_reject_broadcasting_and_preserve_global_relative_l2():
    pred = jnp.asarray([[2.0, 0.0], [0.0, 2.0]])
    target = jnp.asarray([[1.0, 0.0], [0.0, 10.0]])
    expected_relative_l2 = jnp.sqrt(65.0) / (jnp.sqrt(101.0) + 1e-8)

    assert jnp.allclose(latent_relative_l2(pred, target), expected_relative_l2)
    assert jnp.allclose(spectra_relative_l2({"ky": pred}, {"ky": target})["ky"], expected_relative_l2)
    with pytest.raises(ValueError, match="shapes must match"):
        latent_mse(jnp.ones((2, 1)), jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="shapes must match"):
        rollout_mse_by_step(jnp.ones((2, 3, 1)), jnp.ones((2, 3, 2)))
    with pytest.raises(KeyError, match="unexpected targets"):
        spectra_pearson_corr({"ky": pred}, {"ky": target, "q": target})
    with pytest.raises(ValueError, match="inconsistent shapes"):
        aggregate_metrics([{"error": jnp.ones((2,))}, {"error": jnp.ones((3,))}])
    with pytest.raises(ValueError, match="non-empty"):
        horizon_until_threshold(jnp.asarray([]), 1.0)
