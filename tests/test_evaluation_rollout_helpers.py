from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gk_surrogate.evaluation.reports import (
    _jsonable,
    save_basic_rollout_plot,
    save_metrics_by_step,
    save_metrics_json,
    save_rollout_plots,
)
from gk_surrogate.evaluation.rollout import (
    autoregressive_rollout,
    evaluate_rollout_batches,
    horizon_until_threshold,
    latent_rollout_metrics,
    observed_diagnostic_persistence,
    persistence_rollout,
    trajectory_balanced_rollout_metrics,
)


def test_evaluation_rollout_helpers_and_reports(tmp_path):
    context = jnp.zeros((2, 3, 4), dtype=jnp.float32)

    def apply_fn(variables, value, *, train=False):
        del variables, train
        return value[:, -1, :] + 1.0

    rollout = autoregressive_rollout(apply_fn, {}, context, 2)
    assert rollout.shape == (2, 2, 4)
    persisted = persistence_rollout(context, 2)
    observed = observed_diagnostic_persistence(jnp.asarray([[2.0, 3.0], [4.0, 5.0]]), 2)
    np.testing.assert_allclose(
        np.asarray(observed),
        [[[2.0, 3.0], [2.0, 3.0]], [[4.0, 5.0], [4.0, 5.0]]],
    )
    with pytest.raises(ValueError, match="positive"):
        observed_diagnostic_persistence(jnp.ones((1, 1)), 0)
    with pytest.raises(ValueError, match="shape"):
        observed_diagnostic_persistence(jnp.ones((1, 2, 1)), 2)
    metrics = latent_rollout_metrics(persisted, jnp.ones((2, 2, 4)))
    assert metrics["stable"]
    assert horizon_until_threshold(metrics["mse_by_step"], 0.5) == 1
    assert horizon_until_threshold(jnp.asarray([0.1, jnp.nan, 0.2]), 0.5) == 2
    with pytest.raises(ValueError, match="non-empty"):
        horizon_until_threshold(jnp.asarray([]), 0.5)
    with pytest.raises(ValueError, match="finite and non-negative"):
        horizon_until_threshold(jnp.asarray([0.1]), float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        horizon_until_threshold(jnp.asarray([0.1]), -1.0)
    paths = {
        "json": save_metrics_json(metrics, tmp_path),
        "csv": save_metrics_by_step(metrics, tmp_path),
    }
    assert paths["json"].exists()
    assert paths["csv"].exists()
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["step"] == "1"
    batch_metrics = evaluate_rollout_batches(
        apply_fn,
        {},
        [{"z_context": context, "z_target": jnp.ones((2, 2, 4))}],
        rollout_steps=2,
        use_persistence=True,
    )
    assert np.asarray(batch_metrics["mse_by_step"]).shape == (2,)
    with pytest.raises(ValueError):
        latent_rollout_metrics(jnp.ones((1, 1, 1)), jnp.ones((1, 2, 1)))
    jitted = jax.jit(lambda x: autoregressive_rollout(apply_fn, {}, x, 1))
    assert jitted(context).shape == (2, 1, 4)


def test_trajectory_balanced_metrics_do_not_overweight_more_windows():
    target = jnp.zeros((3, 2, 1), dtype=jnp.float32)
    pred = jnp.asarray([[[1.0], [1.0]], [[1.0], [1.0]], [[3.0], [3.0]]], dtype=jnp.float32)

    pooled = latent_rollout_metrics(pred, target)
    balanced = trajectory_balanced_rollout_metrics(pred, target, ["long", "long", "short"])

    assert float(pooled["mse"]) == pytest.approx(11.0 / 3.0)
    assert float(balanced["mse"]) == pytest.approx(5.0)
    assert float(balanced["mse_std_by_step"][0]) == pytest.approx(4.0)
    assert int(balanced["num_trajectories"]) == 2


def test_relative_l2_uses_global_norm_per_trajectory_and_horizon():
    target = jnp.asarray([[[1.0]], [[100.0]], [[10.0]]], dtype=jnp.float32)
    error = jnp.asarray([[[1.0]], [[10.0]], [[20.0]]], dtype=jnp.float32)
    pred = target + error

    pooled = latent_rollout_metrics(pred, target)
    balanced = trajectory_balanced_rollout_metrics(pred, target, ["long", "long", "short"])

    expected_pooled = np.sqrt(501.0) / np.sqrt(10101.0)
    expected_long = np.sqrt(101.0) / np.sqrt(10001.0)
    assert float(pooled["relative_l2"]) == pytest.approx(expected_pooled)
    assert float(balanced["relative_l2"]) == pytest.approx((expected_long + 2.0) / 2.0)
    assert float(balanced["relative_l2_std_by_step"][0]) == pytest.approx((2.0 - expected_long) / 2.0)


def test_rollout_cosine_is_stable_for_zero_and_small_vectors():
    small = jnp.asarray([[[1.0e-5, -2.0e-5, 3.0e-5]]], dtype=jnp.float32)
    small_metrics = latent_rollout_metrics(small, small)
    assert float(small_metrics["cosine"]) == pytest.approx(1.0, abs=1e-6)

    zeros = jnp.zeros_like(small)
    zero_metrics = latent_rollout_metrics(zeros, zeros)
    assert float(zero_metrics["cosine"]) == pytest.approx(0.0)
    assert np.isfinite(float(zero_metrics["cosine"]))


def test_report_serialization_and_baseline_plot_contract(tmp_path):
    assert _jsonable(Path("metrics.json")) == "metrics.json"
    assert _jsonable({"value": np.float32(1.5)}) == {"value": 1.5}
    assert _jsonable(np.asarray([1, 2])) == [1, 2]
    with pytest.raises(ValueError, match="no one-dimensional"):
        save_metrics_by_step({"scalar": np.asarray(1.0)}, tmp_path / "empty")

    plots = save_rollout_plots({"flux_mse_by_step": np.asarray([1.0, 0.5])}, tmp_path / "plots")
    assert set(plots) == {"flux_mse"}
    assert plots["flux_mse"].exists()
    assert save_basic_rollout_plot({"flux_mse_by_step": np.asarray([1.0])}, tmp_path / "basic") is None
