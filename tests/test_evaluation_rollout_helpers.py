from __future__ import annotations

import csv

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gk_surrogate.evaluation.reports import save_metrics_by_step, save_metrics_json
from gk_surrogate.evaluation.rollout import (
    autoregressive_rollout,
    evaluate_rollout_batches,
    horizon_until_threshold,
    latent_rollout_metrics,
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
    metrics = latent_rollout_metrics(persisted, jnp.ones((2, 2, 4)))
    assert metrics["stable"]
    assert horizon_until_threshold(metrics["mse_by_step"], 0.5) == 1
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
