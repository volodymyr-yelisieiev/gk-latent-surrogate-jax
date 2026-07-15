"""Diagnostic evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from gk_surrogate.metrics.diagnostics import (
    flux_mae,
    flux_mse,
    flux_relative_error,
    flux_rmse,
    spectra_log_mse,
    spectra_mse,
    spectra_relative_l2,
    spectra_shape_corr,
)


def diagnostic_metrics(
    *,
    flux_pred: Array | None = None,
    flux_target: Array | None = None,
    spectra_pred: Mapping[str, Array] | None = None,
    spectra_target: Mapping[str, Array] | None = None,
) -> dict[str, Array | dict[str, Array]]:
    """Compute available flux and spectra metrics."""

    metrics: dict[str, Array | dict[str, Array]] = {}
    if flux_pred is not None and flux_target is not None:
        metrics["flux/mse"] = flux_mse(flux_pred, flux_target)
        metrics["flux/rmse"] = flux_rmse(flux_pred, flux_target)
        metrics["flux/mae"] = flux_mae(flux_pred, flux_target)
        metrics["flux/relative_error"] = flux_relative_error(flux_pred, flux_target)
    if spectra_pred is not None and spectra_target is not None:
        metrics["spectra/mse"] = spectra_mse(spectra_pred, spectra_target)
        metrics["spectra/log_mse"] = spectra_log_mse(spectra_pred, spectra_target)
        metrics["spectra/relative_l2"] = spectra_relative_l2(spectra_pred, spectra_target)
        metrics["spectra/shape_corr"] = spectra_shape_corr(spectra_pred, spectra_target)
    return metrics


def diagnostic_metrics_numpy(**kwargs: Array | Mapping[str, Array] | None) -> dict[str, object]:
    """Compute diagnostic metrics and move JAX arrays to Python/NumPy containers."""

    metrics = diagnostic_metrics(**kwargs)  # type: ignore[arg-type]
    return jax.tree_util.tree_map(lambda value: jax.device_get(jnp.asarray(value)), metrics)
