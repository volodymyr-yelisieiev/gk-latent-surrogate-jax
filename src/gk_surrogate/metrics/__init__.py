"""Metrics for latent rollout and diagnostic evaluation."""

from gk_surrogate.metrics.aggregate import (
    aggregate_metrics,
    flatten_metrics,
    save_metrics_csv,
    save_metrics_json,
)
from gk_surrogate.metrics.diagnostics import (
    flux_mae,
    flux_mse,
    flux_relative_error,
    flux_rmse,
    spectra_log_mse,
    spectra_mean_absolute_relative_error,
    spectra_mse,
    spectra_pearson_corr,
    spectra_relative_l2,
    spectra_shape_corr,
    time_average_flux_error,
)
from gk_surrogate.metrics.latent import (
    latent_cosine_similarity,
    latent_mae,
    latent_mse,
    latent_relative_l2,
    rollout_cosine_by_step,
    rollout_mse_by_step,
)
from gk_surrogate.metrics.rollout import (
    autoregressive_rollout,
    horizon_until_threshold,
    rollout_stability,
    summarize_rollout,
)

__all__ = [
    "aggregate_metrics",
    "autoregressive_rollout",
    "flatten_metrics",
    "flux_mae",
    "flux_mse",
    "flux_relative_error",
    "flux_rmse",
    "horizon_until_threshold",
    "latent_cosine_similarity",
    "latent_mae",
    "latent_mse",
    "latent_relative_l2",
    "rollout_cosine_by_step",
    "rollout_mse_by_step",
    "rollout_stability",
    "save_metrics_csv",
    "save_metrics_json",
    "spectra_log_mse",
    "spectra_mean_absolute_relative_error",
    "spectra_mse",
    "spectra_pearson_corr",
    "spectra_relative_l2",
    "spectra_shape_corr",
    "summarize_rollout",
    "time_average_flux_error",
]
