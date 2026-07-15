"""Loss functions for latent surrogate training."""

from gk_surrogate.losses.diagnostics import (
    diagnostic_prediction_loss,
    flux_mae,
    flux_mse,
    flux_relative_l2,
    spectra_mse,
    spectra_relative_l2,
)
from gk_surrogate.losses.latent import latent_prediction_loss
from gk_surrogate.losses.simsiam import negative_cosine_similarity, simsiam_loss
from gk_surrogate.losses.total import encoder_total_loss, sequence_total_loss

__all__ = [
    "diagnostic_prediction_loss",
    "encoder_total_loss",
    "flux_mae",
    "flux_mse",
    "flux_relative_l2",
    "latent_prediction_loss",
    "negative_cosine_similarity",
    "sequence_total_loss",
    "simsiam_loss",
    "spectra_mse",
    "spectra_relative_l2",
]
