"""Small loss-composition helpers used by training code."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from gk_surrogate.losses.diagnostics import diagnostic_prediction_loss
from gk_surrogate.losses.latent import latent_prediction_loss
from gk_surrogate.losses.simsiam import simsiam_loss

Array = jnp.ndarray


def encoder_total_loss(
    *,
    simsiam_outputs=None,
    diagnostics=None,
    flux_target: Array | None = None,
    spectra_target: Mapping[str, Array] | None = None,
    simsiam_weight: float = 0.0,
    flux_weight: float = 0.0,
    spectra_weight: float = 0.0,
    log_spectra: bool = False,
    spectra_eps: float = 1e-8,
) -> tuple[Array, dict[str, Array]]:
    total = jnp.asarray(0.0)
    metrics: dict[str, Array] = {}

    if simsiam_weight:
        if simsiam_outputs is None:
            raise ValueError("SimSiam loss requested but outputs are missing.")
        loss = simsiam_loss(
            simsiam_outputs.q1,
            simsiam_outputs.p2,
            simsiam_outputs.q2,
            simsiam_outputs.p1,
        )
        metrics["loss/simsiam"] = loss
        total = total + simsiam_weight * loss

    if flux_weight or spectra_weight:
        if diagnostics is None:
            raise ValueError("Diagnostic loss requested but predictions are missing.")
        diagnostic_loss, diagnostic_metrics = diagnostic_prediction_loss(
            diagnostics,
            flux_target=flux_target,
            spectra_target=spectra_target,
            flux_weight=flux_weight,
            spectra_weight=spectra_weight,
            log_spectra=log_spectra,
            spectra_eps=spectra_eps,
        )
        metrics.update(diagnostic_metrics)
        total = total + diagnostic_loss

    metrics["loss/total"] = total
    return total, metrics


def sequence_total_loss(
    pred_latent: Array,
    target_latent: Array,
    *,
    latent_weight: float = 1.0,
    latent_loss_mode: str = "mse",
) -> tuple[Array, dict[str, Array]]:
    loss = latent_prediction_loss(pred_latent, target_latent, mode=latent_loss_mode)
    total = latent_weight * loss
    return total, {"loss/latent": loss, "loss/total": total}
