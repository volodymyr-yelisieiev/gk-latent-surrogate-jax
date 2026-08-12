"""Evaluation helper for the direct snapshot diagnostic baseline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp

from gk_surrogate.evaluation.diagnostics import diagnostic_metrics
from gk_surrogate.models.diagnostics import DiagnosticPredictions

Array = jnp.ndarray


def evaluate_direct_snapshot_diagnostics(
    model_apply: Callable[..., DiagnosticPredictions],
    variables: Any,
    x: Array,
    *,
    flux_target: Array | None = None,
    spectra_target: Mapping[str, Array] | None = None,
) -> dict[str, Array | dict[str, Array]]:
    """Evaluate same-time ``x_t -> flux/spectra`` predictions on one batch."""

    predictions = model_apply(variables, x, train=False)
    if flux_target is not None and predictions.flux is None:
        raise ValueError("flux target was provided but the direct baseline has no flux output")
    if spectra_target is not None:
        missing = set(spectra_target) - set(predictions.spectra)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"direct baseline is missing spectra outputs: {joined}")
    return diagnostic_metrics(
        flux_pred=predictions.flux,
        flux_target=flux_target,
        spectra_pred=predictions.spectra if spectra_target is not None else None,
        spectra_target=spectra_target,
    )
