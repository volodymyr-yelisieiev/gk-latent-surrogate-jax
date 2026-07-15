"""Losses for scalar flux and one-dimensional spectra diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

Array = jnp.ndarray


def _require_same_shape(pred: Array, target: Array, *, label: str) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"{label} prediction and target shapes must match, got {pred.shape} and {target.shape}.")


def _require_matching_keys(pred: Mapping[str, Array], target: Mapping[str, Array]) -> None:
    pred_keys = set(pred)
    target_keys = set(target)
    if pred_keys != target_keys:
        missing = sorted(pred_keys - target_keys)
        unexpected = sorted(target_keys - pred_keys)
        raise KeyError(f"Spectra keys do not match; missing targets={missing}, unexpected targets={unexpected}.")


def _relative_l2(pred: Array, target: Array, eps: float) -> Array:
    _require_same_shape(pred, target, label="Relative-L2")
    if pred.ndim == 0:
        raise ValueError("Relative-L2 inputs must have at least one dimension.")
    return jnp.linalg.norm(pred - target) / (jnp.linalg.norm(target) + eps)


def flux_mse(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target, label="Flux")
    return jnp.mean(jnp.square(pred - target))


def flux_mae(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target, label="Flux")
    return jnp.mean(jnp.abs(pred - target))


def flux_huber(pred: Array, target: Array, delta: float = 1.0) -> Array:
    _require_same_shape(pred, target, label="Flux")
    if delta <= 0:
        raise ValueError("Huber delta must be positive.")
    error = pred - target
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * quadratic**2 + delta * linear)


def flux_relative_l2(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    return _relative_l2(pred, target, eps)


def _maybe_log(x: Array, *, log_space: bool, eps: float) -> Array:
    if not log_space:
        return x
    return jnp.log(jnp.maximum(x, 0.0) + eps)


def spectra_mse(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
    *,
    log_space: bool = False,
    eps: float = 1e-8,
) -> Array:
    _require_matching_keys(pred, target)
    if not pred:
        return jnp.asarray(0.0)
    losses = []
    for key, pred_value in sorted(pred.items()):
        target_value = target[key]
        _require_same_shape(pred_value, target_value, label=f"Spectrum {key!r}")
        losses.append(
            jnp.mean(
                jnp.square(
                    _maybe_log(pred_value, log_space=log_space, eps=eps)
                    - _maybe_log(target_value, log_space=log_space, eps=eps)
                )
            )
        )
    return jnp.mean(jnp.stack(losses))


def spectra_mae(pred: Mapping[str, Array], target: Mapping[str, Array]) -> Array:
    _require_matching_keys(pred, target)
    if not pred:
        return jnp.asarray(0.0)
    losses = []
    for key, pred_value in sorted(pred.items()):
        target_value = target[key]
        _require_same_shape(pred_value, target_value, label=f"Spectrum {key!r}")
        losses.append(jnp.mean(jnp.abs(pred_value - target_value)))
    return jnp.mean(jnp.stack(losses))


def spectra_relative_l2(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
    *,
    eps: float = 1e-8,
) -> Array:
    _require_matching_keys(pred, target)
    if not pred:
        return jnp.asarray(0.0)
    losses = []
    for key, pred_value in sorted(pred.items()):
        target_value = target[key]
        losses.append(_relative_l2(pred_value, target_value, eps))
    return jnp.mean(jnp.stack(losses))


def diagnostic_prediction_loss(
    pred,
    *,
    flux_target: Array | None = None,
    spectra_target: Mapping[str, Array] | None = None,
    flux_weight: float = 1.0,
    spectra_weight: float = 1.0,
    log_spectra: bool = False,
    spectra_eps: float = 1e-8,
) -> tuple[Array, dict[str, Array]]:
    """Compose weighted diagnostic losses from ``DiagnosticPredictions``-like objects."""

    total = jnp.asarray(0.0)
    metrics: dict[str, Array] = {}

    if flux_weight:
        if getattr(pred, "flux", None) is None or flux_target is None:
            raise ValueError("Flux loss requested but prediction or target is missing.")
        loss = flux_mse(pred.flux, flux_target)
        metrics["loss/flux"] = loss
        total = total + flux_weight * loss

    pred_spectra = getattr(pred, "spectra", {})
    if spectra_weight:
        if not pred_spectra or spectra_target is None:
            raise ValueError("Spectra loss requested but prediction or target is missing.")
        loss = spectra_mse(
            pred_spectra,
            spectra_target,
            log_space=log_spectra,
            eps=spectra_eps,
        )
        metrics["loss/spectra"] = loss
        total = total + spectra_weight * loss

    metrics["loss/diagnostics"] = total
    return total, metrics
