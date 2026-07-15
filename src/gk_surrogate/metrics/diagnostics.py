"""Physics diagnostic metrics."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

Array = jnp.ndarray


def _require_same_shape(pred: Array, target: Array, *, label: str) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"{label} prediction and target shapes must match, got {pred.shape} and {target.shape}.")


def flux_mse(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target, label="Flux")
    return jnp.mean(jnp.square(pred - target))


def flux_rmse(pred: Array, target: Array) -> Array:
    return jnp.sqrt(flux_mse(pred, target))


def flux_mae(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target, label="Flux")
    return jnp.mean(jnp.abs(pred - target))


def flux_relative_error(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    _require_same_shape(pred, target, label="Flux")
    return jnp.mean(jnp.abs(pred - target) / (jnp.abs(target) + eps))


def time_average_flux_error(pred: Array, target: Array) -> Array:
    _require_same_shape(pred, target, label="Flux")
    if pred.ndim == 0:
        raise ValueError("Flux arrays must have at least one dimension.")
    return jnp.mean(jnp.abs(jnp.mean(pred, axis=0) - jnp.mean(target, axis=0)))


def _per_key(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
    fn,
) -> dict[str, Array]:
    pred_keys = set(pred)
    target_keys = set(target)
    if pred_keys != target_keys:
        missing = sorted(pred_keys - target_keys)
        unexpected = sorted(target_keys - pred_keys)
        raise KeyError(f"Spectra keys do not match; missing targets={missing}, unexpected targets={unexpected}.")
    values = {}
    for key, pred_value in sorted(pred.items()):
        target_value = target[key]
        _require_same_shape(pred_value, target_value, label=f"Spectrum {key!r}")
        values[key] = fn(pred_value, target_value)
    return values


def spectra_mse(pred: Mapping[str, Array], target: Mapping[str, Array]) -> dict[str, Array]:
    return _per_key(pred, target, lambda p, t: jnp.mean(jnp.square(p - t)))


def spectra_log_mse(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
    *,
    eps: float = 1e-8,
) -> dict[str, Array]:
    return _per_key(
        pred,
        target,
        lambda p, t: jnp.mean(jnp.square(jnp.log(jnp.maximum(p, 0.0) + eps) - jnp.log(jnp.maximum(t, 0.0) + eps))),
    )


def spectra_relative_l2(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
    *,
    eps: float = 1e-8,
) -> dict[str, Array]:
    def relative_l2(p: Array, t: Array) -> Array:
        if p.ndim == 0:
            raise ValueError("Spectrum arrays must have at least one dimension.")
        return jnp.linalg.norm(p - t) / (jnp.linalg.norm(t) + eps)

    return _per_key(pred, target, relative_l2)


def _pearson(pred: Array, target: Array, eps: float = 1e-8) -> Array:
    pred_centered = pred - jnp.mean(pred, axis=-1, keepdims=True)
    target_centered = target - jnp.mean(target, axis=-1, keepdims=True)
    numerator = jnp.sum(pred_centered * target_centered, axis=-1)
    pred_norm = jnp.sqrt(jnp.sum(pred_centered**2, axis=-1) + eps**2)
    target_norm = jnp.sqrt(jnp.sum(target_centered**2, axis=-1) + eps**2)
    denominator = pred_norm * target_norm
    return jnp.mean(numerator / denominator)


def spectra_pearson_corr(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
) -> dict[str, Array]:
    return _per_key(pred, target, _pearson)


def spectra_shape_corr(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
) -> dict[str, Array]:
    return spectra_pearson_corr(pred, target)


def spectra_mean_absolute_relative_error(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
    *,
    eps: float = 1e-8,
) -> dict[str, Array]:
    return _per_key(pred, target, lambda p, t: jnp.mean(jnp.abs(p - t) / (jnp.abs(t) + eps)))
