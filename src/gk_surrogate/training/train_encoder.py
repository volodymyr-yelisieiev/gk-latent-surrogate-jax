"""Encoder train and evaluation steps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gk_surrogate.data.augmentations import AugmentationConfig, make_positive_pair
from gk_surrogate.losses.diagnostics import flux_mse
from gk_surrogate.losses.diagnostics import spectra_mse as diagnostic_spectra_mse
from gk_surrogate.losses.simsiam import simsiam_loss as shared_simsiam_loss
from gk_surrogate.training.checkpointing import save_checkpoint
from gk_surrogate.training.logging import MetricsLogger
from gk_surrogate.training.state import TrainState


def _cfg(config: Mapping[str, Any], name: str, default: Any) -> Any:
    loss_cfg = config.get("loss", {})
    if isinstance(loss_cfg, Mapping) and name in loss_cfg:
        return loss_cfg[name]
    return config.get(name, default)


def _batch_get(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    raise KeyError(f"batch is missing one of: {', '.join(names)}")


def _maybe_batch_get(batch: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def _call_apply(apply_fn: Any, params: Any, x: Array, *, train: bool, rng: Array | None = None) -> Any:
    variables = {"params": params}
    if rng is not None:
        try:
            return apply_fn(variables, x, train=train, rngs={"dropout": rng})
        except TypeError:
            pass
    try:
        return apply_fn(variables, x, train=train)
    except TypeError:
        try:
            return apply_fn(variables, x)
        except TypeError:
            return apply_fn(params, x)


def _as_mapping(output: Any) -> dict[str, Any]:
    if isinstance(output, Mapping):
        return dict(output)
    if hasattr(output, "z") and hasattr(output, "diagnostics"):
        diagnostics = output.diagnostics
        result: dict[str, Any] = {"z": output.z, "diagnostics": diagnostics}
        if hasattr(output, "projection"):
            result["projection"] = output.projection
        if hasattr(output, "prediction"):
            result["prediction"] = output.prediction
        if diagnostics is not None:
            result["flux"] = getattr(diagnostics, "flux", None)
            result["spectra"] = getattr(diagnostics, "spectra", {})
        return result
    if all(hasattr(output, name) for name in ("p1", "p2", "q1", "q2")):
        return {
            "projection": output.p1,
            "prediction": output.q1,
            "p1": output.p1,
            "p2": output.p2,
            "q1": output.q1,
            "q2": output.q2,
        }
    if isinstance(output, tuple):
        if len(output) == 2 and isinstance(output[1], Mapping):
            result = dict(output[1])
            result.setdefault("z", output[0])
            return result
        if len(output) == 3:
            return {"z": output[0], "projection": output[1], "prediction": output[2]}
    return {"z": output}


def _spectra_loss(pred: Mapping[str, Array], target: Mapping[str, Array], log_space: bool, eps: float) -> Array:
    if not target:
        return jnp.asarray(0.0, dtype=jnp.float32)
    selected = {}
    for key in target:
        if key not in pred:
            raise KeyError(f"spectra prediction missing key {key!r}")
        selected[key] = pred[key]
    return diagnostic_spectra_mse(selected, target, log_space=log_space, eps=eps)


def _simsiam_loss(out1: Mapping[str, Any], out2: Mapping[str, Any]) -> Array:
    p1 = out1.get("prediction", out1.get("p", out1.get("z")))
    z1 = out1.get("projection", out1.get("proj", out1.get("z")))
    p2 = out2.get("prediction", out2.get("p", out2.get("z")))
    z2 = out2.get("projection", out2.get("proj", out2.get("z")))
    return shared_simsiam_loss(p1, z2, p2, z1)


def _augmentation_config(config: Mapping[str, Any]) -> AugmentationConfig:
    raw = config.get("augmentations", {})
    if not isinstance(raw, Mapping):
        raw = {}
    max_shift = int(raw.get("max_periodic_shift", 0) or 0) if bool(raw.get("periodic_shift", False)) else 0
    return AugmentationConfig(
        gaussian_noise_sigma=float(raw.get("gaussian_noise_std", config.get("augmentation_noise_std", 0.0)) or 0.0),
        element_mask_probability=float(raw.get("mask_probability", 0.0) or 0.0),
        channel_dropout_probability=float(raw.get("channel_dropout_probability", 0.0) or 0.0),
        max_periodic_shift=max_shift,
        amplitude_jitter=float(raw.get("amplitude_jitter_std", 0.0) or 0.0),
    )


def _augment_pair(x: Array, rng: Array, config: Mapping[str, Any] | float) -> tuple[Array, Array]:
    if isinstance(config, Mapping):
        aug_config = _augmentation_config(config)
    else:
        aug_config = AugmentationConfig(gaussian_noise_sigma=float(config))
    if (
        aug_config.gaussian_noise_sigma <= 0.0
        and aug_config.element_mask_probability <= 0.0
        and aug_config.channel_dropout_probability <= 0.0
        and aug_config.max_periodic_shift <= 0
        and aug_config.amplitude_jitter <= 0.0
    ):
        return x, x
    keys = jax.random.split(rng, x.shape[0])
    return jax.vmap(lambda sample, key: make_positive_pair(sample, key, aug_config))(x, keys)


def _encoder_loss(
    params: Any, state: TrainState, batch: Mapping[str, Any], rng: Array, *, train: bool
) -> tuple[Array, dict[str, Array]]:
    x = _batch_get(batch, "x", "inputs", "snapshots")
    output = _as_mapping(_call_apply(state.apply_fn, params, x, train=train, rng=rng))

    flux_weight = float(_cfg(state.model_config, "flux_weight", 1.0))
    spectra_weight = float(_cfg(state.model_config, "spectra_weight", 1.0))
    simsiam_weight = float(_cfg(state.model_config, "simsiam_weight", 0.0))
    use_log_spectra = bool(_cfg(state.model_config, "use_log_spectra", False))
    spectra_epsilon = float(_cfg(state.model_config, "spectra_epsilon", 1e-6))

    flux_loss = jnp.asarray(0.0, dtype=jnp.float32)
    target_flux = _maybe_batch_get(batch, "flux", "target_flux")
    if target_flux is not None and flux_weight != 0.0:
        pred_flux = output.get("flux")
        if pred_flux is None and isinstance(output.get("diagnostics"), Mapping):
            pred_flux = output["diagnostics"].get("flux")
        if pred_flux is None:
            raise KeyError("flux target was provided but model output has no 'flux'")
        flux_loss = flux_mse(pred_flux, target_flux)

    spectra_loss = jnp.asarray(0.0, dtype=jnp.float32)
    target_spectra = _maybe_batch_get(batch, "spectra", "target_spectra")
    if target_spectra is not None and spectra_weight != 0.0:
        pred_spectra = output.get("spectra")
        if pred_spectra is None and isinstance(output.get("diagnostics"), Mapping):
            pred_spectra = output["diagnostics"].get("spectra")
        if pred_spectra is None:
            raise KeyError("spectra target was provided but model output has no 'spectra'")
        spectra_loss = _spectra_loss(pred_spectra, target_spectra, use_log_spectra, spectra_epsilon)

    simsiam_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if simsiam_weight != 0.0:
        view1 = _maybe_batch_get(batch, "view1", "x1")
        view2 = _maybe_batch_get(batch, "view2", "x2")
        if view1 is None or view2 is None:
            view1, view2 = _augment_pair(x, rng, state.model_config)
        out1 = _as_mapping(_call_apply(state.apply_fn, params, view1, train=train, rng=rng))
        out2 = _as_mapping(_call_apply(state.apply_fn, params, view2, train=train, rng=rng))
        simsiam_loss = _simsiam_loss(out1, out2)

    total = flux_weight * flux_loss + spectra_weight * spectra_loss + simsiam_weight * simsiam_loss
    metrics = {
        "loss": total,
        "flux_loss": flux_loss,
        "spectra_loss": spectra_loss,
        "simsiam_loss": simsiam_loss,
    }
    return total, metrics


@jax.jit
def train_encoder_step(state: TrainState, batch: Mapping[str, Any]) -> tuple[TrainState, dict[str, Array]]:
    rng, step_rng = jax.random.split(state.rng)

    def loss_fn(params: Any) -> tuple[Array, dict[str, Array]]:
        return _encoder_loss(params, state, batch, step_rng, train=True)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads).replace(rng=rng)
    metrics = dict(metrics)
    metrics["loss"] = loss
    return new_state, metrics


@jax.jit
def eval_encoder_step(state: TrainState, batch: Mapping[str, Any]) -> dict[str, Array]:
    _, metrics = _encoder_loss(state.params, state, batch, state.rng, train=False)
    return metrics


def train_encoder_loop(
    state: TrainState,
    batches: Iterable[Mapping[str, Any]],
    *,
    max_steps: int,
    output_dir: str | None = None,
    log_every: int = 1,
    checkpoint_every: int | None = None,
) -> tuple[TrainState, list[dict[str, float]]]:
    """Run a small Python training loop over an iterable of batches."""

    logger = MetricsLogger(output_dir) if output_dir else None
    history: list[dict[str, float]] = []
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    while int(state.step) < max_steps:
        pass_start = int(state.step)
        for batch in batches:
            state, metrics = train_encoder_step(state, batch)
            step = int(state.step)
            if step % log_every == 0 or step == max_steps:
                row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
                history.append(row)
                if logger is not None:
                    logger.log(row, prefix="train")
            if output_dir and checkpoint_every and step % checkpoint_every == 0:
                save_checkpoint(state, output_dir, step=step)
            if step >= max_steps:
                break
        if int(state.step) == pass_start:
            raise ValueError("batch iterable is empty or exhausted; provide a non-empty re-iterable batch source")
    if output_dir and int(state.step) > 0:
        save_checkpoint(state, output_dir)
    if logger is not None and history:
        logger.write_summary(history[-1])
    return state, history
