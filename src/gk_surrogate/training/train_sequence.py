"""Latent sequence train and evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gk_surrogate.losses.latent import latent_prediction_loss as _latent_prediction_loss
from gk_surrogate.training.checkpointing import save_checkpoint
from gk_surrogate.training.logging import MetricsLogger
from gk_surrogate.training.state import TrainState


def _batch_get(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    raise KeyError(f"batch is missing one of: {', '.join(names)}")


def _cfg(config: Mapping[str, Any], name: str, default: Any) -> Any:
    loss_cfg = config.get("loss", {})
    if isinstance(loss_cfg, Mapping) and name in loss_cfg:
        return loss_cfg[name]
    return config.get(name, default)


def _call_apply(
    apply_fn: Any,
    params: Any,
    context: Array,
    *,
    train: bool,
    prediction_length: int,
    rng: Array | None = None,
) -> Any:
    variables = {"params": params}
    rngs = {"dropout": rng} if rng is not None else None
    if rngs is not None:
        try:
            return apply_fn(variables, context, train=train, prediction_length=prediction_length, rngs=rngs)
        except TypeError:
            try:
                return apply_fn(variables, context, train=train, rngs=rngs)
            except TypeError:
                pass
    try:
        return apply_fn(variables, context, train=train, prediction_length=prediction_length)
    except TypeError:
        try:
            return apply_fn(variables, context, train=train)
        except TypeError:
            try:
                return apply_fn(variables, context)
            except TypeError:
                return apply_fn(params, context)


def _prediction_array(output: Any) -> Array:
    if isinstance(output, Mapping):
        for key in ("z", "prediction", "pred", "latent"):
            if key in output:
                return output[key]
        raise KeyError("sequence model output mapping has no prediction key")
    return output


def _align_prediction(pred: Array, target: Array) -> tuple[Array, Array]:
    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    if pred.ndim == 2 and target.ndim == 3 and target.shape[1] == 1:
        pred = pred[:, None, :]
    if pred.ndim == 3 and target.ndim == 2 and pred.shape[1] == 1:
        target = target[:, None, :]
    return pred, target


def _predict_sequence(params: Any, state: TrainState, context: Array, target: Array, *, train: bool) -> Array:
    prediction_length = target.shape[1] if target.ndim == 3 else 1
    rng = state.rng
    output = _call_apply(
        state.apply_fn,
        params,
        context,
        train=train,
        prediction_length=int(prediction_length),
        rng=rng,
    )
    pred = _prediction_array(output)
    pred_aligned, target_aligned = _align_prediction(pred, target)
    if pred_aligned.shape == target_aligned.shape:
        return pred_aligned
    if target.ndim != 3:
        raise ValueError(f"sequence prediction shape {pred.shape} is incompatible with target shape {target.shape}")

    carry = context
    preds = []
    for step in range(int(prediction_length)):
        step_rng = jax.random.fold_in(rng, step)
        output = _call_apply(
            state.apply_fn,
            params,
            carry,
            train=train,
            prediction_length=1,
            rng=step_rng,
        )
        step_pred = _prediction_array(output)
        if step_pred.ndim == 3:
            step_pred = step_pred[:, -1, :]
        if step_pred.ndim != 2:
            raise ValueError(f"Expected one-step prediction [B, Z], got {step_pred.shape}.")
        preds.append(step_pred)
        carry = jnp.concatenate([carry[:, 1:, :], step_pred[:, None, :]], axis=1)
    return jnp.stack(preds, axis=1)


def latent_prediction_loss(pred: Array, target: Array, mode: str = "mse") -> Array:
    pred, target = _align_prediction(pred, target)
    return _latent_prediction_loss(pred, target, mode)


def _sequence_loss(
    params: Any, state: TrainState, batch: Mapping[str, Any], *, train: bool
) -> tuple[Array, dict[str, Array]]:
    context = _batch_get(batch, "z_context", "context", "inputs")
    target = _batch_get(batch, "z_target", "target", "targets")
    pred = _predict_sequence(params, state, context, target, train=train)
    mode = str(_cfg(state.model_config, "latent_loss", "mse"))
    latent_loss = latent_prediction_loss(pred, target, mode)
    latent_weight = float(_cfg(state.model_config, "latent_weight", 1.0))
    total = latent_weight * latent_loss
    pred_aligned, target_aligned = _align_prediction(pred, target)
    metrics = {
        "loss": total,
        "latent_loss": latent_loss,
        "latent_mse": jnp.mean(jnp.square(pred_aligned - target_aligned)),
    }
    return total, metrics


@jax.jit
def train_sequence_step(state: TrainState, batch: Mapping[str, Any]) -> tuple[TrainState, dict[str, Array]]:
    rng, step_rng = jax.random.split(state.rng)
    step_state = state.replace_rng(step_rng)

    def loss_fn(params: Any) -> tuple[Array, dict[str, Array]]:
        return _sequence_loss(params, step_state, batch, train=True)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads).replace(rng=rng)
    metrics = dict(metrics)
    metrics["loss"] = loss
    return new_state, metrics


@jax.jit
def eval_sequence_step(state: TrainState, batch: Mapping[str, Any]) -> dict[str, Array]:
    _, metrics = _sequence_loss(state.params, state, batch, train=False)
    return metrics


def persistence_predict(context: Array, prediction_length: int = 1) -> Array:
    last = context[:, -1:, :]
    return jnp.repeat(last, prediction_length, axis=1)


def train_sequence_loop(
    state: TrainState,
    batches: Iterable[Mapping[str, Any]],
    *,
    max_steps: int,
    output_dir: str | None = None,
    log_every: int = 1,
    checkpoint_every: int | None = None,
) -> tuple[TrainState, list[dict[str, float]]]:
    logger = MetricsLogger(output_dir) if output_dir else None
    history: list[dict[str, float]] = []
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    while int(state.step) < max_steps:
        pass_start = int(state.step)
        for batch in batches:
            state, metrics = train_sequence_step(state, batch)
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
