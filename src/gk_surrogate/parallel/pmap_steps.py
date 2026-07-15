"""Pmapped encoder and sequence train/eval steps."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache, partial
from typing import Any

import jax
from jax import Array

from gk_surrogate.training.state import TrainState
from gk_surrogate.training.train_encoder import _encoder_loss
from gk_surrogate.training.train_sequence import _sequence_loss

DEFAULT_AXIS_NAME = "devices"


def pmap_encoder_train_step(state: TrainState, batch: Mapping[str, Any]) -> tuple[TrainState, dict[str, Array]]:
    return make_pmap_encoder_train_step(DEFAULT_AXIS_NAME)(state, batch)


def pmap_encoder_eval_step(state: TrainState, batch: Mapping[str, Any]) -> dict[str, Array]:
    return make_pmap_encoder_eval_step(DEFAULT_AXIS_NAME)(state, batch)


def pmap_sequence_train_step(state: TrainState, batch: Mapping[str, Any]) -> tuple[TrainState, dict[str, Array]]:
    return make_pmap_sequence_train_step(DEFAULT_AXIS_NAME)(state, batch)


def pmap_sequence_eval_step(state: TrainState, batch: Mapping[str, Any]) -> dict[str, Array]:
    return make_pmap_sequence_eval_step(DEFAULT_AXIS_NAME)(state, batch)


@cache
def make_pmap_encoder_train_step(axis_name: str):
    @partial(jax.pmap, axis_name=axis_name)
    def step(state: TrainState, batch: Mapping[str, Any]) -> tuple[TrainState, dict[str, Array]]:
        rng, step_rng = jax.random.split(state.rng)

        def loss_fn(params: Any) -> tuple[Array, dict[str, Array]]:
            return _encoder_loss(params, state, batch, step_rng, train=True)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name)
        new_state = state.apply_gradients(grads=grads).replace(rng=rng)
        metrics = jax.lax.pmean(metrics, axis_name)
        metrics = dict(metrics)
        metrics["loss"] = jax.lax.pmean(loss, axis_name)
        return new_state, metrics

    return step


@cache
def make_pmap_encoder_eval_step(axis_name: str):
    @partial(jax.pmap, axis_name=axis_name)
    def step(state: TrainState, batch: Mapping[str, Any]) -> dict[str, Array]:
        _, metrics = _encoder_loss(state.params, state, batch, state.rng, train=False)
        return jax.lax.pmean(metrics, axis_name)

    return step


@cache
def make_pmap_sequence_train_step(axis_name: str):
    @partial(jax.pmap, axis_name=axis_name)
    def step(state: TrainState, batch: Mapping[str, Any]) -> tuple[TrainState, dict[str, Array]]:
        rng, step_rng = jax.random.split(state.rng)
        step_state = state.replace_rng(step_rng)

        def loss_fn(params: Any) -> tuple[Array, dict[str, Array]]:
            return _sequence_loss(params, step_state, batch, train=True)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name)
        new_state = state.apply_gradients(grads=grads).replace(rng=rng)
        metrics = jax.lax.pmean(metrics, axis_name)
        metrics = dict(metrics)
        metrics["loss"] = jax.lax.pmean(loss, axis_name)
        return new_state, metrics

    return step


@cache
def make_pmap_sequence_eval_step(axis_name: str):
    @partial(jax.pmap, axis_name=axis_name)
    def step(state: TrainState, batch: Mapping[str, Any]) -> dict[str, Array]:
        _, metrics = _sequence_loss(state.params, state, batch, train=False)
        return jax.lax.pmean(metrics, axis_name)

    return step
