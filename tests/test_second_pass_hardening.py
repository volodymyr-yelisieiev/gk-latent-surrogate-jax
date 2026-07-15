from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.evaluation import rollout as eval_rollout
from gk_surrogate.models.encoders import (
    ConvNDEncoder,
    FlattenMLPEncoder,
    PatchTransformerEncoder,
    _activation,
    _as_tuple,
)
from gk_surrogate.training import train_encoder as encoder_train
from gk_surrogate.training import train_sequence as sequence_train


def _snapshot_batch():
    return jnp.ones((2, 2, 4, 4, 4, 4, 4), dtype=jnp.float32)


def _latent_context():
    return jnp.ones((2, 3, 4), dtype=jnp.float32)


def test_config_loader_and_validation_remaining_command_edges(tiny_config_path):
    with pytest.raises(ValueError, match="override must use key=value"):
        load_config(tiny_config_path, overrides=["training.max_steps"], command="train-encoder")
    with pytest.raises(ValueError, match="non-mapping"):
        load_config(tiny_config_path, overrides=["data.backend.kind=h5"], command="train-encoder")
    with pytest.raises(ValueError, match="simsiam loss"):
        load_config(tiny_config_path, overrides=["loss.simsiam_weight=1.0"])
    with pytest.raises(ValueError, match="latent_cache.path"):
        load_config(
            tiny_config_path,
            overrides=[
                "model.sequence={type: mlp_delta, latent_dim: 32, context_length: 4, hidden_dims: [16], extra: {}}"
            ],
            command="train-sequence",
        )
    with pytest.raises(ValueError, match="evaluate-rollout requires latent_cache.path"):
        load_config(tiny_config_path, command="evaluate-rollout")
    with pytest.raises(ValueError, match="sequence_checkpoint_path"):
        load_config(
            tiny_config_path,
            overrides=["latent_cache.path=/tmp/nonexistent_latent_cache.h5"],
            command="evaluate-rollout",
        )
    config = load_config(
        tiny_config_path,
        overrides=[
            "latent_cache.path=/tmp/nonexistent_latent_cache.h5",
            "latent_cache.use_persistence_baseline=true",
        ],
        command="evaluate-rollout",
    )
    assert config.latent_cache.use_persistence_baseline


def test_train_sequence_private_branches_are_shape_safe():
    context = _latent_context()
    target = jnp.ones((2, 1, 4), dtype=jnp.float32)

    assert sequence_train._batch_get({"inputs": context}, "context", "inputs") is context
    with pytest.raises(KeyError, match="batch is missing"):
        sequence_train._batch_get({}, "context")
    assert sequence_train._cfg({"loss": {"latent_loss": "huber"}}, "latent_loss", "mse") == "huber"
    assert sequence_train._cfg({"latent_loss": "cosine"}, "latent_loss", "mse") == "cosine"

    def with_prediction_length(variables, value, *, train, prediction_length):
        del variables, train
        return {"prediction": jnp.repeat(value[:, -1:, :], prediction_length, axis=1)}

    def without_prediction_length(variables, value, *, train):
        del variables, train
        return {"pred": value[:, -1, :]}

    def without_train(variables, value):
        del variables
        return {"latent": value[:, -1, :]}

    def raw_params_only(params, value):
        if isinstance(params, dict) and "params" in params:
            raise TypeError("raw params only")
        return value[:, -1, :]

    assert sequence_train._prediction_array(
        sequence_train._call_apply(with_prediction_length, {}, context, train=True, prediction_length=2)
    ).shape == (2, 2, 4)
    assert sequence_train._prediction_array(
        sequence_train._call_apply(without_prediction_length, {}, context, train=True, prediction_length=2)
    ).shape == (2, 4)
    assert sequence_train._prediction_array(
        sequence_train._call_apply(without_train, {}, context, train=True, prediction_length=2)
    ).shape == (2, 4)
    assert sequence_train._call_apply(raw_params_only, {"w": 1}, context, train=True, prediction_length=2).shape == (
        2,
        4,
    )
    with pytest.raises(KeyError, match="no prediction key"):
        sequence_train._prediction_array({"bad": context})

    assert jnp.isfinite(sequence_train.latent_prediction_loss(context[:, -1, :], target, mode="mse"))
    assert jnp.isfinite(sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="huber"))
    assert jnp.isfinite(sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="cosine"))
    assert jnp.isfinite(sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="mse_plus_cosine"))
    with pytest.raises(ValueError, match="Unsupported latent loss"):
        sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="bad")
    assert sequence_train.persistence_predict(context, prediction_length=2).shape == (2, 2, 4)


def test_train_encoder_helper_branches_are_explicit():
    z = jnp.ones((2, 4), dtype=jnp.float32)
    diagnostics = SimpleNamespace(flux=z[:, :1], spectra={"ky": z})
    output = SimpleNamespace(z=z, diagnostics=diagnostics, projection=z + 1.0, prediction=z + 2.0)
    mapped = encoder_train._as_mapping(output)
    assert mapped["flux"].shape == (2, 1)
    assert mapped["projection"].shape == (2, 4)
    assert encoder_train._as_mapping((z, {"flux": z}))["z"].shape == (2, 4)
    assert encoder_train._as_mapping((z, z + 1.0, z + 2.0))["prediction"].shape == (2, 4)
    assert encoder_train._as_mapping(z)["z"].shape == (2, 4)
    assert encoder_train._maybe_batch_get({}, "missing") is None
    with pytest.raises(KeyError, match="batch is missing"):
        encoder_train._batch_get({}, "x")

    assert encoder_train._spectra_loss({}, {}, log_space=False, eps=1e-6) == 0.0
    with pytest.raises(KeyError, match="missing key"):
        encoder_train._spectra_loss({}, {"ky": z}, log_space=False, eps=1e-6)
    assert jnp.isfinite(encoder_train._spectra_loss({"ky": -z}, {"ky": z}, log_space=True, eps=1e-6))
    view1, view2 = encoder_train._augment_pair(z, jax.random.PRNGKey(0), 0.0)
    assert jnp.allclose(view1, z)
    assert jnp.allclose(view2, z)
    noisy1, noisy2 = encoder_train._augment_pair(
        z.reshape((2, 1, 2, 2)),
        jax.random.PRNGKey(0),
        {
            "augmentations": {
                "gaussian_noise_std": 0.1,
                "amplitude_jitter_std": 0.1,
                "mask_probability": 0.1,
                "channel_dropout_probability": 0.1,
            }
        },
    )
    assert noisy1.shape == noisy2.shape == (2, 1, 2, 2)

    def with_dropout(variables, value, *, train, rngs):
        del variables, train, rngs
        return value

    def without_rng(variables, value, *, train):
        del variables, train
        return value

    def without_train(variables, value):
        del variables
        return value

    def raw_params_only(params, value):
        if isinstance(params, dict) and "params" in params:
            raise TypeError("raw params only")
        return value

    assert encoder_train._call_apply(with_dropout, {}, z, train=True, rng=jax.random.PRNGKey(1)).shape == z.shape
    assert encoder_train._call_apply(without_rng, {}, z, train=True, rng=jax.random.PRNGKey(1)).shape == z.shape
    assert encoder_train._call_apply(without_train, {}, z, train=True).shape == z.shape
    assert encoder_train._call_apply(raw_params_only, {"w": 1}, z, train=True).shape == z.shape


def test_evaluation_rollout_error_and_fallback_branches():
    context = _latent_context()

    def with_train(variables, value, *, train=False):
        del variables, train
        return {"prediction": value[:, -1, :] + 1.0}

    def without_train(variables, value):
        del variables
        return {"z": value[:, -1, :]}

    def raw_params_only(params, value):
        if isinstance(params, dict) and "params" in params:
            raise TypeError("raw params only")
        return {"latent": value[:, -1, :]}

    assert eval_rollout._prediction_array(eval_rollout._call_model(with_train, {}, context, train=False)).shape == (
        2,
        4,
    )
    assert eval_rollout._prediction_array(eval_rollout._call_model(without_train, {}, context, train=False)).shape == (
        2,
        4,
    )
    assert eval_rollout._prediction_array(
        eval_rollout._call_model(raw_params_only, {"w": 1}, context, train=False)
    ).shape == (2, 4)
    with pytest.raises(KeyError, match="no prediction key"):
        eval_rollout._prediction_array({"bad": context})
    with pytest.raises(ValueError, match="positive"):
        eval_rollout.autoregressive_rollout(with_train, {}, context, rollout_steps=0)
    with pytest.raises(KeyError, match="context/target"):
        eval_rollout.evaluate_rollout_batches(with_train, {}, [{"context": context}], rollout_steps=1)
    with pytest.raises(ValueError, match="no rollout batches"):
        eval_rollout.evaluate_rollout_batches(with_train, {}, [], rollout_steps=1)
    metrics = eval_rollout.evaluate_rollout_batches(
        with_train,
        {},
        [{"context": context, "target": jnp.ones((2, 2, 4), dtype=jnp.float32)}],
        rollout_steps=2,
    )
    assert metrics["mse_by_step"].shape == (2,)
    assert eval_rollout.horizon_until_threshold(jnp.asarray([0.1, 0.2]), threshold=1.0) == 2


def test_encoder_validation_and_dropout_branches():
    x = _snapshot_batch()
    for name in ("relu", "swish", "silu", "tanh", "linear", "identity", "none"):
        assert _activation(name)(jnp.asarray([1.0])).shape == (1,)
    with pytest.raises(ValueError, match="Unsupported activation"):
        _activation("bad")
    assert _as_tuple(2, length=5, name="patch") == (2, 2, 2, 2, 2)
    with pytest.raises(ValueError, match="must have length"):
        _as_tuple((1, 2), length=5, name="patch")

    with pytest.raises(ValueError, match="shape"):
        FlattenMLPEncoder(latent_dim=4).init(jax.random.PRNGKey(0), jnp.ones((2, 2)), train=False)
    with pytest.raises(ValueError, match="latent_dim"):
        FlattenMLPEncoder(latent_dim=0).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="latent_dim"):
        ConvNDEncoder(latent_dim=0).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="channels"):
        ConvNDEncoder(latent_dim=4, channels=()).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="kernel_size"):
        ConvNDEncoder(latent_dim=4, channels=(4,), kernel_size=(1, 2)).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="one entry"):
        ConvNDEncoder(
            latent_dim=4,
            channels=(4,),
            strides=((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="length 5"):
        ConvNDEncoder(latent_dim=4, channels=(4,), strides=((1, 1),)).init(jax.random.PRNGKey(0), x, train=False)

    with pytest.raises(ValueError, match="latent_dim"):
        PatchTransformerEncoder(latent_dim=0).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="patch_size"):
        PatchTransformerEncoder(latent_dim=4, patch_size=(2, 2)).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="num_heads"):
        PatchTransformerEncoder(latent_dim=4, embed_dim=10, num_heads=3).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="exceeds"):
        PatchTransformerEncoder(latent_dim=4, embed_dim=8, num_heads=2, max_token_count=1).init(
            jax.random.PRNGKey(0), x, train=False
        )
    model = PatchTransformerEncoder(
        latent_dim=4,
        patch_size=(2, 2, 2, 2, 2),
        embed_dim=8,
        depth=1,
        num_heads=2,
        dropout_rate=0.1,
        use_cls_token=True,
        max_token_count=1,
        allow_large_token_count=True,
    )
    variables = model.init(jax.random.PRNGKey(1), x, train=True)
    out = model.apply(variables, x, train=True, rngs={"dropout": jax.random.PRNGKey(2)})
    assert out.shape == (2, 4)
