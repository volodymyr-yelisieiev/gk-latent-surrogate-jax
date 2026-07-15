import jax
import jax.numpy as jnp
import pytest

from gk_surrogate.metrics.rollout import autoregressive_rollout
from gk_surrogate.models.sequence import (
    CausalTransformerSequenceModel,
    GRUSequenceModel,
    GuppyLatentTransformer,
    MLPDeltaSequenceModel,
    PersistenceBaseline,
)


def _context():
    return jnp.arange(2 * 4 * 6, dtype=jnp.float32).reshape(2, 4, 6) / 100.0


def test_sequence_model_shapes():
    z = _context()
    models = [
        PersistenceBaseline(latent_dim=6),
        MLPDeltaSequenceModel(latent_dim=6, context_length=4, hidden_dims=(16,)),
        GRUSequenceModel(latent_dim=6, hidden_dim=10, num_layers=1),
        CausalTransformerSequenceModel(
            latent_dim=6,
            context_length=4,
            embed_dim=12,
            depth=1,
            num_heads=3,
        ),
        GuppyLatentTransformer(
            latent_dim=6,
            context_length=4,
            model_dim=12,
            depth=1,
            num_heads=3,
        ),
    ]
    for i, model in enumerate(models):
        variables = model.init(jax.random.PRNGKey(i), z, train=False)
        out = model.apply(variables, z, train=False)
        assert out.shape == (2, 6)


def test_gru_sequence_model_shares_recurrent_parameters_across_timesteps():
    z = _context()
    model = GRUSequenceModel(latent_dim=6, hidden_dim=10, num_layers=1)
    variables = model.init(jax.random.PRNGKey(10), z, train=False)
    layer_params = variables["params"]["gru_layer_0"]
    assert "gates" in layer_params
    assert "candidate" in layer_params
    assert not any(name.startswith("gates_") or name.startswith("candidate_") for name in layer_params)


def test_sequence_models_reject_bad_shapes_and_config_mismatches():
    z = _context()
    with pytest.raises(ValueError, match="shape"):
        PersistenceBaseline(latent_dim=6).init(jax.random.PRNGKey(0), jnp.ones((2, 6)), train=False)
    with pytest.raises(ValueError, match="latent dimension"):
        PersistenceBaseline(latent_dim=7).init(jax.random.PRNGKey(0), z, train=False)
    with pytest.raises(ValueError, match="at least one timestep"):
        PersistenceBaseline(latent_dim=6).init(jax.random.PRNGKey(0), jnp.ones((2, 0, 6)), train=False)
    with pytest.raises(ValueError, match="context_length"):
        MLPDeltaSequenceModel(latent_dim=6, context_length=3).init(jax.random.PRNGKey(0), z, train=False)
    with pytest.raises(ValueError, match="hidden_dims"):
        MLPDeltaSequenceModel(latent_dim=6, context_length=4, hidden_dims=(0,)).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="dropout_rate"):
        MLPDeltaSequenceModel(latent_dim=6, context_length=4, dropout_rate=1.0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="num_layers"):
        GRUSequenceModel(latent_dim=6, num_layers=0).init(jax.random.PRNGKey(0), z, train=False)
    with pytest.raises(ValueError, match="hidden_dim"):
        GRUSequenceModel(latent_dim=6, hidden_dim=0).init(jax.random.PRNGKey(0), z, train=False)
    with pytest.raises(ValueError, match="context_length"):
        CausalTransformerSequenceModel(latent_dim=6, context_length=3).init(jax.random.PRNGKey(0), z, train=False)
    with pytest.raises(ValueError, match="num_heads"):
        CausalTransformerSequenceModel(latent_dim=6, context_length=4, embed_dim=10, num_heads=3).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="num_heads"):
        CausalTransformerSequenceModel(latent_dim=6, context_length=4, num_heads=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="embed_dim"):
        CausalTransformerSequenceModel(latent_dim=6, context_length=4, embed_dim=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="depth"):
        CausalTransformerSequenceModel(latent_dim=6, context_length=4, depth=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="mlp_ratio"):
        CausalTransformerSequenceModel(latent_dim=6, context_length=4, mlp_ratio=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="at least one MLP feature"):
        CausalTransformerSequenceModel(latent_dim=6, context_length=4, embed_dim=8, mlp_ratio=0.01).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="context_length"):
        GuppyLatentTransformer(latent_dim=6, context_length=3).init(jax.random.PRNGKey(0), z, train=False)
    with pytest.raises(ValueError, match="model_dim"):
        GuppyLatentTransformer(latent_dim=6, context_length=4, model_dim=10, num_heads=3).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="model_dim"):
        GuppyLatentTransformer(latent_dim=6, context_length=4, model_dim=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="depth"):
        GuppyLatentTransformer(latent_dim=6, context_length=4, depth=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="num_heads"):
        GuppyLatentTransformer(latent_dim=6, context_length=4, num_heads=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="mlp_ratio"):
        GuppyLatentTransformer(latent_dim=6, context_length=4, mlp_ratio=0).init(
            jax.random.PRNGKey(0), z, train=False
        )
    with pytest.raises(ValueError, match="at least one MLP feature"):
        GuppyLatentTransformer(latent_dim=6, context_length=4, model_dim=8, mlp_ratio=0.01).init(
            jax.random.PRNGKey(0), z, train=False
        )


def test_guppy_latent_transformer_has_projection_contract_and_delta_mode():
    z = _context()
    model = GuppyLatentTransformer(
        latent_dim=6,
        context_length=4,
        model_dim=12,
        depth=2,
        num_heads=3,
        predict_delta=True,
    )
    variables = model.init(jax.random.PRNGKey(12), z, train=False)
    params = variables["params"]
    out = model.apply(variables, z, train=False)
    residual = out - z[:, -1, :]

    assert out.shape == (2, 6)
    assert residual.shape == (2, 6)
    assert params["up_projection"]["kernel"].shape == (6, 12)
    assert params["down_projection"]["kernel"].shape == (12, 6)
    assert "block_0" in params
    assert "block_1" in params


def test_autoregressive_rollout_shape_persistence_and_jit():
    z = _context()

    def apply_fn(params, context, *, train):
        del params, train
        return context[:, -1, :]

    rollout = autoregressive_rollout(apply_fn, {}, z, 3)
    assert rollout.shape == (2, 3, 6)
    assert jnp.allclose(rollout, jnp.repeat(z[:, -1:, :], 3, axis=1))

    jitted = jax.jit(lambda value: autoregressive_rollout(apply_fn, {}, value, 2))
    assert jitted(z).shape == (2, 2, 6)


def test_public_rollout_helper_accepts_real_flax_apply_and_sequence_outputs():
    z = _context()
    model = MLPDeltaSequenceModel(latent_dim=6, context_length=4, hidden_dims=(8,))
    variables = model.init(jax.random.PRNGKey(11), z, train=False)
    rollout = autoregressive_rollout(model.apply, variables["params"], z, 2)
    assert rollout.shape == (2, 2, 6)

    def sequence_apply(params, context, *, train):
        del params, train
        return jnp.stack([context[:, -1, :] + 1.0, context[:, -1, :] + 2.0], axis=1)

    seq_rollout = autoregressive_rollout(sequence_apply, {}, z, 1)
    assert jnp.allclose(seq_rollout[:, 0, :], z[:, -1, :] + 2.0)
