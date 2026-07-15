import jax
import jax.numpy as jnp
import pytest

from gk_surrogate.models.encoders import ConvNDEncoder, FlattenMLPEncoder, PatchTransformerEncoder


def _snapshot_batch():
    return jnp.ones((2, 2, 4, 4, 4, 4, 4), dtype=jnp.float32)


def test_flatten_mlp_encoder_shape_and_jit():
    x = _snapshot_batch()
    model = FlattenMLPEncoder(latent_dim=16, hidden_dims=(32, 16))
    variables = model.init(jax.random.PRNGKey(0), x, train=False)

    out = model.apply(variables, x, train=False)
    assert out.shape == (2, 16)

    jitted = jax.jit(lambda value: model.apply(variables, value, train=False))
    assert jitted(x).shape == (2, 16)


def test_conv_nd_encoder_shape():
    x = _snapshot_batch()
    model = ConvNDEncoder(
        latent_dim=12,
        channels=(4, 8),
        kernel_size=(3, 3, 3, 3, 3),
        strides=((1, 1, 1, 1, 1), (2, 2, 2, 2, 2)),
    )
    variables = model.init(jax.random.PRNGKey(1), x, train=False)
    assert model.apply(variables, x, train=False).shape == (2, 12)


def test_conv_nd_pointwise_encoder_shape_and_jit():
    x = _snapshot_batch()
    model = ConvNDEncoder(
        latent_dim=12,
        channels=(4, 8),
        kernel_size=(1, 1, 1, 1, 1),
        strides=((2, 1, 1, 2, 1), (1, 2, 2, 1, 2)),
    )
    variables = model.init(jax.random.PRNGKey(11), x, train=False)
    assert model.apply(variables, x, train=False).shape == (2, 12)
    jitted = jax.jit(lambda value: model.apply(variables, value, train=False))
    assert jitted(x).shape == (2, 12)

    with pytest.raises(ValueError, match="positive"):
        ConvNDEncoder(latent_dim=12, channels=(4,), kernel_size=-1).init(jax.random.PRNGKey(12), x, train=False)


def test_patch_transformer_encoder_shape_and_invalid_spatial_dims():
    x = _snapshot_batch()
    model = PatchTransformerEncoder(
        latent_dim=10,
        patch_size=(2, 2, 2, 2, 2),
        embed_dim=16,
        depth=1,
        num_heads=4,
    )
    variables = model.init(jax.random.PRNGKey(2), x, train=False)
    assert model.apply(variables, x, train=False).shape == (2, 10)

    bad_x = jnp.ones((2, 2, 5, 4, 4, 4, 4), dtype=jnp.float32)
    with pytest.raises(ValueError, match="divisible"):
        model.init(jax.random.PRNGKey(3), bad_x, train=False)

    with pytest.raises(ValueError, match="positive"):
        PatchTransformerEncoder(latent_dim=10, patch_size=(2, 2, 2, 2, 0)).init(
            jax.random.PRNGKey(4), x, train=False
        )
    with pytest.raises(ValueError, match="num_heads"):
        PatchTransformerEncoder(latent_dim=10, num_heads=0).init(jax.random.PRNGKey(5), x, train=False)


@pytest.mark.parametrize(
    "model",
    [
        FlattenMLPEncoder(latent_dim=4, dropout_rate=-0.1),
        ConvNDEncoder(latent_dim=4, channels=(4,), kernel_size=(1,) * 5, dropout_rate=1.0),
        PatchTransformerEncoder(latent_dim=4, embed_dim=8, num_heads=2, attention_dropout_rate=1.0),
    ],
)
def test_encoders_reject_invalid_dropout_rates(model):
    with pytest.raises(ValueError, match="rate < 1"):
        model.init(jax.random.PRNGKey(6), _snapshot_batch(), train=False)


def test_patch_transformer_rejects_zero_width_mlp():
    with pytest.raises(ValueError, match="at least one MLP feature"):
        PatchTransformerEncoder(latent_dim=4, embed_dim=8, num_heads=2, mlp_ratio=0.01).init(
            jax.random.PRNGKey(7), _snapshot_batch(), train=False
        )
