import jax
import jax.numpy as jnp
import pytest

from gk_surrogate.losses.simsiam import negative_cosine_similarity, simsiam_loss
from gk_surrogate.models.encoders import FlattenMLPEncoder
from gk_surrogate.models.simsiam import PredictionHead, ProjectionHead, SimSiamModel


def test_simsiam_loss_finite_and_identical_lower_than_random():
    key = jax.random.PRNGKey(0)
    p = jax.random.normal(key, (8, 16))
    z_same = p
    z_random = jax.random.normal(jax.random.PRNGKey(1), (8, 16))

    same_loss = negative_cosine_similarity(p, z_same)
    random_loss = negative_cosine_similarity(p, z_random)
    assert jnp.isfinite(same_loss)
    assert same_loss < random_loss

    zero_loss = simsiam_loss(jnp.zeros((2, 4)), jnp.zeros((2, 4)), jnp.zeros((2, 4)), jnp.zeros((2, 4)))
    assert jnp.isfinite(zero_loss)


def test_simsiam_model_output_shapes():
    x = jnp.ones((2, 1, 4, 4, 4, 4, 4), dtype=jnp.float32)
    model = SimSiamModel(
        encoder=FlattenMLPEncoder(latent_dim=8, hidden_dims=(16,)),
        projection_head=ProjectionHead(output_dim=12, hidden_dim=16, num_layers=2),
        prediction_head=PredictionHead(output_dim=12, hidden_dim=8),
    )
    variables = model.init(jax.random.PRNGKey(2), x, x, train=False)
    out = model.apply(variables, x, x, train=False)
    assert out.z1.shape == (2, 8)
    assert out.p1.shape == (2, 12)
    assert out.q1.shape == (2, 12)


def test_negative_cosine_stops_gradient_through_target():
    p = jnp.asarray([[1.0, 0.0, 0.0]])
    z = jnp.asarray([[0.5, 0.5, 0.0]])
    grad_p = jax.grad(lambda value: negative_cosine_similarity(value, z))(p)
    grad_z = jax.grad(lambda value: negative_cosine_similarity(p, value))(z)
    assert jnp.linalg.norm(grad_p) > 0
    assert jnp.allclose(grad_z, 0.0)


def test_negative_cosine_rejects_shape_broadcasting():
    with pytest.raises(ValueError, match="shapes must match"):
        negative_cosine_similarity(jnp.ones((2, 1)), jnp.ones((2, 3)))
