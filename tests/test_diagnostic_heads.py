import jax
import jax.numpy as jnp
import pytest

from gk_surrogate.models.diagnostics import DiagnosticHeads


def test_diagnostic_heads_shapes_and_jit():
    z = jnp.ones((3, 8), dtype=jnp.float32)
    model = DiagnosticHeads(
        flux_dim=2,
        spectra_dims={"ky": 5, "q": 7},
        hidden_dims=(16,),
    )
    variables = model.init(jax.random.PRNGKey(0), z, train=False)

    out = model.apply(variables, z, train=False)
    assert out.flux.shape == (3, 2)
    assert out.spectra["ky"].shape == (3, 5)
    assert out.spectra["q"].shape == (3, 7)

    jitted = jax.jit(lambda value: model.apply(variables, value, train=False))
    assert jitted(z).spectra["ky"].shape == (3, 5)


def test_diagnostic_heads_allow_no_spectra():
    z = jnp.ones((3, 8), dtype=jnp.float32)
    model = DiagnosticHeads(flux_dim=1, spectra_dims={}, hidden_dims=())
    variables = model.init(jax.random.PRNGKey(1), z, train=False)
    out = model.apply(variables, z, train=False)
    assert out.flux.shape == (3, 1)
    assert out.spectra == {}


def test_diagnostic_heads_reject_invalid_or_colliding_names():
    z = jnp.ones((2, 4), dtype=jnp.float32)
    with pytest.raises(ValueError, match="non-negative"):
        DiagnosticHeads(flux_dim=-1).init(jax.random.PRNGKey(2), z, train=False)
    with pytest.raises(ValueError, match="remain unique"):
        DiagnosticHeads(spectra_dims={"a-b": 2, "a_b": 2}).init(jax.random.PRNGKey(3), z, train=False)
