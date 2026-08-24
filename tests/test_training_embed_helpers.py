from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gk_surrogate.training.embed_dataset import encode_snapshots


def _apply_fn(params, x, *, train=False):
    del train
    scale = params.get("scale", params["params"]["scale"])
    return {"z": jnp.mean(x.reshape((x.shape[0], -1)), axis=-1, keepdims=True) * scale}


class _OutputWithZ:
    def __init__(self, z):
        self.z = z


def test_encode_snapshots_batches_to_numpy():
    snapshots = np.ones((3, 2, 4, 4, 4, 4, 4), dtype=np.float32)
    z = encode_snapshots(_apply_fn, {"scale": jnp.asarray(2.0)}, snapshots, batch_size=2)
    assert z.shape == (3, 1)
    assert z.dtype == np.float32


def test_encode_snapshots_chunks_object_outputs():
    calls = []

    def apply_fn(params, x, *, train=False):
        del params, train
        calls.append(int(x.shape[0]))
        return _OutputWithZ(jnp.ones((x.shape[0], 2), dtype=jnp.float32))

    snapshots = np.ones((5, 2, 2), dtype=np.float32)
    z = encode_snapshots(apply_fn, {}, snapshots, batch_size=2)
    assert z.shape == (5, 2)
    assert calls == [2, 2, 1]


def test_encode_snapshots_errors_are_explicit():
    with pytest.raises(ValueError, match="empty"):
        encode_snapshots(_apply_fn, {"scale": jnp.asarray(1.0)}, np.ones((0, 2, 2), dtype=np.float32))
    with pytest.raises(KeyError, match="latent key"):
        encode_snapshots(lambda params, x, train=False: {"bad": x}, {}, np.ones((1, 2), dtype=np.float32))
