"""SimSiam projection and prediction heads."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from flax import struct

Array = jnp.ndarray


@struct.dataclass
class SimSiamOutput:
    z1: Array
    z2: Array
    p1: Array
    p2: Array
    q1: Array
    q2: Array


class ProjectionHead(nn.Module):
    output_dim: int
    hidden_dim: int = 2048
    num_layers: int = 2

    @nn.compact
    def __call__(self, z: Array, *, train: bool) -> Array:
        del train
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape [B, Z], got {z.shape}.")
        if self.output_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        y = z
        for i in range(max(self.num_layers - 1, 0)):
            y = nn.Dense(self.hidden_dim, name=f"dense_{i}")(y)
            y = nn.relu(y)
        return nn.Dense(self.output_dim, name="output")(y)


class PredictionHead(nn.Module):
    output_dim: int
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, p: Array, *, train: bool) -> Array:
        del train
        if p.ndim != 2:
            raise ValueError(f"Expected projection with shape [B, P], got {p.shape}.")
        if self.output_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive.")
        y = nn.Dense(self.hidden_dim, name="dense_0")(p)
        y = nn.relu(y)
        return nn.Dense(self.output_dim, name="output")(y)


class SimSiamModel(nn.Module):
    encoder: nn.Module
    projection_head: ProjectionHead
    prediction_head: PredictionHead

    @nn.compact
    def __call__(self, view1: Array, view2: Array, *, train: bool) -> SimSiamOutput:
        z1 = self.encoder(view1, train=train)
        z2 = self.encoder(view2, train=train)
        p1 = self.projection_head(z1, train=train)
        p2 = self.projection_head(z2, train=train)
        q1 = self.prediction_head(p1, train=train)
        q2 = self.prediction_head(p2, train=train)
        return SimSiamOutput(z1=z1, z2=z2, p1=p1, p2=p2, q1=q1, q2=q2)
