"""Snapshot encoders for channel-first 5D gyrokinetic fields.

All encoders accept ``x`` with shape ``[B, C, S1, S2, S3, S4, S5]`` and
return latent vectors with shape ``[B, latent_dim]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import linen as nn

from gk_surrogate.models.patching import as_5d_tuple, validate_token_count

Array = jnp.ndarray


def _activation(name: str):
    match name:
        case "gelu":
            return nn.gelu
        case "relu":
            return nn.relu
        case "swish" | "silu":
            return nn.swish
        case "tanh":
            return nn.tanh
        case "linear" | "identity" | "none":
            return lambda x: x
        case _:
            raise ValueError(f"Unsupported activation {name!r}.")


def _ensure_snapshot_shape(x: Array) -> None:
    if x.ndim != 7:
        raise ValueError(
            f"Expected snapshot batch with shape [B, C, S1, S2, S3, S4, S5], got rank {x.ndim} and shape {x.shape}."
        )


def _as_tuple(value: int | Sequence[int], *, length: int, name: str) -> tuple[int, ...]:
    result = (value,) * length if isinstance(value, int) else tuple(int(v) for v in value)
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}, got {result}.")
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} values must be positive, got {result}.")
    return result


class FlattenMLPEncoder(nn.Module):
    """Flattening MLP baseline for small synthetic snapshots."""

    latent_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "gelu"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        _ensure_snapshot_shape(x)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("hidden_dims values must be positive.")

        y = x.reshape((x.shape[0], -1))
        act = _activation(self.activation)
        for i, hidden_dim in enumerate(self.hidden_dims):
            y = nn.Dense(hidden_dim, name=f"dense_{i}")(y)
            y = act(y)
            if self.dropout_rate:
                y = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{i}")(y, deterministic=not train)
        return nn.Dense(self.latent_dim, name="latent")(y)


class ConvNDEncoder(nn.Module):
    """N-dimensional convolutional encoder for channel-first 5D snapshots."""

    latent_dim: int
    channels: tuple[int, ...] = (16, 32, 64)
    kernel_size: int | tuple[int, ...] = (3, 3, 3, 3, 3)
    strides: tuple[tuple[int, ...], ...] | None = None
    activation: str = "gelu"
    dropout_rate: float = 0.0
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        _ensure_snapshot_shape(x)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not self.channels:
            raise ValueError("channels must contain at least one layer width.")
        if any(features <= 0 for features in self.channels):
            raise ValueError("channels values must be positive.")

        kernel_size = _as_tuple(self.kernel_size, length=5, name="kernel_size")
        if self.strides is None:
            strides = ((1, 1, 1, 1, 1),) * len(self.channels)
        else:
            strides = tuple(tuple(int(v) for v in stride) for stride in self.strides)
            if len(strides) != len(self.channels):
                raise ValueError("strides must have one entry per convolution layer.")
            for stride in strides:
                if len(stride) != 5:
                    raise ValueError(f"Each stride must have length 5, got {stride}.")
                if any(step <= 0 for step in stride):
                    raise ValueError(f"Stride values must be positive, got {stride}.")

        y = jnp.moveaxis(x, 1, -1)
        act = _activation(self.activation)
        if all(size == 1 for size in kernel_size):
            for i, (features, stride) in enumerate(zip(self.channels, strides, strict=True)):
                spatial_slices = tuple(slice(None, None, step) for step in stride)
                y = y[(slice(None), *spatial_slices, slice(None))]
                y = nn.Dense(features, name=f"pointwise_{i}")(y)
                y = act(y)
                if self.dropout_rate:
                    y = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{i}")(y, deterministic=not train)
            y = jnp.mean(y, axis=tuple(range(1, y.ndim - 1)))
            return nn.Dense(self.latent_dim, name="latent")(y)

        for i, (features, stride) in enumerate(zip(self.channels, strides, strict=True)):
            y = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                strides=stride,
                padding=self.padding,
                name=f"conv_{i}",
            )(y)
            y = act(y)
            if self.dropout_rate:
                y = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{i}")(y, deterministic=not train)

        y = jnp.mean(y, axis=tuple(range(1, y.ndim - 1)))
        return nn.Dense(self.latent_dim, name="latent")(y)


class _TransformerEncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    dropout_rate: float
    attention_dropout_rate: float
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        deterministic = not train
        y = nn.LayerNorm(name="attn_norm")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            name="attention",
        )(y, y, deterministic=deterministic)
        if self.dropout_rate:
            y = nn.Dropout(rate=self.dropout_rate, name="attn_dropout")(y, deterministic=deterministic)
        x = x + y

        y = nn.LayerNorm(name="mlp_norm")(x)
        y = nn.Dense(int(self.embed_dim * self.mlp_ratio), name="mlp_in")(y)
        y = _activation(self.activation)(y)
        if self.dropout_rate:
            y = nn.Dropout(rate=self.dropout_rate, name="mlp_dropout")(y, deterministic=deterministic)
        y = nn.Dense(self.embed_dim, name="mlp_out")(y)
        return x + y


class PatchTransformerEncoder(nn.Module):
    """Patch-token Transformer encoder for 5D snapshots."""

    latent_dim: int
    patch_size: tuple[int, int, int, int, int] = (2, 2, 2, 2, 2)
    embed_dim: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    use_cls_token: bool = False
    max_token_count: int = 4096
    allow_large_token_count: bool = False
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        _ensure_snapshot_shape(x)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        patch_size = as_5d_tuple(self.patch_size, name="patch_size")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        spatial_shape = x.shape[2:]
        token_count = validate_token_count(
            spatial_shape,
            patch_size,
            max_token_count=self.max_token_count,
            allow_large_token_count=self.allow_large_token_count,
        )

        y = jnp.moveaxis(x, 1, -1)
        y = nn.Conv(
            features=self.embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            name="patch_embed",
        )(y)

        y = y.reshape((y.shape[0], token_count, self.embed_dim))
        if self.use_cls_token:
            cls = self.param("cls_token", nn.initializers.zeros, (1, 1, self.embed_dim))
            y = jnp.concatenate([jnp.broadcast_to(cls, (y.shape[0], 1, self.embed_dim)), y], axis=1)

        pos = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, y.shape[1], self.embed_dim),
        )
        y = y + pos
        if self.dropout_rate:
            y = nn.Dropout(rate=self.dropout_rate, name="pos_dropout")(y, deterministic=not train)

        for i in range(self.depth):
            y = _TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                activation=self.activation,
                name=f"block_{i}",
            )(y, train=train)

        y = y[:, 0] if self.use_cls_token else jnp.mean(y, axis=1)
        return nn.Dense(self.latent_dim, name="latent")(nn.LayerNorm(name="final_norm")(y))


class ExternalEncoderAdapter(nn.Module):
    """Unsupported external encoder hook kept behind explicit config opt-in."""

    name: str
    latent_dim: int

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        del x, train
        raise NotImplementedError(f"External encoder adapter {self.name!r} is not implemented in v0.")
