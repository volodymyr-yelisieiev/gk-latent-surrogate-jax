"""Latent sequence models for autoregressive surrogate rollouts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from gk_surrogate.models.encoders import _activation

Array = jnp.ndarray


def _ensure_latent_context(z_context: Array, latent_dim: int | None = None) -> None:
    if z_context.ndim != 3:
        raise ValueError(f"Expected z_context with shape [B, T, Z], got {z_context.shape}.")
    if latent_dim is not None and z_context.shape[-1] != latent_dim:
        raise ValueError(f"Expected latent dimension {latent_dim}, got {z_context.shape[-1]}.")
    if z_context.shape[1] <= 0:
        raise ValueError("Latent context must contain at least one timestep.")


def _validate_dropout_rate(dropout_rate: float) -> None:
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must satisfy 0 <= rate < 1, got {dropout_rate}.")


class PersistenceBaseline(nn.Module):
    """Parameter-free baseline: predict the next latent as the last context latent."""

    latent_dim: int

    @nn.compact
    def __call__(self, z_context: Array, *, train: bool) -> Array:
        del train
        _ensure_latent_context(z_context, self.latent_dim)
        return z_context[:, -1, :]


class MLPDeltaSequenceModel(nn.Module):
    """Flatten context, predict a residual delta, and add it to the last latent."""

    latent_dim: int
    context_length: int | None = None
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "gelu"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z_context: Array, *, train: bool) -> Array:
        _ensure_latent_context(z_context, self.latent_dim)
        _validate_dropout_rate(self.dropout_rate)
        if self.context_length is not None and z_context.shape[1] != self.context_length:
            raise ValueError(f"Expected context_length={self.context_length}, got {z_context.shape[1]}.")
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("hidden_dims values must be positive.")
        y = z_context.reshape((z_context.shape[0], -1))
        act = _activation(self.activation)
        for i, hidden_dim in enumerate(self.hidden_dims):
            y = nn.Dense(hidden_dim, name=f"dense_{i}")(y)
            y = act(y)
            if self.dropout_rate:
                y = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{i}")(y, deterministic=not train)
        delta = nn.Dense(self.latent_dim, name="delta")(y)
        return z_context[:, -1, :] + delta


class _GRULayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, inputs: Array) -> tuple[Array, Array]:
        batch_size, timesteps, _ = inputs.shape
        h = jnp.zeros((batch_size, self.hidden_dim), dtype=inputs.dtype)
        gates_layer = nn.Dense(2 * self.hidden_dim, name="gates")
        candidate_layer = nn.Dense(self.hidden_dim, name="candidate")
        outputs = []
        for t in range(timesteps):
            x_t = inputs[:, t, :]
            gates = gates_layer(jnp.concatenate([x_t, h], axis=-1))
            reset, update = jnp.split(jax.nn.sigmoid(gates), 2, axis=-1)
            candidate = candidate_layer(jnp.concatenate([x_t, reset * h], axis=-1))
            candidate = jnp.tanh(candidate)
            h = update * h + (1.0 - update) * candidate
            outputs.append(h)
        return jnp.stack(outputs, axis=1), h


class GRUSequenceModel(nn.Module):
    """Small custom GRU stack for latent one-step prediction."""

    latent_dim: int
    hidden_dim: int = 256
    num_layers: int = 1
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z_context: Array, *, train: bool) -> Array:
        _ensure_latent_context(z_context, self.latent_dim)
        _validate_dropout_rate(self.dropout_rate)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        y = z_context
        h = None
        for i in range(self.num_layers):
            y, h = _GRULayer(self.hidden_dim, name=f"gru_layer_{i}")(y)
            if self.dropout_rate and i < self.num_layers - 1:
                y = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{i}")(y, deterministic=not train)
        if h is None:
            raise ValueError("GRU did not produce a hidden state.")
        return nn.Dense(self.latent_dim, name="output")(h)


class _CausalTransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    dropout_rate: float
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: Array, mask: Array, *, train: bool) -> Array:
        deterministic = not train
        y = nn.LayerNorm(name="attn_norm")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name="attention",
        )(y, y, mask=mask, deterministic=deterministic)
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


class CausalTransformerSequenceModel(nn.Module):
    """Causal Transformer over latent context, predicting the next latent."""

    latent_dim: int
    context_length: int
    embed_dim: int = 128
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    activation: str = "gelu"

    @nn.compact
    def __call__(self, z_context: Array, *, train: bool) -> Array:
        _ensure_latent_context(z_context, self.latent_dim)
        _validate_dropout_rate(self.dropout_rate)
        if z_context.shape[1] != self.context_length:
            raise ValueError(f"Expected context_length={self.context_length}, got {z_context.shape[1]}.")
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

        y = nn.Dense(self.embed_dim, name="input_projection")(z_context)
        pos = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.context_length, self.embed_dim),
        )
        y = y + pos
        if self.dropout_rate:
            y = nn.Dropout(rate=self.dropout_rate, name="pos_dropout")(y, deterministic=not train)

        mask = nn.make_causal_mask(jnp.ones((1, self.context_length), dtype=bool))
        for i in range(self.depth):
            y = _CausalTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                name=f"block_{i}",
            )(y, mask, train=train)

        y = nn.LayerNorm(name="final_norm")(y[:, -1, :])
        return nn.Dense(self.latent_dim, name="output")(y)


class GuppyLatentTransformer(nn.Module):
    """GPT/Guppy-style causal Transformer over continuous latent vectors.

    Latents are up-projected to ``model_dim``, processed over time by causal
    Transformer blocks, then down-projected to the next latent vector.
    """

    latent_dim: int
    context_length: int
    model_dim: int = 128
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    activation: str = "gelu"
    predict_delta: bool = False

    @nn.compact
    def __call__(self, z_context: Array, *, train: bool) -> Array:
        _ensure_latent_context(z_context, self.latent_dim)
        _validate_dropout_rate(self.dropout_rate)
        if z_context.shape[1] != self.context_length:
            raise ValueError(f"Expected context_length={self.context_length}, got {z_context.shape[1]}.")
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        y = nn.Dense(self.model_dim, name="up_projection")(z_context)
        pos = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.context_length, self.model_dim),
        )
        y = y + pos
        if self.dropout_rate:
            y = nn.Dropout(rate=self.dropout_rate, name="pos_dropout")(y, deterministic=not train)

        mask = nn.make_causal_mask(jnp.ones((1, self.context_length), dtype=bool))
        for i in range(self.depth):
            y = _CausalTransformerBlock(
                embed_dim=self.model_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                name=f"block_{i}",
            )(y, mask, train=train)

        y = nn.LayerNorm(name="final_norm")(y[:, -1, :])
        next_latent = nn.Dense(self.latent_dim, name="down_projection")(y)
        if self.predict_delta:
            return z_context[:, -1, :] + next_latent
        return next_latent


class GPT2Adapter(nn.Module):
    """Unsupported external language-model adapter kept behind explicit config opt-in."""

    latent_dim: int

    @nn.compact
    def __call__(self, z_context: Array, *, train: bool) -> Array:
        del z_context, train
        raise NotImplementedError("External language-model adapter is not implemented in v0.")
