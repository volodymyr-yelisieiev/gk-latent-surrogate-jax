"""Diagnostic prediction heads from latent vectors."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from gk_surrogate.models.encoders import _activation, _validate_dropout_rate

Array = jnp.ndarray


@struct.dataclass
class DiagnosticPredictions:
    """Predicted physics diagnostics from a latent batch."""

    flux: Array | None
    spectra: Mapping[str, Array]


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


class DiagnosticHeads(nn.Module):
    """Shared MLP trunk with separate flux and spectra outputs."""

    flux_dim: int = 0
    spectra_dims: Mapping[str, int] = struct.field(default_factory=dict)
    hidden_dims: tuple[int, ...] = (128,)
    dropout_rate: float = 0.0
    activation: str = "gelu"

    @nn.compact
    def __call__(self, z: Array, *, train: bool) -> DiagnosticPredictions:
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape [B, Z], got shape {z.shape}.")
        if self.flux_dim < 0:
            raise ValueError("flux_dim must be non-negative.")
        _validate_dropout_rate(self.dropout_rate)
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("hidden_dims values must be positive.")

        y = z
        act = _activation(self.activation)
        for i, hidden_dim in enumerate(self.hidden_dims):
            y = nn.Dense(hidden_dim, name=f"trunk_dense_{i}")(y)
            y = act(y)
            if self.dropout_rate:
                y = nn.Dropout(rate=self.dropout_rate, name=f"trunk_dropout_{i}")(y, deterministic=not train)

        flux = None
        if self.flux_dim > 0:
            flux = nn.Dense(self.flux_dim, name="flux")(y)

        spectra: dict[str, Array] = {}
        module_names: set[str] = set()
        for key, dim in sorted(self.spectra_dims.items()):
            if dim <= 0:
                raise ValueError(f"Spectra dimension for {key!r} must be positive.")
            module_name = f"spectra_{_safe_name(key)}"
            if module_name in module_names:
                raise ValueError("Spectra keys must remain unique after conversion to Flax module names.")
            module_names.add(module_name)
            spectra[key] = nn.Dense(dim, name=module_name)(y)

        return DiagnosticPredictions(flux=flux, spectra=spectra)


class DirectSnapshotDiagnosticBaseline(nn.Module):
    """Predict same-time diagnostics directly from channel-first 5D snapshots.

    This intentionally has no learned latent bottleneck or temporal context.
    Per-channel spatial mean, RMS, and maximum magnitude keep the parameter
    count independent of the five-dimensional grid size, so the control stays
    feasible for server snapshots. It distinguishes same-time diagnostic
    learnability from representation and rollout quality.
    """

    flux_dim: int = 0
    spectra_dims: Mapping[str, int] = struct.field(default_factory=dict)
    hidden_dims: tuple[int, ...] = (128,)
    dropout_rate: float = 0.0
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> DiagnosticPredictions:
        if x.ndim != 7:
            raise ValueError(
                "Expected channel-first snapshots with shape [B, C, S1, S2, S3, S4, S5], "
                f"got shape {x.shape}."
            )
        spatial_axes = tuple(range(2, x.ndim))
        channel_mean = jnp.mean(x, axis=spatial_axes)
        channel_rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=spatial_axes))
        channel_max_magnitude = jnp.max(jnp.abs(x), axis=spatial_axes)
        features = jnp.concatenate((channel_mean, channel_rms, channel_max_magnitude), axis=-1)
        return DiagnosticHeads(
            flux_dim=self.flux_dim,
            spectra_dims=self.spectra_dims,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            name="direct_diagnostic_heads",
        )(features, train=train)
