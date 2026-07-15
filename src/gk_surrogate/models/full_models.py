"""Composed encoder models."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from gk_surrogate.models.diagnostics import DiagnosticHeads, DiagnosticPredictions
from gk_surrogate.models.simsiam import PredictionHead, ProjectionHead

Array = jnp.ndarray


@struct.dataclass
class EncoderOutput:
    z: Array
    diagnostics: DiagnosticPredictions | None


@struct.dataclass
class EncoderTrainingOutput:
    z: Array
    diagnostics: DiagnosticPredictions | None
    projection: Array | None = None
    prediction: Array | None = None


class EncoderWithDiagnostics(nn.Module):
    encoder: nn.Module
    diagnostic_heads: DiagnosticHeads | None = None

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> EncoderOutput:
        z = self.encoder(x, train=train)
        diagnostics = self.diagnostic_heads(z, train=train) if self.diagnostic_heads is not None else None
        return EncoderOutput(z=z, diagnostics=diagnostics)


class SimSiamEncoderWithDiagnostics(nn.Module):
    """Single-view encoder with SimSiam projection/prediction heads."""

    encoder: nn.Module
    projection_head: ProjectionHead
    prediction_head: PredictionHead
    diagnostic_heads: DiagnosticHeads | None = None

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> EncoderTrainingOutput:
        z = self.encoder(x, train=train)
        projection = self.projection_head(z, train=train)
        prediction = self.prediction_head(projection, train=train)
        diagnostics = self.diagnostic_heads(z, train=train) if self.diagnostic_heads is not None else None
        return EncoderTrainingOutput(
            z=z,
            diagnostics=diagnostics,
            projection=projection,
            prediction=prediction,
        )
