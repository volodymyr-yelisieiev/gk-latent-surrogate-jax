"""Factories that build models from validated experiment configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flax import linen as nn

from gk_surrogate.config.schema import (
    DiagnosticHeadConfig,
    EncoderConfig,
    ModelConfig,
    SequenceModelConfig,
    SimSiamConfig,
)
from gk_surrogate.models.diagnostics import DiagnosticHeads, DirectSnapshotDiagnosticBaseline
from gk_surrogate.models.encoders import (
    ConvNDEncoder,
    ExternalEncoderAdapter,
    FlattenMLPEncoder,
    PatchTransformerEncoder,
)
from gk_surrogate.models.full_models import EncoderWithDiagnostics, SimSiamEncoderWithDiagnostics
from gk_surrogate.models.sequence import (
    CausalTransformerSequenceModel,
    GPT2Adapter,
    GRUSequenceModel,
    GuppyLatentTransformer,
    MLPDeltaSequenceModel,
    PersistenceBaseline,
)
from gk_surrogate.models.simsiam import PredictionHead, ProjectionHead, SimSiamModel


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _extra(config: EncoderConfig | SequenceModelConfig) -> Mapping[str, Any]:
    return config.extra or {}


def build_encoder(config: EncoderConfig) -> nn.Module:
    """Build one snapshot encoder from an ``EncoderConfig``."""

    kind = config.type
    extra = _extra(config)
    if kind == "flatten_mlp":
        return FlattenMLPEncoder(
            latent_dim=config.latent_dim,
            hidden_dims=tuple(int(v) for v in config.hidden_dims),
            activation=config.activation,
            dropout_rate=config.dropout_rate,
        )
    if kind == "conv_nd":
        channels = extra.get("channels", config.hidden_dims or (16, 32))
        kernel_size = extra.get("kernel_size", (3, 3, 3, 3, 3))
        strides = extra.get("strides")
        return ConvNDEncoder(
            latent_dim=config.latent_dim,
            channels=tuple(int(v) for v in channels),
            kernel_size=tuple(int(v) for v in _tuple(kernel_size)),
            strides=None if strides is None else tuple(tuple(int(v) for v in row) for row in strides),
            activation=config.activation,
            dropout_rate=config.dropout_rate,
        )
    if kind == "patch_transformer":
        return PatchTransformerEncoder(
            latent_dim=config.latent_dim,
            patch_size=tuple(int(v) for v in extra.get("patch_size", (2, 2, 2, 2, 2))),
            embed_dim=int(extra.get("embed_dim", 64)),
            depth=int(extra.get("depth", 2)),
            num_heads=int(extra.get("num_heads", 4)),
            mlp_ratio=float(extra.get("mlp_ratio", 4.0)),
            dropout_rate=float(extra.get("dropout_rate", config.dropout_rate)),
            attention_dropout_rate=float(extra.get("attention_dropout_rate", 0.0)),
            use_cls_token=bool(extra.get("use_cls_token", False)),
            max_token_count=int(extra.get("max_token_count", 4096)),
            allow_large_token_count=bool(extra.get("allow_large_token_count", False)),
            activation=config.activation,
        )
    if kind == "external_adapter":
        return ExternalEncoderAdapter(
            name=str(extra.get("name", "external")),
            latent_dim=config.latent_dim,
        )
    raise ValueError(f"unknown encoder type: {kind}")


def build_diagnostic_heads(config: DiagnosticHeadConfig) -> DiagnosticHeads | None:
    """Build diagnostic heads, or ``None`` if all diagnostics are disabled."""

    flux_dim = int(config.flux_dim or 0)
    spectra_dims = {str(k): int(v) for k, v in config.spectra_dims.items()}
    if flux_dim <= 0 and not spectra_dims:
        return None
    return DiagnosticHeads(
        flux_dim=flux_dim,
        spectra_dims=spectra_dims,
        hidden_dims=tuple(int(v) for v in config.hidden_dims),
        dropout_rate=config.dropout_rate,
    )


def build_direct_diagnostic_baseline(config: DiagnosticHeadConfig) -> DirectSnapshotDiagnosticBaseline:
    """Build the direct same-time snapshot-to-diagnostics control model."""

    return DirectSnapshotDiagnosticBaseline(
        flux_dim=int(config.flux_dim or 0),
        spectra_dims={str(k): int(v) for k, v in config.spectra_dims.items()},
        hidden_dims=tuple(int(v) for v in config.hidden_dims),
        dropout_rate=config.dropout_rate,
    )


def build_encoder_with_diagnostics(config: ModelConfig) -> EncoderWithDiagnostics:
    return EncoderWithDiagnostics(
        encoder=build_encoder(config.encoder),
        diagnostic_heads=build_diagnostic_heads(config.diagnostics),
    )


def build_simsiam(config: SimSiamConfig, encoder: nn.Module) -> SimSiamModel:
    projection = ProjectionHead(
        output_dim=config.projection_dim,
        hidden_dim=config.projection_hidden_dim,
        num_layers=config.projection_layers,
    )
    prediction = PredictionHead(
        output_dim=config.projection_dim,
        hidden_dim=config.prediction_hidden_dim,
    )
    return SimSiamModel(encoder=encoder, projection_head=projection, prediction_head=prediction)


def build_simsiam_encoder_with_diagnostics(config: ModelConfig) -> SimSiamEncoderWithDiagnostics:
    if config.simsiam is None:
        raise ValueError("model.simsiam is required")
    projection = ProjectionHead(
        output_dim=config.simsiam.projection_dim,
        hidden_dim=config.simsiam.projection_hidden_dim,
        num_layers=config.simsiam.projection_layers,
    )
    prediction = PredictionHead(
        output_dim=config.simsiam.projection_dim,
        hidden_dim=config.simsiam.prediction_hidden_dim,
    )
    return SimSiamEncoderWithDiagnostics(
        encoder=build_encoder(config.encoder),
        projection_head=projection,
        prediction_head=prediction,
        diagnostic_heads=build_diagnostic_heads(config.diagnostics),
    )


def build_sequence_model(config: SequenceModelConfig) -> nn.Module:
    kind = config.type
    extra = _extra(config)
    if kind == "persistence":
        return PersistenceBaseline(latent_dim=config.latent_dim)
    if kind == "mlp_delta":
        return MLPDeltaSequenceModel(
            latent_dim=config.latent_dim,
            context_length=config.context_length,
            hidden_dims=tuple(int(v) for v in config.hidden_dims),
            activation=str(extra.get("activation", "gelu")),
            dropout_rate=float(extra.get("dropout_rate", 0.0)),
        )
    if kind == "gru":
        return GRUSequenceModel(
            latent_dim=config.latent_dim,
            hidden_dim=int(extra.get("hidden_dim", config.hidden_dims[0] if config.hidden_dims else 128)),
            num_layers=int(extra.get("num_layers", 1)),
            dropout_rate=float(extra.get("dropout_rate", 0.0)),
        )
    if kind == "causal_transformer":
        return CausalTransformerSequenceModel(
            latent_dim=config.latent_dim,
            context_length=config.context_length,
            embed_dim=int(extra.get("embed_dim", 128)),
            depth=int(extra.get("depth", 2)),
            num_heads=int(extra.get("num_heads", 4)),
            mlp_ratio=float(extra.get("mlp_ratio", 4.0)),
            dropout_rate=float(extra.get("dropout_rate", 0.0)),
            activation=str(extra.get("activation", "gelu")),
        )
    if kind in {"guppy_latent_transformer", "gpt_latent_transformer"}:
        return GuppyLatentTransformer(
            latent_dim=config.latent_dim,
            context_length=config.context_length,
            model_dim=int(extra.get("model_dim", extra.get("embed_dim", 128))),
            depth=int(extra.get("depth", 2)),
            num_heads=int(extra.get("num_heads", 4)),
            mlp_ratio=float(extra.get("mlp_ratio", 4.0)),
            dropout_rate=float(extra.get("dropout_rate", 0.0)),
            activation=str(extra.get("activation", "gelu")),
            predict_delta=bool(extra.get("predict_delta", False)),
        )
    if kind == "gpt2_adapter":
        return GPT2Adapter(latent_dim=config.latent_dim)
    raise ValueError(f"unknown sequence model type: {kind}")
