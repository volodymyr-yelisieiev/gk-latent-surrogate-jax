"""Model components for latent gyrokinetic surrogates."""

from gk_surrogate.models.diagnostics import DiagnosticHeads, DiagnosticPredictions
from gk_surrogate.models.encoders import (
    ConvNDEncoder,
    ExternalEncoderAdapter,
    FlattenMLPEncoder,
    PatchTransformerEncoder,
)
from gk_surrogate.models.full_models import EncoderOutput, EncoderWithDiagnostics
from gk_surrogate.models.sequence import (
    CausalTransformerSequenceModel,
    GPT2Adapter,
    GRUSequenceModel,
    GuppyLatentTransformer,
    MLPDeltaSequenceModel,
    PersistenceBaseline,
)
from gk_surrogate.models.simsiam import (
    PredictionHead,
    ProjectionHead,
    SimSiamModel,
    SimSiamOutput,
)

__all__ = [
    "CausalTransformerSequenceModel",
    "ConvNDEncoder",
    "DiagnosticHeads",
    "DiagnosticPredictions",
    "EncoderOutput",
    "EncoderWithDiagnostics",
    "ExternalEncoderAdapter",
    "FlattenMLPEncoder",
    "GPT2Adapter",
    "GRUSequenceModel",
    "GuppyLatentTransformer",
    "MLPDeltaSequenceModel",
    "PatchTransformerEncoder",
    "PersistenceBaseline",
    "PredictionHead",
    "ProjectionHead",
    "SimSiamModel",
    "SimSiamOutput",
]
