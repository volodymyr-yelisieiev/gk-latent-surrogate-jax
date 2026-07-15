"""Validated configuration schema for data and bootstrap CLI workflows."""

from __future__ import annotations

import math
from string import Formatter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NormalizationConfig(StrictModel):
    mode: Literal["none", "sample", "trajectory", "dataset", "fixed"] = "none"
    mean: float | list[float] | None = None
    std: float | list[float] | None = None
    max_samples: int = Field(default=128, gt=0)

    @model_validator(mode="after")
    def validate_fixed_stats(self) -> NormalizationConfig:
        if self.mode == "fixed" and (self.mean is None or self.std is None):
            msg = "fixed normalization requires mean and std"
            raise ValueError(msg)
        if self.mode == "fixed":
            assert self.mean is not None and self.std is not None
            if isinstance(self.mean, list) != isinstance(self.std, list):
                msg = "fixed normalization mean and std must both be scalars or channel lists"
                raise ValueError(msg)
            if isinstance(self.mean, list):
                assert isinstance(self.std, list)
                mean_values = self.mean
                std_values = self.std
            else:
                assert not isinstance(self.std, list)
                mean_values = [float(self.mean)]
                std_values = [float(self.std)]
            if not mean_values or len(mean_values) != len(std_values):
                msg = "fixed normalization channel mean and std must have the same non-zero length"
                raise ValueError(msg)
            if any(not math.isfinite(value) for value in mean_values):
                msg = "fixed normalization mean must contain finite values"
                raise ValueError(msg)
            if any(not math.isfinite(value) or value <= 0.0 for value in std_values):
                msg = "fixed normalization std must contain positive finite values"
                raise ValueError(msg)
        return self


class AugmentationConfig(StrictModel):
    gaussian_noise_std: float = Field(default=0.0, ge=0.0)
    amplitude_jitter_std: float = Field(default=0.0, ge=0.0)
    mask_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    channel_dropout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    periodic_shift: bool = False
    max_periodic_shift: int = Field(default=0, ge=0)


class SyntheticDataConfig(StrictModel):
    num_trajectories: int = Field(gt=0)
    timesteps: int = Field(gt=0)
    channels: int = Field(gt=0)
    spatial_shape: tuple[int, int, int, int, int]
    flux_dim: int = Field(gt=0)
    spectra_dims: dict[str, int] = Field(default_factory=dict)

    @field_validator("spatial_shape")
    @classmethod
    def spatial_shape_positive(cls, value: tuple[int, int, int, int, int]) -> tuple[int, ...]:
        if any(dim <= 0 for dim in value):
            msg = "all synthetic spatial dimensions must be positive"
            raise ValueError(msg)
        return value


class H5SchemaConfig(StrictModel):
    trajectory_glob: str = "*.h5"
    data_group: str = "data"
    timestep_key_template: str = "timestep_{t:05d}"
    phi_key_template: str | None = None
    metadata_group: str = "metadata"
    flux_key: str | None = None
    timestep_key: str | None = None
    spectra_keys: dict[str, str] = Field(default_factory=dict)
    geometry_group: str | None = None
    channel_indices: tuple[int, ...] | None = None
    dtype: Literal["float16", "float32", "float64"] = "float32"

    @field_validator("channel_indices")
    @classmethod
    def channel_indices_non_negative(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None and any(index < 0 for index in value):
            msg = "channel_indices must be non-negative"
            raise ValueError(msg)
        if value is not None and len(set(value)) != len(value):
            msg = "channel_indices must not contain duplicates"
            raise ValueError(msg)
        return value

    @field_validator("timestep_key_template", "phi_key_template")
    @classmethod
    def timestep_templates_require_t(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            fields = [field for _, field, _, _ in Formatter().parse(value) if field is not None]
            value.format(t=0)
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            msg = "HDF5 timestep templates must be valid format strings using only {t}"
            raise ValueError(msg) from exc
        if fields != ["t"]:
            msg = "HDF5 timestep templates must contain exactly one {t} field"
            raise ValueError(msg)
        return value


class CycloneKvikIOConfig(StrictModel):
    trajectories: tuple[str, ...] | None = None
    fields_to_load: tuple[str, ...] = ("df",)
    conditions: tuple[str, ...] = ("itg", "dg", "s_hat", "q")
    normalization: str | None = None
    normalization_scope: Literal["dataset", "trajectory", "sample", "none"] = "dataset"
    normalization_stats: str | None = None
    spatial_ifft: bool = True
    real_potens: bool = True
    bundle_seq_length: int = Field(default=1, ge=1)
    offset: int = Field(default=0, ge=0)
    tail_offset: int = Field(default=0, ge=0)
    subsample: int = Field(default=1, ge=1)
    separate_zf: bool = True
    decouple_mu: bool = False
    prefer_dtype: Literal["float32"] = "float32"
    use_kvikio: bool = False
    return_jax: bool = False

    @field_validator("fields_to_load", "conditions")
    @classmethod
    def non_empty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item for item in value):
            msg = "cyclone fields and conditions must be non-empty strings"
            raise ValueError(msg)
        return value


class DataConfig(StrictModel):
    backend: Literal["synthetic", "h5", "cyclone_kvikio"]
    root: str | None = None
    split: Literal["train", "val", "test", "all"] = "train"
    input_fields: tuple[str, ...] = ("df",)
    target_flux: bool = True
    target_spectra: tuple[str, ...] = ()
    context_length: int = Field(default=1, ge=1)
    prediction_length: int = Field(default=1, ge=1)
    batch_size: int = Field(default=1, gt=0)
    shuffle: bool = True
    num_workers: int = Field(default=0, ge=0)
    seed: int = 42
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    h5_schema: H5SchemaConfig | None = None
    cyclone: CycloneKvikIOConfig | None = None
    synthetic: SyntheticDataConfig | None = None
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @field_validator("input_fields", "target_spectra")
    @classmethod
    def unique_non_empty_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value):
            msg = "data field and target names must be non-empty"
            raise ValueError(msg)
        if len(set(value)) != len(value):
            msg = "data field and target names must not contain duplicates"
            raise ValueError(msg)
        return value

    @field_validator("input_fields")
    @classmethod
    def input_fields_required(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            msg = "data.input_fields must contain at least one field"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_backend_payload(self) -> DataConfig:
        if self.backend == "synthetic" and self.synthetic is None:
            msg = "synthetic backend requires data.synthetic"
            raise ValueError(msg)
        if self.backend == "synthetic" and self.input_fields != ("df",):
            msg = "synthetic backend supports only data.input_fields=[df]"
            raise ValueError(msg)
        if self.backend == "h5" and (self.root is None or self.h5_schema is None):
            msg = "h5 backend requires data.root and data.h5_schema"
            raise ValueError(msg)
        if self.backend == "cyclone_kvikio" and (self.root is None or self.cyclone is None):
            msg = "cyclone_kvikio backend requires data.root and data.cyclone"
            raise ValueError(msg)
        return self


class EncoderConfig(StrictModel):
    type: str = "flatten_mlp"
    latent_dim: int = Field(default=64, gt=0)
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "gelu"
    dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    extra: dict[str, Any] = Field(default_factory=dict)


class DiagnosticHeadConfig(StrictModel):
    flux_dim: int | None = Field(default=1, gt=0)
    spectra_dims: dict[str, int] = Field(default_factory=dict)
    hidden_dims: tuple[int, ...] = (128,)
    dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)


class SimSiamConfig(StrictModel):
    projection_dim: int = Field(default=128, gt=0)
    projection_hidden_dim: int = Field(default=256, gt=0)
    projection_layers: int = Field(default=2, ge=1)
    prediction_hidden_dim: int = Field(default=64, gt=0)


class SequenceModelConfig(StrictModel):
    type: str = "mlp_delta"
    latent_dim: int = Field(default=64, gt=0)
    context_length: int = Field(default=4, ge=1)
    hidden_dims: tuple[int, ...] = (256, 256)
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(StrictModel):
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    diagnostics: DiagnosticHeadConfig = Field(default_factory=DiagnosticHeadConfig)
    simsiam: SimSiamConfig | None = None
    sequence: SequenceModelConfig | None = None


class TrainingConfig(StrictModel):
    max_steps: int = Field(default=100, ge=0)
    epochs: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    log_every: int = Field(default=10, ge=1)
    eval_every: int = Field(default=50, ge=1)
    checkpoint_every: int = Field(default=100, ge=1)
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    jit: bool = True
    seed: int = 42


class LossConfig(StrictModel):
    simsiam_weight: float = Field(default=0.0, ge=0.0)
    flux_weight: float = Field(default=1.0, ge=0.0)
    spectra_weight: float = Field(default=1.0, ge=0.0)
    latent_weight: float = Field(default=1.0, ge=0.0)
    use_log_spectra: bool = False
    spectra_epsilon: float = Field(default=1e-6, gt=0.0)
    latent_loss: Literal["mse", "huber", "cosine", "mse_plus_cosine"] = "mse"


class EvaluationConfig(StrictModel):
    rollout_steps: int = Field(default=4, ge=1)
    batch_size: int | None = Field(default=None, gt=0)
    metrics: tuple[str, ...] = ("latent_mse", "flux_mse", "spectra_mse")
    flux_head_ridge_alpha: float = Field(default=1e-3, ge=0.0)
    tsne_perplexities: tuple[float, ...] = (5.0, 30.0)
    tsne_max_iter: int = Field(default=1000, ge=250)
    representation_max_points: int | None = Field(default=2000, gt=0)


class LatentCacheConfig(StrictModel):
    path: str | None = None
    encoder_checkpoint_path: str | None = None
    sequence_checkpoint_path: str | None = None
    use_persistence_baseline: bool = False
    latent_normalization: Literal["none", "cache"] = "none"
    latent_normalization_split: Literal["train", "selected", "all"] = "train"
    latent_normalization_epsilon: float = Field(default=1e-6, gt=0.0)


class WandbConfig(StrictModel):
    enabled: bool = False
    project: str = "gk-latent-surrogate"
    entity: str | None = None
    mode: Literal["disabled", "offline", "online"] = "disabled"
    group: str | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()
    log_artifacts: bool = False
    directory: str | None = None

    @field_validator("project")
    @classmethod
    def project_non_empty(cls, value: str) -> str:
        if not value:
            msg = "logging.wandb.project must be non-empty"
            raise ValueError(msg)
        return value


class LoggingConfig(StrictModel):
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class ParallelConfig(StrictModel):
    mode: Literal["auto", "single", "pmap"] = "auto"
    axis_name: str = "devices"
    min_devices: int = Field(default=1, ge=1)
    require_all_visible_devices: bool = False
    drop_remainder: bool = True
    auto_scale_learning_rate: bool = False
    log_device_summary: bool = True

    @field_validator("axis_name")
    @classmethod
    def axis_name_non_empty(cls, value: str) -> str:
        if not value:
            msg = "parallel.axis_name must be non-empty"
            raise ValueError(msg)
        return value


class ExperimentConfig(StrictModel):
    name: str = "unnamed"
    output_dir: str = "outputs/default"
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    latent_cache: LatentCacheConfig = Field(default_factory=LatentCacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
