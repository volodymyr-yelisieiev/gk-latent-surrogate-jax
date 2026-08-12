"""Cross-field validation beyond the Pydantic field schema."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py

from gk_surrogate.config.schema import ExperimentConfig

_UNRESOLVED_ENV_RE = re.compile(r"\$(?:\{[^}]+\}|\([^)]+\)|[A-Za-z_][A-Za-z0-9_]*)")


def validate_config(config: ExperimentConfig, *, command: str | None = None) -> None:
    """Reject scientifically unsafe or command-incompatible configs."""

    data = config.data
    _reject_unresolved_paths(config)
    if config.loss.flux_weight > 0 and not data.target_flux:
        msg = "flux loss is enabled but data.target_flux is false"
        raise ValueError(msg)
    if config.loss.flux_weight > 0 and not config.model.diagnostics.flux_dim:
        msg = "flux loss is enabled but model.diagnostics.flux_dim is disabled"
        raise ValueError(msg)
    if config.loss.spectra_weight > 0 and not data.target_spectra:
        msg = "spectra loss is enabled but data.target_spectra is empty"
        raise ValueError(msg)
    if config.loss.simsiam_weight > 0 and config.model.simsiam is None:
        msg = "simsiam loss is enabled but model.simsiam is missing"
        raise ValueError(msg)
    if command == "train-sequence" and config.model.sequence is None:
        msg = "train-sequence requires model.sequence"
        raise ValueError(msg)
    if command == "train-sequence" and not config.latent_cache.path:
        msg = "train-sequence requires latent_cache.path"
        raise ValueError(msg)
    if command == "evaluate-rollout" and not config.latent_cache.path:
        msg = "evaluate-rollout requires latent_cache.path"
        raise ValueError(msg)
    if command == "evaluate-flux-head" and not config.latent_cache.path:
        msg = "evaluate-flux-head requires latent_cache.path"
        raise ValueError(msg)
    if command == "evaluate-flux-head" and not data.target_flux:
        msg = "evaluate-flux-head requires data.target_flux"
        raise ValueError(msg)
    if command == "plot-representation" and not config.latent_cache.path:
        msg = "plot-representation requires latent_cache.path"
        raise ValueError(msg)
    if command == "plot-representation" and not data.target_flux:
        msg = "plot-representation requires data.target_flux"
        raise ValueError(msg)
    if (
        command == "evaluate-rollout"
        and not config.latent_cache.sequence_checkpoint_path
        and not config.latent_cache.use_persistence_baseline
    ):
        msg = "evaluate-rollout requires latent_cache.sequence_checkpoint_path or persistence baseline"
        raise ValueError(msg)
    _validate_model_data_contract(config, command=command)
    _validate_existing_latent_cache(config, command=command)

    if data.backend in {"h5", "cyclone_kvikio"} and data.root is not None:
        output_dir = Path(config.output_dir).expanduser().resolve()
        root = Path(data.root).expanduser().resolve()
        if _is_relative_to(output_dir, root):
            msg = "output_dir must not be inside raw dataset root"
            raise ValueError(msg)
        if config.latent_cache.path:
            latent_cache_path = Path(config.latent_cache.path).expanduser().resolve()
            if _is_relative_to(latent_cache_path, root):
                msg = "latent_cache.path must not be inside raw dataset root"
                raise ValueError(msg)

    diagnostic_spectra = set(config.model.diagnostics.spectra_dims)
    requested_spectra = set(data.target_spectra)
    missing_heads = requested_spectra - diagnostic_spectra
    if config.loss.spectra_weight > 0 and missing_heads:
        missing = ", ".join(sorted(missing_heads))
        msg = f"diagnostic spectra heads missing for requested targets: {missing}"
        raise ValueError(msg)


def _validate_model_data_contract(config: ExperimentConfig, *, command: str | None) -> None:
    data = config.data
    sequence = config.model.sequence
    sequence_commands = {"train-sequence", "evaluate-rollout"}
    if command in sequence_commands and sequence is not None:
        if sequence.context_length != data.context_length:
            msg = (
                "model.sequence.context_length must equal data.context_length "
                f"({sequence.context_length} != {data.context_length})"
            )
            raise ValueError(msg)
        if sequence.latent_dim != config.model.encoder.latent_dim:
            msg = (
                "model.sequence.latent_dim must equal model.encoder.latent_dim "
                f"({sequence.latent_dim} != {config.model.encoder.latent_dim})"
            )
            raise ValueError(msg)
    synthetic = data.synthetic
    if data.backend != "synthetic" or synthetic is None:
        return
    if synthetic.timesteps < data.context_length + data.prediction_length:
        msg = "synthetic timesteps must cover data.context_length + data.prediction_length"
        raise ValueError(msg)
    diagnostics = config.model.diagnostics
    if data.target_flux and diagnostics.flux_dim is not None and diagnostics.flux_dim != synthetic.flux_dim:
        msg = (
            "model.diagnostics.flux_dim must match data.synthetic.flux_dim "
            f"({diagnostics.flux_dim} != {synthetic.flux_dim})"
        )
        raise ValueError(msg)
    for name in data.target_spectra:
        source_dim = synthetic.spectra_dims.get(name)
        head_dim = diagnostics.spectra_dims.get(name)
        if source_dim is None:
            raise ValueError(f"data.synthetic.spectra_dims is missing requested target {name!r}")
        if head_dim is not None and head_dim != source_dim:
            msg = (
                f"model.diagnostics.spectra_dims[{name!r}] must match "
                f"data.synthetic.spectra_dims[{name!r}] ({head_dim} != {source_dim})"
            )
            raise ValueError(msg)


def _validate_existing_latent_cache(config: ExperimentConfig, *, command: str | None) -> None:
    if command not in {"train-sequence", "evaluate-rollout", "evaluate-flux-head", "plot-representation"}:
        return
    cache_path = config.latent_cache.path
    if not cache_path or not Path(cache_path).is_file():
        return
    try:
        with h5py.File(cache_path, "r") as handle:
            cache_latent_dim = int(handle["metadata"].attrs["latent_dim"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"latent cache metadata is invalid: {cache_path}") from exc
    expected = (
        config.model.sequence.latent_dim
        if config.model.sequence is not None
        else config.model.encoder.latent_dim
    )
    if cache_latent_dim != expected:
        msg = f"latent cache dimension must match configured latent dimension ({cache_latent_dim} != {expected})"
        raise ValueError(msg)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_unresolved_paths(config: ExperimentConfig) -> None:
    for name, value in _string_leaves(config.model_dump(mode="json")):
        if _UNRESOLVED_ENV_RE.search(value):
            msg = f"{name} contains unresolved environment variables: {value}"
            raise ValueError(msg)


def _string_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _string_leaves(item, name)
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        for index, item in enumerate(value):
            yield from _string_leaves(item, f"{prefix}[{index}]")
