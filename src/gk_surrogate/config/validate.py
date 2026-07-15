"""Cross-field validation beyond the Pydantic field schema."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

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
    if command == "train-encoder" and config.loss.simsiam_weight > 0 and config.model.simsiam is None:
        msg = "SimSiam encoder training requires model.simsiam"
        raise ValueError(msg)

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
