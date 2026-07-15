"""YAML config loading and dotted-key overrides."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from gk_surrogate.config.schema import ExperimentConfig
from gk_surrogate.config.validate import validate_config


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
    command: str | None = None,
) -> ExperimentConfig:
    """Load, merge, validate, and return an experiment config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "data" in raw and "name" not in raw:
        raw = _wrap_data_config(raw)

    merged = deepcopy(raw)
    for override in overrides or []:
        _apply_override(merged, override)
    merged = _expand_env_vars(merged)

    config = ExperimentConfig.model_validate(merged)
    validate_config(config, command=command)
    return config


def config_to_yaml(config: ExperimentConfig) -> str:
    """Serialize a resolved config with stable key ordering disabled for readability."""

    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)


def _wrap_data_config(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": raw.get("name", "data_only"),
        "output_dir": raw.get("output_dir", "outputs/data_only"),
        "data": raw["data"],
        "model": raw.get("model", {}),
        "training": raw.get("training", {}),
        "loss": raw.get(
            "loss",
            {
                "simsiam_weight": 0.0,
                "flux_weight": 0.0,
                "spectra_weight": 0.0,
                "latent_weight": 0.0,
                "use_log_spectra": False,
                "spectra_epsilon": 1e-6,
                "latent_loss": "mse",
            },
        ),
        "evaluation": raw.get("evaluation", {}),
    }


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        msg = f"override must use key=value syntax: {override}"
        raise ValueError(msg)
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        next_value = cursor.setdefault(key, {})
        if not isinstance(next_value, dict):
            msg = f"cannot override nested key below non-mapping: {'.'.join(keys[:-1])}"
            raise ValueError(msg)
        cursor = next_value
    cursor[keys[-1]] = yaml.safe_load(raw_value)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value
