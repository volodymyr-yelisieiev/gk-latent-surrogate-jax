from __future__ import annotations

import os
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gk_surrogate.config.schema import ExperimentConfig


@pytest.fixture(autouse=True)
def _cpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JAX_PLATFORM_NAME", os.environ["JAX_PLATFORM_NAME"])


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "experiment" / "smoke_encoder_supervised.yaml"


@pytest.fixture
def params_changed():
    return _params_changed


def _params_changed(before, after) -> bool:
    leaves_before = jax.tree_util.tree_leaves(before)
    leaves_after = jax.tree_util.tree_leaves(after)
    return any(not bool(jnp.allclose(a, b)) for a, b in zip(leaves_before, leaves_after, strict=True))


def with_output(config: ExperimentConfig, output_dir: Path, *, max_steps: int | None = None) -> ExperimentConfig:
    update: dict[str, object] = {"output_dir": str(output_dir)}
    if max_steps is not None:
        update["training"] = config.training.model_copy(update={"max_steps": max_steps})
    return config.model_copy(update=update)
