from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from gk_surrogate.config.load import config_to_yaml, load_config
from gk_surrogate.config.schema import DataConfig, NormalizationConfig
from gk_surrogate.data.latent_cache import LatentCacheWriter


@pytest.mark.parametrize(
    "override",
    (
        "training.epochs=1",
        "training.jit=false",
        "training.dtype=float16",
        "data.num_workers=1",
        "evaluation.batch_size=8",
    ),
)
def test_removed_no_op_config_knobs_are_rejected(tiny_config_path: Path, override: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(tiny_config_path, overrides=[override], command="train-encoder")


def test_resolved_config_has_one_batch_size_and_no_execution_dtype(tiny_config_path: Path) -> None:
    dumped = config_to_yaml(load_config(tiny_config_path, command="train-encoder"))
    assert dumped.count("batch_size:") == 1
    assert "num_workers:" not in dumped
    assert "epochs:" not in dumped
    assert "jit:" not in dumped
    assert "dtype:" not in dumped


def test_sequence_contract_rejects_context_and_latent_dimension_mismatches(tiny_config_path: Path) -> None:
    sequence = "{type: mlp_delta, latent_dim: 32, context_length: 3, hidden_dims: [16], extra: {}}"
    with pytest.raises(ValueError, match="sequence.context_length"):
        load_config(
            tiny_config_path,
            overrides=[f"model.sequence={sequence}", "latent_cache.path=missing.h5"],
            command="train-sequence",
        )

    sequence = "{type: mlp_delta, latent_dim: 16, context_length: 4, hidden_dims: [16], extra: {}}"
    with pytest.raises(ValueError, match="sequence.latent_dim"):
        load_config(
            tiny_config_path,
            overrides=[f"model.sequence={sequence}", "latent_cache.path=missing.h5"],
            command="train-sequence",
        )


def test_synthetic_diagnostic_dimensions_are_cross_checked(tiny_config_path: Path) -> None:
    with pytest.raises(ValueError, match="synthetic.flux_dim"):
        load_config(
            tiny_config_path,
            overrides=["data.synthetic.flux_dim=2"],
            command="train-encoder",
        )
    with pytest.raises(ValueError, match=r"spectra_dims\['ky'\]"):
        load_config(
            tiny_config_path,
            overrides=["data.synthetic.spectra_dims.ky=7"],
            command="train-encoder",
        )
    with pytest.raises(ValueError, match="missing requested target"):
        load_config(
            tiny_config_path,
            overrides=["data.synthetic.spectra_dims={q: 8}"],
            command="train-encoder",
        )


def test_synthetic_sequence_window_must_fit_trajectory(tiny_config_path: Path) -> None:
    with pytest.raises(ValueError, match="timesteps must cover"):
        load_config(
            tiny_config_path,
            overrides=["data.synthetic.timesteps=4"],
            command="train-encoder",
        )


def test_existing_latent_cache_dimension_is_cross_checked(tiny_config_path: Path, tmp_path: Path) -> None:
    cache_path = tmp_path / "latent_cache.h5"
    LatentCacheWriter(cache_path, latent_dim=7)
    sequence = "{type: mlp_delta, latent_dim: 32, context_length: 4, hidden_dims: [16], extra: {}}"
    with pytest.raises(ValueError, match="latent cache dimension"):
        load_config(
            tiny_config_path,
            overrides=[f"model.sequence={sequence}", f"latent_cache.path={cache_path}"],
            command="train-sequence",
        )


@pytest.mark.parametrize("command", ("evaluate-flux-head", "plot-representation"))
def test_diagnostic_commands_require_a_latent_cache(tiny_config_path: Path, command: str) -> None:
    with pytest.raises(ValueError, match="requires latent_cache.path"):
        load_config(
            tiny_config_path,
            overrides=["latent_cache.path=null"],
            command=command,
        )


@pytest.mark.parametrize("command", ("evaluate-flux-head", "plot-representation"))
def test_diagnostic_commands_require_flux_targets(tiny_config_path: Path, command: str) -> None:
    with pytest.raises(ValueError, match="requires data.target_flux"):
        load_config(
            tiny_config_path,
            overrides=[
                "loss.flux_weight=0",
                "data.target_flux=false",
                "latent_cache.path=missing.h5",
            ],
            command=command,
        )


def test_invalid_existing_latent_cache_metadata_is_rejected(
    tiny_config_path: Path,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "invalid_cache.h5"
    with h5py.File(cache_path, "w"):
        pass
    with pytest.raises(ValueError, match="latent cache metadata is invalid"):
        load_config(
            tiny_config_path,
            overrides=[f"latent_cache.path={cache_path}"],
            command="evaluate-flux-head",
        )


def test_schema_rejects_nonfinite_fixed_mean_and_missing_synthetic_payload() -> None:
    with pytest.raises(ValueError, match="mean must contain finite"):
        NormalizationConfig(mode="fixed", mean=float("nan"), std=1.0)
    with pytest.raises(ValueError, match="synthetic backend requires"):
        DataConfig(backend="synthetic", synthetic=None)
