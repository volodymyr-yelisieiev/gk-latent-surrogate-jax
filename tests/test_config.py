from __future__ import annotations

import pytest

from gk_surrogate.config.load import config_to_yaml, load_config
from gk_surrogate.config.schema import (
    DataConfig,
    DiagnosticHeadConfig,
    EncoderConfig,
    H5SchemaConfig,
    NormalizationConfig,
    WandbConfig,
)


def test_valid_config_loads_and_roundtrips(tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.backend == "synthetic"
    assert config.model.encoder.latent_dim == 32
    dumped = config_to_yaml(config)
    assert "smoke_encoder_supervised" in dumped


def test_invalid_backend_and_missing_h5_schema_rejected():
    with pytest.raises(ValueError):
        DataConfig.model_validate({"backend": "bogus"})
    with pytest.raises(ValueError, match="h5 backend"):
        DataConfig.model_validate({"backend": "h5", "root": "/tmp/data"})
    with pytest.raises(ValueError, match="logging.wandb.project"):
        WandbConfig(project="")
    with pytest.raises(ValueError, match="less than 1"):
        EncoderConfig(dropout_rate=1.0)
    with pytest.raises(ValueError, match="less than 1"):
        DiagnosticHeadConfig(dropout_rate=1.0)
    assert H5SchemaConfig(dtype="float16").dtype == "float16"
    with pytest.raises(ValueError, match="Input should be"):
        H5SchemaConfig(dtype="complex64")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both be scalars or channel lists"):
        NormalizationConfig(mode="fixed", mean=[0.0, 1.0], std=1.0)
    with pytest.raises(ValueError, match="same non-zero length"):
        NormalizationConfig(mode="fixed", mean=[0.0], std=[1.0, 2.0])
    with pytest.raises(ValueError, match="positive finite"):
        NormalizationConfig(mode="fixed", mean=0.0, std=0.0)
    with pytest.raises(ValueError, match="at least one field"):
        DataConfig.model_validate(
            {
                "backend": "synthetic",
                "input_fields": [],
                "synthetic": {
                    "num_trajectories": 1,
                    "timesteps": 1,
                    "channels": 1,
                    "spatial_shape": [1, 1, 1, 1, 1],
                    "flux_dim": 1,
                },
            }
        )
    with pytest.raises(ValueError, match="duplicates"):
        DataConfig.model_validate(
            {
                "backend": "synthetic",
                "target_spectra": ["ky", "ky"],
                "synthetic": {
                    "num_trajectories": 1,
                    "timesteps": 1,
                    "channels": 1,
                    "spatial_shape": [1, 1, 1, 1, 1],
                    "flux_dim": 1,
                },
            }
        )


def test_cli_override_changes_resolved_config(tiny_config_path):
    config = load_config(
        tiny_config_path,
        overrides=["training.max_steps=2", "data.batch_size=3"],
        command="train-encoder",
    )
    assert config.training.max_steps == 2
    assert config.data.batch_size == 3


@pytest.mark.parametrize("override", ("training..max_steps=2", "=2", "training.max_steps="))
def test_cli_override_rejects_ambiguous_paths_and_values(tiny_config_path, override):
    with pytest.raises(ValueError, match="override"):
        load_config(tiny_config_path, overrides=[override], command="train-encoder")


def test_h5_schema_rejects_ambiguous_templates_and_channel_selection():
    with pytest.raises(ValueError, match="exactly one"):
        H5SchemaConfig(timestep_key_template="snapshot")
    with pytest.raises(ValueError, match=r"only \{t\}"):
        H5SchemaConfig(timestep_key_template="snapshot_{index:05d}")
    with pytest.raises(ValueError, match="duplicates"):
        H5SchemaConfig(channel_indices=(0, 0))


def test_synthetic_backend_rejects_unavailable_input_fields():
    with pytest.raises(ValueError, match="supports only"):
        DataConfig.model_validate(
            {
                "backend": "synthetic",
                "input_fields": ["phi"],
                "synthetic": {
                    "num_trajectories": 1,
                    "timesteps": 1,
                    "channels": 1,
                    "spatial_shape": [1, 1, 1, 1, 1],
                    "flux_dim": 1,
                },
            }
        )


def test_wandb_config_is_disabled_by_default_and_serializes(tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.logging.wandb.enabled is False
    dumped = config_to_yaml(config)
    assert "logging:" in dumped
    assert "enabled: false" in dumped


def test_wandb_config_can_be_enabled_by_override(tiny_config_path, tmp_path):
    config = load_config(
        tiny_config_path,
        overrides=[
            "logging.wandb.enabled=true",
            "logging.wandb.mode=offline",
            "logging.wandb.project=unit-test",
            f"logging.wandb.directory={tmp_path}",
            "logging.wandb.tags=[smoke, wandb]",
        ],
        command="train-encoder",
    )
    assert config.logging.wandb.enabled is True
    assert config.logging.wandb.mode == "offline"
    assert config.logging.wandb.project == "unit-test"
    assert config.logging.wandb.directory == str(tmp_path)
    assert config.logging.wandb.tags == ("smoke", "wandb")


def test_invalid_loss_target_combo_rejected(tiny_config_path):
    with pytest.raises(ValueError, match="spectra loss"):
        load_config(
            tiny_config_path,
            overrides=["data.target_spectra=[]", "loss.spectra_weight=1.0"],
            command="train-encoder",
        )


def test_command_specific_validation(tiny_config_path):
    with pytest.raises(ValueError, match="requires model.sequence"):
        load_config(tiny_config_path, command="train-sequence")
    with pytest.raises(ValueError, match="model.simsiam"):
        load_config(
            tiny_config_path,
            overrides=["loss.simsiam_weight=1.0"],
            command="train-encoder",
        )


def test_env_expansion_and_flux_target_validation(repo_root, tiny_config_path, tmp_path, monkeypatch):
    monkeypatch.setenv("GK_LOCAL_PC_DATA_ROOT", str(tmp_path / "raw"))
    config = load_config(repo_root / "configs/data/local_pc_template.yaml", command="inspect-data")
    assert config.data.root == str(tmp_path / "raw")
    assert config.data.h5_schema is not None
    assert config.data.h5_schema.spectra_keys == {"ky": "ky_spectrum", "q": "q_spectrum"}
    with pytest.raises(ValueError, match="flux loss"):
        load_config(
            tiny_config_path,
            overrides=["data.target_flux=false", "loss.flux_weight=1.0"],
            command="train-encoder",
        )
    with pytest.raises(ValueError, match="flux_dim"):
        load_config(
            tiny_config_path,
            overrides=["model.diagnostics.flux_dim=null", "loss.flux_weight=1.0"],
            command="train-encoder",
        )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    with pytest.raises(ValueError, match="latent_cache.path"):
        load_config(
            tiny_config_path,
            overrides=[
                "data.backend=h5",
                f"data.root={raw_root}",
                "data.synthetic=null",
                "data.h5_schema={trajectory_glob: '*.h5'}",
                f"latent_cache.path={raw_root / 'latent_cache.h5'}",
            ],
            command="train-encoder",
        )


def test_unresolved_env_paths_are_rejected(repo_root, monkeypatch):
    monkeypatch.delenv("GK_LOCAL_PC_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="data.root contains unresolved environment variables"):
        load_config(repo_root / "configs/data/local_pc_template.yaml", command="inspect-data")
    monkeypatch.delenv("GK_MISSING_OUTPUT_ROOT", raising=False)
    with pytest.raises(ValueError, match="output_dir contains unresolved environment variables"):
        load_config(
            repo_root / "configs/experiment/smoke_encoder_supervised.yaml",
            overrides=["output_dir=$GK_MISSING_OUTPUT_ROOT/out"],
            command="train-encoder",
        )
    monkeypatch.delenv("GK_CYCLONE_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match=r"data\.root contains unresolved"):
        load_config(
            repo_root / "configs/experiment/server_encoder_simsiam_medium.yaml",
            command="train-encoder",
        )
