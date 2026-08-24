from __future__ import annotations

from pathlib import Path

import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import DataConfig


def test_cyclone_template_config_expands_env_and_keeps_spectra_disabled(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "cyclone_root"
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", str(root))

    config = load_config(repo_root / "configs/data/cyclone_kvikio_template.yaml", command="inspect-data")

    assert config.data.backend == "cyclone_kvikio"
    assert config.data.root == str(root)
    assert config.data.target_flux is True
    assert config.data.target_spectra == ()
    assert config.data.cyclone is not None
    assert config.data.cyclone.prefer_dtype == "float32"
    assert config.data.cyclone.use_kvikio is False
    assert config.data.cyclone.return_jax is False


def test_cyclone_real_smoke_config_uses_confirmed_stored_spectra_and_tiny_subset(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", str(tmp_path / "raw"))

    config = load_config(repo_root / "configs/experiment/smoke_real_encoder_flux.yaml", command="train-encoder")

    assert config.data.backend == "cyclone_kvikio"
    assert config.data.split == "train"
    assert config.data.batch_size == 1
    assert config.data.target_spectra == ("kyspec", "fluxspec")
    assert config.data.cyclone is not None
    assert config.data.cyclone.trajectories == ("iteration_0",)
    assert config.data.cyclone.offset == 80
    assert config.data.cyclone.subsample == 32
    assert config.model.encoder.type == "conv_nd"
    assert config.model.diagnostics.flux_dim == 1
    assert config.model.diagnostics.spectra_dims == {"kyspec": 32, "fluxspec": 32}
    assert config.loss.flux_weight == 1.0
    assert config.loss.spectra_weight == 1.0
    assert config.training.max_steps == 2


@pytest.mark.parametrize(
    ("config_name", "command"),
    [
        ("smoke_real_embed_dataset.yaml", "embed-dataset"),
        ("smoke_real_sequence.yaml", "train-sequence"),
        ("smoke_real_evaluate_rollout.yaml", "evaluate-rollout"),
    ],
)
def test_cyclone_real_pipeline_smoke_configs_validate(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    config_name: str,
    command: str,
) -> None:
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", str(tmp_path / "raw"))

    config = load_config(repo_root / "configs/experiment" / config_name, command=command)

    assert config.data.backend == "cyclone_kvikio"
    assert config.data.target_flux is True
    assert config.data.target_spectra == ("kyspec", "fluxspec")
    assert config.loss.spectra_weight == 1.0
    assert config.latent_cache.path is not None


def test_cyclone_config_rejects_missing_payload_and_outputs_inside_raw_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    with pytest.raises(ValueError, match="cyclone_kvikio backend"):
        DataConfig.model_validate({"backend": "cyclone_kvikio", "root": str(raw_root)})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: bad_cyclone_paths
output_dir: {raw_root / "outputs"}
data:
  backend: cyclone_kvikio
  root: {raw_root}
  target_flux: true
  target_spectra: []
  cyclone:
    fields_to_load: [df]
    conditions: [q]
loss:
  flux_weight: 0.0
  spectra_weight: 0.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output_dir"):
        load_config(
            config_path,
            command="inspect-data",
        )
