from __future__ import annotations

from pathlib import Path

from gk_surrogate.config.load import load_config

CANONICAL_CONFIGS = {
    "encoder": ("server_encoder_simsiam_medium.yaml", "train-encoder", "train"),
    "embed": ("server_embed_dataset_medium.yaml", "embed-dataset", "all"),
    "gru_train": ("server_sequence_gru_medium.yaml", "train-sequence", "train"),
    "transformer_train": ("server_sequence_transformer_medium.yaml", "train-sequence", "train"),
    "gru_eval": ("server_evaluate_rollout_medium.yaml", "evaluate-rollout", "test"),
    "transformer_eval": ("server_evaluate_transformer_rollout_medium.yaml", "evaluate-rollout", "test"),
    "latent_persistence_eval": (
        "server_evaluate_latent_persistence_medium.yaml",
        "evaluate-rollout",
        "test",
    ),
    "observed_persistence_eval": (
        "server_evaluate_observed_persistence_medium.yaml",
        "evaluate-rollout",
        "test",
    ),
}


def test_canonical_server_configs_load_without_personal_paths(repo_root, monkeypatch):
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    config_dir = repo_root / "configs" / "experiment"
    loaded = {}
    for role, (filename, command, expected_split) in CANONICAL_CONFIGS.items():
        config = load_config(config_dir / filename, command=command)
        loaded[role] = config
        assert config.data.backend == "cyclone_kvikio"
        assert config.data.root == "/tmp/gk-cyclone-root"
        assert config.data.split == expected_split
        assert config.data.cyclone is not None
        assert config.data.cyclone.offset == 80
        assert config.data.cyclone.subsample == 8
        assert str(config.output_dir).startswith("outputs/")

    assert loaded["gru_train"].model.sequence is not None
    assert loaded["gru_train"].model.sequence.type == "gru"
    assert loaded["transformer_train"].model.sequence is not None
    assert loaded["transformer_train"].model.sequence.type == "causal_transformer"
    assert loaded["latent_persistence_eval"].evaluation.baseline_mode == "latent_state_persistence_decoded"
    assert loaded["observed_persistence_eval"].evaluation.baseline_mode == "observed_diagnostic_persistence"


def test_canonical_server_configs_share_representation_and_dataset_contract(repo_root, monkeypatch):
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    config_dir = repo_root / "configs" / "experiment"
    loaded = {
        role: load_config(config_dir / filename, command=command)
        for role, (filename, command, _split) in CANONICAL_CONFIGS.items()
    }
    reference = loaded["encoder"]
    for _role, config in loaded.items():
        normalized_config = config.data.model_copy(update={"split": "all", "shuffle": False, "batch_size": 1})
        normalized_reference = reference.data.model_copy(
            update={
                "split": "all",
                "shuffle": False,
                "batch_size": 1,
                "augmentations": config.data.augmentations,
            }
        )
        assert normalized_config == normalized_reference
        assert config.model.encoder == reference.model.encoder
        assert config.model.diagnostics == reference.model.diagnostics
        assert (
            config.loss.flux_weight,
            config.loss.spectra_weight,
            config.loss.use_log_spectra,
            config.loss.spectra_epsilon,
        ) == (
            reference.loss.flux_weight,
            reference.loss.spectra_weight,
            reference.loss.use_log_spectra,
            reference.loss.spectra_epsilon,
        )


def test_core_docs_describe_current_evidence_contract(repo_root: Path):
    docs = {
        "data_contract.md": "observed_diagnostic_persistence",
        "result_status.md": "diagnostic-head oracle",
        "wandb_tracking.md": "post-hoc runs",
    }
    for filename, required_term in docs.items():
        path = repo_root / "docs" / filename
        assert path.is_file()
        assert len(path.read_text(encoding="utf-8").splitlines()) >= 10
        assert required_term in path.read_text(encoding="utf-8")
