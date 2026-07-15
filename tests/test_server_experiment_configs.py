from __future__ import annotations

from pathlib import Path

from gk_surrogate.config.load import load_config

SERVER_CONFIG_COMMANDS = {
    "server_encoder_supervised_small.yaml": "train-encoder",
    "server_encoder_simsiam_small.yaml": "train-encoder",
    "server_embed_dataset_small.yaml": "embed-dataset",
    "server_sequence_mlp_delta_small.yaml": "train-sequence",
    "server_sequence_gru_small.yaml": "train-sequence",
    "server_sequence_transformer_small.yaml": "train-sequence",
    "server_evaluate_flux_head_small.yaml": "evaluate-flux-head",
    "server_plot_representation_small.yaml": "plot-representation",
    "server_evaluate_rollout_small.yaml": "evaluate-rollout",
    "server_evaluate_rollout_small_transformer.yaml": "evaluate-rollout",
    "server_evaluate_persistence_baseline_small.yaml": "evaluate-rollout",
    "server_encoder_supervised_medium.yaml": "train-encoder",
    "server_encoder_simsiam_medium.yaml": "train-encoder",
    "server_embed_dataset_medium.yaml": "embed-dataset",
    "server_sequence_gru_medium.yaml": "train-sequence",
    "server_sequence_transformer_medium.yaml": "train-sequence",
    "server_evaluate_rollout_medium.yaml": "evaluate-rollout",
    "server_encoder_supervised_medium_logspec.yaml": "train-encoder",
    "server_encoder_simsiam_medium_logspec.yaml": "train-encoder",
    "server_embed_dataset_medium_logspec.yaml": "embed-dataset",
    "server_sequence_gru_medium_logspec.yaml": "train-sequence",
    "server_sequence_transformer_medium_logspec.yaml": "train-sequence",
    "server_evaluate_rollout_medium_logspec_gru.yaml": "evaluate-rollout",
    "server_evaluate_rollout_medium_logspec_transformer.yaml": "evaluate-rollout",
    "server_encoder_simsiam_longsignal.yaml": "train-encoder",
    "server_embed_dataset_longsignal.yaml": "embed-dataset",
    "server_sequence_gru_longsignal.yaml": "train-sequence",
    "server_sequence_transformer_longsignal.yaml": "train-sequence",
    "server_evaluate_rollout_longsignal_gru.yaml": "evaluate-rollout",
    "server_evaluate_rollout_longsignal_persistence.yaml": "evaluate-rollout",
    "server_evaluate_rollout_longsignal_transformer.yaml": "evaluate-rollout",
}

SMALL_VALIDATION_TRAJECTORIES = ("traj-0", "traj-1", "traj-2", "traj-3")


def test_server_experiment_configs_load_without_personal_paths(repo_root, monkeypatch):
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    for index, trajectory_id in enumerate(SMALL_VALIDATION_TRAJECTORIES):
        monkeypatch.setenv(f"GK_SMALL_VALIDATION_TRAJ_{index}", trajectory_id)
    config_dir = repo_root / "configs" / "experiment"
    for filename, command in SERVER_CONFIG_COMMANDS.items():
        config = load_config(config_dir / filename, command=command)
        assert config.data.backend == "cyclone_kvikio"
        assert config.data.root == "/tmp/gk-cyclone-root"
        assert config.parallel.mode == "auto"
        assert str(config.output_dir).startswith("outputs/")
        assert config.data.cyclone is not None
        assert config.data.cyclone.offset == 80
        assert config.data.cyclone.subsample in {4, 8, 16}
        if "logspec" in filename:
            assert config.loss.use_log_spectra is True
        if "longsignal" in filename:
            assert config.data.cyclone.subsample == 4
            assert config.data.context_length == 16
            assert config.evaluation.rollout_steps == 16
        if "_small" in filename:
            assert config.data.cyclone.trajectories == SMALL_VALIDATION_TRAJECTORIES


def test_server_reproducibility_docs_exist(repo_root):
    docs = {
        "server_gpu_setup.md": "server",
        "real_data_binding_checklist.md": "Cyclone",
        "data_contract.md": "[B, C, S1, S2, S3, S4, S5]",
        "verification_matrix.md": "make smoke-all",
        "gyroswin_comparison.md": "GyroSwin",
        "pretrained_guppy_sft_feasibility.md": "THESIS-19 completed",
    }
    for filename, required_term in docs.items():
        path = Path(repo_root / "docs" / filename)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) >= 10
        assert required_term in text


def test_small_validation_bundle_uses_validation_flux_rmse(repo_root, monkeypatch):
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    for index, trajectory_id in enumerate(SMALL_VALIDATION_TRAJECTORIES):
        monkeypatch.setenv(f"GK_SMALL_VALIDATION_TRAJ_{index}", trajectory_id)
    config_dir = repo_root / "configs" / "experiment"

    flux_head = load_config(config_dir / "server_evaluate_flux_head_small.yaml", command="evaluate-flux-head")
    persistence = load_config(
        config_dir / "server_evaluate_persistence_baseline_small.yaml",
        command="evaluate-rollout",
    )
    transformer = load_config(config_dir / "server_evaluate_rollout_small_transformer.yaml", command="evaluate-rollout")
    representation = load_config(config_dir / "server_plot_representation_small.yaml", command="plot-representation")

    assert flux_head.data.split == "val"
    assert flux_head.evaluation.metrics == ("flux_rmse",)
    assert persistence.data.split == "val"
    assert transformer.data.split == "val"
    assert flux_head.data.cyclone is not None
    assert flux_head.data.cyclone.trajectories == SMALL_VALIDATION_TRAJECTORIES
    assert persistence.data.cyclone is not None
    assert persistence.data.cyclone.trajectories == SMALL_VALIDATION_TRAJECTORIES
    assert transformer.data.cyclone is not None
    assert transformer.data.cyclone.trajectories == SMALL_VALIDATION_TRAJECTORIES
    assert persistence.latent_cache.use_persistence_baseline is True
    assert transformer.model.sequence is not None
    assert transformer.model.sequence.type == "causal_transformer"
    assert "flux_rmse" in transformer.evaluation.metrics
    assert "latent_mse" in transformer.evaluation.metrics
    assert representation.data.split == "all"
    assert len(representation.evaluation.tsne_perplexities) >= 2


def test_small_validation_runbook_documents_acceptance(repo_root):
    text = (repo_root / "docs" / "small_validation_experiment.md").read_text(encoding="utf-8")
    required_terms = [
        "3-5 trajectories",
        "held-out validation",
        "persistence baseline",
        "flux_rmse",
        "latent MSE",
        "server_evaluate_flux_head_small.yaml",
        "server_plot_representation_small.yaml",
    ]
    for term in required_terms:
        assert term in text


def test_pretrained_guppy_feasibility_documents_decision(repo_root):
    text = (repo_root / "docs" / "pretrained_guppy_sft_feasibility.md").read_text(encoding="utf-8")
    required_terms = [
        "trained from scratch",
        "arman-bd/guppylm",
        "guppylm-9M",
        "75 source tensors into 99 Flax target tensors",
        "silently falling back",
        "17.8415",
        "11.5816",
        "current tracked main tree",
    ]
    for term in required_terms:
        assert term in text
