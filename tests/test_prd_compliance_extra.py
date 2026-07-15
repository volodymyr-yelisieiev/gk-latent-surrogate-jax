from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import H5SchemaConfig, NormalizationConfig, SyntheticDataConfig
from gk_surrogate.data.factory import build_dataset
from gk_surrogate.data.h5_loader import H5TrajectoryDataset, write_synthetic_h5
from gk_surrogate.data.normalization import NormalizationStats, normalize_snapshot
from gk_surrogate.data.sequence_dataset import valid_sequence_starts
from gk_surrogate.data.split import split_trajectory_ids
from gk_surrogate.evaluation.diagnostics import diagnostic_metrics, diagnostic_metrics_numpy
from gk_surrogate.factory import build_diagnostic_heads, build_encoder, build_sequence_model, build_simsiam
from gk_surrogate.losses.latent import latent_prediction_loss
from gk_surrogate.losses.total import encoder_total_loss
from gk_surrogate.metrics.diagnostics import (
    spectra_log_mse,
    spectra_mean_absolute_relative_error,
    spectra_shape_corr,
    time_average_flux_error,
)
from gk_surrogate.metrics.latent import (
    latent_cosine_similarity,
    latent_mae,
    latent_relative_l2,
    rollout_cosine_by_step,
)
from gk_surrogate.models.encoders import ExternalEncoderAdapter
from gk_surrogate.models.patching import patch_grid_shape, patch_token_count, validate_token_count
from gk_surrogate.models.sequence import GPT2Adapter
from gk_surrogate.training.loops import cycle_batches, run_fixed_steps
from gk_surrogate.training.train_encoder import train_encoder_loop
from gk_surrogate.training.train_sequence import train_sequence_loop
from gk_surrogate.utils.arrays import assert_rank, assert_shape_suffix, cosine_similarity, ensure_finite_tree
from gk_surrogate.utils.tree import tree_allclose, tree_l2_norm


def test_prd_is_checked_in_and_not_local_path_placeholder(repo_root):
    text = Path(repo_root / "PRD.md").read_text(encoding="utf-8")
    assert "JAX Latent Time-Series Surrogate" in text
    assert "Definition of done for the MacBook stage" in text
    assert len(text.splitlines()) > 500
    assert "/Users/" not in text
    assert "Latent Surrogate JAX.md" not in text


def test_config_validation_edge_cases(tiny_config_path, tmp_path):
    with pytest.raises(ValueError, match="fixed normalization"):
        NormalizationConfig(mode="fixed")
    with pytest.raises(ValueError, match="spatial"):
        SyntheticDataConfig(
            num_trajectories=1,
            timesteps=1,
            channels=1,
            spatial_shape=(1, 1, 1, 1, 0),
            flux_dim=1,
        )
    with pytest.raises(ValueError, match="channel_indices"):
        H5SchemaConfig(channel_indices=(-1,))
    root = tmp_path / "raw"
    root.mkdir()
    with pytest.raises(ValueError, match="output_dir"):
        load_config(
            tiny_config_path,
            overrides=[
                "data.backend=h5",
                f"data.root={root}",
                "data.synthetic=null",
                "data.h5_schema={trajectory_glob: '*.h5'}",
                f"output_dir={root / 'outputs'}",
            ],
            command="train-encoder",
        )
    with pytest.raises(ValueError, match="diagnostic spectra heads"):
        load_config(
            tiny_config_path,
            overrides=["model.diagnostics.spectra_dims={ky: 8}"],
            command="train-encoder",
        )


def test_h5_and_synthetic_error_edges(tmp_path, tiny_config_path):
    config = load_config(tiny_config_path, command="train-encoder")
    assert config.data.synthetic is not None
    with pytest.raises(FileNotFoundError):
        H5TrajectoryDataset(tmp_path, H5SchemaConfig())
    write_synthetic_h5(tmp_path, config.data.synthetic)
    dataset = H5TrajectoryDataset(
        tmp_path,
        H5SchemaConfig(flux_key=None, timestep_key=None, spectra_keys={}),
        target_flux=False,
    )
    assert dataset.num_timesteps(dataset.trajectory_ids()[0]) == 16
    assert dataset.snapshot_shape() == (2, 4, 4, 4, 4, 4)
    with pytest.raises(KeyError):
        dataset.get_snapshot("missing", 0)
    synthetic = build_dataset(config.data)
    with pytest.raises(KeyError):
        synthetic.get_snapshot("missing", 0)
    with pytest.raises(IndexError):
        synthetic.get_snapshot(synthetic.trajectory_ids()[0], 100)


def test_normalization_split_sequence_and_patch_edges(tmp_path):
    x = np.ones((2, 2, 2, 2, 2, 2), dtype=np.float32)
    stats = NormalizationStats(mean=np.asarray(1.0, dtype=np.float32), std=np.asarray(2.0, dtype=np.float32))
    assert normalize_snapshot(x, mode="dataset", stats=stats).shape == x.shape
    assert normalize_snapshot(x, mode="trajectory", stats=stats).shape == x.shape
    with pytest.raises(ValueError, match="unknown normalization"):
        normalize_snapshot(x, mode="bad", stats=stats)
    with pytest.raises(ValueError, match="at least two"):
        split_trajectory_ids(["only"])
    with pytest.raises(ValueError, match="positive"):
        split_trajectory_ids(["a", "b"], ratios=(0.0, 0.0, 0.0))
    assert patch_grid_shape((4, 4, 4, 4, 4), (2, 2, 2, 2, 2)) == (2, 2, 2, 2, 2)
    assert patch_token_count((4, 4, 4, 4, 4), 2) == 32
    with pytest.raises(ValueError, match="divisible"):
        patch_grid_shape((5, 4, 4, 4, 4), 2)
    with pytest.raises(ValueError, match="exceeds"):
        validate_token_count((8, 8, 8, 8, 8), 1, max_token_count=10)


def test_factories_and_adapter_placeholders():
    assert (
        build_diagnostic_heads(
            type("Cfg", (), {"flux_dim": 0, "spectra_dims": {}, "hidden_dims": (), "dropout_rate": 0.0})()
        )
        is None
    )
    encoder_cfg = type(
        "Cfg",
        (),
        {
            "type": "flatten_mlp",
            "latent_dim": 4,
            "hidden_dims": (8,),
            "activation": "gelu",
            "dropout_rate": 0.0,
            "extra": {},
        },
    )()
    encoder = build_encoder(encoder_cfg)
    simsiam = build_simsiam(
        type(
            "SimCfg",
            (),
            {
                "projection_dim": 4,
                "projection_hidden_dim": 8,
                "projection_layers": 1,
                "prediction_hidden_dim": 4,
            },
        )(),
        encoder,
    )
    x = jnp.ones((1, 1, 2, 2, 2, 2, 2))
    variables = simsiam.init(jax.random.PRNGKey(0), x, x, train=False)
    assert simsiam.apply(variables, x, x, train=False).q1.shape == (1, 4)
    with pytest.raises(NotImplementedError):
        ExternalEncoderAdapter(name="x", latent_dim=4).apply({}, x, train=False)
    with pytest.raises(NotImplementedError):
        GPT2Adapter(latent_dim=4).apply({}, jnp.ones((1, 2, 4)), train=False)
    with pytest.raises(ValueError):
        seq_cfg = type(
            "SeqCfg",
            (),
            {"type": "bad", "latent_dim": 4, "context_length": 2, "hidden_dims": (8,), "extra": {}},
        )()
        build_sequence_model(seq_cfg)


def test_loop_helpers_and_training_loops(tiny_config_path, tmp_path):
    config = load_config(tiny_config_path, command="train-encoder")
    from gk_surrogate.pipeline import _init_encoder_state, _snapshot_batches, benchmark_step_time

    state, _ = _init_encoder_state(config)
    batch = next(_snapshot_batches(config, repeat=False))
    state, history = train_encoder_loop(
        state,
        [batch],
        max_steps=1,
        output_dir=str(tmp_path / "enc_loop"),
        log_every=1,
        checkpoint_every=1,
    )
    assert history and (tmp_path / "enc_loop" / "metrics.csv").exists()

    def step_fn(value, item):
        return value + 1, {"loss": item["x"]}

    _, fixed_history = run_fixed_steps(0, [{"x": 1.0}], step_fn, max_steps=2)
    assert len(fixed_history) == 2
    with pytest.raises(ValueError):
        next(cycle_batches([]))
    report = benchmark_step_time(config, measured_steps=1)
    assert report["benchmark"] == "encoder_train_step"
    assert report["mean_step_seconds"] >= 0.0
    assert np.isfinite(report["loss"])


def test_sequence_loop_and_valid_empty_window(repo_root, tmp_path):
    from gk_surrogate.pipeline import _init_sequence_state, _latent_cache_for, _sequence_batches

    config = load_config(repo_root / "configs/experiment/smoke_sequence.yaml", command="train-sequence")
    from gk_surrogate.data.latent_cache import LatentCacheWriter

    cache_path = tmp_path / "latent_cache.h5"
    writer = LatentCacheWriter(cache_path, latent_dim=32)
    writer.write_trajectory("traj", np.ones((12, 32), dtype=np.float32))
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "seq"),
            "latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)}),
        }
    )
    cache = _latent_cache_for(config)
    state, _ = _init_sequence_state(config)
    batch = next(_sequence_batches(config, cache, repeat=False))
    _, history = train_sequence_loop(state, [batch], max_steps=1, output_dir=str(tmp_path / "seq_loop"))
    assert history
    from gk_surrogate.data.latent_cache import LatentCacheDataset

    dataset = LatentCacheDataset(cache)
    assert valid_sequence_starts(dataset, dataset.trajectory_ids()[0], context_length=100, prediction_length=1) == ()


def test_more_metrics_losses_and_utils():
    pred = jnp.asarray([[1.0, 2.0]])
    target = jnp.asarray([[2.0, 1.0]])
    assert latent_mae(pred, target) > 0
    assert latent_relative_l2(pred, target) > 0
    assert latent_cosine_similarity(pred, target) <= 1
    rollout_pred = pred[:, None, :]
    rollout_target = target[:, None, :]
    assert rollout_cosine_by_step(rollout_pred, rollout_target).shape == (1,)
    spectra = {"ky": pred}
    assert spectra_log_mse(spectra, {"ky": target})["ky"] >= 0
    assert spectra_shape_corr(spectra, {"ky": target})["ky"] <= 1
    assert spectra_mean_absolute_relative_error(spectra, {"ky": target})["ky"] >= 0
    assert time_average_flux_error(pred, target) >= 0
    metrics = diagnostic_metrics(
        flux_pred=pred,
        flux_target=target,
        spectra_pred=spectra,
        spectra_target={"ky": target},
    )
    assert "flux/mse" in metrics
    assert "spectra/shape_corr" in metrics
    assert "spectra/mse" in diagnostic_metrics_numpy(
        spectra_pred=spectra,
        spectra_target={"ky": target},
    )
    with pytest.raises(ValueError, match="SimSiam"):
        encoder_total_loss(simsiam_weight=1.0)
    with pytest.raises(ValueError, match="Diagnostic"):
        encoder_total_loss(flux_weight=1.0)
    with pytest.raises(ValueError, match="Unsupported latent"):
        latent_prediction_loss(pred, target, mode="bad")
    with pytest.raises(ValueError):
        assert_rank(pred, 3)
    with pytest.raises(ValueError):
        assert_shape_suffix(pred, (3,))
    with pytest.raises(ValueError):
        ensure_finite_tree({"bad": jnp.asarray([jnp.nan])})
    assert cosine_similarity(pred, target).shape == (1,)
    assert tree_l2_norm({"x": pred}) > 0
    assert not tree_allclose({"x": pred}, {"x": target})
