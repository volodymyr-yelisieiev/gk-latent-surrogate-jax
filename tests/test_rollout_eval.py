from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from gk_surrogate import pipeline as pipeline_module
from gk_surrogate.config.load import load_config
from gk_surrogate.data.latent_cache import LatentCacheWriter
from gk_surrogate.pipeline import (
    _shape_correlation,
    _trajectory_relative_l2_by_step,
    _trajectory_time_average_errors,
    _trajectory_values_by_step,
    embed_dataset,
    evaluate_rollout,
)


def _assert_finite_metric(result: dict[str, object], key: str) -> None:
    value = result[key]
    if isinstance(value, list):
        assert value, key
        assert all(math.isfinite(float(item)) for item in value), key
    else:
        assert math.isfinite(float(value)), key


def _write_rollout_cache(path: Path, *, seed: int = 42) -> None:
    trajectory_ids = tuple(f"traj-{index}" for index in range(4))
    encoder_run = path.parent / "encoder"
    encoder_checkpoint = encoder_run / "checkpoints" / "step_000001"
    encoder_checkpoint.mkdir(parents=True)
    checkpoint_file = encoder_checkpoint / "checkpoint.pkl"
    checkpoint_file.write_bytes(pickle.dumps({"step": 1, "params": {}}))
    train_ids = pipeline_module.split_trajectory_ids(trajectory_ids, seed=seed)["train"]
    (encoder_run / "config_resolved.json").write_text(
        json.dumps(
            {
                "data": {"backend": "cyclone_kvikio", "seed": seed, "split": "train"},
                "training": {"seed": seed},
            }
        ),
        encoding="utf-8",
    )
    (encoder_run / "metrics.json").write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "artifact_role": "encoder_checkpoint",
                "data_backend": "cyclone_kvikio",
                "data_split": "train",
                "data_split_seed": seed,
                "training_seed": seed,
                "selected_trajectory_ids": list(train_ids),
                "trajectory_manifest_sha256": pipeline_module._trajectory_manifest_sha256(train_ids),
                "universe_trajectory_ids": list(trajectory_ids),
                "universe_manifest_sha256": pipeline_module._trajectory_manifest_sha256(trajectory_ids),
                "checkpoint": str(encoder_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    protocol = {
        "protocol_version": 1,
        "artifact_role": "latent_cache",
        "data_backend": "cyclone_kvikio",
        "data_split": "all",
        "data_split_seed": seed,
        "training_seed": seed,
        "selected_trajectory_ids": list(trajectory_ids),
        "trajectory_manifest_sha256": pipeline_module._trajectory_manifest_sha256(trajectory_ids),
        "universe_trajectory_ids": list(trajectory_ids),
        "universe_manifest_sha256": pipeline_module._trajectory_manifest_sha256(trajectory_ids),
        "encoder_checkpoint": str(encoder_checkpoint),
        "encoder_checkpoint_sha256": pipeline_module._sha256_file(checkpoint_file),
    }
    cache_config = {
        "data": {"backend": "cyclone_kvikio", "seed": seed, "split": "all"},
        "training": {"seed": seed},
        "latent_cache": {"encoder_checkpoint_path": str(encoder_checkpoint)},
    }
    writer = LatentCacheWriter(
        path,
        latent_dim=3,
        config_yaml=yaml.safe_dump(cache_config),
        encoder_checkpoint_path=str(encoder_checkpoint),
        protocol_metadata=protocol,
    )
    for index, trajectory_id in enumerate(trajectory_ids):
        t = np.linspace(0.0, 1.0, 12, dtype=np.float32)
        z = np.stack((t + index, t * t, np.cos(t * np.pi)), axis=1).astype(np.float32)
        flux = (z[:, :1] * 0.5 + 1.0).astype(np.float32)
        writer.write_trajectory(trajectory_id, z, flux=flux)


def test_diagnostic_rollout_aggregation_weights_trajectories_equally():
    values = np.asarray([[1.0, 1.0], [1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
    ids = ["long", "long", "short"]
    by_trajectory = _trajectory_values_by_step(values, ids)
    np.testing.assert_allclose(np.asarray(by_trajectory), [[1.0, 1.0], [9.0, 9.0]])
    np.testing.assert_allclose(np.mean(np.asarray(by_trajectory), axis=0), [5.0, 5.0])

    pred = np.asarray([[[1.0]], [[1.0]], [[9.0]]], dtype=np.float32)
    target = np.zeros_like(pred)
    np.testing.assert_allclose(np.asarray(_trajectory_time_average_errors(pred, target, ids)), [1.0, 9.0])

    relative_target = np.asarray([[[1.0]], [[100.0]], [[10.0]]], dtype=np.float32)
    relative_error = np.asarray([[[1.0]], [[10.0]], [[20.0]]], dtype=np.float32)
    relative = _trajectory_relative_l2_by_step(relative_error, relative_target, ids, eps=1e-8)
    np.testing.assert_allclose(np.asarray(relative[:, 0]), [np.sqrt(101.0 / 10001.0), 2.0], rtol=1e-6)


def test_spectral_shape_correlation_is_stable_for_zero_and_small_vectors():
    small = np.asarray([[[1.0e-5, 2.0e-5, 4.0e-5]]], dtype=np.float32)
    correlation = _shape_correlation(small, small, eps=1e-8)
    np.testing.assert_allclose(np.asarray(correlation), [[1.0]], atol=1e-6)

    zeros = np.zeros_like(small)
    zero_correlation = _shape_correlation(zeros, zeros, eps=1e-8)
    np.testing.assert_allclose(np.asarray(zero_correlation), [[0.0]])
    assert np.all(np.isfinite(np.asarray(zero_correlation)))


def test_rollout_eval_outputs_metrics_json_csv_and_plot(repo_root, tmp_path):
    embed_cfg = load_config(repo_root / "configs/experiment/smoke_embed_dataset.yaml", command="embed-dataset")
    enc_cfg = load_config(repo_root / "configs/experiment/smoke_encoder_supervised.yaml", command="train-encoder")
    enc_cfg = enc_cfg.model_copy(update={"output_dir": str(tmp_path / "enc")})
    from gk_surrogate.pipeline import train_encoder

    enc = train_encoder(enc_cfg)
    embed_cfg = embed_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "smoke_embed_dataset"),
            "latent_cache": embed_cfg.latent_cache.model_copy(
                update={
                    "path": str(tmp_path / "smoke_embed_dataset" / "latent_cache.h5"),
                    "encoder_checkpoint_path": enc["checkpoint"],
                }
            ),
        }
    )
    embed_dataset(embed_cfg)
    config = load_config(
        repo_root / "configs/experiment/smoke_evaluate_rollout.yaml",
        command="evaluate-rollout",
    )
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "eval"),
        "latent_cache": config.latent_cache.model_copy(
                update={
                    "path": str(tmp_path / "smoke_embed_dataset" / "latent_cache.h5"),
                    "encoder_checkpoint_path": enc["checkpoint"],
                    "sequence_checkpoint_path": None,
                }
            ),
            "evaluation": config.evaluation.model_copy(
                update={"baseline_mode": "latent_state_persistence_decoded"}
            ),
        }
    )
    result = evaluate_rollout(config)
    assert Path(result["metrics_json"]).exists()
    assert Path(result["metrics_by_step_csv"]).exists()
    assert Path(result["diagnostic_samples_npz"]).exists()
    assert len(result["mse_by_step"]) == 4
    assert len(result["mse_by_trajectory"]) == result["num_trajectories"]
    assert list(result["latent_trajectory_ids"]) == list(result["selected_trajectory_ids"])
    assert len(result["mse_std_by_step"]) == 4
    assert len(result["relative_l2_by_step"]) == 4
    assert result["num_trajectories"] == 1
    assert result["num_rollout_windows"] > 1
    assert result["diagnostic_heads_loaded"] is True
    assert result["diagnostic_metrics_requested"] is True
    assert result["flux_metrics_computed"] is True
    assert result["spectra_metrics_computed"] is True
    assert result["diagnostic_warnings"] == []
    assert result["num_configured_trajectories"] == 4
    assert result["num_selected_trajectories"] == 1
    assert len(result["selected_trajectory_ids"]) == 1
    assert result["data_split"] == config.data.split
    assert result["data_split_seed"] == config.data.seed
    assert result["rollout_horizon"] == config.evaluation.rollout_steps
    assert result["aggregation"] == "trajectory_balanced_mean_with_between_trajectory_std"
    assert len(result["trajectory_manifest_sha256"]) == 64
    assert result["latent_cache"] == config.latent_cache.path
    assert "flux_mse_by_step" in result
    assert "flux_mse_std_by_step" in result
    assert "flux_rmse" in result
    assert result["trajectory_balanced_flux_rmse"] == pytest.approx(
        float(np.mean(result["flux_rmse_by_trajectory"]))
    )
    assert math.isfinite(float(result["flux_rmse_pooled"]))
    assert "flux_mae" in result
    assert "flux_relative_error" in result
    assert "flux_time_average_error" in result
    assert result["rollout_method"] == "latent_state_persistence_decoded"
    assert result["baseline_mode"] == "latent_state_persistence_decoded"
    assert result["sequence_checkpoint"] == "latent_state_persistence_decoded"
    assert len(result["latent_cache_sha256"]) == 64
    assert len(result["encoder_checkpoint_sha256"]) == 64
    assert "observed_diagnostic_persistence_flux_rmse" in result
    assert "observed_diagnostic_persistence_flux_rmse_by_trajectory" in result
    assert "diagnostic_head_oracle_flux_rmse" in result
    assert len(result["flux_rmse_by_trajectory"]) == result["num_trajectories"]
    assert list(result["flux_trajectory_ids"]) == list(result["selected_trajectory_ids"])
    assert "spectra_mse_by_step" in result
    assert "spectra_mse_std_by_step" in result
    assert "spectra_log_mse" in result
    assert "spectra_relative_l2" in result
    assert "spectra_shape_corr" in result
    assert "observed_diagnostic_persistence_spectra_ky_rmse" in result
    assert "observed_diagnostic_persistence_spectra_q_rmse" in result
    assert "diagnostic_head_oracle_spectra_ky_rmse" in result
    assert "diagnostic_head_oracle_spectra_q_rmse" in result
    for key in (
        "flux_mse_by_step",
        "flux_rmse_by_step",
        "flux_mae_by_step",
        "flux_relative_error_by_step",
        "flux_mse",
        "flux_rmse",
        "flux_mae",
        "flux_relative_error",
        "flux_time_average_error",
        "spectra_mse_by_step",
        "spectra_log_mse_by_step",
        "spectra_relative_l2_by_step",
        "spectra_shape_corr_by_step",
        "spectra_mse",
        "spectra_log_mse",
        "spectra_relative_l2",
        "spectra_shape_corr",
        "observed_diagnostic_persistence_flux_rmse",
        "diagnostic_head_oracle_flux_rmse",
        "observed_diagnostic_persistence_spectra_ky_rmse",
        "diagnostic_head_oracle_spectra_ky_rmse",
    ):
        _assert_finite_metric(result, key)
    assert result["stable"] is True
    assert set(result["plots"]) == {"latent_mse", "flux_mse", "spectra_mse"}
    assert all(Path(path).exists() for path in result["plots"].values())


def test_rollout_eval_honors_configured_cache_trajectories(repo_root, tmp_path, monkeypatch):
    monkeypatch.setattr("gk_surrogate.pipeline._requires_complete_protocol", lambda _config: False)
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    for index in range(4):
        monkeypatch.setenv(f"GK_VALIDATION_TRAJ_{index}", f"traj-{index}")
    cache_path = tmp_path / "latent_cache.h5"
    _write_rollout_cache(cache_path)
    config = load_config(
        repo_root / "configs/experiment/server_evaluate_latent_persistence_medium.yaml",
        command="evaluate-rollout",
    )
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "rollout_subset"),
            "data": config.data.model_copy(
                update={
                    "seed": 42,
                    "cyclone": config.data.cyclone.model_copy(update={"trajectories": ("traj-0", "traj-1", "traj-2")})
                }
            ),
            "latent_cache": config.latent_cache.model_copy(
                update={
                    "path": str(cache_path),
                    "encoder_checkpoint_path": None,
                    "sequence_checkpoint_path": None,
                }
            ),
            "evaluation": config.evaluation.model_copy(
                update={
                    "baseline_mode": "latent_state_persistence_decoded",
                    "rollout_steps": 3,
                    "metrics": ("latent_mse",),
                }
            ),
        }
    )

    result = evaluate_rollout(config)

    assert result["num_configured_trajectories"] == 3
    assert result["num_selected_trajectories"] == 1
    assert set(result["configured_trajectory_ids"]) == {"traj-0", "traj-1", "traj-2"}
    assert set(result["selected_trajectory_ids"]).issubset({"traj-0", "traj-1", "traj-2"})
    assert Path(result["metrics_json"]).exists()


def test_rollout_eval_uses_cache_lineage_without_configured_checkpoint(repo_root, tmp_path):
    enc_cfg = load_config(repo_root / "configs/experiment/smoke_encoder_supervised.yaml", command="train-encoder")
    enc_cfg = enc_cfg.model_copy(update={"output_dir": str(tmp_path / "enc")})
    from gk_surrogate.pipeline import train_encoder

    enc = train_encoder(enc_cfg)
    embed_cfg = load_config(repo_root / "configs/experiment/smoke_embed_dataset.yaml", command="embed-dataset")
    embed_cfg = embed_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "embed"),
            "latent_cache": embed_cfg.latent_cache.model_copy(
                update={
                    "path": str(tmp_path / "embed" / "latent_cache.h5"),
                    "encoder_checkpoint_path": enc["checkpoint"],
                }
            ),
        }
    )
    embedded = embed_dataset(embed_cfg)

    config = load_config(repo_root / "configs/experiment/smoke_evaluate_rollout.yaml", command="evaluate-rollout")
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "eval_no_diag"),
            "latent_cache": config.latent_cache.model_copy(
                update={
                    "path": embedded["latent_cache"],
                    "encoder_checkpoint_path": None,
                    "sequence_checkpoint_path": None,
                }
            ),
            "evaluation": config.evaluation.model_copy(
                update={"baseline_mode": "latent_state_persistence_decoded"}
            ),
        }
    )
    result = evaluate_rollout(config)
    assert result["diagnostic_heads_loaded"] is True
    assert result["diagnostic_metrics_requested"] is True
    assert result["flux_metrics_computed"] is True
    assert result["spectra_metrics_computed"] is True
    assert result["diagnostic_warnings"] == []
    assert "flux_mse_by_step" in result
    assert "spectra_mse_by_step" in result

    observed_config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "eval_observed_persistence"),
            "evaluation": config.evaluation.model_copy(
                update={"baseline_mode": "observed_diagnostic_persistence"}
            ),
        }
    )
    observed = evaluate_rollout(observed_config)
    assert observed["rollout_method"] == "observed_diagnostic_persistence"
    assert observed["baseline_mode"] == "observed_diagnostic_persistence"
    assert observed["sequence_checkpoint"] == "observed_diagnostic_persistence"
    assert observed["stable"] is True
    assert list(observed["flux_trajectory_ids"]) == list(observed["selected_trajectory_ids"])
    assert observed["flux_metrics_computed"] is False
    assert observed["spectra_metrics_computed"] is False
    assert observed["diagnostic_warnings"] == []
    _assert_finite_metric(observed, "flux_mse")
    _assert_finite_metric(observed, "observed_diagnostic_persistence_flux_rmse")
    _assert_finite_metric(observed, "diagnostic_head_oracle_flux_rmse")
    _assert_finite_metric(observed, "observed_diagnostic_persistence_spectra_ky_rmse")
    _assert_finite_metric(observed, "diagnostic_head_oracle_spectra_ky_rmse")

    diagnostics_disabled = config.model_copy(
        update={
            "output_dir": str(tmp_path / "eval_diagnostics_disabled"),
            "model": config.model.model_copy(
                update={
                    "diagnostics": config.model.diagnostics.model_copy(
                        update={"flux_dim": None, "spectra_dims": {}}
                    )
                }
            ),
        }
    )
    disabled = evaluate_rollout(diagnostics_disabled)
    assert disabled["diagnostic_heads_loaded"] is False
    assert disabled["flux_metrics_computed"] is False
    assert disabled["spectra_metrics_computed"] is False
    assert len(disabled["diagnostic_warnings"]) == 2

    no_windows = config.model_copy(
        update={
            "output_dir": str(tmp_path / "eval_no_windows"),
            "evaluation": config.evaluation.model_copy(update={"rollout_steps": 100}),
        }
    )
    import pytest

    with pytest.raises(ValueError, match="no valid rollout windows"):
        evaluate_rollout(no_windows)


def test_embed_dataset_accepts_simsiam_checkpoint(repo_root, tmp_path):
    from gk_surrogate.pipeline import train_encoder

    sim_cfg = load_config(repo_root / "configs/experiment/smoke_encoder_simsiam.yaml", command="train-encoder")
    sim_cfg = sim_cfg.model_copy(update={"output_dir": str(tmp_path / "simsiam")})
    sim = train_encoder(sim_cfg)

    embed_cfg = load_config(repo_root / "configs/experiment/smoke_embed_dataset.yaml", command="embed-dataset")
    embed_cfg = embed_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "embed_simsiam"),
            "latent_cache": embed_cfg.latent_cache.model_copy(
                update={
                    "path": str(tmp_path / "embed_simsiam" / "latent_cache.h5"),
                    "encoder_checkpoint_path": sim["checkpoint"],
                }
            ),
        }
    )
    result = embed_dataset(embed_cfg)
    assert Path(result["latent_cache"]).exists()


def test_rollout_rejects_configured_encoder_mismatch(repo_root, tmp_path):
    enc_cfg = load_config(repo_root / "configs/experiment/smoke_encoder_supervised.yaml", command="train-encoder")
    enc_cfg = enc_cfg.model_copy(update={"output_dir": str(tmp_path / "enc")})
    from gk_surrogate.pipeline import train_encoder

    enc = train_encoder(enc_cfg)
    embed_cfg = load_config(repo_root / "configs/experiment/smoke_embed_dataset.yaml", command="embed-dataset")
    embed_cfg = embed_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "embed"),
            "latent_cache": embed_cfg.latent_cache.model_copy(
                update={"path": str(tmp_path / "embed/cache.h5"), "encoder_checkpoint_path": enc["checkpoint"]}
            ),
        }
    )
    embedded = embed_dataset(embed_cfg)
    config = load_config(repo_root / "configs/experiment/smoke_evaluate_rollout.yaml", command="evaluate-rollout")
    config = config.model_copy(
        update={
            "latent_cache": config.latent_cache.model_copy(
                update={
                    "path": embedded["latent_cache"],
                    "encoder_checkpoint_path": str(tmp_path / "different-checkpoint"),
                }
            )
        }
    )
    import pytest

    with pytest.raises(ValueError, match="does not match the latent cache lineage"):
        evaluate_rollout(config)
