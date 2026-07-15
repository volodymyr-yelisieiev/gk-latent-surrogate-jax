from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from gk_surrogate import pipeline as pipeline_module
from gk_surrogate.config.load import load_config
from gk_surrogate.data.latent_cache import LatentCacheDataset, LatentCacheWriter
from gk_surrogate.evaluation.flux_head import evaluate_flux_head, fit_ridge_flux_head
from gk_surrogate.pipeline import evaluate_flux_head as evaluate_flux_head_pipeline


def _write_linear_flux_cache(path: Path) -> None:
    writer = LatentCacheWriter(path, latent_dim=3)
    for index in range(4):
        base = np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0
        z = base + float(index)
        flux = (2.0 * z[:, :1] - 0.5 * z[:, 1:2] + 0.25).astype(np.float32)
        writer.write_trajectory(f"traj-{index}", z, flux=flux)


def test_ridge_flux_head_fits_linear_flux():
    z = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0]], dtype=np.float32)
    flux = (1.5 * z[:, :1] - 2.0 * z[:, 1:2] + 0.75).astype(np.float32)
    head = fit_ridge_flux_head(z, flux, alpha=0.0)
    pred = head.predict(z)
    assert np.max(np.abs(pred - flux)) < 1e-5


def test_ridge_flux_head_accepts_1d_targets_and_validates_shapes(monkeypatch):
    z = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    flux = np.asarray([1.0, 2.0], dtype=np.float32)

    def singular_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced fallback")

    monkeypatch.setattr(np.linalg, "solve", singular_solve)
    assert fit_ridge_flux_head(z, flux).predict(z).shape == (2, 1)
    with pytest.raises(ValueError, match="sample counts"):
        fit_ridge_flux_head(z, flux[:1])
    with pytest.raises(ValueError, match="zero samples"):
        fit_ridge_flux_head(np.empty((0, 2), dtype=np.float32), np.empty((0, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="z must have shape"):
        fit_ridge_flux_head(np.ones((1, 1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="flux must have shape"):
        fit_ridge_flux_head(z, np.ones((2, 1, 1), dtype=np.float32))


def test_flux_head_reports_validation_rmse(tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_linear_flux_cache(cache_path)
    cache = LatentCacheDataset(cache_path)
    result = evaluate_flux_head(
        cache,
        train_ids=("traj-0", "traj-1"),
        eval_ids=("traj-2",),
        eval_split="val",
        alpha=1e-6,
    )
    assert result["primary_metric"] == "val_flux_rmse"
    assert result["flux_head"] == "ridge_linear"
    assert result["num_train_samples"] == 12
    assert result["num_eval_samples"] == 6
    assert math.isfinite(result["flux_rmse"])
    assert result["flux_rmse"] < 1e-3


def test_flux_head_rejects_missing_cache_flux(tmp_path):
    cache_path = tmp_path / "latent_cache_missing_flux.h5"
    writer = LatentCacheWriter(cache_path, latent_dim=2)
    writer.write_trajectory("traj", np.ones((3, 2), dtype=np.float32))
    cache = LatentCacheDataset(cache_path)
    with pytest.raises(ValueError, match="flux targets are missing"):
        evaluate_flux_head(cache, train_ids=("traj",), eval_ids=("traj",), eval_split="val")


def test_flux_head_pipeline_writes_metrics_and_predictions(repo_root, tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_linear_flux_cache(cache_path)
    config = load_config(repo_root / "configs/experiment/smoke_evaluate_flux_head.yaml", command="evaluate-flux-head")
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "eval"),
            "latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)}),
        }
    )
    result = evaluate_flux_head_pipeline(config)
    assert result["primary_metric"] == "val_flux_rmse"
    assert result["eval_split"] == "val"
    assert result["train_trajectories"]
    assert result["eval_trajectories"]
    assert set(result["train_trajectories"]).isdisjoint(result["eval_trajectories"])
    assert math.isfinite(float(result["flux_rmse"]))
    assert Path(result["metrics_json"]).exists()
    assert Path(result["predictions_npz"]).exists()


def test_flux_head_pipeline_honors_configured_cache_trajectories(repo_root, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_requires_complete_protocol", lambda _config: False)
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    for index in range(4):
        monkeypatch.setenv(f"GK_SMALL_VALIDATION_TRAJ_{index}", f"traj-{index}")
    cache_path = tmp_path / "latent_cache.h5"
    _write_linear_flux_cache(cache_path)
    config = load_config(
        repo_root / "configs/experiment/server_evaluate_flux_head_small.yaml",
        command="evaluate-flux-head",
    )
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "eval_subset"),
            "data": config.data.model_copy(
                update={
                    "cyclone": config.data.cyclone.model_copy(
                        update={"trajectories": ("traj-0", "traj-1", "traj-2")}
                    )
                }
            ),
            "latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)}),
        }
    )
    result = evaluate_flux_head_pipeline(config)
    assert result["num_configured_trajectories"] == 3
    assert set(result["configured_trajectories"]) == {"traj-0", "traj-1", "traj-2"}
    assert "traj-3" not in result["train_trajectories"]
    assert "traj-3" not in result["eval_trajectories"]

    missing = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "cyclone": config.data.cyclone.model_copy(
                        update={"trajectories": ("traj-0", "missing-traj")}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="missing from latent cache"):
        evaluate_flux_head_pipeline(missing)

    duplicate = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "cyclone": config.data.cyclone.model_copy(
                        update={"trajectories": ("traj-0", "traj-0", "traj-1")}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="must be distinct"):
        evaluate_flux_head_pipeline(duplicate)

    unresolved = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "cyclone": config.data.cyclone.model_copy(
                        update={"trajectories": ("${GK_SMALL_VALIDATION_TRAJ_0}", "traj-1", "traj-2")}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="unresolved environment variables"):
        evaluate_flux_head_pipeline(unresolved)
