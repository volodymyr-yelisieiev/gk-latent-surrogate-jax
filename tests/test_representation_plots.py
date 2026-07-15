from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gk_surrogate import pipeline as pipeline_module
from gk_surrogate.config.load import load_config
from gk_surrogate.data.latent_cache import LatentCacheDataset, LatentCacheWriter
from gk_surrogate.evaluation.representation import evaluate_representation, pca_project, tsne_project
from gk_surrogate.pipeline import plot_representation


def _write_representation_cache(path: Path) -> None:
    writer = LatentCacheWriter(path, latent_dim=4)
    for index in range(4):
        t = np.linspace(0.0, 1.0, 12, dtype=np.float32)
        z = np.stack(
            [
                t + index,
                np.sin(t * np.pi) + index * 0.1,
                np.cos(t * np.pi),
                t * t,
            ],
            axis=1,
        ).astype(np.float32)
        flux = (z[:, :1] * 0.7 + z[:, 1:2] * 0.3).astype(np.float32)
        writer.write_trajectory(f"traj-{index}", z, flux=flux)


def test_pca_and_tsne_projection_shapes():
    z = np.arange(60, dtype=np.float32).reshape(15, 4)
    assert pca_project(z).shape == (15, 2)
    assert pca_project(np.arange(5, dtype=np.float32).reshape(5, 1)).shape == (5, 2)
    assert tsne_project(z, perplexity=3, max_iter=250, seed=0).shape == (15, 2)
    with pytest.raises(ValueError, match="perplexity"):
        tsne_project(z, perplexity=15, max_iter=250, seed=0)
    with pytest.raises(ValueError, match="latents must have shape"):
        pca_project(np.ones((3, 1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="at least three"):
        pca_project(np.ones((2, 1), dtype=np.float32))


def test_representation_plots_write_pca_tsne_and_points(tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    cache = LatentCacheDataset(cache_path)

    result = evaluate_representation(
        cache,
        tmp_path / "repr",
        split_seed=0,
        perplexities=(3, 5),
        tsne_max_iter=250,
        max_points=None,
    )

    assert result["num_points"] == 48
    assert result["flux_color_component"] == 0
    assert result["held_out_points"] > 0
    assert Path(result["pca_plot"]).exists()
    assert len(result["tsne_plots"]) == 2
    assert all(Path(path).exists() for path in result["tsne_plots"])
    assert Path(result["points_npz"]).exists()
    assert Path(result["points_csv"]).exists()
    assert set(result["split_counts"]) >= {"train", "val", "test"}


def test_representation_plots_honor_configured_trajectories(tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    cache = LatentCacheDataset(cache_path)

    result = evaluate_representation(
        cache,
        tmp_path / "repr_subset",
        split_seed=0,
        trajectory_ids=("traj-0", "traj-1", "traj-2"),
        perplexities=(3, 5),
        tsne_max_iter=250,
        max_points=None,
    )

    assert result["num_configured_trajectories"] == 3
    assert set(result["configured_trajectories"]) == {"traj-0", "traj-1", "traj-2"}
    assert result["num_points"] == 36
    points = np.load(result["points_npz"])
    assert set(points["trajectory_id"].astype(str)) == {"traj-0", "traj-1", "traj-2"}

    with pytest.raises(ValueError, match="missing from latent cache"):
        evaluate_representation(
            cache,
            tmp_path / "repr_missing",
            split_seed=0,
            trajectory_ids=("traj-0", "missing-traj"),
            perplexities=(3, 5),
            tsne_max_iter=250,
        )

    with pytest.raises(ValueError, match="must be distinct"):
        evaluate_representation(
            cache,
            tmp_path / "repr_duplicate",
            split_seed=0,
            trajectory_ids=("traj-0", "traj-0", "traj-1"),
            perplexities=(3, 5),
            tsne_max_iter=250,
        )


def test_representation_plots_validate_inputs_and_subsample(tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    cache = LatentCacheDataset(cache_path)
    result = evaluate_representation(
        cache,
        tmp_path / "subsampled",
        split_seed=0,
        perplexities=(3, 5),
        tsne_max_iter=250,
        max_points=10,
    )
    assert result["num_points"] == 10

    with pytest.raises(ValueError, match="at least two distinct"):
        evaluate_representation(
            cache,
            tmp_path / "bad_perplexity",
            split_seed=0,
            perplexities=(5, 5, 99),
            tsne_max_iter=250,
            max_points=10,
        )

    missing_flux = tmp_path / "missing_flux.h5"
    writer = LatentCacheWriter(missing_flux, latent_dim=2)
    writer.write_trajectory("traj-a", np.ones((4, 2), dtype=np.float32))
    writer.write_trajectory("traj-b", np.ones((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="flux targets are missing"):
        evaluate_representation(
            LatentCacheDataset(missing_flux),
            tmp_path / "missing_flux_repr",
            perplexities=(2, 3),
            tsne_max_iter=250,
        )

    bad_flux = tmp_path / "bad_flux.h5"
    writer = LatentCacheWriter(bad_flux, latent_dim=2)
    writer.write_trajectory("traj-a", np.ones((4, 2), dtype=np.float32), flux=np.ones((4, 1, 1), dtype=np.float32))
    writer.write_trajectory("traj-b", np.ones((4, 2), dtype=np.float32), flux=np.ones((4, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="flux must have shape"):
        evaluate_representation(
            LatentCacheDataset(bad_flux),
            tmp_path / "bad_flux_repr",
            perplexities=(2, 3),
            tsne_max_iter=250,
        )

    no_held_out = tmp_path / "no_held_out.h5"
    writer = LatentCacheWriter(no_held_out, latent_dim=2)
    writer.write_trajectory("traj-a", np.ones((8, 2), dtype=np.float32), flux=np.ones((8, 1), dtype=np.float32))
    writer.write_trajectory("traj-b", np.ones((8, 2), dtype=np.float32), flux=np.ones((8, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty validation or test"):
        evaluate_representation(
            LatentCacheDataset(no_held_out),
            tmp_path / "no_held_out_repr",
            perplexities=(2, 3),
            tsne_max_iter=250,
        )


def test_representation_pipeline_writes_metrics(repo_root, tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    config = load_config(repo_root / "configs/experiment/smoke_plot_representation.yaml", command="plot-representation")
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "repr_pipeline"),
            "latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)}),
            "evaluation": config.evaluation.model_copy(
                update={
                    "tsne_perplexities": (3.0, 5.0),
                    "representation_max_points": None,
                }
            ),
        }
    )
    result = plot_representation(config)
    assert Path(result["metrics_json"]).exists()
    assert Path(result["pca_plot"]).exists()
    assert len(result["tsne_plots"]) == 2


def test_representation_pipeline_honors_configured_cache_trajectories(repo_root, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_requires_complete_protocol", lambda _config: False)
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    for index in range(4):
        monkeypatch.setenv(f"GK_SMALL_VALIDATION_TRAJ_{index}", f"traj-{index}")
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    config = load_config(
        repo_root / "configs/experiment/server_plot_representation_small.yaml",
        command="plot-representation",
    )
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "repr_subset_pipeline"),
            "data": config.data.model_copy(
                update={
                    "cyclone": config.data.cyclone.model_copy(
                        update={"trajectories": ("traj-0", "traj-1", "traj-2")}
                    )
                }
            ),
            "evaluation": config.evaluation.model_copy(
                update={
                    "tsne_perplexities": (3.0, 5.0),
                    "tsne_max_iter": 250,
                    "representation_max_points": None,
                }
            ),
            "latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)}),
        }
    )

    result = plot_representation(config)

    assert result["num_configured_trajectories"] == 3
    assert set(result["configured_trajectories"]) == {"traj-0", "traj-1", "traj-2"}
    assert result["split_counts"]["val"] > 0
    assert Path(result["metrics_json"]).exists()
