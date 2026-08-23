from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gk_surrogate import pipeline as pipeline_module
from gk_surrogate.config.load import load_config
from gk_surrogate.data.latent_cache import LatentCacheDataset, LatentCacheWriter
from gk_surrogate.data.split import TrajectorySplits
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


def test_representation_plots_use_exact_trajectory_splits_and_are_deterministic(tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    cache = LatentCacheDataset(cache_path)
    splits = TrajectorySplits(
        train=("traj-0", "traj-1"),
        val=("traj-2",),
        test=("traj-3",),
        strategy="explicit_manifest",
        manifest_path="outer_fold_0.json",
        manifest_sha256="a" * 64,
        fold_id="outer-0",
    )

    first = evaluate_representation(
        cache,
        tmp_path / "first",
        split_seed=52,
        trajectory_splits=splits,
        perplexities=(3, 5),
        tsne_max_iter=250,
        max_points=None,
    )
    second = evaluate_representation(
        cache,
        tmp_path / "second",
        split_seed=52,
        trajectory_splits=splits,
        perplexities=(3, 5),
        tsne_max_iter=250,
        max_points=None,
    )

    assert first["split_source"] == "explicit_manifest"
    assert first["split_manifest_sha256"] == "a" * 64
    assert first["split_fold_id"] == "outer-0"
    assert first["split_counts"] == {"test": 12, "train": 24, "val": 12}
    first_points = np.load(first["points_npz"])
    second_points = np.load(second["points_npz"])
    assert set(first_points["split"].astype(str)) == {"train", "val", "test"}
    assert np.array_equal(first_points["pca"], second_points["pca"])
    assert np.array_equal(first_points["tsne_perplexity_3"], second_points["tsne_perplexity_3"])


def test_representation_plots_reject_incomplete_or_overlapping_exact_splits(tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    cache = LatentCacheDataset(cache_path)

    incomplete = TrajectorySplits(
        train=("traj-0", "traj-1"),
        val=("traj-2",),
        test=(),
        strategy="explicit_manifest",
    )
    with pytest.raises(ValueError, match="missing configured trajectories"):
        evaluate_representation(
            cache,
            tmp_path / "incomplete",
            trajectory_splits=incomplete,
            perplexities=(3, 5),
            tsne_max_iter=250,
        )

    overlapping = TrajectorySplits(
        train=("traj-0", "traj-1"),
        val=("traj-1", "traj-2"),
        test=("traj-3",),
        strategy="explicit_manifest",
    )
    with pytest.raises(ValueError, match="assignments overlap"):
        evaluate_representation(
            cache,
            tmp_path / "overlap",
            trajectory_splits=overlapping,
            perplexities=(3, 5),
            tsne_max_iter=250,
        )

    extra = TrajectorySplits(
        train=("traj-0", "traj-1", "extra-traj"),
        val=("traj-2",),
        test=("traj-3",),
        strategy="explicit_manifest",
    )
    with pytest.raises(ValueError, match="contain extra trajectories"):
        evaluate_representation(
            cache,
            tmp_path / "extra",
            trajectory_splits=extra,
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


def test_representation_pipeline_uses_explicit_split_manifest(repo_root, tmp_path, monkeypatch):
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    monkeypatch.setattr(
        pipeline_module,
        "_validate_latent_cache_protocol",
        lambda *_args, **_kwargs: ("traj-0", "traj-1", "traj-2", "traj-3"),
    )
    monkeypatch.setattr(pipeline_module, "_optional_cache_encoder_lineage", lambda *_args: None)
    split_manifest = tmp_path / "outer_fold_0.json"
    split_manifest.write_text(
        json.dumps(
            {
                "fold_id": "outer-0",
                "splits": {
                    "train": ["traj-0", "traj-1"],
                    "val": ["traj-2"],
                    "test": ["traj-3"],
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(repo_root / "configs/experiment/smoke_plot_representation.yaml", command="plot-representation")
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "repr_manifest"),
            "data": config.data.model_copy(update={"split_manifest": str(split_manifest), "seed": 52}),
            "training": config.training.model_copy(update={"seed": 52}),
            "latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)}),
            "evaluation": config.evaluation.model_copy(
                update={"tsne_perplexities": (3.0, 5.0), "representation_max_points": None}
            ),
        }
    )

    result = plot_representation(config)

    assert result["split_source"] == "explicit_manifest"
    assert result["split_fold_id"] == "outer-0"
    assert result["split_counts"] == {"test": 12, "train": 24, "val": 12}
    assert result["split_manifest_sha256"] == hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    assert result["latent_cache_sha256"] == hashlib.sha256(cache_path.read_bytes()).hexdigest()
    assert result["encoder_checkpoint_sha256"] is None
    assert result["embed_config_resolved_sha256"] is None
    assert result["encoder_config_resolved_sha256"] is None


def test_representation_pipeline_honors_configured_cache_trajectories(repo_root, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_requires_complete_protocol", lambda _config: False)
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", "/tmp/gk-cyclone-root")
    for index in range(4):
        monkeypatch.setenv(f"GK_VALIDATION_TRAJ_{index}", f"traj-{index}")
    cache_path = tmp_path / "latent_cache.h5"
    _write_representation_cache(cache_path)
    config = load_config(
        repo_root / "configs/experiment/server_evaluate_observed_persistence_medium.yaml",
        command="plot-representation",
    )
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "repr_subset_pipeline"),
            "data": config.data.model_copy(
                update={
                    "split": "all",
                    "cyclone": config.data.cyclone.model_copy(update={"trajectories": ("traj-0", "traj-1", "traj-2")}),
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
