from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _figure_module(repo_root: Path):
    path = repo_root / "thesis" / "scripts" / "build_thesis_figures.py"
    spec = importlib.util.spec_from_file_location("build_thesis_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_methodology_shape_trace_matches_medium_encoder(repo_root, tmp_path):
    module = _figure_module(repo_root)
    contract = module.load_methodology_contract()

    assert contract["trace"] == (
        (4, 32, 8, 16, 85, 32),
        (8, 16, 4, 8, 22, 16),
        (16, 8, 4, 4, 6, 8),
        (32, 4, 4, 4, 3, 4),
    )
    output = module.build_methodology_figure(
        repo_root / "configs" / "experiment" / "server_encoder_simsiam_medium.yaml",
        tmp_path / "methodology.png",
    )
    assert output.is_file()
    assert output.stat().st_size > 10_000


def test_methodology_shape_trace_rejects_invalid_contract(repo_root):
    module = _figure_module(repo_root)
    with pytest.raises(ValueError, match="input_shape"):
        module.pointwise_shape_trace((4, 2), (8,), ((2, 2, 2, 2, 2),))
    with pytest.raises(ValueError, match="same non-zero"):
        module.pointwise_shape_trace((4, 2, 2, 2, 2, 2), (8, 16), ((2, 2, 2, 2, 2),))


def test_latent_figure_requires_canonical_metadata(repo_root, tmp_path):
    module = _figure_module(repo_root)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"num_points": 48}), encoding="utf-8")
    points = tmp_path / "points.npz"
    np.savez_compressed(points, flux=np.ones((48, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="not the canonical thesis run"):
        module.build_latent_space_figure(points, metrics, tmp_path / "latent.png")


def test_latent_figure_and_provenance_use_verified_points(repo_root, tmp_path):
    module = _figure_module(repo_root)
    release = json.loads(module.DEFAULT_RELEASE_MANIFEST.read_text(encoding="utf-8"))
    stage_hashes = release["stage_config_resolved_sha256"]
    rng = np.random.default_rng(52)
    points = tmp_path / "points.npz"
    split = np.asarray(["train"] * 713 + ["val"] * 230 + ["test"] * 230)
    np.savez_compressed(
        points,
        flux=rng.normal(size=(1173, 1)).astype(np.float32),
        split=split,
        pca=rng.normal(size=(1173, 2)).astype(np.float32),
        tsne_perplexity_5=rng.normal(size=(1173, 2)).astype(np.float32),
        tsne_perplexity_30=rng.normal(size=(1173, 2)).astype(np.float32),
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "num_points": 1173,
                "latent_dim": 128,
                "split_source": "explicit_manifest",
                "split_fold_id": "outer-0",
                "split_manifest_sha256": module._sha256(module.DEFAULT_FOLD_MANIFEST),
                "latent_cache_sha256": "b" * 64,
                "encoder_checkpoint_sha256": "c" * 64,
                "embed_config_resolved_sha256": stage_hashes["outer_fold_0/seed_52/embed"],
                "encoder_config_resolved_sha256": stage_hashes["outer_fold_0/seed_52/encoder"],
                "protocol_version": 1,
                "data_split_seed": 52,
                "training_seed": 52,
                "perplexities": [5.0, 30.0],
            }
        ),
        encoding="utf-8",
    )

    figure = module.build_latent_space_figure(points, metrics, tmp_path / "latent.png")
    provenance = module.write_latent_provenance(points, metrics, tmp_path / "provenance.json")
    payload = json.loads(provenance.read_text(encoding="utf-8"))

    assert figure.is_file()
    assert figure.stat().st_size > 10_000
    assert payload["protocol_id"] == "multiseed-v1"
    assert payload["outer_fold"] == 0
    assert payload["training_seed"] == 52
    assert payload["latent_cache_sha256"] == "b" * 64
    assert payload["encoder_checkpoint_sha256"] == "c" * 64
    assert payload["embed_config_resolved_sha256"] == stage_hashes["outer_fold_0/seed_52/embed"]
    assert payload["encoder_config_resolved_sha256"] == stage_hashes["outer_fold_0/seed_52/encoder"]
    assert len(payload["points_sha256"]) == 64
