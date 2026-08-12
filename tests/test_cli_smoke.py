from __future__ import annotations

from pathlib import Path

from gk_surrogate.cli import main


def test_cli_smoke_full_pipeline_uses_temp_dirs(repo_root, tmp_path):
    data_config = repo_root / "configs/data/tiny_dummy.yaml"
    assert main(["inspect-data", "--config", str(data_config), "--dry-run", "--max-target-samples", "4"]) == 0

    supervised = repo_root / "configs/experiment/smoke_encoder_supervised.yaml"
    assert (
        main(
            [
                "train-encoder",
                "--config",
                str(supervised),
                "--override",
                "training.max_steps=2",
                "--output-dir",
                str(tmp_path / "enc"),
            ]
        )
        == 0
    )
    assert (tmp_path / "enc" / "metrics.json").exists()

    assert (
        main(
            [
                "train-direct-diagnostics",
                "--config",
                str(supervised),
                "--override",
                "training.max_steps=1",
                "--override",
                "training.eval_every=1",
                "--output-dir",
                str(tmp_path / "direct_diagnostics"),
            ]
        )
        == 0
    )
    assert (tmp_path / "direct_diagnostics" / "metrics.json").exists()

    enc_ckpt = tmp_path / "enc" / "checkpoints" / "step_000002"
    embed = repo_root / "configs/experiment/smoke_embed_dataset.yaml"
    assert (
        main(
            [
                "embed-dataset",
                "--config",
                str(embed),
                "--output-dir",
                str(tmp_path / "embed"),
                "--override",
                f"latent_cache.encoder_checkpoint_path={enc_ckpt}",
            ]
        )
        == 0
    )
    assert (tmp_path / "embed" / "latent_cache.h5").exists()

    flux_head = repo_root / "configs/experiment/smoke_evaluate_flux_head.yaml"
    assert (
        main(
            [
                "evaluate-flux-head",
                "--config",
                str(flux_head),
                "--output-dir",
                str(tmp_path / "flux_head"),
                "--override",
                f"latent_cache.path={tmp_path / 'embed' / 'latent_cache.h5'}",
                "--override",
                f"latent_cache.encoder_checkpoint_path={enc_ckpt}",
            ]
        )
        == 0
    )
    assert Path(tmp_path / "flux_head" / "metrics.json").exists()

    representation = repo_root / "configs/experiment/smoke_plot_representation.yaml"
    assert (
        main(
            [
                "plot-representation",
                "--config",
                str(representation),
                "--output-dir",
                str(tmp_path / "representation"),
                "--override",
                f"latent_cache.path={tmp_path / 'embed' / 'latent_cache.h5'}",
                "--override",
                f"latent_cache.encoder_checkpoint_path={enc_ckpt}",
                "--override",
                "evaluation.tsne_perplexities=[3,5]",
            ]
        )
        == 0
    )
    assert Path(tmp_path / "representation" / "plots" / "pca_flux.png").exists()

    sequence = repo_root / "configs/experiment/smoke_sequence.yaml"
    assert (
        main(
            [
                "train-sequence",
                "--config",
                str(sequence),
                "--override",
                "training.max_steps=2",
                "--override",
                f"latent_cache.path={tmp_path / 'embed' / 'latent_cache.h5'}",
                "--override",
                f"latent_cache.encoder_checkpoint_path={enc_ckpt}",
                "--output-dir",
                str(tmp_path / "seq"),
            ]
        )
        == 0
    )

    evaluate = repo_root / "configs/experiment/smoke_evaluate_rollout.yaml"
    assert (
        main(
            [
                "evaluate-rollout",
                "--config",
                str(evaluate),
                "--output-dir",
                str(tmp_path / "eval"),
                "--override",
                f"latent_cache.path={tmp_path / 'embed' / 'latent_cache.h5'}",
                "--override",
                f"latent_cache.encoder_checkpoint_path={enc_ckpt}",
                "--override",
                f"latent_cache.sequence_checkpoint_path={tmp_path / 'seq' / 'checkpoints' / 'step_000002'}",
            ]
        )
        == 0
    )
    assert Path(tmp_path / "eval" / "metrics_by_step.csv").exists()

    h5_out = tmp_path / "h5"
    assert main(["make-synthetic-h5", "--config", str(data_config), "--output-dir", str(h5_out)]) == 0
    assert list(h5_out.glob("*.h5"))
    assert main(["benchmark-step-time", "--config", str(supervised), "--dry-run"]) == 0
