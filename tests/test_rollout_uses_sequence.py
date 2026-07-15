from __future__ import annotations

from pathlib import Path

from gk_surrogate.config.load import load_config
from gk_surrogate.pipeline import embed_dataset, evaluate_rollout, train_encoder, train_sequence


def test_evaluate_rollout_uses_sequence_checkpoint_not_persistence(repo_root, tmp_path, monkeypatch):
    enc_cfg = load_config(repo_root / "configs/experiment/smoke_encoder_supervised.yaml", command="train-encoder")
    enc_cfg = enc_cfg.model_copy(update={"output_dir": str(tmp_path / "enc")})
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

    seq_cfg = load_config(repo_root / "configs/experiment/smoke_sequence.yaml", command="train-sequence")
    seq_cfg = seq_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "seq"),
            "latent_cache": seq_cfg.latent_cache.model_copy(update={"path": embedded["latent_cache"]}),
        }
    )
    seq = train_sequence(seq_cfg)

    def fail_persistence(*args, **kwargs):
        raise AssertionError("persistence baseline should not be used")

    monkeypatch.setattr("gk_surrogate.pipeline.persistence_rollout", fail_persistence)

    eval_cfg = load_config(repo_root / "configs/experiment/smoke_evaluate_rollout.yaml", command="evaluate-rollout")
    eval_cfg = eval_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "eval"),
            "latent_cache": eval_cfg.latent_cache.model_copy(
                update={
                    "path": embedded["latent_cache"],
                    "sequence_checkpoint_path": seq["checkpoint"],
                    "use_persistence_baseline": False,
                }
            ),
        }
    )
    result = evaluate_rollout(eval_cfg)
    assert Path(result["metrics_json"]).exists()
