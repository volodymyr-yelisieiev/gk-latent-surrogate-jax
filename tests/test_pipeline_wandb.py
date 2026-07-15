from __future__ import annotations

from pathlib import Path
from typing import Any

from gk_surrogate import pipeline
from gk_surrogate.config.load import load_config


class FakeMetricsLogger:
    instances: list[FakeMetricsLogger] = []

    def __init__(
        self,
        output_dir: str | Path,
        *,
        wandb_config: dict[str, Any] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.wandb_config = dict(wandb_config or {})
        self.run_config = dict(run_config or {})
        self.logs: list[tuple[str | None, dict[str, Any]]] = []
        self.summary: dict[str, Any] | None = None
        self.finished_artifacts: tuple[Path, ...] = ()
        FakeMetricsLogger.instances.append(self)

    def log(self, metrics: dict[str, Any], *, prefix: str | None = None) -> None:
        self.logs.append((prefix, dict(metrics)))

    def write_summary(self, metrics: dict[str, Any]) -> Path:
        self.summary = dict(metrics)
        self.finish(metrics, artifact_paths=(self.output_dir / "metrics.json", self.output_dir / "metrics.jsonl"))
        return self.output_dir / "metrics.json"

    def finish(
        self,
        summary: dict[str, Any] | None = None,
        *,
        artifact_paths: tuple[str | Path | None, ...] = (),
    ) -> None:
        if summary is not None:
            self.summary = dict(summary)
        self.finished_artifacts = tuple(Path(path) for path in artifact_paths if path is not None)

    def wandb_status(self) -> dict[str, Any]:
        requested = bool(self.wandb_config.get("enabled")) and self.wandb_config.get("mode") != "disabled"
        return {
            "enabled": requested,
            "requested": requested,
            "mode": self.wandb_config.get("mode", "disabled"),
            "run_url": "https://wandb.local/pipeline-test" if requested else None,
            "run_dir": str(self.output_dir / "wandb") if requested else None,
        }


def test_pipeline_passes_wandb_config_to_train_and_eval_loggers(repo_root, tmp_path, monkeypatch):
    FakeMetricsLogger.instances.clear()
    monkeypatch.setattr(pipeline, "MetricsLogger", FakeMetricsLogger)

    train_cfg = load_config(
        repo_root / "configs/experiment/smoke_encoder_supervised.yaml",
        overrides=[
            "logging.wandb.enabled=true",
            "logging.wandb.mode=offline",
            "logging.wandb.project=unit-test",
            "logging.wandb.tags=[pipeline,wandb]",
            "logging.wandb.log_artifacts=true",
        ],
        command="train-encoder",
    )
    train_cfg = train_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "enc"),
            "training": train_cfg.training.model_copy(update={"max_steps": 1, "log_every": 1, "checkpoint_every": 1}),
        }
    )
    train_result = pipeline.train_encoder(train_cfg)
    train_logger = FakeMetricsLogger.instances[-1]

    assert train_logger.wandb_config["enabled"] is True
    assert train_logger.wandb_config["mode"] == "offline"
    assert train_logger.run_config["logging"]["wandb"]["project"] == "unit-test"
    assert train_logger.run_config["logging"]["wandb"]["tags"] == ["pipeline", "wandb"]
    assert train_logger.logs
    train_prefix, train_metrics = train_logger.logs[0]
    assert train_prefix == "train"
    assert "lr" in train_metrics
    assert "loss" in train_metrics
    assert train_result["wandb"]["requested"] is True
    assert train_result["system/device_count"] >= 1
    assert train_result["system/global_batch_size"] == train_result["global_batch_size"]

    embed_cfg = load_config(repo_root / "configs/experiment/smoke_embed_dataset.yaml", command="embed-dataset")
    embed_cfg = embed_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "embed"),
            "latent_cache": embed_cfg.latent_cache.model_copy(
                update={
                    "path": str(tmp_path / "embed" / "latent_cache.h5"),
                    "encoder_checkpoint_path": train_result["checkpoint"],
                }
            ),
        }
    )
    embedded = pipeline.embed_dataset(embed_cfg)

    eval_cfg = load_config(
        repo_root / "configs/experiment/smoke_evaluate_rollout.yaml",
        overrides=[
            "logging.wandb.enabled=true",
            "logging.wandb.mode=offline",
            "logging.wandb.project=unit-test",
            "logging.wandb.log_artifacts=true",
        ],
        command="evaluate-rollout",
    )
    eval_cfg = eval_cfg.model_copy(
        update={
            "output_dir": str(tmp_path / "eval"),
            "latent_cache": eval_cfg.latent_cache.model_copy(
                update={
                    "path": embedded["latent_cache"],
                    "encoder_checkpoint_path": train_result["checkpoint"],
                    "sequence_checkpoint_path": None,
                    "use_persistence_baseline": True,
                }
            ),
        }
    )
    eval_result = pipeline.evaluate_rollout(eval_cfg)
    eval_logger = FakeMetricsLogger.instances[-1]

    assert eval_result["wandb"]["requested"] is True
    assert eval_logger.logs
    eval_prefix, eval_metrics = eval_logger.logs[0]
    assert eval_prefix == "eval"
    assert eval_metrics["metrics_json"] == eval_result["metrics_json"]
    assert eval_metrics["metrics_by_step_csv"] == eval_result["metrics_by_step_csv"]
    assert any(path.name == "metrics.json" for path in eval_logger.finished_artifacts)
    assert any(path.name == "metrics_by_step.csv" for path in eval_logger.finished_artifacts)
    assert any(path.suffix == ".png" for path in eval_logger.finished_artifacts)
