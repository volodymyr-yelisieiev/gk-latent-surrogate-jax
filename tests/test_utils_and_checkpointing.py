from __future__ import annotations

import builtins
import hashlib
import json
import pickle
import subprocess
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gk_surrogate.training import logging as logging_module
from gk_surrogate.training.checkpointing import latest_checkpoint, load_checkpoint, restore_train_state, save_checkpoint
from gk_surrogate.training.logging import MetricsLogger, collect_git_info
from gk_surrogate.training.rng import PRNGSequence, fold_in_rng, make_rng, split_rng
from gk_surrogate.training.state import TrainState
from gk_surrogate.utils.arrays import as_float32, assert_rank, ensure_finite_tree, mean_squared_error, to_numpy
from gk_surrogate.utils.paths import ensure_dir, timestamped_run_dir
from gk_surrogate.utils.pretty import flatten_dict, scalarize
from gk_surrogate.utils.timing import timed
from gk_surrogate.utils.tree import tree_allclose


def test_utils_training_state_checkpoint_and_logging(tmp_path):
    assert as_float32([1]).dtype == jnp.float32
    assert_rank(jnp.ones((2, 3)), 2)
    ensure_finite_tree({"x": jnp.ones((2, 3))})
    assert float(mean_squared_error(jnp.ones((1,)), jnp.zeros((1,)))) == 1.0
    assert to_numpy(jnp.ones((1,))).shape == (1,)
    assert ensure_dir(tmp_path / "a" / "b").exists()
    assert timestamped_run_dir(tmp_path, "run").exists()
    assert flatten_dict({"a": {"b": 1}}) == {"a/b": 1}
    assert scalarize(jnp.asarray(2.0)) == 2.0
    with timed() as elapsed:
        _ = elapsed

    rng = make_rng(0)
    assert len(split_rng(rng, 3)) == 3
    assert fold_in_rng(rng, 1).shape == rng.shape
    seq = PRNGSequence(0)
    assert seq.next().shape == rng.shape

    def apply_fn(params, x):
        return x @ params["w"]

    params = {"w": jnp.ones((2, 1))}
    state = TrainState.create(
        apply_fn=apply_fn,
        params=params,
        tx=optax.sgd(0.1),
        rng=rng,
        model_config={"loss": "test"},
    )
    ckpt = save_checkpoint(state, tmp_path, step=0)
    assert latest_checkpoint(tmp_path) == ckpt
    (tmp_path / "checkpoints" / "step_999999").mkdir()
    assert latest_checkpoint(tmp_path) == ckpt
    with pytest.raises(ValueError, match="does not match train state step"):
        save_checkpoint(state, tmp_path, step=1)
    restored = restore_train_state(state, ckpt)
    assert tree_allclose(restored.params, state.params)

    logger = MetricsLogger(tmp_path / "logs")
    logger.log({"loss": 1.0})
    logger.write_summary({"loss": 1.0})
    assert collect_git_info(tmp_path)["git_available"] in {True, False}


def test_collect_git_info_records_exact_patch_hash_and_untracked_paths(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    clean = collect_git_info(tmp_path)
    assert clean["tracked_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert clean["has_untracked_paths"] is False
    assert clean["untracked_path_count"] == 0
    assert clean["untracked_paths"] == []

    tracked.write_text("after\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("untracked\n", encoding="utf-8")
    expected_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout
    dirty = collect_git_info(tmp_path)
    assert dirty["dirty"] is True
    assert dirty["tracked_diff_sha256"] == hashlib.sha256(expected_diff).hexdigest()
    assert dirty["has_untracked_paths"] is True
    assert dirty["untracked_path_count"] == 1
    assert dirty["untracked_paths"] == ["new.txt"]


def test_metrics_logger_disabled_wandb_does_not_import(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled W&B logging must not import wandb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    logger = MetricsLogger(
        tmp_path / "disabled",
        wandb_config={"enabled": False, "mode": "disabled"},
        run_config={"name": "disabled-run"},
    )
    logger.log({"step": 1, "loss": 1.0}, prefix="train")
    logger.write_summary({"loss": 1.0})
    assert logger.wandb_status() == {"enabled": False, "requested": False, "mode": "disabled"}
    assert json.loads((tmp_path / "disabled" / "wandb_status.json").read_text(encoding="utf-8")) == {
        "enabled": False,
        "mode": "disabled",
        "requested": False,
    }


def test_metrics_logger_uses_optional_wandb_when_enabled(tmp_path, monkeypatch):
    calls: list[tuple[str, object]] = []

    class FakeRun:
        url = "https://wandb.local/unit-test/run"
        dir = "/tmp/wandb-offline"

        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def log(self, payload, *, step=None):
            calls.append(("log", (payload, step)))

        def save(self, path):
            calls.append(("save", path))

        def finish(self):
            calls.append(("finish", None))

    fake_run = FakeRun()

    def fake_init(**kwargs):
        calls.append(("init", kwargs))
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    logger = MetricsLogger(
        tmp_path / "wandb_logs",
        wandb_config={
            "enabled": True,
            "mode": "offline",
            "project": "unit-test",
            "tags": ("one-traj",),
            "log_artifacts": True,
        },
        run_config={"name": "fake-run"},
    )
    logger.log({"step": 1, "loss": 2.0, "checkpoint": "/tmp/model", "curve": [1.0, 2.0]}, prefix="train")
    logger.write_summary({"loss": 1.0})

    init = next(payload for name, payload in calls if name == "init")
    assert init["project"] == "unit-test"
    assert init["mode"] == "offline"
    assert init["name"] == "fake-run"
    logged, step = next(payload for name, payload in calls if name == "log")
    assert logged["train/loss"] == 2.0
    assert "train/checkpoint" not in logged
    assert "train/curve" not in logged
    assert step == 1
    assert init["job_type"] == "experiment"
    assert ("finish", None) in calls
    assert fake_run.summary["loss"] == 1.0
    assert (tmp_path / "wandb_logs" / "wandb_status.json").exists()


def test_metrics_logger_wandb_import_failure_is_non_fatal(tmp_path, monkeypatch):
    def missing_wandb(name):
        if name == "wandb":
            raise ModuleNotFoundError("no wandb")
        raise AssertionError(name)

    monkeypatch.setattr(logging_module.importlib, "import_module", missing_wandb)
    logger = MetricsLogger(
        tmp_path / "missing_wandb",
        wandb_config={"enabled": True, "mode": "offline"},
        run_config={"name": "missing-run"},
    )

    logger.log({"loss": 1.0})
    logger.write_summary({"loss": 1.0})
    status = json.loads((tmp_path / "missing_wandb" / "wandb_status.json").read_text())
    assert status["requested"] is True
    assert status["available"] is False
    assert "wandb import failed" in status["warning"]


def test_metrics_logger_wandb_init_failure_is_non_fatal(tmp_path, monkeypatch):
    def failing_init(**kwargs):
        raise RuntimeError("offline init failed")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=failing_init))
    logger = MetricsLogger(
        tmp_path / "failing_wandb",
        wandb_config={"enabled": True, "mode": "offline", "project": "unit-test"},
        run_config={"name": "failing-run"},
    )

    logger.log({"loss": 1.0})
    logger.write_summary({"loss": 1.0})
    status = json.loads((tmp_path / "failing_wandb" / "wandb_status.json").read_text())
    assert status["requested"] is True
    assert status["available"] is True
    assert "wandb init failed" in status["warning"]


def test_metrics_logger_registers_wandb_tables_plots_and_run_artifact(tmp_path, monkeypatch):
    logged: list[dict[str, object]] = []
    artifacts: list[object] = []
    init_kwargs: dict[str, object] = {}

    class FakeTable:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    class FakeImage:
        def __init__(self, path):
            self.path = path

    class FakeArtifact:
        def __init__(self, *, name, type):
            self.name = name
            self.type = type
            self.files: list[tuple[str, str]] = []

        def add_file(self, path, *, name):
            self.files.append((path, name))

    class FakeRun:
        url = "https://wandb.local/eval"
        dir = "/tmp/wandb-eval"

        def __init__(self):
            self.summary: dict[str, object] = {}

        def log(self, payload, *, step=None):
            del step
            logged.append(payload)

        def log_artifact(self, artifact):
            artifacts.append(artifact)

        def finish(self):
            pass

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fake_init, Table=FakeTable, Image=FakeImage, Artifact=FakeArtifact),
    )
    csv_path = tmp_path / "metrics_by_step.csv"
    csv_path.write_text("step,mse\n1,0.5\n", encoding="utf-8")
    plot_path = tmp_path / "latent_mse_by_step.png"
    plot_path.write_bytes(b"plot")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"mse": 0.5}', encoding="utf-8")

    logger = MetricsLogger(
        tmp_path / "run",
        wandb_config={"enabled": True, "mode": "offline", "log_artifacts": True},
        run_config={"name": "server_evaluate_rollout_medium_gru"},
    )
    logger.finish(
        {"mse": 0.5, "mse_by_step": [0.5]},
        artifact_paths=(metrics_path, csv_path, plot_path),
    )

    assert init_kwargs["job_type"] == "rollout-evaluation"
    assert any("tables/metrics_by_step" in payload for payload in logged)
    assert any("plots/latent_mse_by_step" in payload for payload in logged)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.name == "server_evaluate_rollout_medium_gru-outputs"
    assert artifact.type == "run-output"
    assert {name for _path, name in artifact.files} == {
        "metrics.json",
        "metrics_by_step.csv",
        "latent_mse_by_step.png",
    }


def test_metrics_logging_serialization_and_naming_contracts(tmp_path):
    """Non-scalar artifacts must not leak into the W&B metric namespace."""

    class ArrayConversionFailure:
        def __array__(self):
            raise TypeError("not scalarizable")

    class ListConvertible:
        def tolist(self):
            return [1, np.nan, 3]

    class StableString:
        def __str__(self):
            return "stable-value"

    logger = MetricsLogger(tmp_path / "disabled")
    assert logger._wandb_payload(
        {
            "step": 2,
            "loss": jnp.asarray(1.25),
            "nested/already_namespaced": 3.0,
            "array": np.ones(2),
            "none": None,
            "bad": ArrayConversionFailure(),
            "nonfinite": np.inf,
            "flag": True,
        },
        prefix="train",
    ) == {
        "step": 2,
        "train/loss": 1.25,
        "nested/already_namespaced": 3.0,
        "train/flag": True,
    }
    assert logging_module._flatten_mapping({"eval": {"loss": 0.5}}) == {"eval/loss": 0.5}
    assert logging_module._json_ready(
        {
            "path": tmp_path,
            "values": (1, np.inf, True),
            "array_like": ListConvertible(),
            "label": "validation",
            "missing": None,
            "opaque": StableString(),
        }
    ) == {
        "path": str(tmp_path),
        "values": [1, None, True],
        "array_like": [1, None, 3],
        "label": "validation",
        "missing": None,
        "opaque": "stable-value",
    }

    # Empty/invalid names receive a stable artifact-safe fallback.
    assert logging_module._wandb_safe_name("///") == "gk-surrogate-run"
    assert logging_module._wandb_safe_name("thesis run: 01") == "thesis-run-01"
    assert [logging_module._parse_csv_cell(value) for value in ("", "3", "0.25", "ok")] == [
        None,
        3,
        0.25,
        "ok",
    ]
    assert logging_module._infer_job_type("flux_head_validation") == "diagnostic-evaluation"
    assert logging_module._infer_job_type("representation_plot") == "representation-evaluation"
    assert logging_module._infer_job_type("embed_dataset") == "dataset-embedding"
    assert logging_module._infer_job_type("train_sequence") == "sequence-training"
    assert logging_module._infer_job_type("train_encoder") == "encoder-training"

    # These no-op paths are part of the optional dependency contract.
    logger._log_wandb_previews(())
    logger._log_wandb_artifact(())


def test_checkpoint_loader_rejects_malformed_payloads_and_empty_roots(tmp_path):
    assert latest_checkpoint(tmp_path / "missing") is None
    empty_root = tmp_path / "empty"
    (empty_root / "checkpoints").mkdir(parents=True)
    assert latest_checkpoint(empty_root) is None

    scalar_path = tmp_path / "scalar.pkl"
    scalar_path.write_bytes(pickle.dumps(42))
    with pytest.raises(ValueError, match="must be a mapping"):
        load_checkpoint(scalar_path)

    incomplete_path = tmp_path / "incomplete.pkl"
    incomplete_path.write_bytes(pickle.dumps({"step": 1}))
    with pytest.raises(ValueError, match="missing required fields.*params"):
        load_checkpoint(incomplete_path)
