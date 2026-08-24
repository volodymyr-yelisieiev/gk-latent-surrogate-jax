from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import CycloneKvikIOConfig
from gk_surrogate.data.split import resolve_trajectory_splits, split_manifest_assigned_ids
from gk_surrogate.parallel.devices import ParallelPlan
from gk_surrogate.pipeline import (
    _aggregate_eval_metrics,
    _aggregate_eval_metrics_by_trajectory,
    _batch_size,
    _build_universe_dataset,
    _config_with_parallel_optimizer,
    _protocol_fields,
    _relative_artifact_path,
    embed_dataset,
    evaluate_rollout,
    train_direct_diagnostics,
    train_encoder,
    train_sequence,
)


def _write_manifest(path: Path, *, fold_id: str = "outer-0") -> Path:
    path.write_text(
        json.dumps(
            {
                "fold_id": fold_id,
                "train": ["synthetic_0001", "synthetic_0003"],
                "val": ["synthetic_0000"],
                "test": ["synthetic_0002"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_explicit_manifest_rejects_overlap_unknown_and_unassigned_ids(tmp_path: Path) -> None:
    overlap = tmp_path / "overlap.json"
    overlap.write_text(
        json.dumps({"train": ["a"], "val": ["a"], "test": ["b"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        resolve_trajectory_splits(("a", "b"), manifest_path=overlap)

    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps({"train": ["a"], "val": ["b"], "test": ["missing"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing from dataset/cache"):
        resolve_trajectory_splits(("a", "b", "c"), manifest_path=valid)

    valid.write_text(
        json.dumps({"train": ["a"], "val": ["b"], "test": ["c"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not assign"):
        resolve_trajectory_splits(("a", "b", "c", "d"), manifest_path=valid)


def test_split_manifest_parser_rejects_malformed_inputs_and_supports_yaml(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        split_manifest_assigned_ids(missing)
    with pytest.raises(FileNotFoundError):
        resolve_trajectory_splits(("a", "b", "c"), manifest_path=missing)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        split_manifest_assigned_ids(scalar)
    with pytest.raises(ValueError, match="must contain a mapping"):
        resolve_trajectory_splits(("a", "b", "c"), manifest_path=scalar)

    bad_splits = tmp_path / "bad_splits.json"
    bad_splits.write_text(json.dumps({"splits": "invalid"}), encoding="utf-8")
    with pytest.raises(ValueError, match="'splits' must be a mapping"):
        split_manifest_assigned_ids(bad_splits)
    with pytest.raises(ValueError, match="'splits' must be a mapping"):
        resolve_trajectory_splits(("a", "b", "c"), manifest_path=bad_splits)

    valid_yaml = tmp_path / "valid.yaml"
    valid_yaml.write_text(
        "fold_id: outer-0\nsplits:\n  train: [a]\n  val: [b]\n  test: [c]\n",
        encoding="utf-8",
    )
    assert split_manifest_assigned_ids(valid_yaml) == ("a", "b", "c")
    resolved = resolve_trajectory_splits(("a", "b", "c"), manifest_path=valid_yaml)
    assert resolved.as_dict() == {"train": ("a",), "val": ("b",), "test": ("c",)}

    assert resolve_trajectory_splits(("only",)).strategy == "single_trajectory_fallback"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"train": [], "val": ["b"], "test": ["c"]}, "non-empty string list"),
        ({"train": ["a", "a"], "val": ["b"], "test": ["c"]}, "duplicate"),
        ({"fold_id": "", "train": ["a"], "val": ["b"], "test": ["c"]}, "fold_id"),
    ],
)
def test_split_manifest_rejects_invalid_roles_and_fold_id(
    tmp_path: Path,
    payload: dict[str, object],
    match: str,
) -> None:
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        resolve_trajectory_splits(("a", "b", "c"), manifest_path=manifest)


def test_absolute_server_paths_are_portable_and_full_universe_is_preserved(
    repo_root: Path,
) -> None:
    config = load_config(
        repo_root / "configs/experiment/smoke_encoder_supervised.yaml",
        command="train-encoder",
    )
    universe = tuple(f"/system/user/publicdata/run/iteration_{index}_ifft_realpotens" for index in range(3))
    from gk_surrogate import pipeline

    selected = pipeline._selected_ids_from_universe(config, universe)
    fields = _protocol_fields(
        config,
        selected,
        aggregation="test",
        universe_trajectory_ids=universe,
    )
    assert fields["num_universe_trajectories"] == 3
    assert fields["universe_trajectory_ids"] == [
        "iteration_0",
        "iteration_1",
        "iteration_2",
    ]
    assert 0 < len(fields["selected_trajectory_ids"]) < 3
    assert all(not value.startswith("/") for value in fields["selected_trajectory_ids"])


def test_pipeline_validation_helpers_cover_fail_closed_edges(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    config = load_config(
        repo_root / "configs/experiment/smoke_encoder_supervised.yaml",
        command="train-encoder",
    )
    plan = ParallelPlan(
        mode="pmap",
        axis_name="devices",
        num_devices=4,
        local_device_count=4,
        global_batch_size=4,
        per_device_batch_size=1,
        drop_remainder=True,
        auto_scale_learning_rate=True,
        devices=("d0", "d1", "d2", "d3"),
    )
    scaled = _config_with_parallel_optimizer(config, plan)
    assert scaled.training.learning_rate == config.training.learning_rate * 4
    assert not _relative_artifact_path(tmp_path / "artifact").startswith("/")

    with pytest.raises(ValueError, match="no batched array"):
        _batch_size({})
    single = plan.__class__(**{**plan.__dict__, "mode": "single"})
    with pytest.raises(ValueError, match="no evaluable batches"):
        _aggregate_eval_metrics(None, iter(()), plan=single, step_fn=lambda *_: {})
    with pytest.raises(ValueError, match="at least one trajectory"):
        _aggregate_eval_metrics_by_trajectory(
            None,
            (),
            batches_for_trajectory=lambda _: iter(()),
            plan=single,
            step_fn=lambda *_: {},
        )

    def batches_for_trajectory(trajectory_id: str):
        value = 1 if trajectory_id == "a" else 2
        return iter(({"x": np.asarray([value], dtype=np.float32)},))

    def inconsistent_step(_state, batch):
        return {"first" if int(batch["x"][0]) == 1 else "second": 1.0}

    with pytest.raises(ValueError, match="inconsistent metric sets"):
        _aggregate_eval_metrics_by_trajectory(
            None,
            ("a", "b"),
            batches_for_trajectory=batches_for_trajectory,
            plan=single,
            step_fn=inconsistent_step,
        )


def test_explicit_requested_trajectory_missing_from_reader_fails_closed(
    repo_root: Path,
    monkeypatch,
) -> None:
    config = load_config(
        repo_root / "configs/experiment/smoke_encoder_supervised.yaml",
        command="train-encoder",
    )
    requested = tuple(f"iteration_{index}" for index in range(3))
    cyclone = CycloneKvikIOConfig(trajectories=requested)
    config = config.model_copy(
        update={"data": config.data.model_copy(update={"backend": "cyclone_kvikio", "cyclone": cyclone})}
    )

    class IncompleteDataset:
        def trajectory_ids(self):
            return ("/server/iteration_0_ifft_realpotens", "/server/iteration_2_ifft")

    monkeypatch.setattr("gk_surrogate.pipeline.build_dataset", lambda _config: IncompleteDataset())
    with pytest.raises(ValueError, match="iteration_1"):
        _build_universe_dataset(config)


def test_explicit_outer_fold_is_respected_end_to_end(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    manifest = _write_manifest(tmp_path / "outer_fold.json")
    common = [f"data.split_manifest={manifest}", "training.max_steps=1", "training.eval_every=1"]
    normalization_fit_ids: list[tuple[str, ...]] = []
    from gk_surrogate import pipeline

    original_estimate = pipeline.estimate_dataset_stats

    def record_estimate(*args, **kwargs):
        ids = kwargs.get("trajectory_ids")
        if ids is not None:
            normalization_fit_ids.append(tuple(ids))
        return original_estimate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "estimate_dataset_stats", record_estimate)

    encoder_config = load_config(
        repo_root / "configs/experiment/smoke_encoder_supervised.yaml",
        overrides=common,
        command="train-encoder",
    ).model_copy(update={"output_dir": str(tmp_path / "encoder")})
    encoder = train_encoder(encoder_config)
    assert encoder["selected_trajectory_ids"] == ["synthetic_0001", "synthetic_0003"]
    assert encoder["validation_trajectory_ids"] == ["synthetic_0000"]
    assert encoder["test_trajectory_ids"] == ["synthetic_0002"]
    assert ("synthetic_0001", "synthetic_0003") in normalization_fit_ids
    assert all("synthetic_0002" not in ids for ids in normalization_fit_ids)

    direct_config = encoder_config.model_copy(update={"output_dir": str(tmp_path / "direct")})
    direct = train_direct_diagnostics(direct_config)
    assert direct["train_trajectory_ids"] == ["synthetic_0001", "synthetic_0003"]
    assert direct["validation_trajectory_ids"] == ["synthetic_0000"]
    assert direct["test_split_inspected"] is False

    embed_config = load_config(
        repo_root / "configs/experiment/smoke_embed_dataset.yaml",
        overrides=[f"data.split_manifest={manifest}"],
        command="embed-dataset",
    ).model_copy(
        update={
            "output_dir": str(tmp_path / "embed"),
            "latent_cache": encoder_config.latent_cache.model_copy(
                update={
                    "path": str(tmp_path / "embed" / "latent_cache.h5"),
                    "encoder_checkpoint_path": encoder["checkpoint"],
                }
            ),
        }
    )
    embedded = embed_dataset(embed_config)
    with h5py.File(embedded["latent_cache"], "r") as handle:
        metadata = handle["metadata"].attrs
        assert str(tmp_path) not in str(metadata["protocol_json"])
        assert str(tmp_path) not in str(metadata["config_yaml"])
        assert all(not name.startswith("%2F") for name in handle["trajectories"])

    sequence_config = load_config(
        repo_root / "configs/experiment/smoke_sequence.yaml",
        overrides=common,
        command="train-sequence",
    ).model_copy(
        update={
            "output_dir": str(tmp_path / "sequence"),
            "latent_cache": embed_config.latent_cache.model_copy(update={"path": embedded["latent_cache"]}),
        }
    )
    sequence = train_sequence(sequence_config)
    assert sequence["selected_trajectory_ids"] == ["synthetic_0001", "synthetic_0003"]
    assert sequence["validation_trajectory_ids"] == ["synthetic_0000"]
    assert sequence["test_trajectory_ids"] == ["synthetic_0002"]

    rollout_config_latent = sequence_config.latent_cache.model_copy(
        update={
            "path": embedded["latent_cache"],
            "sequence_checkpoint_path": sequence["checkpoint"],
        }
    )
    rollout_config = load_config(
        repo_root / "configs/experiment/smoke_evaluate_rollout.yaml",
        overrides=[f"data.split_manifest={manifest}"],
        command="evaluate-rollout",
    ).model_copy(
        update={
            "output_dir": str(tmp_path / "rollout"),
            "latent_cache": rollout_config_latent,
        }
    )
    assert rollout_config_latent.encoder_checkpoint_path == encoder["checkpoint"]
    rollout = evaluate_rollout(rollout_config)
    assert rollout["selected_trajectory_ids"] == ["synthetic_0002"]

    other_manifest = _write_manifest(tmp_path / "same_ids_different_fold.json", fold_id="outer-1")
    mismatched = rollout_config.model_copy(
        update={
            "data": rollout_config.data.model_copy(update={"split_manifest": str(other_manifest)}),
        }
    )
    with pytest.raises(ValueError, match="split-manifest lineage"):
        evaluate_rollout(mismatched)
