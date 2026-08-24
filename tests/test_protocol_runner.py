from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gk_surrogate import protocol_runner


def _frozen_protocol(source: Path, manifest: Path | None, universe: Path) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["status"] = "frozen"
    payload["source"]["commit"] = "a" * 40
    payload["source"]["tracked_diff_sha256"] = "b" * 64
    payload["data"]["dataset_revision"] = "dataset-v1"
    payload["data"]["universe_manifest_sha256"] = hashlib.sha256(universe.read_bytes()).hexdigest()
    payload["data"]["evaluation_route"] = "final_unseen_test"
    if manifest is not None:
        payload["data"]["final_test_rule"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return payload


def _universe_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_revision": "dataset-v1",
                "trajectory_ids": [f"unseen-{index}" for index in range(10)]
                + [f"development-{index}" for index in range(10)],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _held_out_manifest(path: Path, universe: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "protocol_id": "multiseed-v1",
                "dataset_revision": "dataset-v1",
                "universe_manifest_sha256": hashlib.sha256(universe.read_bytes()).hexdigest(),
                "trajectory_ids": [f"unseen-{index}" for index in range(10)],
                "not_used_for_training": True,
                "not_used_for_model_selection": True,
                "attested_by": "dataset custodian",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _configs(tmp_path: Path) -> dict[str, Path]:
    return {role: tmp_path / f"{role}.yaml" for role in protocol_runner.CONFIG_ROLES}


def _write_success_evidence(command: list[str], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data_seed = int(command[command.index("--seed") + 1])
    training_seed = (
        int(command[command.index("--training-seed") + 1])
        if "--training-seed" in command
        else data_seed
    )
    cli_command = command[3]
    split = "all" if cli_command == "embed-dataset" else "test" if cli_command == "evaluate-rollout" else "train"
    if "data.split=val" in command:
        split = "val"
    if "--split-manifest" in command:
        manifest_path = Path(command[command.index("--split-manifest") + 1])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = manifest["splits"]
        universe_ids = [*splits["train"], *splits["val"], *splits["test"]]
        selected_ids = universe_ids if split == "all" else splits[split]
        fold_fields = {
            "split_fold_id": manifest["fold_id"],
            "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    else:
        splits = {"train": ["train-a"], "val": ["val-a"], "test": ["test-a"]}
        universe_ids = [*splits["train"], *splits["val"], *splits["test"]]
        selected_ids = universe_ids if split == "all" else splits[split]
        fold_fields = {"split_fold_id": None, "split_manifest_sha256": None}
    def trajectory_hash(values: list[str]) -> str:
        return hashlib.sha256(
            json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    payload: dict[str, object] = {
        "protocol_version": 1,
        "data_split": split,
        "data_split_seed": data_seed,
        "training_seed": training_seed,
        "selected_trajectory_ids": selected_ids,
        "trajectory_manifest_sha256": trajectory_hash(selected_ids),
        "universe_trajectory_ids": universe_ids,
        "universe_manifest_sha256": trajectory_hash(universe_ids),
        **fold_fields,
    }
    if cli_command in {"train-encoder", "train-sequence"}:
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        checkpoint_file = checkpoint / "checkpoint.pkl"
        checkpoint_file.write_bytes(f"checkpoint:{output}".encode())
        payload.update(
            checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint_file.read_bytes()).hexdigest(),
            artifact_role="encoder_checkpoint" if cli_command == "train-encoder" else "sequence_checkpoint",
        )
    elif cli_command == "embed-dataset":
        cache = output / "latent_cache.h5"
        cache.write_bytes(b"cache")
        payload.update(
            latent_cache=str(cache),
            latent_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            artifact_role="latent_cache",
        )
    else:
        baseline_mode = (
            "observed_diagnostic_persistence"
            if output.name == "observed_persistence_eval"
            else "latent_state_persistence_decoded"
            if output.name == "latent_persistence_eval"
            else "none"
        )
        payload.update(
            flux_rmse=1.0 if "gru" in output.name else 2.0,
            trajectory_balanced_flux_rmse=1.0 if "gru" in output.name else 2.0,
            stable=True,
            num_trajectories=len(selected_ids),
            baseline_mode=baseline_mode,
        )
    (output / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_planned_protocol_defaults_to_blocked_preflight(
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol_path = tmp_path / "planned-protocol.json"
    protocol = json.loads((repo_root / "experiment_protocols" / "multiseed_v1.json").read_text(encoding="utf-8"))
    protocol["status"] = "planned"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    result = protocol_runner.main(
        [
            "--protocol",
            str(protocol_path),
            "--repo-root",
            str(repo_root),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 2
    assert payload["mode"] == "preflight"
    assert payload["execution"] == []
    assert any("status must be 'frozen'" in blocker for blocker in payload["blockers"])
    assert "--universe-manifest is required to verify the dataset universe" in payload["blockers"]
    assert all(stage.get("status") != "completed" for stage in payload["stages"])


def test_default_stage_runner_captures_child_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage = protocol_runner.Stage(
        "probe",
        52,
        (sys.executable, "-c", "print('child-output')"),
        tmp_path / "probe",
    )
    monkeypatch.setattr(protocol_runner, "_load_stage_evidence", lambda _stage: {"metrics_sha256": "a" * 64})

    result = protocol_runner.execute_stages([stage], repo_root=tmp_path, resume=False)

    assert result[0]["status"] == "completed"
    assert capsys.readouterr().out == ""
    assert (stage.output_dir / "command.stdout.log").read_text(encoding="utf-8") == "child-output\n"
    assert (stage.output_dir / "command.stderr.log").read_text(encoding="utf-8") == ""


def test_preflight_builds_matched_seed_plan_when_evidence_is_complete(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_out = tmp_path / "held_out.json"
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    _held_out_manifest(held_out, universe)
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", held_out, universe)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    configs = _configs(tmp_path)

    monkeypatch.setattr(protocol_runner, "_validate_configs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_runner, "_validate_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_runner, "_verify_dataset_bytes", lambda **_kwargs: None)
    report, stages = protocol_runner.preflight(
        protocol_path,
        configs=configs,
        output_root=tmp_path / "runs",
        held_out_manifest=held_out,
        universe_manifest=universe,
        fold_manifest=None,
        repo_root=repo_root,
        execute=False,
        resume=False,
        environment={
            "gpu_available": True,
            "jax_platforms": ["gpu"],
            "kvikio_available": True,
            "cupy_available": True,
        },
    )

    assert report.ready is True
    assert len(stages) == 51
    assert {stage.seed for stage in stages if stage.seed is not None} == {52, 53, 54, 55, 56}
    assert all(item["status"] == "planned" for item in report.stages)
    assert report.execution == []
    transformer = next(stage for stage in stages if stage.seed == 52 and stage.name == "transformer_eval")
    assert "--seed" in transformer.command
    assert transformer.command[transformer.command.index("--seed") + 1] == "52"
    assert transformer.command[transformer.command.index("--training-seed") + 1] == "52"
    assert any("{transformer_train.checkpoint}" in part for part in transformer.command)


def test_final_test_proof_fails_closed_on_hash_count_and_attestation(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "held_out.json"
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "multiseed-v1",
                "dataset_revision": "dataset-v1",
                "universe_manifest_sha256": hashlib.sha256(universe.read_bytes()).hexdigest(),
                "trajectory_ids": ["a", "a"],
                "not_used_for_training": False,
                "not_used_for_model_selection": True,
                "attested_by": "",
            }
        ),
        encoding="utf-8",
    )
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", manifest, universe)
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(
        protocol,
        manifest,
        report,
        resume=False,
        universe_ids={f"unseen-{index}" for index in range(10)},
    )

    assert any("at least 10 unique" in blocker for blocker in report.blockers)
    assert any("duplicate" in blocker for blocker in report.blockers)
    assert any("non-use attestations" in blocker for blocker in report.blockers)
    assert any("attested_by" in blocker for blocker in report.blockers)


def test_historical_51_trajectory_universe_requires_nested_cv(repo_root: Path, tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "dataset_revision": "dataset-v1",
                "trajectory_ids": [f"historical-{index}" for index in range(51)],
            }
        ),
        encoding="utf-8",
    )
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", None, universe)
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_universe_manifest(protocol, universe, report)
    assert any("historical 51-trajectory universe" in blocker for blocker in report.blockers)

    protocol["data"]["evaluation_route"] = "nested_group_holdout_cross_validation"
    report = protocol_runner.PreflightReport()
    ids = protocol_runner._validate_universe_manifest(protocol, universe, report)
    protocol_runner._validate_held_out_manifest(
        protocol,
        None,
        report,
        resume=False,
        universe_ids=ids,
    )
    assert report.blockers == []


def test_execute_requires_gpu_but_preflight_only_warns(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_out = tmp_path / "held_out.json"
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    _held_out_manifest(held_out, universe)
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", held_out, universe)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(protocol_runner, "_validate_configs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_runner, "_validate_source", lambda *_args, **_kwargs: None)
    kwargs = {
        "configs": _configs(tmp_path),
        "output_root": tmp_path / "runs",
        "held_out_manifest": held_out,
        "universe_manifest": universe,
        "fold_manifest": None,
        "repo_root": repo_root,
        "resume": False,
        "environment": {
            "gpu_available": False,
            "jax_platforms": ["cpu"],
            "kvikio_available": False,
            "cupy_available": False,
        },
    }
    dry_report, _ = protocol_runner.preflight(protocol_path, execute=False, **kwargs)
    execute_report, _ = protocol_runner.preflight(protocol_path, execute=True, **kwargs)

    assert dry_report.ready is True
    assert any("CPU only" in warning for warning in dry_report.warnings)
    assert execute_report.ready is False
    assert any("CPU only" in blocker for blocker in execute_report.blockers)
    assert "KvikIO is unavailable" in execute_report.warnings
    assert "CuPy is unavailable" in execute_report.warnings


def test_execute_and_resume_require_verified_stage_evidence(tmp_path: Path) -> None:
    configs = _configs(tmp_path)
    stages = protocol_runner.build_stages(configs, tmp_path / "runs", [52])
    calls: list[list[str]] = []

    def successful_runner(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, check, text
        calls.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_success_evidence(command, output_dir)
        return subprocess.CompletedProcess(command, 0)

    first = protocol_runner.execute_stages(
        stages,
        repo_root=tmp_path,
        resume=False,
        command_runner=successful_runner,
    )
    assert len(calls) == 9
    assert any(item["status"] == "completed_selection_barrier" for item in first)
    assert sum(item["status"] == "skipped_unselected" for item in first) == 1
    assert not any("{" in argument for command in calls for argument in command)
    outputs = [command[command.index("--output-dir") + 1] for command in calls]
    first_test = min(
        index for index, output in enumerate(outputs) if output.endswith(("gru_eval", "observed_persistence_eval"))
    )
    assert max(index for index, output in enumerate(outputs) if "validation" in output) < first_test
    assert not any(output.endswith("transformer_eval") for output in outputs)

    calls.clear()
    second = protocol_runner.execute_stages(
        stages,
        repo_root=tmp_path,
        resume=True,
        command_runner=successful_runner,
    )
    assert calls == []
    assert all(
        item["status"] in {"resumed_existing", "completed_selection_barrier", "skipped_unselected"} for item in second
    )


def test_protocol_stage_keeps_development_split_seed_separate_from_training_seed(tmp_path: Path) -> None:
    stages = protocol_runner.build_stages(
        _configs(tmp_path),
        tmp_path / "runs",
        [53],
        data_seed=52,
    )
    encoder = next(stage for stage in stages if stage.name == "encoder")
    assert encoder.data_seed == 52
    assert encoder.command[encoder.command.index("--seed") + 1] == "52"
    assert encoder.command[encoder.command.index("--training-seed") + 1] == "53"


def test_success_without_stage_evidence_is_rejected(tmp_path: Path) -> None:
    stage = protocol_runner.Stage(
        "encoder",
        52,
        ("python", "fake"),
        tmp_path / "encoder",
    )

    def no_evidence(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0)

    with pytest.raises(RuntimeError, match="without valid evidence"):
        protocol_runner.execute_stages(
            [stage],
            repo_root=tmp_path,
            resume=False,
            command_runner=no_evidence,
        )


def test_source_tag_can_authoritatively_bind_head_without_self_referential_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {
        "source": {
            "commit": None,
            "tag": "multiseed-v1-frozen",
            "tracked_diff_sha256": "b" * 64,
        }
    }
    monkeypatch.setattr(
        protocol_runner,
        "inspect_source",
        lambda _root: {
            "commit": "a" * 40,
            "tracked_diff_sha256": "b" * 64,
            "untracked_files": [],
        },
    )
    monkeypatch.setattr(protocol_runner, "_git_output", lambda *_args: "a" * 40)
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_source(protocol, tmp_path, report)
    assert report.blockers == []


def test_source_tag_must_resolve_exactly_to_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = {
        "source": {
            "commit": None,
            "tag": "multiseed-v1-frozen",
            "tracked_diff_sha256": "b" * 64,
        }
    }
    monkeypatch.setattr(
        protocol_runner,
        "inspect_source",
        lambda _root: {
            "commit": "a" * 40,
            "tracked_diff_sha256": "b" * 64,
            "untracked_files": [],
        },
    )
    monkeypatch.setattr(protocol_runner, "_git_output", lambda *_args: "c" * 40)
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_source(protocol, tmp_path, report)
    assert report.blockers == ["protocol.source.tag does not resolve exactly to checked-out HEAD"]


@pytest.mark.parametrize("contents", ("{", "[]"))
def test_json_inputs_must_be_readable_objects(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read JSON|root must be an object"):
        protocol_runner._read_json(path)
    with pytest.raises(ValueError, match="cannot read JSON"):
        protocol_runner._read_json(tmp_path / "missing.json")


def test_environment_reports_jax_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import jax

    monkeypatch.setattr(jax, "devices", lambda: (_ for _ in ()).throw(RuntimeError("no runtime")))
    monkeypatch.setattr(protocol_runner.importlib.util, "find_spec", lambda name: object() if name == "cupy" else None)
    result = protocol_runner.inspect_environment()
    assert result["gpu_available"] is False
    assert result["kvikio_available"] is False
    assert result["cupy_available"] is True
    assert result["jax_error"] == "RuntimeError: no runtime"


def test_protocol_rejects_unsafe_identity_seed_and_selection_variants(repo_root: Path) -> None:
    base = json.loads((repo_root / "experiment_protocols" / "multiseed_v1.json").read_text(encoding="utf-8"))
    base["status"] = "frozen"
    base["source"]["tag"] = "frozen-v1"
    base["data"]["dataset_revision"] = "dataset-v1"
    base["data"]["universe_manifest_sha256"] = "a" * 64
    mutations = [
        lambda value: value.update(schema_version="bad", created_before_runs=False),
        lambda value: value.update(source=None),
        lambda value: value.update(data=None),
        lambda value: value["data"].update(normalization_fit_split="all"),
        lambda value: value.update(selection=None),
        lambda value: value["selection"].update(architecture_aggregation="best_seed"),
        lambda value: value.update(models=None),
        lambda value: value["models"].update(training_seeds=[]),
        lambda value: value["models"].update(training_seeds=[52, 52]),
        lambda value: value["models"].update(families=["transformer"]),
        lambda value: value.update(accepted_runs=[{"status": "accepted"}]),
    ]
    messages: set[str] = set()
    for mutate in mutations:
        payload = deepcopy(base)
        mutate(payload)
        report = protocol_runner.PreflightReport()
        seeds = protocol_runner._validate_protocol(payload, report, resume=False)
        messages.update(report.blockers)
        if payload.get("models") is None or payload.get("models", {}).get("training_seeds") == []:
            assert seeds == []
    assert any("schema_version" in message for message in messages)
    assert any("created_before_runs" in message for message in messages)
    assert any("source is missing" in message for message in messages)
    assert any("models is missing" in message for message in messages)
    assert any("unique" in message for message in messages)
    assert any("aggregate validation" in message for message in messages)
    assert any("--resume" in message for message in messages)


def test_held_out_manifest_rejects_missing_route_file_and_identity(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", None, universe)

    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest({}, None, report, resume=False, universe_ids=None)
    assert report.blockers == []

    report = protocol_runner.PreflightReport()
    bad_rule = deepcopy(protocol)
    bad_rule["data"]["final_test_rule"] = None
    protocol_runner._validate_held_out_manifest(bad_rule, None, report, resume=False, universe_ids=None)
    assert any("final_test_rule" in item for item in report.blockers)

    nested = deepcopy(protocol)
    nested["data"]["evaluation_route"] = "nested_group_holdout_cross_validation"
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(nested, universe, report, resume=False, universe_ids=None)
    assert any("must not supply" in item for item in report.blockers)

    invalid_route = deepcopy(protocol)
    invalid_route["data"]["evaluation_route"] = "ordinary_test"
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(invalid_route, None, report, resume=False, universe_ids=None)
    assert any("evaluation_route" in item for item in report.blockers)

    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(protocol, None, report, resume=False, universe_ids=None)
    assert any("hash is not frozen" in item for item in report.blockers)
    assert any("--held-out-manifest" in item for item in report.blockers)

    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(
        protocol,
        tmp_path / "absent.json",
        report,
        resume=False,
        universe_ids=None,
    )
    assert any("does not exist" in item for item in report.blockers)


def test_held_out_manifest_rejects_tampering_and_open_state(repo_root: Path, tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    held_out = tmp_path / "held_out.json"
    _universe_manifest(universe)
    _held_out_manifest(held_out, universe)
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", held_out, universe)
    held_out.write_text("{", encoding="utf-8")
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(protocol, held_out, report, resume=False, universe_ids=set())
    assert any("SHA-256" in item for item in report.blockers)
    assert any("cannot read JSON" in item for item in report.blockers)

    _held_out_manifest(held_out, universe)
    manifest = json.loads(held_out.read_text(encoding="utf-8"))
    manifest.update(
        protocol_id="wrong",
        dataset_revision="wrong",
        universe_manifest_sha256="wrong",
        trajectory_ids=[1],
    )
    held_out.write_text(json.dumps(manifest), encoding="utf-8")
    protocol["data"]["final_test_rule"]["manifest_sha256"] = hashlib.sha256(held_out.read_bytes()).hexdigest()
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(protocol, held_out, report, resume=True, universe_ids=set())
    assert any("trajectory_ids" in item for item in report.blockers)

    _held_out_manifest(held_out, universe)
    protocol["data"]["final_test_rule"]["manifest_sha256"] = hashlib.sha256(held_out.read_bytes()).hexdigest()
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(protocol, held_out, report, resume=True, universe_ids=set())
    assert any("not all present" in item for item in report.blockers)
    assert any("opened_at_utc" in item for item in report.blockers)
    protocol["data"]["final_test_rule"]["opened_at_utc"] = "2026-01-01T00:00:00Z"
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(protocol, held_out, report, resume=False, universe_ids=None)
    assert any("already-opened" in item for item in report.blockers)


def test_held_out_manifest_checks_frozen_identity_fields(repo_root: Path, tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    held_out = tmp_path / "held_out.json"
    _universe_manifest(universe)
    _held_out_manifest(held_out, universe)
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", held_out, universe)
    payload = json.loads(held_out.read_text(encoding="utf-8"))
    payload.update(protocol_id="wrong", dataset_revision="wrong", universe_manifest_sha256="wrong")
    held_out.write_text(json.dumps(payload), encoding="utf-8")
    protocol["data"]["final_test_rule"]["manifest_sha256"] = hashlib.sha256(held_out.read_bytes()).hexdigest()
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_held_out_manifest(
        protocol,
        held_out,
        report,
        resume=False,
        universe_ids={f"unseen-{index}" for index in range(10)},
    )
    assert any("protocol_id" in item for item in report.blockers)
    assert any("dataset_revision" in item for item in report.blockers)
    assert any("universe hash" in item for item in report.blockers)


def test_outer_fold_manifest_requires_frozen_dataset_and_universe_hash(repo_root: Path, tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    protocol_path = tmp_path / "protocol.json"
    protocol = _nested_protocol(repo_root, universe)
    protocol["data"]["dataset_revision"] = None
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_revision"):
        protocol_runner.write_outer_fold_manifest(protocol_path, universe, tmp_path / "folds.json")

    protocol["data"]["dataset_revision"] = "dataset-v1"
    protocol["data"]["universe_manifest_sha256"] = "0" * 64
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="universe manifest SHA"):
        protocol_runner.write_outer_fold_manifest(protocol_path, universe, tmp_path / "folds.json")


def test_universe_manifest_rejects_bad_files_ids_and_nested_rules(repo_root: Path, tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    protocol = _frozen_protocol(repo_root / "experiment_protocols" / "multiseed_v1.json", None, universe)

    report = protocol_runner.PreflightReport()
    protocol_runner._validate_universe_manifest(protocol, tmp_path / "absent.json", report)
    assert any("does not exist" in item for item in report.blockers)

    universe.write_text("{", encoding="utf-8")
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_universe_manifest(protocol, universe, report)
    assert any("SHA-256" in item for item in report.blockers)
    assert any("cannot read JSON" in item for item in report.blockers)

    universe.write_text(json.dumps({"dataset_revision": "wrong", "trajectory_ids": ["a", "a"]}), encoding="utf-8")
    protocol["data"]["universe_manifest_sha256"] = hashlib.sha256(universe.read_bytes()).hexdigest()
    report = protocol_runner.PreflightReport()
    ids = protocol_runner._validate_universe_manifest(protocol, universe, report)
    assert ids == {"a"}
    assert any("duplicate" in item for item in report.blockers)
    assert any("dataset_revision" in item for item in report.blockers)

    universe.write_text(json.dumps({"dataset_revision": "dataset-v1", "trajectory_ids": []}), encoding="utf-8")
    report = protocol_runner.PreflightReport()
    assert protocol_runner._validate_universe_manifest(protocol, universe, report) is None
    assert any("non-empty strings" in item for item in report.blockers)

    _universe_manifest(universe)
    protocol["data"]["universe_manifest_sha256"] = hashlib.sha256(universe.read_bytes()).hexdigest()
    protocol["data"]["evaluation_route"] = "nested_group_holdout_cross_validation"
    protocol["data"]["fallback_rule"] = None
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_universe_manifest(protocol, universe, report)
    assert any("frozen nested" in item for item in report.blockers)
    protocol["data"]["fallback_rule"] = {
        "method": "nested_group_holdout_cross_validation",
        "outer_folds": 4,
    }
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_universe_manifest(protocol, universe, report)
    assert any("exactly five" in item for item in report.blockers)


def test_source_rejects_commit_tag_diff_and_untracked_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = {
        "commit": "a" * 40,
        "tracked_diff_sha256": "b" * 64,
        "untracked_files": ["results/private.json"],
    }
    monkeypatch.setattr(protocol_runner, "inspect_source", lambda _root: actual)
    monkeypatch.setattr(protocol_runner, "_git_output", lambda *_args: None)
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_source(
        {
            "source": {
                "commit": "c" * 40,
                "tag": 123,
                "tracked_diff_sha256": None,
            }
        },
        tmp_path,
        report,
    )
    assert any("commit does not match" in item for item in report.blockers)
    assert any("tag must be" in item for item in report.blockers)
    assert any("not frozen" in item for item in report.blockers)
    assert any("untracked" in item for item in report.blockers)

    monkeypatch.setattr(protocol_runner, "_git_output", lambda *_args: "d" * 40)
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_source(
        {"source": {"commit": "a" * 40, "tag": "v1", "tracked_diff_sha256": "c" * 64}},
        tmp_path,
        report,
    )
    assert any("different commits" in item for item in report.blockers)
    assert any("tracked Git diff" in item for item in report.blockers)

    report = protocol_runner.PreflightReport()
    protocol_runner._validate_source({}, tmp_path, report)
    assert report.blockers == []


@pytest.mark.parametrize("value", ("bad", "unknown=x", "encoder="))
def test_config_argument_requires_known_role_and_path(value: str) -> None:
    with pytest.raises(Exception, match="ROLE=PATH"):
        protocol_runner._config_arg(value)
    assert protocol_runner._config_arg("encoder=config.yaml") == ("encoder", Path("config.yaml"))


def test_config_validation_checks_roles_splits_baselines_and_kvikio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = _configs(tmp_path)
    for path in configs.values():
        path.touch()
    configs["missing_extra"] = tmp_path / "absent.yaml"

    def fake_load(path: Path, *, command: str) -> SimpleNamespace:
        del command
        role = path.stem
        if role == "encoder":
            raise ValueError("invalid encoder")
        split = "wrong" if role == "embed" else "test" if role.endswith("eval") else "train"
        baseline_mode = (
            "latent_state_persistence_decoded"
            if role == "latent_persistence_eval"
            else "observed_diagnostic_persistence"
            if role == "observed_persistence_eval"
            else "latent_state_persistence_decoded"
            if role in {"gru_eval", "transformer_eval"}
            else "none"
        )
        cyclone = SimpleNamespace(use_kvikio=True)
        return SimpleNamespace(
            data=SimpleNamespace(split=split, backend="cyclone_kvikio", cyclone=cyclone),
            evaluation=SimpleNamespace(baseline_mode=baseline_mode),
        )

    monkeypatch.setattr(protocol_runner, "load_config", fake_load)
    report = protocol_runner.PreflightReport(environment={"kvikio_available": False})
    protocol_runner._validate_configs(configs, report)
    assert any("does not exist" in item for item in report.blockers)
    assert any("failed validation" in item for item in report.blockers)
    assert any("data.split=all" in item for item in report.blockers)
    assert any("evaluation.baseline_mode=none" in item for item in report.blockers)
    assert any("config gru_eval must use evaluation.baseline_mode=none" in item for item in report.blockers)
    assert any("requires KvikIO" in item for item in report.blockers)


def test_invalid_resume_evidence_dependency_and_stage_failure(tmp_path: Path) -> None:
    encoder = protocol_runner.Stage("encoder", 52, ("python", "fake"), tmp_path / "encoder")
    encoder.output_dir.mkdir()
    (encoder.output_dir / "metrics.json").write_text("{", encoding="utf-8")
    assert protocol_runner._load_stage_evidence(encoder) is None
    (encoder.output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    assert protocol_runner._load_stage_evidence(encoder) is None

    embed = protocol_runner.Stage("embed", 52, ("python", "fake"), tmp_path / "embed")
    embed.output_dir.mkdir()
    (embed.output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    assert protocol_runner._load_stage_evidence(embed) is None
    evaluation = protocol_runner.Stage(
        "gru_eval",
        52,
        ("python", "fake"),
        tmp_path / "evaluation",
        phase="test",
        family="gru",
    )
    evaluation.output_dir.mkdir()
    (evaluation.output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    assert protocol_runner._load_stage_evidence(evaluation) is None
    with pytest.raises(ValueError, match="unresolved stage dependency"):
        protocol_runner._resolve_command(("checkpoint={encoder.checkpoint}",), {})

    def failed(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 7)

    with pytest.raises(RuntimeError, match="exit code 7"):
        protocol_runner.execute_stages([encoder], repo_root=tmp_path, resume=False, command_runner=failed)


def test_stage_evidence_contract_is_fail_closed_for_roles_splits_and_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "gru_eval"
    command = (
        "uv",
        "run",
        "gks",
        "evaluate-rollout",
        "data.split=val",
        "--seed",
        "52",
        "--output-dir",
        str(output),
    )
    stage = protocol_runner.Stage(
        "gru_eval",
        52,
        command,
        output,
        phase="validation",
        family="gru",
    )
    _write_success_evidence(list(command), output)
    assert protocol_runner._load_stage_evidence(stage) is not None

    metrics_path = output / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.pop("protocol_version")
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(stage) is None

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["protocol_version"] = 1
    payload["data_split"] = "train"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(stage) is None

    payload["data_split"] = "val"
    payload["selected_trajectory_ids"] = []
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(stage) is None

    encoder_output = tmp_path / "encoder"
    encoder_command = (
        "uv",
        "run",
        "gks",
        "train-encoder",
        "--seed",
        "52",
        "--output-dir",
        str(encoder_output),
    )
    encoder = protocol_runner.Stage("encoder", 52, encoder_command, encoder_output)
    _write_success_evidence(list(encoder_command), encoder_output)
    encoder_metrics = encoder_output / "metrics.json"
    encoder_payload = json.loads(encoder_metrics.read_text(encoding="utf-8"))
    encoder_payload["artifact_role"] = "wrong-role"
    encoder_metrics.write_text(json.dumps(encoder_payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(encoder) is None

    encoder_payload["artifact_role"] = "encoder_checkpoint"
    encoder_payload.pop("checkpoint")
    encoder_metrics.write_text(json.dumps(encoder_payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(encoder) is None

    encoder_payload["checkpoint"] = str(encoder_output / "checkpoint")
    encoder_payload["checkpoint_sha256"] = hashlib.sha256(
        (encoder_output / "checkpoint" / "checkpoint.pkl").read_bytes()
    ).hexdigest()
    encoder_payload["checkpoint_sha256"] = "0" * 64
    encoder_metrics.write_text(json.dumps(encoder_payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(encoder) is None

    payload["selected_trajectory_ids"] = ["val-a"]
    payload["flux_rmse"] = "not-finite"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(stage) is None


def test_stage_evidence_checks_exact_fold_manifest_lineage(tmp_path: Path) -> None:
    manifest_path = tmp_path / "fold.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fold_id": "fold-0",
                "splits": {"train": ["train-a"], "val": ["val-a"], "test": ["test-a"]},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gru_eval"
    command = (
        "uv",
        "run",
        "gks",
        "evaluate-rollout",
        "data.split=val",
        "--split-manifest",
        str(manifest_path),
        "--seed",
        "52",
        "--output-dir",
        str(output),
    )
    stage = protocol_runner.Stage("gru_eval", 52, command, output, phase="validation", family="gru")
    _write_success_evidence(list(command), output)
    assert protocol_runner._load_stage_evidence(stage) is not None

    manifest_path.write_text(json.dumps({"fold_id": "fold-0"}), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(stage) is None

    manifest_path.write_text(
        json.dumps(
            {
                "fold_id": "fold-0",
                "splits": {"train": ["train-a"], "val": ["other"], "test": ["test-a"]},
            }
        ),
        encoding="utf-8",
    )
    assert protocol_runner._load_stage_evidence(stage) is None

    manifest_path.write_text(
        json.dumps(
            {
                "fold_id": "fold-0",
                "splits": {"train": ["train-a"], "val": ["val-a"], "test": ["test-a"]},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    payload["split_manifest_sha256"] = "0" * 64
    (output / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    assert protocol_runner._load_stage_evidence(stage) is None


def test_semantic_config_validation_rejects_mislabeled_transformer(repo_root: Path) -> None:
    from gk_surrogate.config.load import load_config

    base = load_config(repo_root / "configs/experiment/smoke_sequence.yaml", command="train-sequence")
    latent_persistence = base.model_copy(
        update={"evaluation": base.evaluation.model_copy(update={"baseline_mode": "latent_state_persistence_decoded"})}
    )
    observed_persistence = base.model_copy(
        update={"evaluation": base.evaluation.model_copy(update={"baseline_mode": "observed_diagnostic_persistence"})}
    )
    loaded = {
        "encoder": base,
        "embed": base,
        "gru_train": base,
        "transformer_train": base,
        "gru_eval": base,
        "transformer_eval": base,
        "latent_persistence_eval": latent_persistence,
        "observed_persistence_eval": observed_persistence,
    }
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_config_semantics(loaded, report)
    assert any("transformer roles require sequence type" in blocker for blocker in report.blockers)


def test_schema_contract_rejects_null_source_and_outer_fold(repo_root: Path) -> None:
    payload = json.loads((repo_root / "experiment_protocols/multiseed_v1.json").read_text(encoding="utf-8"))
    payload["status"] = "frozen"
    payload["source"].update(commit=None, tag=None, tracked_diff_sha256="a" * 64)
    payload["accepted_runs"] = [
        {
            "outer_fold": None,
            "stage": "test",
            "model": "transformer",
            "training_seed": 52,
            "wandb_run_id": None,
            "artifact_manifest_sha256": None,
            "status": "accepted",
        }
    ]
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_protocol(payload, report, resume=True)
    assert any("valid source commit or tag" in blocker for blocker in report.blockers)
    assert any("outer_fold must be an integer" in blocker for blocker in report.blockers)


def test_protocol_schema_reports_missing_fields_estimands_and_run_shapes(repo_root: Path) -> None:
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_protocol({}, report, resume=False)
    assert any("schema missing required fields" in blocker for blocker in report.blockers)

    payload = json.loads((repo_root / "experiment_protocols/multiseed_v1.json").read_text(encoding="utf-8"))
    payload["status"] = "frozen"
    payload["source"]["tag"] = "frozen-v1"
    payload["data"]["dataset_revision"] = "dataset-v1"
    payload["data"]["universe_manifest_sha256"] = "a" * 64
    payload["evaluation"]["primary_estimand"] = "wrong"
    payload["evaluation"]["secondary_estimand"] = "wrong"
    payload["accepted_runs"] = []
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_protocol(payload, report, resume=False)
    assert any("selected-model-minus-observed" in blocker for blocker in report.blockers)
    assert any("latent-state-persistence" in blocker for blocker in report.blockers)

    payload["accepted_runs"] = {}
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_protocol(payload, report, resume=False)
    assert any("accepted_runs to be an array" in blocker for blocker in report.blockers)

    payload["accepted_runs"] = []
    payload["data"].pop("dataset_revision")
    payload["data"].pop("universe_manifest_sha256")
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_protocol(payload, report, resume=False)
    assert any("dataset_revision is not frozen" in blocker for blocker in report.blockers)
    assert any("universe_manifest_sha256 is not frozen" in blocker for blocker in report.blockers)


def test_main_reports_preflight_error_and_executes_ready_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        protocol_runner,
        "preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert protocol_runner.main(["--protocol", str(tmp_path / "bad.json")]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "bad"

    stage = protocol_runner.Stage("encoder", 52, ("python", "fake"), tmp_path / "encoder")
    report = protocol_runner.PreflightReport(ready=True, mode="execute")
    monkeypatch.setattr(protocol_runner, "preflight", lambda *_args, **_kwargs: (report, [stage]))
    monkeypatch.setattr(
        protocol_runner,
        "execute_stages",
        lambda *_args, **_kwargs: [{"name": "encoder", "status": "completed"}],
    )
    assert protocol_runner.main(["--execute"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution"] == [{"name": "encoder", "status": "completed"}]


def _nested_protocol(repo_root: Path, universe: Path) -> dict[str, object]:
    protocol = _frozen_protocol(
        repo_root / "experiment_protocols" / "multiseed_v1.json",
        None,
        universe,
    )
    protocol["data"]["evaluation_route"] = "nested_group_holdout_cross_validation"
    return protocol


def test_outer_fold_generation_is_deterministic_disjoint_and_complete(repo_root: Path, tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.json"
    _universe_manifest(universe_path)
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    protocol = _nested_protocol(repo_root, universe_path)

    first = protocol_runner.generate_outer_fold_manifest(protocol, universe)
    universe["trajectory_ids"] = list(reversed(universe["trajectory_ids"]))
    second = protocol_runner.generate_outer_fold_manifest(protocol, universe)
    assert first == second
    assert first["assignment_algorithm"] == "sha256_rank_round_robin_cyclic_inner_validation_v1"
    assert first["outer_folds"] == 5

    all_ids = set(universe["trajectory_ids"])
    all_test_ids: list[str] = []
    for fold in first["folds"]:
        train = set(fold["train_trajectory_ids"])
        validation = set(fold["validation_trajectory_ids"])
        test = set(fold["test_trajectory_ids"])
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert train | validation | test == all_ids
        assert fold["inner_validation_fold"] == (fold["outer_fold"] + 1) % 5
        all_test_ids.extend(test)
    assert len(all_test_ids) == len(set(all_test_ids)) == len(all_ids)


def test_fold_generation_rejects_unsafe_inputs(repo_root: Path, tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.json"
    _universe_manifest(universe_path)
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    protocol = _nested_protocol(repo_root, universe_path)
    cases = [
        ({}, universe, "protocol.data"),
        ({"data": {"fallback_rule": None}}, universe, "nested_group"),
        (
            {"data": {"fallback_rule": {"method": "nested_group_holdout_cross_validation", "outer_folds": 4}}},
            universe,
            "five",
        ),
        (
            {
                "data": {
                    "fallback_rule": {"method": "nested_group_holdout_cross_validation", "outer_folds": 5},
                    "development_split_seed": None,
                }
            },
            universe,
            "split_seed",
        ),
        (protocol, {"trajectory_ids": []}, "non-empty"),
        (protocol, {"trajectory_ids": ["a", "a"]}, "duplicate"),
        (protocol, {"trajectory_ids": ["a"]}, "at least one"),
    ]
    for candidate_protocol, candidate_universe, message in cases:
        with pytest.raises(ValueError, match=message):
            protocol_runner.generate_outer_fold_manifest(candidate_protocol, candidate_universe)


def test_frozen_fold_manifest_is_consumed_and_derives_two_phase_exact_commands(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    protocol = _nested_protocol(repo_root, universe)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    fold_manifest = tmp_path / "outer_folds.json"
    protocol_runner.write_outer_fold_manifest(protocol_path, universe, fold_manifest)
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = hashlib.sha256(
        fold_manifest.read_bytes()
    ).hexdigest()
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    report = protocol_runner.PreflightReport()
    folds = protocol_runner._validate_outer_fold_manifest(protocol, fold_manifest, universe, report)
    assert report.blockers == []
    assert folds == (0, 1, 2, 3, 4)

    monkeypatch.setattr(protocol_runner, "_validate_configs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_runner, "_validate_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_runner, "exact_fold_cli_supported", lambda: False)
    monkeypatch.setattr(protocol_runner, "_verify_dataset_bytes", lambda **_kwargs: None)
    preflight, stages = protocol_runner.preflight(
        protocol_path,
        configs=_configs(tmp_path),
        output_root=tmp_path / "runs",
        held_out_manifest=None,
        universe_manifest=universe,
        fold_manifest=fold_manifest,
        repo_root=repo_root,
        execute=False,
        resume=False,
        environment={
            "gpu_available": True,
            "kvikio_available": True,
            "cupy_available": True,
        },
    )
    assert preflight.ready is True
    assert len(stages) == 255
    assert {stage.outer_fold for stage in stages} == {0, 1, 2, 3, 4}
    assert any("lacks --split-manifest" in warning for warning in preflight.warnings)
    stage = next(item for item in stages if item.outer_fold == 3 and item.seed == 55 and item.name == "gru_train")
    assert "--split-manifest" in stage.command
    assert "outer_fold_3" in stage.command[stage.command.index("--split-manifest") + 1]
    assert "outer_fold_3/seed_55" in str(stage.output_dir)

    monkeypatch.setattr(protocol_runner, "exact_fold_cli_supported", lambda: True)
    execute_report, _ = protocol_runner.preflight(
        protocol_path,
        configs=_configs(tmp_path),
        output_root=tmp_path / "runs",
        held_out_manifest=None,
        universe_manifest=universe,
        fold_manifest=fold_manifest,
        repo_root=repo_root,
        execute=True,
        resume=False,
        environment={
            "gpu_available": True,
            "kvikio_available": True,
            "cupy_available": True,
        },
        dataset_root=tmp_path,
    )
    assert execute_report.ready is True


def test_fold_manifest_tampering_and_wrong_route_are_rejected(repo_root: Path, tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    protocol = _nested_protocol(repo_root, universe)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    folds = tmp_path / "folds.json"
    protocol_runner.write_outer_fold_manifest(protocol_path, universe, folds)
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = "a" * 64
    payload = json.loads(folds.read_text(encoding="utf-8"))
    payload["folds"][0]["test_trajectory_ids"].append("tampered")
    folds.write_text(json.dumps(payload), encoding="utf-8")
    report = protocol_runner.PreflightReport()
    assert protocol_runner._validate_outer_fold_manifest(protocol, folds, universe, report) == ()
    assert any("SHA-256" in item for item in report.blockers)
    assert any("deterministic regeneration" in item for item in report.blockers)

    protocol["data"]["evaluation_route"] = "final_unseen_test"
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_outer_fold_manifest(protocol, folds, universe, report)
    assert any("must not supply" in item for item in report.blockers)


def test_current_cli_exposes_exact_fold_support() -> None:
    assert protocol_runner.exact_fold_cli_supported() is True


def test_generate_fold_manifest_cli_writes_only_manifest(
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    universe = tmp_path / "universe.json"
    _universe_manifest(universe)
    protocol = _nested_protocol(repo_root, universe)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "generated" / "folds.json"
    result = protocol_runner.main(
        [
            "--protocol",
            str(protocol_path),
            "--universe-manifest",
            str(universe),
            "--generate-fold-manifest",
            str(output),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output.is_file()
    assert payload["outer_folds"] == 5
    assert payload["execution"] == []
    assert payload["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert all(
        output.with_name(f"{output.stem}.outer_fold_{outer_fold}{output.suffix}").is_file() for outer_fold in range(5)
    )

    assert protocol_runner.main(["--generate-fold-manifest", str(tmp_path / "bad.json")]) == 2
    assert "requires --universe-manifest" in json.loads(capsys.readouterr().out)["error"]


def test_exact_fold_execution_and_resume_keep_evidence_isolated(tmp_path: Path) -> None:
    fold_manifest = tmp_path / "folds.json"
    fold_manifest.write_text("{}", encoding="utf-8")
    for outer_fold in (0, 1):
        protocol_runner._fold_cli_manifest_path(fold_manifest, outer_fold).write_text(
            json.dumps(
                {
                    "fold_id": f"outer-{outer_fold}",
                    "splits": {
                        "train": [f"train-{outer_fold}"],
                        "val": [f"val-{outer_fold}"],
                        "test": [f"test-{outer_fold}"],
                    },
                }
            ),
            encoding="utf-8",
        )
    stages = protocol_runner.build_stages(
        _configs(tmp_path),
        tmp_path / "runs",
        [52],
        fold_manifest=fold_manifest,
        outer_folds=(0, 1),
    )
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        _write_success_evidence(command, output)
        return subprocess.CompletedProcess(command, 0)

    completed = protocol_runner.execute_stages(
        stages,
        repo_root=tmp_path,
        resume=False,
        command_runner=runner,
    )
    assert len(commands) == 18
    assert {item["outer_fold"] for item in completed} == {0, 1}
    for command in commands:
        split_manifest = command[command.index("--split-manifest") + 1]
        fold = "0" if "outer_fold_0" in split_manifest else "1"
        assert f"outer_fold_{fold}" in command[command.index("--output-dir") + 1]
        assert not any("{" in argument for argument in command)

    commands.clear()
    resumed = protocol_runner.execute_stages(
        stages,
        repo_root=tmp_path,
        resume=True,
        command_runner=runner,
    )
    assert commands == []
    assert all(
        item["status"] in {"resumed_existing", "completed_selection_barrier", "skipped_unselected"} for item in resumed
    )
