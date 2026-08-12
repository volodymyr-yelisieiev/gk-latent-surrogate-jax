from __future__ import annotations

import hashlib
import json
import subprocess
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


def test_repository_protocol_defaults_to_blocked_preflight(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = protocol_runner.main(
        [
            "--protocol",
            str(repo_root / "experiment_protocols" / "multiseed_v1.json"),
            "--repo-root",
            str(repo_root),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 2
    assert payload["mode"] == "preflight"
    assert payload["execution"] == []
    assert any("status must be 'frozen'" in blocker for blocker in payload["blockers"])
    assert any("dataset_revision" in blocker for blocker in payload["blockers"])
    assert all(stage.get("status") != "completed" for stage in payload["stages"])


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
    report, stages = protocol_runner.preflight(
        protocol_path,
        configs=configs,
        output_root=tmp_path / "runs",
        held_out_manifest=held_out,
        universe_manifest=universe,
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
    assert len(stages) == 35
    assert {stage.seed for stage in stages} == {52, 53, 54, 55, 56}
    assert all(item["status"] == "planned" for item in report.stages)
    assert report.execution == []
    transformer = next(stage for stage in stages if stage.seed == 52 and stage.name == "transformer_eval")
    assert "--seed" in transformer.command
    assert "52" in transformer.command
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

    protocol["data"]["evaluation_route"] = "nested_group_cross_validation"
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
        output_dir.mkdir(parents=True, exist_ok=True)
        cli_command = command[3]
        payload: dict[str, str] = {}
        if cli_command in {"train-encoder", "train-sequence"}:
            checkpoint = output_dir / "checkpoints" / "best"
            checkpoint.mkdir(parents=True, exist_ok=True)
            payload["checkpoint"] = str(checkpoint)
        elif cli_command == "embed-dataset":
            cache = output_dir / "latent_cache.h5"
            cache.touch()
            payload["latent_cache"] = str(cache)
        (output_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    first = protocol_runner.execute_stages(
        stages,
        repo_root=tmp_path,
        resume=False,
        command_runner=successful_runner,
    )
    assert len(calls) == 7
    assert all(item["status"] == "completed" for item in first)
    assert not any("{" in argument for command in calls for argument in command)

    calls.clear()
    second = protocol_runner.execute_stages(
        stages,
        repo_root=tmp_path,
        resume=True,
        command_runner=successful_runner,
    )
    assert calls == []
    assert all(item["status"] == "resumed_existing" for item in second)


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
    nested["data"]["evaluation_route"] = "nested_group_cross_validation"
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
    protocol["data"]["evaluation_route"] = "nested_group_cross_validation"
    protocol["data"]["fallback_rule"] = None
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_universe_manifest(protocol, universe, report)
    assert any("frozen nested" in item for item in report.blockers)
    protocol["data"]["fallback_rule"] = {"method": "nested_group_cross_validation", "outer_folds": 4}
    report = protocol_runner.PreflightReport()
    protocol_runner._validate_universe_manifest(protocol, universe, report)
    assert any("at least five" in item for item in report.blockers)


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
        persistence = role in {"gru_eval", "transformer_eval"}
        cyclone = SimpleNamespace(use_kvikio=True)
        return SimpleNamespace(
            data=SimpleNamespace(split=split, backend="cyclone_kvikio", cyclone=cyclone),
            latent_cache=SimpleNamespace(use_persistence_baseline=persistence),
        )

    monkeypatch.setattr(protocol_runner, "load_config", fake_load)
    report = protocol_runner.PreflightReport(environment={"kvikio_available": False})
    protocol_runner._validate_configs(configs, report)
    assert any("does not exist" in item for item in report.blockers)
    assert any("failed validation" in item for item in report.blockers)
    assert any("data.split=all" in item for item in report.blockers)
    assert any("persistence_eval config must enable" in item for item in report.blockers)
    assert any("gru_eval must not" in item for item in report.blockers)
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
    with pytest.raises(ValueError, match="unresolved stage dependency"):
        protocol_runner._resolve_command(("checkpoint={encoder.checkpoint}",), {})

    def failed(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 7)

    with pytest.raises(RuntimeError, match="exit code 7"):
        protocol_runner.execute_stages([encoder], repo_root=tmp_path, resume=False, command_runner=failed)


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
