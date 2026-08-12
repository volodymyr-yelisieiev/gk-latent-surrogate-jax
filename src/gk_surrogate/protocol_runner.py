"""Fail-closed preflight and execution for frozen multi-seed protocols."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gk_surrogate.config.load import load_config

CONFIG_ROLES: dict[str, str] = {
    "encoder": "train-encoder",
    "embed": "embed-dataset",
    "gru_train": "train-sequence",
    "transformer_train": "train-sequence",
    "gru_eval": "evaluate-rollout",
    "transformer_eval": "evaluate-rollout",
    "persistence_eval": "evaluate-rollout",
}


@dataclass(frozen=True)
class Stage:
    name: str
    seed: int
    command: tuple[str, ...]
    output_dir: Path
    dependencies: tuple[str, ...] = ()


@dataclass
class PreflightReport:
    protocol_id: str | None = None
    mode: str = "preflight"
    ready: bool = False
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, object] = field(default_factory=dict)
    stages: list[dict[str, object]] = field(default_factory=list)
    execution: list[dict[str, object]] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_environment() -> dict[str, object]:
    """Report accelerator and optional data-stack availability without mutating state."""

    platforms: list[str] = []
    error: str | None = None
    try:
        import jax

        platforms = sorted({device.platform for device in jax.devices()})
    except Exception as exc:  # pragma: no cover - depends on the host JAX installation
        error = f"{type(exc).__name__}: {exc}"
    result: dict[str, object] = {
        "jax_platforms": platforms,
        "gpu_available": "gpu" in platforms,
        "kvikio_available": importlib.util.find_spec("kvikio") is not None,
        "cupy_available": importlib.util.find_spec("cupy") is not None,
    }
    if error:
        result["jax_error"] = error
    return result


def _git_output(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_source(repo_root: Path) -> dict[str, object]:
    head = _git_output(repo_root, "rev-parse", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    diff_sha = hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
    return {
        "commit": head,
        "tracked_diff_sha256": diff_sha,
        "untracked_files": [line[3:] for line in (status or "").splitlines() if line.startswith("?? ")],
    }


def _validate_protocol(payload: Mapping[str, Any], report: PreflightReport, *, resume: bool) -> list[int]:
    report.protocol_id = str(payload.get("protocol_id")) if payload.get("protocol_id") else None
    if payload.get("schema_version") != "1.0.0":
        report.blockers.append("protocol.schema_version must be 1.0.0")
    if payload.get("status") != "frozen":
        report.blockers.append("protocol.status must be 'frozen' before execution")
    if payload.get("created_before_runs") is not True:
        report.blockers.append("protocol must attest created_before_runs=true")
    source = payload.get("source")
    data = payload.get("data")
    models = payload.get("models")
    selection = payload.get("selection")
    if not isinstance(source, Mapping):
        report.blockers.append("protocol.source is missing")
    elif not source.get("commit") and not source.get("tag"):
        report.blockers.append("protocol.source must freeze either commit or tag")
    if not isinstance(data, Mapping):
        report.blockers.append("protocol.data is missing")
    else:
        if not data.get("dataset_revision"):
            report.blockers.append("protocol.data.dataset_revision is not frozen")
        if not data.get("universe_manifest_sha256"):
            report.blockers.append("protocol.data.universe_manifest_sha256 is not frozen")
        if data.get("normalization_fit_split") != "train_only":
            report.blockers.append("normalization_fit_split must be train_only")
    if not isinstance(selection, Mapping) or selection.get("test_used_for_selection") is not False:
        report.blockers.append("protocol must prohibit test use for model selection")
    elif (
        selection.get("architecture_rule")
        != "lowest_mean_validation_primary_metric_over_all_matched_training_seeds"
        or selection.get("architecture_aggregation") != "arithmetic_mean_over_matched_training_seeds"
    ):
        report.blockers.append("architecture selection must aggregate validation performance over all matched seeds")
    if not isinstance(models, Mapping):
        report.blockers.append("protocol.models is missing")
        return []
    seeds = models.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        report.blockers.append("protocol.models.training_seeds must be a non-empty integer list")
        return []
    if len(set(seeds)) != len(seeds):
        report.blockers.append("protocol training seeds must be unique")
    if models.get("families") != ["persistence", "gru", "transformer"]:
        report.blockers.append("protocol model families must be persistence, gru, and transformer")
    accepted = payload.get("accepted_runs")
    if not resume and accepted:
        report.blockers.append("new execution requires an empty accepted_runs list; use --resume for existing runs")
    return seeds


def _validate_held_out_manifest(
    protocol: Mapping[str, Any],
    manifest_path: Path | None,
    report: PreflightReport,
    *,
    resume: bool,
    universe_ids: set[str] | None,
) -> None:
    data = protocol.get("data")
    if not isinstance(data, Mapping):
        return
    rule = data.get("final_test_rule")
    if not isinstance(rule, Mapping):
        report.blockers.append("protocol.data.final_test_rule is missing")
        return
    route = data.get("evaluation_route")
    if route == "nested_group_cross_validation":
        if manifest_path is not None:
            report.blockers.append("nested-CV route must not supply or open a final-test manifest")
        return
    if route != "final_unseen_test":
        report.blockers.append("protocol.data.evaluation_route is invalid or missing")
        return
    expected_hash = rule.get("manifest_sha256")
    if not expected_hash:
        report.blockers.append("final-test manifest hash is not frozen in the protocol")
    if manifest_path is None:
        report.blockers.append("--held-out-manifest is required to prove final-test eligibility")
        return
    if not manifest_path.is_file():
        report.blockers.append(f"held-out manifest does not exist: {manifest_path}")
        return
    actual_hash = _sha256(manifest_path)
    if expected_hash and actual_hash != expected_hash:
        report.blockers.append("held-out manifest SHA-256 does not match the frozen protocol")
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        report.blockers.append(str(exc))
        return
    trajectory_ids = manifest.get("trajectory_ids")
    if not isinstance(trajectory_ids, list) or any(not isinstance(item, str) or not item for item in trajectory_ids):
        report.blockers.append("held-out manifest trajectory_ids must be non-empty strings")
        return
    minimum = rule.get("minimum_unseen_trajectories")
    if not isinstance(minimum, int) or len(set(trajectory_ids)) < minimum:
        report.blockers.append(f"held-out manifest does not prove at least {minimum} unique trajectories")
    if len(set(trajectory_ids)) != len(trajectory_ids):
        report.blockers.append("held-out manifest contains duplicate trajectory IDs")
    if universe_ids is not None and not set(trajectory_ids).issubset(universe_ids):
        report.blockers.append("held-out trajectories are not all present in the frozen universe manifest")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        report.blockers.append("held-out manifest protocol_id does not match")
    if manifest.get("dataset_revision") != data.get("dataset_revision"):
        report.blockers.append("held-out manifest dataset_revision does not match")
    if manifest.get("universe_manifest_sha256") != data.get("universe_manifest_sha256"):
        report.blockers.append("held-out manifest universe hash does not match")
    if manifest.get("not_used_for_training") is not True or manifest.get("not_used_for_model_selection") is not True:
        report.blockers.append("held-out manifest lacks explicit non-use attestations")
    if not isinstance(manifest.get("attested_by"), str) or not manifest["attested_by"].strip():
        report.blockers.append("held-out manifest requires a non-empty attested_by")
    opened_at = rule.get("opened_at_utc")
    if resume and not opened_at:
        report.blockers.append("resume requires protocol.final_test_rule.opened_at_utc")
    if not resume and opened_at:
        report.blockers.append("new execution cannot use an already-opened final-test protocol; use --resume")


def _validate_universe_manifest(
    protocol: Mapping[str, Any],
    manifest_path: Path | None,
    report: PreflightReport,
) -> set[str] | None:
    data = protocol.get("data")
    expected_hash = data.get("universe_manifest_sha256") if isinstance(data, Mapping) else None
    if manifest_path is None:
        report.blockers.append("--universe-manifest is required to verify the dataset universe")
        return None
    if not manifest_path.is_file():
        report.blockers.append(f"universe manifest does not exist: {manifest_path}")
        return None
    if expected_hash and _sha256(manifest_path) != expected_hash:
        report.blockers.append("universe manifest SHA-256 does not match the frozen protocol")
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        report.blockers.append(str(exc))
        return None
    ids = manifest.get("trajectory_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids):
        report.blockers.append("universe manifest trajectory_ids must be non-empty strings")
        return None
    if len(set(ids)) != len(ids):
        report.blockers.append("universe manifest contains duplicate trajectory IDs")
    if isinstance(data, Mapping) and manifest.get("dataset_revision") != data.get("dataset_revision"):
        report.blockers.append("universe manifest dataset_revision does not match")
    if isinstance(data, Mapping):
        route = data.get("evaluation_route")
        fallback = data.get("fallback_rule")
        if len(set(ids)) == 51 and route != "nested_group_cross_validation":
            report.blockers.append(
                "the historical 51-trajectory universe cannot prove an unseen final test; "
                "freeze evaluation_route=nested_group_cross_validation"
            )
        if route == "nested_group_cross_validation":
            if not isinstance(fallback, Mapping) or fallback.get("method") != "nested_group_cross_validation":
                report.blockers.append("nested-CV route requires the frozen nested_group_cross_validation fallback")
            elif not isinstance(fallback.get("outer_folds"), int) or fallback["outer_folds"] < 5:
                report.blockers.append("nested-CV route requires at least five frozen outer folds")
    return set(ids)


def _validate_source(protocol: Mapping[str, Any], repo_root: Path, report: PreflightReport) -> None:
    actual = inspect_source(repo_root)
    report.environment["source"] = actual
    source = protocol.get("source")
    if not isinstance(source, Mapping):
        return
    expected_commit = source.get("commit")
    expected_tag = source.get("tag")
    if expected_commit and expected_commit != actual["commit"]:
        report.blockers.append("checked-out Git commit does not match protocol.source.commit")
    if expected_tag:
        if not isinstance(expected_tag, str) or not expected_tag.strip():
            report.blockers.append("protocol.source.tag must be a non-empty Git tag name")
        else:
            resolved_tag = _git_output(repo_root, "rev-parse", "--verify", f"refs/tags/{expected_tag}^{{commit}}")
            if resolved_tag is None:
                report.blockers.append(f"protocol.source.tag does not resolve locally: {expected_tag}")
            elif resolved_tag != actual["commit"]:
                report.blockers.append("protocol.source.tag does not resolve exactly to checked-out HEAD")
    if expected_commit and expected_tag:
        resolved_tag = _git_output(repo_root, "rev-parse", "--verify", f"refs/tags/{expected_tag}^{{commit}}")
        if resolved_tag is not None and resolved_tag != expected_commit:
            report.blockers.append("protocol.source.commit and protocol.source.tag resolve to different commits")
    expected_diff = source.get("tracked_diff_sha256")
    if expected_diff is None:
        report.blockers.append("protocol.source.tracked_diff_sha256 is not frozen")
    elif expected_diff != actual["tracked_diff_sha256"]:
        report.blockers.append("tracked Git diff does not match protocol.source.tracked_diff_sha256")
    if actual["untracked_files"]:
        report.blockers.append("untracked files are not covered by protocol source provenance")


def _config_arg(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or role not in CONFIG_ROLES or not raw_path:
        choices = ", ".join(CONFIG_ROLES)
        raise argparse.ArgumentTypeError(f"config must be ROLE=PATH where ROLE is one of: {choices}")
    return role, Path(raw_path)


def _validate_configs(configs: Mapping[str, Path], report: PreflightReport) -> None:
    for role in CONFIG_ROLES:
        if role not in configs:
            report.blockers.append(f"missing experiment config role: {role}")
    loaded: dict[str, Any] = {}
    for role, path in configs.items():
        if not path.is_file():
            report.blockers.append(f"config for {role} does not exist: {path}")
            continue
        try:
            loaded[role] = load_config(path, command=CONFIG_ROLES[role])
        except Exception as exc:
            report.blockers.append(f"config for {role} failed validation: {exc}")
    expected_splits = {
        "encoder": "train",
        "embed": "all",
        "gru_train": "train",
        "transformer_train": "train",
        "gru_eval": "test",
        "transformer_eval": "test",
        "persistence_eval": "test",
    }
    for role, split in expected_splits.items():
        if role in loaded and loaded[role].data.split != split:
            report.blockers.append(f"config {role} must use data.split={split}")
    if "persistence_eval" in loaded and not loaded["persistence_eval"].latent_cache.use_persistence_baseline:
        report.blockers.append("persistence_eval config must enable the persistence baseline")
    for role in ("gru_eval", "transformer_eval"):
        if role in loaded and loaded[role].latent_cache.use_persistence_baseline:
            report.blockers.append(f"config {role} must not enable the persistence baseline")
    if any(
        config.data.backend == "cyclone_kvikio" and config.data.cyclone and config.data.cyclone.use_kvikio
        for config in loaded.values()
    ) and not report.environment.get("kvikio_available"):
        report.blockers.append("a config requires KvikIO, but the kvikio package is unavailable")


def build_stages(configs: Mapping[str, Path], output_root: Path, seeds: Sequence[int]) -> list[Stage]:
    """Create deterministic argv plans; dependency artifacts remain explicit placeholders."""

    stages: list[Stage] = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"

        def command(role: str, output: Path, *overrides: str, run_seed: int = seed) -> tuple[str, ...]:
            argv = [
                sys.executable,
                "-m",
                "gk_surrogate.cli",
                CONFIG_ROLES[role],
                "--config",
                str(configs[role]),
                "--seed",
                str(run_seed),
                "--output-dir",
                str(output),
            ]
            for override in overrides:
                argv.extend(("--override", override))
            return tuple(argv)

        encoder_out = seed_root / "encoder"
        embed_out = seed_root / "embed"
        cache = embed_out / "latent_cache.h5"
        encoder_ref = "latent_cache.encoder_checkpoint_path={encoder.checkpoint}"
        stages.append(Stage("encoder", seed, command("encoder", encoder_out), encoder_out))
        stages.append(
            Stage("embed", seed, command("embed", embed_out, encoder_ref), embed_out, ("encoder",))
        )
        for family in ("gru", "transformer"):
            train_role = f"{family}_train"
            train_out = seed_root / train_role
            stages.append(
                Stage(
                    train_role,
                    seed,
                    command(
                        train_role,
                        train_out,
                        f"latent_cache.path={cache}",
                        encoder_ref,
                    ),
                    train_out,
                    ("encoder", "embed"),
                )
            )
            eval_role = f"{family}_eval"
            eval_out = seed_root / eval_role
            stages.append(
                Stage(
                    eval_role,
                    seed,
                    command(
                        eval_role,
                        eval_out,
                        f"latent_cache.path={cache}",
                        encoder_ref,
                        f"latent_cache.sequence_checkpoint_path={{{train_role}.checkpoint}}",
                    ),
                    eval_out,
                    ("encoder", "embed", train_role),
                )
            )
        persistence_out = seed_root / "persistence_eval"
        stages.append(
            Stage(
                "persistence_eval",
                seed,
                command(
                    "persistence_eval",
                    persistence_out,
                    f"latent_cache.path={cache}",
                    encoder_ref,
                ),
                persistence_out,
                ("encoder", "embed"),
            )
        )
    return stages


def _stage_payload(stage: Stage) -> dict[str, object]:
    return {
        "name": stage.name,
        "seed": stage.seed,
        "command": list(stage.command),
        "output_dir": str(stage.output_dir),
        "dependencies": list(stage.dependencies),
        "status": "planned",
    }


def _load_stage_evidence(stage: Stage) -> dict[str, str] | None:
    metrics_path = stage.output_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        payload = _read_json(metrics_path)
    except ValueError:
        return None
    evidence = {"metrics": str(metrics_path)}
    for key in ("checkpoint", "latent_cache"):
        raw = payload.get(key)
        if isinstance(raw, str) and Path(raw).exists():
            evidence[key] = raw
    if stage.name in {"encoder", "gru_train", "transformer_train"} and "checkpoint" not in evidence:
        return None
    if stage.name == "embed" and "latent_cache" not in evidence:
        return None
    return evidence


def _resolve_command(command: Sequence[str], evidence: Mapping[str, Mapping[str, str]]) -> list[str]:
    resolved = []
    for argument in command:
        value = argument
        for stage, fields in evidence.items():
            for evidence_field, replacement in fields.items():
                value = value.replace(f"{{{stage}.{evidence_field}}}", replacement)
        if "{" in value or "}" in value:
            raise ValueError(f"unresolved stage dependency in command argument: {value}")
        resolved.append(value)
    return resolved


def execute_stages(
    stages: Sequence[Stage],
    *,
    repo_root: Path,
    resume: bool,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    by_seed: dict[int, dict[str, dict[str, str]]] = {}
    for stage in stages:
        evidence = by_seed.setdefault(stage.seed, {})
        existing = _load_stage_evidence(stage) if resume else None
        if existing is not None:
            evidence[stage.name] = existing
            results.append({"name": stage.name, "seed": stage.seed, "status": "resumed_existing", **existing})
            continue
        command = _resolve_command(stage.command, evidence)
        completed = command_runner(command, cwd=repo_root, check=False, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"stage {stage.name} seed {stage.seed} failed with exit code {completed.returncode}")
        produced = _load_stage_evidence(stage)
        if produced is None:
            raise RuntimeError(f"stage {stage.name} seed {stage.seed} returned success without valid evidence")
        evidence[stage.name] = produced
        results.append({"name": stage.name, "seed": stage.seed, "status": "completed", **produced})
    return results


def preflight(
    protocol_path: Path,
    *,
    configs: Mapping[str, Path],
    output_root: Path,
    held_out_manifest: Path | None,
    universe_manifest: Path | None,
    repo_root: Path,
    execute: bool,
    resume: bool,
    environment: Mapping[str, object] | None = None,
) -> tuple[PreflightReport, list[Stage]]:
    protocol = _read_json(protocol_path)
    report = PreflightReport(mode="resume" if resume else "execute" if execute else "preflight")
    report.environment.update(environment or inspect_environment())
    seeds = _validate_protocol(protocol, report, resume=resume)
    _validate_source(protocol, repo_root, report)
    universe_ids = _validate_universe_manifest(protocol, universe_manifest, report)
    _validate_held_out_manifest(
        protocol,
        held_out_manifest,
        report,
        resume=resume,
        universe_ids=universe_ids,
    )
    _validate_configs(configs, report)
    data = protocol.get("data")
    if isinstance(data, Mapping) and data.get("evaluation_route") == "nested_group_cross_validation":
        message = (
            "nested group CV requires exact frozen outer-fold assignments, which the current training CLI "
            "cannot consume; do not substitute its ordinary train/validation/test split"
        )
        if execute or resume:
            report.blockers.append(message)
        else:
            report.warnings.append(message)
    if not report.environment.get("gpu_available"):
        message = "JAX reports no GPU; the accepted multi-seed protocol cannot execute on CPU only"
        if execute or resume:
            report.blockers.append(message)
        else:
            report.warnings.append(message)
    if not report.environment.get("kvikio_available"):
        report.warnings.append("KvikIO is unavailable")
    if not report.environment.get("cupy_available"):
        report.warnings.append("CuPy is unavailable")
    stages = build_stages(configs, output_root, seeds) if not set(CONFIG_ROLES).difference(configs) else []
    report.stages = [_stage_payload(stage) for stage in stages]
    report.ready = not report.blockers
    return report, stages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gks-protocol")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiment_protocols/multiseed_v1.json"),
    )
    parser.add_argument("--config", action="append", default=[], type=_config_arg, metavar="ROLE=PATH")
    parser.add_argument("--held-out-manifest", type=Path)
    parser.add_argument("--universe-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multiseed-v1"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Execute a ready frozen protocol.")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Execute while reusing stages with verified local evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configs = dict(args.config)
    try:
        report, stages = preflight(
            args.protocol,
            configs=configs,
            output_root=args.output_root,
            held_out_manifest=args.held_out_manifest,
            universe_manifest=args.universe_manifest,
            repo_root=args.repo_root,
            execute=args.execute,
            resume=args.resume,
        )
        if (args.execute or args.resume) and report.ready:
            report.execution = execute_stages(
                stages,
                repo_root=args.repo_root,
                resume=args.resume,
            )
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
