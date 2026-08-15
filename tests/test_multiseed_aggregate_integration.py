from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gk_surrogate import multiseed_aggregate as aggregate_module
from gk_surrogate import protocol_runner


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _folds(ids: list[str]) -> dict[int, dict[str, Any]]:
    # The aggregation contract cares about exact pairing and fold bookkeeping;
    # repeated deterministic partitions keep this fixture compact while still
    # exercising all five outer folds and all five matched seeds.
    return {
        fold: {
            "fold_id": f"outer-{fold}",
            "splits": {
                "train": ids[:2],
                "val": [ids[2]],
                "test": ids[3:],
            },
        }
        for fold in range(5)
    }


def _stage_plan(
    output_root: Path,
    cli_paths: dict[int, Path],
    seeds: tuple[int, ...],
) -> list[protocol_runner.Stage]:
    stages: list[protocol_runner.Stage] = []
    for outer_fold in range(5):
        for seed in seeds:
            seed_root = output_root / f"outer_fold_{outer_fold}" / f"seed_{seed}"
            stages.extend(
                [
                    protocol_runner.Stage(
                        "encoder",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "encoder",
                        outer_fold=outer_fold,
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "embed",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "embed",
                        outer_fold=outer_fold,
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "gru_train",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "gru_train",
                        outer_fold=outer_fold,
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "transformer_train",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "transformer_train",
                        outer_fold=outer_fold,
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "gru_validation",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "gru_validation",
                        outer_fold=outer_fold,
                        phase="validation",
                        family="gru",
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "transformer_validation",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "transformer_validation",
                        outer_fold=outer_fold,
                        phase="validation",
                        family="transformer",
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "gru_eval",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "gru_eval",
                        outer_fold=outer_fold,
                        phase="test",
                        family="gru",
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "transformer_eval",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "transformer_eval",
                        outer_fold=outer_fold,
                        phase="test",
                        family="transformer",
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "latent_persistence_eval",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "latent_persistence_eval",
                        outer_fold=outer_fold,
                        phase="test",
                        family="latent_state_persistence_decoded",
                        data_seed=52,
                    ),
                    protocol_runner.Stage(
                        "observed_persistence_eval",
                        seed,
                        ("fake", "--split-manifest", str(cli_paths[outer_fold])),
                        seed_root / "observed_persistence_eval",
                        outer_fold=outer_fold,
                        phase="test",
                        family="observed_diagnostic_persistence",
                        data_seed=52,
                    ),
                ]
            )
        stages.append(
            protocol_runner.Stage(
                "architecture_selection",
                None,
                (),
                output_root / f"outer_fold_{outer_fold}" / "selection",
                outer_fold=outer_fold,
                phase="selection",
                data_seed=52,
            )
        )
    # The production plan has 255 slots: 50 validation, 20 selection/train/
    # representation slots per fold, and 20 test slots per fold.
    assert len(stages) == 255
    return stages


def _payload(
    stage: protocol_runner.Stage,
    *,
    ids: list[str],
    universe: list[str],
    split_hash: str,
    encoder_hash: str,
    cache_hash: str,
    sequence_hash: str,
) -> dict[str, Any]:
    split = (
        "val"
        if stage.phase == "validation"
        else "test"
        if stage.phase == "test"
        else "all"
        if stage.name == "embed"
        else "train"
    )
    if split == "train":
        baseline = "none"
    elif stage.name == "latent_persistence_eval":
        baseline = "latent_state_persistence_decoded"
    elif stage.name == "observed_persistence_eval":
        baseline = "observed_diagnostic_persistence"
    else:
        baseline = "none"
    payload: dict[str, Any] = {
        "protocol_version": 1,
        "data_split": split,
        "data_split_seed": 52,
        "training_seed": stage.seed,
        "selected_trajectory_ids": ids,
        "trajectory_manifest_sha256": protocol_runner._trajectory_manifest_sha256(ids),
        "universe_trajectory_ids": universe,
        "universe_manifest_sha256": protocol_runner._trajectory_manifest_sha256(universe),
        "split_fold_id": f"outer-{stage.outer_fold}",
        "split_manifest_sha256": split_hash,
        "baseline_mode": baseline,
    }
    if stage.name == "encoder":
        payload.update(
            artifact_role="encoder_checkpoint",
            checkpoint=str(stage.output_dir / "checkpoint.pkl"),
            checkpoint_sha256=encoder_hash,
            checkpoint_selection="minimum_validation_trajectory_balanced_flux_rmse",
            best_validation_metric=1.0,
        )
    elif stage.name == "embed":
        payload.update(
            artifact_role="latent_cache",
            latent_cache=str(stage.output_dir / "latent_cache.h5"),
            latent_cache_sha256=cache_hash,
            encoder_checkpoint_sha256=encoder_hash,
        )
    elif stage.name in {"gru_train", "transformer_train"}:
        payload.update(
            artifact_role="sequence_checkpoint",
            checkpoint=str(stage.output_dir / "checkpoint.pkl"),
            checkpoint_sha256=sequence_hash,
            checkpoint_selection="minimum_validation_trajectory_balanced_latent_rmse",
            best_validation_metric=0.5,
        )
    elif stage.phase == "validation":
        metric = 1.0 if stage.family == "gru" else 2.0
        payload.update(
            num_trajectories=len(ids),
            stable=True,
            flux_rmse=metric,
            flux_trajectory_ids=ids,
            flux_rmse_by_trajectory=[metric for _ in ids],
            trajectory_balanced_flux_rmse=metric,
            latent_cache_sha256=cache_hash,
            encoder_checkpoint_sha256=encoder_hash,
            sequence_checkpoint_sha256=sequence_hash,
        )
    elif stage.phase == "test":
        is_observed = stage.name == "observed_persistence_eval"
        is_latent = stage.name == "latent_persistence_eval"
        learned_flux = [2.0, 3.0]
        observed_flux = [1.0, 1.0]
        latent_flux = [1.5, 1.6]
        oracle_flux = [1.8, 2.0]
        flux_values = observed_flux if is_observed else latent_flux if is_latent else learned_flux
        payload.update(
            num_trajectories=len(ids),
            stable=True,
            rollout_method=baseline if (is_observed or is_latent) else "learned_sequence_model",
            latent_cache_sha256=cache_hash,
            encoder_checkpoint_sha256=encoder_hash,
            sequence_checkpoint_sha256=None if (is_observed or is_latent) else sequence_hash,
            flux_trajectory_ids=ids,
            latent_trajectory_ids=ids,
            flux_rmse_by_trajectory=flux_values,
            trajectory_balanced_flux_rmse=sum(flux_values) / len(ids),
            flux_rmse=sum(flux_values) / len(ids),
            observed_diagnostic_persistence_flux_rmse_by_trajectory=observed_flux,
            diagnostic_head_oracle_flux_rmse_by_trajectory=oracle_flux,
            mse_by_trajectory=[0.4, 0.5] if is_latent else [0.2, 0.3],
            flux_rmse_by_step=[1.0, 1.1],
            diagnostic_head_oracle_flux_rmse_by_step=[1.2, 1.3],
            mse_by_step=[0.2, 0.3] if not is_latent else [0.4, 0.5],
            flux_mae=0.5 if is_observed else 0.6,
            spectra_kyspec_relative_l2=0.1,
            spectra_kyspec_shape_corr=0.9,
            spectra_fluxspec_relative_l2=0.2,
            spectra_fluxspec_shape_corr=0.8,
            observed_diagnostic_persistence_spectra_kyspec_relative_l2=0.3,
            observed_diagnostic_persistence_spectra_kyspec_shape_corr=0.7,
            observed_diagnostic_persistence_spectra_fluxspec_relative_l2=0.4,
            observed_diagnostic_persistence_spectra_fluxspec_shape_corr=0.6,
        )
    return payload


def test_aggregate_reconstructs_full_nested_cv_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    protocol_path = repo_root / "protocol.json"
    universe_path = repo_root / "universe.json"
    fold_manifest_path = repo_root / "folds.json"
    output_root = repo_root / "result"
    universe_ids = [f"traj-{index}" for index in range(5)]
    universe_path.write_text(
        json.dumps({"dataset_revision": "dataset-v1", "trajectory_ids": universe_ids}),
        encoding="utf-8",
    )
    fold_manifest_path.write_text("fold-index", encoding="utf-8")
    protocol = json.loads((Path(__file__).parents[1] / "experiment_protocols/multiseed_v1.json").read_text())
    protocol = copy.deepcopy(protocol)
    protocol["status"] = "frozen"
    protocol["source"] = {
        "repository": "https://example.invalid/repo",
        "commit": "0" * 40,
        "tag": "test-freeze",
        "tracked_diff_sha256": _digest(""),
    }
    protocol["data"]["dataset_revision"] = "dataset-v1"
    protocol["data"]["universe_manifest_sha256"] = _digest(universe_path.read_text(encoding="utf-8"))
    protocol["data"]["fallback_rule"]["outer_fold_manifest_sha256"] = _digest("fold-index")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    seeds = (52, 53, 54, 55, 56)
    folds = _folds(universe_ids)
    cli_paths = {fold: repo_root / f"outer-{fold}.json" for fold in range(5)}
    cli_hashes = {}
    for fold, path in cli_paths.items():
        path.write_text(f"cli-{fold}", encoding="utf-8")
        cli_hashes[fold] = _digest(f"cli-{fold}")
    stages = _stage_plan(output_root, cli_paths, seeds)
    for stage in stages:
        if stage.phase == "selection" or (stage.phase == "test" and stage.family == "transformer"):
            continue
        stage.output_dir.mkdir(parents=True, exist_ok=True)
        (stage.output_dir / "config_resolved.json").write_text("{}", encoding="utf-8")

    failure_mode = {"name": None}

    def fake_payload(stage: protocol_runner.Stage) -> tuple[dict[str, Any], dict[str, str]]:
        fold = int(stage.outer_fold)
        split = (
            "val"
            if stage.phase == "validation"
            else "test"
            if stage.phase == "test"
            else "all"
            if stage.name == "embed"
            else "train"
        )
        ids = folds[fold]["splits"][split] if split != "all" else universe_ids
        encoder_hash = _digest(f"encoder-{fold}-{stage.seed}")
        cache_hash = _digest(f"cache-{fold}-{stage.seed}")
        sequence_hash = _digest(f"{stage.name.split('_')[0]}-{fold}-{stage.seed}")
        payload = _payload(
            stage,
            ids=ids,
            universe=universe_ids,
            split_hash=cli_hashes[fold],
            encoder_hash=encoder_hash,
            cache_hash=cache_hash,
            sequence_hash=sequence_hash,
        )
        # The lineage checks intentionally use the same seed-specific hashes
        # across all roles while preserving distinct GRU/Transformer hashes.
        if stage.name in {"gru_validation", "transformer_validation"}:
            payload["sequence_checkpoint_sha256"] = _digest(
                f"{stage.family}-{fold}-{stage.seed}"
            )
        if stage.name == "embed":
            payload["encoder_checkpoint_sha256"] = encoder_hash
        if failure_mode["name"] == "baseline" and stage.name == "gru_eval":
            payload["observed_diagnostic_persistence_flux_rmse_by_trajectory"] = [9.0, 9.0]
        elif failure_mode["name"] == "pairing" and stage.name == "latent_persistence_eval":
            reversed_ids = list(reversed(ids))
            payload["selected_trajectory_ids"] = reversed_ids
            payload["flux_trajectory_ids"] = reversed_ids
            payload["latent_trajectory_ids"] = reversed_ids
            payload["trajectory_manifest_sha256"] = protocol_runner._trajectory_manifest_sha256(reversed_ids)
        elif failure_mode["name"] == "spectra_relative" and stage.name == "latent_persistence_eval":
            payload.pop("spectra_kyspec_relative_l2")
        elif failure_mode["name"] == "spectra_shape" and stage.name == "latent_persistence_eval":
            payload.pop("spectra_kyspec_shape_corr")
        evidence = {
            "metrics": str(stage.output_dir / "metrics.json"),
            "metrics_sha256": _digest(f"metrics-{stage.name}-{fold}-{stage.seed}"),
        }
        if stage.name == "encoder":
            evidence["checkpoint_sha256"] = encoder_hash
        elif stage.name == "embed":
            evidence["latent_cache_sha256"] = cache_hash
        elif stage.name in {"gru_train", "transformer_train"}:
            evidence["checkpoint_sha256"] = sequence_hash
        return payload, evidence

    config_paths = {}
    for role in protocol_runner.CONFIG_ROLES:
        config_path = repo_root / f"{role}.yaml"
        config_path.write_text(f"role: {role}\n", encoding="utf-8")
        config_paths[role] = config_path

    for fold in range(5):
        selection_path = output_root / f"outer_fold_{fold}" / "selection" / "metrics.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_bytes(
            protocol_runner._canonical_json_bytes(
                {
                    "protocol_version": 1,
                    "outer_fold": fold,
                    "selection_split": "val",
                    "primary_metric": "trajectory_balanced_flux_rmse",
                    "matched_training_seeds": list(seeds),
                    "candidate_mean_validation_trajectory_balanced_flux_rmse": {"gru": 1.0, "transformer": 2.0},
                    "candidate_validation_metrics_sha256": {
                        family: {
                            str(seed): _digest(f"metrics-{family}_validation-{fold}-{seed}")
                            for seed in seeds
                        }
                        for family in ("gru", "transformer")
                    },
                    "tie_break_rule": "lexicographic_family_name",
                    "selected_family": "gru",
                    "test_evidence_opened": False,
                }
            )
        )

    monkeypatch.setattr(aggregate_module.protocol_runner, "build_stages", lambda *args, **kwargs: stages)
    monkeypatch.setattr(aggregate_module, "_fold_manifests", lambda *args, **kwargs: folds)
    monkeypatch.setattr(aggregate_module, "_metrics_and_evidence", fake_payload)
    monkeypatch.setattr(aggregate_module, "_source_tag_commit", lambda *args, **kwargs: ("test-freeze", "0" * 40))
    monkeypatch.setattr(aggregate_module, "_validate_wandb_contract", lambda *_args, **_kwargs: None)

    def fake_git_show(args: list[str], **kwargs: Any) -> Any:
        if args[:2] == ["git", "show"]:
            return SimpleNamespace(returncode=0, stdout=protocol_path.read_bytes())
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(aggregate_module.subprocess, "run", fake_git_show)
    result = aggregate_module.aggregate(
        protocol_path,
        fold_manifest_path=fold_manifest_path,
        universe_manifest_path=universe_path,
        output_root=output_root,
        configs=config_paths,
        repo_root=repo_root,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    assert result["status"] == "accepted"
    assert result["selection"]["selected_family_by_outer_fold"] == {str(index): "gru" for index in range(5)}
    assert result["primary"]["num_paired_trajectory_runs"] == 50
    assert result["secondary"]["mean_latent_mse_difference"] == pytest.approx(-0.2)
    assert len(result["stage_ledger"]) == 255
    assert result["wandb_status"]["enabled"] is False
    for artifact in (
        "aggregate_results_json",
        "outer_fold_summary_csv",
        "paired_trajectory_results_csv",
        "primary_difference_figure",
    ):
        assert Path(result["artifacts"][artifact]).is_file()
    assert json.loads((output_root / "aggregate_results.json").read_text(encoding="utf-8"))["status"] == "accepted"

    for mode, message in (
        ("baseline", "observed baseline disagrees"),
        ("pairing", "trajectory pairing mismatch"),
        ("spectra_relative", "missing spectral metric"),
        ("spectra_shape", "missing spectral metric"),
    ):
        failure_mode["name"] = mode
        with pytest.raises(aggregate_module.AggregationError, match=message):
            aggregate_module.aggregate(
                protocol_path,
                fold_manifest_path=fold_manifest_path,
                universe_manifest_path=universe_path,
                output_root=output_root,
                configs=config_paths,
                repo_root=repo_root,
                bootstrap_replicates=100,
                bootstrap_seed=7,
            )
    failure_mode["name"] = None

    missing_config = output_root / "outer_fold_0" / "seed_52" / "encoder" / "config_resolved.json"
    missing_config.unlink()
    with pytest.raises(aggregate_module.AggregationError, match="missing resolved config"):
        aggregate_module.aggregate(
            protocol_path,
            fold_manifest_path=fold_manifest_path,
            universe_manifest_path=universe_path,
            output_root=output_root,
            configs=config_paths,
            repo_root=repo_root,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
    missing_config.write_text("{}", encoding="utf-8")

    unselected_dir = output_root / "outer_fold_0" / "seed_52" / "transformer_eval"
    unselected_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(aggregate_module.AggregationError, match="unselected test stage"):
        aggregate_module.aggregate(
            protocol_path,
            fold_manifest_path=fold_manifest_path,
            universe_manifest_path=universe_path,
            output_root=output_root,
            configs=config_paths,
            repo_root=repo_root,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
    unselected_dir.rmdir()

    # The production result must contain all 255 slots; a deliberately
    # truncated outer-fold universe is rejected before any summary is written.
    monkeypatch.setattr(aggregate_module, "EXPECTED_FOLDS", (0,))
    with pytest.raises(aggregate_module.AggregationError, match="stage ledger"):
        aggregate_module.aggregate(
            protocol_path,
            fold_manifest_path=fold_manifest_path,
            universe_manifest_path=universe_path,
            output_root=output_root,
            configs=config_paths,
            repo_root=repo_root,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
