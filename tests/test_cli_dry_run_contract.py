from __future__ import annotations

import json
from pathlib import Path

import pytest

from gk_surrogate.cli import main


@pytest.mark.parametrize(
    ("command", "config_name"),
    [
        ("train-encoder", "smoke_encoder_supervised.yaml"),
        ("embed-dataset", "smoke_embed_dataset.yaml"),
        ("train-sequence", "smoke_sequence.yaml"),
        ("evaluate-flux-head", "smoke_evaluate_flux_head.yaml"),
        ("plot-representation", "smoke_plot_representation.yaml"),
        ("evaluate-rollout", "smoke_evaluate_rollout.yaml"),
        ("benchmark-step-time", "smoke_encoder_supervised.yaml"),
    ],
)
def test_cli_dry_run_does_not_write_training_artifacts(repo_root, tmp_path, command, config_name):
    config = repo_root / "configs" / "experiment" / config_name
    output_dir = tmp_path / command
    assert main([command, "--config", str(config), "--dry-run", "--output-dir", str(output_dir)]) == 0
    assert not (output_dir / "checkpoints").exists()
    assert not (output_dir / "latent_cache.h5").exists()
    assert not (output_dir / "plots").exists()
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "config_resolved.yaml").exists()


def test_make_synthetic_h5_dry_run_does_not_write(repo_root, tmp_path, capsys):
    config = repo_root / "configs/data/tiny_dummy.yaml"
    assert main(["make-synthetic-h5", "--config", str(config), "--dry-run", "--output-dir", str(tmp_path)]) == 0
    assert list(Path(tmp_path).glob("*.h5")) == []
    output = capsys.readouterr().out
    assert "Resolved config:" in output
    assert "name: data_only" in output


def test_inspect_data_output_dir_writes_report(repo_root, tmp_path):
    config = repo_root / "configs/data/tiny_dummy.yaml"
    assert main(["inspect-data", "--config", str(config), "--output-dir", str(tmp_path)]) == 0
    report = tmp_path / "data_inspection.json"
    resolved = tmp_path / "config_resolved.yaml"
    assert report.exists()
    assert resolved.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["backend"] == "synthetic"
    assert payload["snapshot_shape"] == [2, 4, 4, 4, 4, 4]


def test_cyclone_inspect_dry_run_reports_permission_blocker(monkeypatch, repo_root, tmp_path):
    config = repo_root / "configs/data/cyclone_kvikio_template.yaml"
    monkeypatch.setenv("GK_CYCLONE_DATA_ROOT", str(tmp_path / "restricted"))

    def blocked_inspection(*_args, **_kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr("gk_surrogate.cli.inspect_dataset", blocked_inspection)
    assert main(["inspect-data", "--config", str(config), "--dry-run"]) == 0


@pytest.mark.parametrize(
    "args",
    (
        ["inspect-data", "--max-trajectories", "0"],
        ["inspect-data", "--max-depth", "-1"],
        ["inspect-data", "--max-target-samples", "0"],
        ["benchmark-step-time", "--measured-steps", "0"],
    ),
)
def test_cli_rejects_invalid_bounded_work_arguments(repo_root, args):
    config = repo_root / "configs/data/tiny_dummy.yaml"
    with pytest.raises(SystemExit) as exc_info:
        main([args[0], "--config", str(config), *args[1:]])
    assert exc_info.value.code == 2
