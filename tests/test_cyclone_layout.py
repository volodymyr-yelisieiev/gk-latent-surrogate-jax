from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gk_surrogate.data.cyclone_layout import inspect_cyclone_layout


def test_inspect_cyclone_layout_reports_binary_shards(tmp_path: Path) -> None:
    traj_a = tmp_path / "traj_a"
    traj_b = tmp_path / "traj_b"
    (traj_a / "data").mkdir(parents=True)
    (traj_b / "data").mkdir(parents=True)
    (traj_a / "metadata.pkl").write_bytes(b"meta")
    (traj_a / "metadata_light.pkl").write_bytes(b"light")
    (traj_a / "data" / "timestep_00000.bin").write_bytes(b"df")
    (traj_a / "data" / "timestep_00000.bf16.bin").write_bytes(b"bf16")
    (traj_a / "data" / "poten_00000.bin").write_bytes(b"phi")
    (traj_a / "data" / "poten_00000.bf16.bin").write_bytes(b"phi-bf16")
    (traj_b / "data" / "timestep_00000.bin").write_bytes(b"df")

    report = inspect_cyclone_layout(tmp_path)
    payload = report.as_dict()

    assert payload["root"] == str(tmp_path)
    assert payload["trajectory_count"] == 2
    assert payload["sample_count_estimate"] == 2
    assert payload["quantized_shards_available"] is True
    assert payload["metadata_pkl_count"] == 1
    assert payload["metadata_light_pkl_count"] == 1
    assert payload["trajectories"][0]["trajectory_id"] == "traj_a"
    assert payload["trajectories"][0]["timestep_bin_count"] == 1
    assert payload["trajectories"][0]["timestep_bf16_bin_count"] == 1
    assert payload["trajectories"][0]["potential_bin_count"] == 1
    assert payload["trajectories"][0]["first_timestep_bin"] == "data/timestep_00000.bin"
    assert payload["trajectories"][0]["first_timestep_bf16_bin"] == "data/timestep_00000.bf16.bin"
    assert payload["trajectories"][0]["first_potential_bin"] == "data/poten_00000.bin"


def test_inspect_cyclone_layout_warns_when_empty_or_truncated(tmp_path: Path) -> None:
    empty = inspect_cyclone_layout(tmp_path)
    assert empty.trajectory_count == 0
    assert "no trajectory directories" in "\n".join(empty.warnings)

    for index in range(3):
        path = tmp_path / f"traj_{index}" / "data"
        path.mkdir(parents=True)
        (path / "timestep_00000.bin").write_bytes(b"df")
    report = inspect_cyclone_layout(tmp_path, max_trajectories=1)
    warnings = "\n".join(report.warnings)
    assert report.trajectory_count == 3
    assert "truncated to 1 of 3" in warnings
    assert "no metadata.pkl" in warnings


def test_inspect_cyclone_layout_script_writes_json(repo_root: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "raw" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "timestep_00000.bin").write_bytes(b"df")
    output = tmp_path / "report.json"

    script = _load_layout_script(repo_root)
    assert script.main(["--root", str(tmp_path / "raw"), "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["trajectory_count"] == 1
    assert payload["trajectories"][0]["trajectory_id"] == "raw"


def test_inspect_cyclone_layout_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_trajectories"):
        inspect_cyclone_layout(tmp_path, max_trajectories=0)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        inspect_cyclone_layout(tmp_path / "missing")


def test_inspect_cyclone_layout_reports_permission_blocker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_exists = Path.exists

    def blocked_exists(path: Path) -> bool:
        if path == tmp_path:
            raise PermissionError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", blocked_exists)
    report = inspect_cyclone_layout(tmp_path)

    assert report.trajectory_count == 0
    assert report.sample_count_estimate == 0
    assert "permission denied" in "\n".join(report.warnings)


def _load_layout_script(repo_root: Path):
    path = repo_root / "scripts" / "inspect_cyclone_layout.py"
    spec = importlib.util.spec_from_file_location("inspect_cyclone_layout_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
