from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from gk_surrogate.cli import main
from gk_surrogate.config.schema import DataConfig
from gk_surrogate.data.inspect import inspect_dataset


class InspectionCycloneDataset:
    def __init__(self, root: str, **kwargs: Any):
        self.root = root
        self.kwargs = kwargs
        self.metadata = {"normalization": {"source": "fixture"}}
        self.geometry = {"kx": np.arange(2), "ky": np.arange(2)}

    def trajectory_ids(self) -> tuple[str]:
        return ("inspection_traj",)

    def num_timesteps(self, trajectory_id: str) -> int:
        if trajectory_id != "inspection_traj":
            raise KeyError(trajectory_id)
        return 2

    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> dict[str, Any]:
        return {
            "df": np.ones((2, 2, 2, 2, 2, 2), dtype=np.float32) * timestep_index,
            "flux": np.asarray([float(timestep_index)], dtype=np.float32),
            "timestep": np.asarray(float(timestep_index), dtype=np.float32),
            "file_index": 0,
            "timestep_index": timestep_index,
            "conditioning": np.asarray([1.0, 2.0], dtype=np.float32),
        }

    def __len__(self) -> int:
        return 2


def test_cyclone_inspection_reports_missing_requested_spectra_without_hiding_flux(monkeypatch, tmp_path: Path) -> None:
    _install_fake_neugk(monkeypatch)
    data = DataConfig.model_validate(
        {
            "backend": "cyclone_kvikio",
            "root": str(tmp_path),
            "target_flux": True,
            "target_spectra": ["ky"],
            "batch_size": 1,
            "shuffle": False,
            "cyclone": {
                "fields_to_load": ["df"],
                "conditions": ["q"],
                "prefer_dtype": "float32",
            },
        }
    )

    inspection = inspect_dataset(data, max_trajectories=1, max_target_samples=2)
    warnings = "\n".join(inspection.warnings)

    assert inspection.backend == "cyclone_kvikio"
    assert inspection.snapshot_shape == (2, 2, 2, 2, 2, 2)
    assert inspection.flux_shape == (1,)
    assert inspection.flux_stats is not None
    assert inspection.flux_stats["count"] == 2
    assert "spectra target 'ky' requested but not found" in warnings
    assert "missing spectra targets: ky" in warnings
    assert inspection.spectra_shapes == {}
    assert inspection.sample_count_estimate == 2


def test_cyclone_inspection_cli_writes_report_under_output_dir(monkeypatch, tmp_path: Path) -> None:
    _install_fake_neugk(monkeypatch)
    root = tmp_path / "raw"
    output_dir = tmp_path / "inspection"
    config_path = tmp_path / "cyclone.yaml"
    config_path.write_text(
        f"""
data:
  backend: cyclone_kvikio
  root: {root}
  input_fields: [df]
  target_flux: true
  target_spectra: []
  batch_size: 1
  shuffle: false
  cyclone:
    fields_to_load: [df]
    conditions: [q]
    prefer_dtype: float32
    use_kvikio: false
loss:
  flux_weight: 0.0
  spectra_weight: 0.0
""",
        encoding="utf-8",
    )

    assert main(["inspect-data", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0

    report = output_dir / "data_inspection.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["backend"] == "cyclone_kvikio"
    assert payload["root"] == str(root)
    assert payload["sample_count_estimate"] == 2
    assert payload["kvikio_enabled"] is False
    assert payload["preferred_dtype"] == "float32"
    assert not (root / "data_inspection.json").exists()


def _install_fake_neugk(monkeypatch) -> None:
    package = types.ModuleType("neugk_jax")
    package.__path__ = []
    dataset_package = types.ModuleType("neugk_jax.dataset")
    dataset_package.__path__ = []
    cyclone_module = types.ModuleType("neugk_jax.dataset.cyclone")
    cyclone_module.CycloneDataset = InspectionCycloneDataset
    dataset_package.CycloneDataset = InspectionCycloneDataset
    dataset_package.cyclone = cyclone_module
    package.dataset = dataset_package
    monkeypatch.setitem(sys.modules, "neugk_jax", package)
    monkeypatch.setitem(sys.modules, "neugk_jax.dataset", dataset_package)
    monkeypatch.setitem(sys.modules, "neugk_jax.dataset.cyclone", cyclone_module)
