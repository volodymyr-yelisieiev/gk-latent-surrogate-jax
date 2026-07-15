from __future__ import annotations

import importlib
from pathlib import Path

import jax


def test_import_package_without_gpu():
    module = importlib.import_module("gk_surrogate")
    assert module.__version__
    assert jax.default_backend() == "cpu"


def test_core_code_has_no_cuda_or_old_repo_dependency(repo_root):
    paths = [
        *list((repo_root / "src").rglob("*.py")),
        *list((repo_root / "configs").rglob("*.yaml")),
        *list((repo_root / "scripts").rglob("*.py")),
        repo_root / "pyproject.toml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    lowered = text.lower()
    assert "cuda" not in lowered
    assert "import ndswin" not in lowered
    assert "pytorch" not in lowered
    assert "/mnt/d" not in lowered
    assert "/users/" not in lowered
    assert "pjit" not in lowered
    assert "cuda_visible_devices" not in lowered
    assert "local_device_count() == 4" not in lowered


def test_templates_are_documented_not_core_paths(repo_root):
    templates = [
        repo_root / "configs/data/local_pc_template.yaml",
        repo_root / "configs/data/student_server_template.yaml",
    ]
    assert all(Path(path).exists() for path in templates)
