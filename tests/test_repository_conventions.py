from __future__ import annotations

import subprocess
import sys

import pytest

from gk_surrogate.utils.pr_title import title_errors


@pytest.mark.parametrize(
    "title",
    (
        "feat(model): add causal latent predictor",
        "fix(eval): preserve trajectory weighting",
        "docs: clarify result provenance",
        "refactor!: remove legacy checkpoint schema",
    ),
)
def test_valid_pull_request_titles(title: str) -> None:
    assert title_errors(title) == ()


@pytest.mark.parametrize(
    "title",
    (
        "[fix] preserve trajectory weighting",
        "agent: polish repository",
        "docs: document phase 7 results",
        "docs: update report 2026-07-15",
        "docs: end with punctuation.",
        "feat(scope): " + "x" * 80,
    ),
)
def test_invalid_pull_request_titles(title: str) -> None:
    assert title_errors(title)


def _run_title_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/validate_pr_title.py", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_pull_request_title_cli_contract() -> None:
    missing = _run_title_validator()
    assert missing.returncode == 2
    assert "usage:" in missing.stderr

    valid = _run_title_validator("fix: reject mixed protocols")
    assert valid.returncode == 0
    assert "valid" in valid.stdout

    invalid = _run_title_validator("WIP mixed protocols")
    assert invalid.returncode == 1
    assert "must follow" in invalid.stderr
