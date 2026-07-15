"""Filesystem path helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def timestamped_run_dir(root: str | Path, name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return ensure_dir(Path(root) / f"{timestamp()}_{safe_name}")
