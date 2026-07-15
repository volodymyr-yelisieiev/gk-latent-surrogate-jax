"""Device discovery and mode resolution for optional data parallelism."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

from gk_surrogate.config.schema import ParallelConfig
from gk_surrogate.utils.paths import ensure_dir


@dataclass(frozen=True)
class ParallelPlan:
    mode: str
    axis_name: str
    num_devices: int
    local_device_count: int
    global_batch_size: int
    per_device_batch_size: int
    drop_remainder: bool
    auto_scale_learning_rate: bool
    devices: tuple[str, ...]

    @property
    def uses_pmap(self) -> bool:
        return self.mode == "pmap"


def get_local_devices() -> tuple[Any, ...]:
    return tuple(jax.local_devices())


def get_device_count() -> int:
    return len(get_local_devices())


def resolve_parallel_mode(config: ParallelConfig, *, batch_size: int) -> ParallelPlan:
    """Resolve config intent to a concrete single-host execution plan."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    local_devices = get_local_devices()
    local_count = len(local_devices)
    if local_count < config.min_devices:
        if config.mode == "pmap":
            msg = f"parallel.mode=pmap requires at least {config.min_devices} usable devices"
            raise ValueError(msg)
        num_devices = 1
        mode = "single"
    elif config.mode == "single":
        num_devices = 1
        mode = "single"
    else:
        candidate_count = local_count
        if config.drop_remainder and batch_size < candidate_count:
            candidate_count = batch_size
        if config.drop_remainder and batch_size % candidate_count != 0:
            if config.require_all_visible_devices:
                msg = (
                    f"data.batch_size={batch_size} is not divisible by {candidate_count} visible devices; "
                    "adjust batch size or disable parallel.require_all_visible_devices"
                )
                raise ValueError(msg)
            candidate_count = _largest_divisor_at_most(batch_size, candidate_count)
        if candidate_count < config.min_devices:
            if config.mode == "pmap":
                msg = f"parallel.mode=pmap requires at least {config.min_devices} usable devices"
                raise ValueError(msg)
            num_devices = 1
            mode = "single"
        else:
            num_devices = candidate_count
            mode = "pmap" if config.mode == "pmap" or num_devices > 1 else "single"
    selected_devices = local_devices[:num_devices]
    per_device = batch_size if mode == "single" else _ceil_div(batch_size, num_devices)
    return ParallelPlan(
        mode=mode,
        axis_name=config.axis_name,
        num_devices=num_devices,
        local_device_count=local_count,
        global_batch_size=batch_size,
        per_device_batch_size=per_device,
        drop_remainder=config.drop_remainder,
        auto_scale_learning_rate=config.auto_scale_learning_rate,
        devices=tuple(str(device) for device in selected_devices),
    )


def write_device_report(
    output_dir: str | Path,
    *,
    config: ParallelConfig,
    plan: ParallelPlan,
) -> Path:
    """Write a small JSON report for reproducibility."""

    path = ensure_dir(output_dir) / "device_report.json"
    payload = {
        "backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in jax.local_devices()],
        "parallel_config": config.model_dump(mode="json"),
        "parallel_plan": asdict(plan),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)
