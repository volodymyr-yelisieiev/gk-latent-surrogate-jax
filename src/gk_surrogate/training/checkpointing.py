"""Simple pickle-based checkpointing for v0."""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import jax

from gk_surrogate.training.state import TrainState
from gk_surrogate.utils.paths import ensure_dir
from gk_surrogate.utils.pretty import flatten_dict


def _step_dir(root: str | Path, step: int) -> Path:
    return ensure_dir(Path(root) / "checkpoints" / f"step_{step:06d}")


def checkpoint_payload(state: TrainState) -> dict[str, Any]:
    return {
        "step": int(state.step),
        "params": jax.device_get(state.params),
        "opt_state": jax.device_get(state.opt_state),
        "rng": jax.device_get(state.rng),
        "model_config": dict(state.model_config),
        "batch_stats": jax.device_get(state.batch_stats) if state.batch_stats is not None else None,
    }


def save_checkpoint(state: TrainState, root: str | Path, *, step: int | None = None) -> Path:
    step_value = int(state.step if step is None else step)
    if step_value != int(state.step):
        raise ValueError(f"checkpoint step {step_value} does not match train state step {int(state.step)}")
    path = _step_dir(root, step_value)
    payload = checkpoint_payload(state)
    checkpoint_path = path / "checkpoint.pkl"
    checkpoint_tmp = path / f".checkpoint.{os.getpid()}.tmp"
    with checkpoint_tmp.open("wb") as f:
        pickle.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(checkpoint_tmp, checkpoint_path)
    metadata_path = path / "metadata.json"
    metadata_tmp = path / f".metadata.{os.getpid()}.tmp"
    with metadata_tmp.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "step": step_value,
                "model_config": flatten_dict(payload["model_config"]),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(metadata_tmp, metadata_path)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoint.pkl"
    with ckpt_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {ckpt_path}")
    missing = {"step", "params"} - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required fields {sorted(missing)}: {ckpt_path}")
    return payload


def restore_train_state(state: TrainState, path: str | Path) -> TrainState:
    payload = load_checkpoint(path)
    restored = state.replace(
        step=payload["step"],
        params=payload["params"],
        opt_state=payload["opt_state"],
        rng=payload["rng"],
        model_config=payload.get("model_config", state.model_config),
        batch_stats=payload.get("batch_stats", state.batch_stats),
    )
    return restored


def latest_checkpoint(root: str | Path) -> Path | None:
    ckpt_root = Path(root) / "checkpoints"
    if not ckpt_root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in ckpt_root.iterdir():
        match = re.fullmatch(r"step_(\d+)", path.name)
        if path.is_dir() and match and (path / "checkpoint.pkl").is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
