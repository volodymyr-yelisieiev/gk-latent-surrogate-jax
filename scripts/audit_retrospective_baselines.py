"""Recompute diagnostic baselines from a retained latent cache.

This is an audit utility for retrospective artifacts. It does not read raw snapshots, train a
model, or publish a result. The observed-flux reference copies the last observed diagnostic;
the oracle applies the frozen diagnostic head to true future latents and is explicitly not a
forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from gk_surrogate.config.load import load_config
from gk_surrogate.factory import build_diagnostic_heads
from gk_surrogate.training.checkpointing import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trajectory_rmse(values: list[np.ndarray]) -> float:
    if not values:
        raise ValueError("no rollout windows were available")
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64))))


def audit(
    *,
    cache_path: Path,
    metrics_path: Path,
    encoder_checkpoint: Path,
    config_path: Path,
    context_length: int,
    rollout_steps: int,
) -> dict[str, object]:
    if context_length < 1 or rollout_steps < 1:
        raise ValueError("context_length and rollout_steps must be positive")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trajectory_ids = metrics.get("selected_trajectory_ids")
    if (
        not isinstance(trajectory_ids, list)
        or not trajectory_ids
        or not all(isinstance(item, str) for item in trajectory_ids)
    ):
        raise ValueError("metrics must contain non-empty selected_trajectory_ids")
    config = load_config(config_path, overrides=["data.root=/tmp/gk-audit-root"])
    diagnostic_model = build_diagnostic_heads(config.model.diagnostics)
    if diagnostic_model is None:
        raise ValueError("the supplied config must define diagnostic heads")
    diagnostic_params = load_checkpoint(encoder_checkpoint)["params"].get("diagnostic_heads")
    if diagnostic_params is None:
        raise ValueError("encoder checkpoint does not contain diagnostic-head parameters")

    observed_by_trajectory: dict[str, float] = {}
    oracle_by_trajectory: dict[str, float] = {}
    windows_by_trajectory: dict[str, int] = {}
    with h5py.File(cache_path, "r") as handle:
        groups = handle["trajectories"]
        for trajectory_id in trajectory_ids:
            group_name = quote(trajectory_id, safe="")
            if group_name not in groups:
                raise KeyError(f"trajectory is absent from cache: {trajectory_id}")
            group = groups[group_name]
            flux = np.asarray(group["flux"], dtype=np.float32)
            latents = np.asarray(group["z"], dtype=np.float32)
            required = context_length + rollout_steps
            if flux.shape[0] != latents.shape[0] or flux.shape[0] < required:
                raise ValueError(f"trajectory has insufficient aligned cache rows: {trajectory_id}")
            observed_errors: list[np.ndarray] = []
            oracle_errors: list[np.ndarray] = []
            for start in range(flux.shape[0] - required + 1):
                target_flux = flux[start + context_length : start + required]
                observed_flux = np.repeat(
                    flux[start + context_length - 1 : start + context_length],
                    rollout_steps,
                    axis=0,
                )
                oracle_flux = np.asarray(
                    diagnostic_model.apply(
                        {"params": diagnostic_params},
                        jnp.asarray(latents[start + context_length : start + required]),
                        train=False,
                    ).flux,
                    dtype=np.float32,
                )
                error_axes = tuple(range(1, target_flux.ndim))
                observed_errors.append(np.mean(np.square(observed_flux - target_flux), axis=error_axes))
                oracle_errors.append(np.mean(np.square(oracle_flux - target_flux), axis=error_axes))
            observed_by_trajectory[trajectory_id] = _trajectory_rmse(observed_errors)
            oracle_by_trajectory[trajectory_id] = _trajectory_rmse(oracle_errors)
            windows_by_trajectory[trajectory_id] = len(observed_errors)

    return {
        "audit": "retrospective_diagnostic_baselines_v1",
        "cache_sha256": _sha256(cache_path),
        "metrics_sha256": _sha256(metrics_path),
        "encoder_checkpoint_sha256": _sha256(
            encoder_checkpoint / "checkpoint.pkl" if encoder_checkpoint.is_dir() else encoder_checkpoint
        ),
        "context_length": context_length,
        "rollout_steps": rollout_steps,
        "trajectory_ids": trajectory_ids,
        "windows_by_trajectory": windows_by_trajectory,
        "observed_flux_persistence_rmse_by_trajectory": observed_by_trajectory,
        "diagnostic_head_oracle_rmse_by_trajectory": oracle_by_trajectory,
        "observed_flux_persistence_flux_rmse": _trajectory_rmse(
            [np.asarray(value) ** 2 for value in observed_by_trajectory.values()]
        ),
        "diagnostic_head_oracle_flux_rmse": _trajectory_rmse(
            [np.asarray(value) ** 2 for value in oracle_by_trajectory.values()]
        ),
        "oracle_is_forecast": False,
        "device": str(jax.default_backend()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/server_evaluate_rollout_medium.yaml"))
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        cache_path=args.cache,
        metrics_path=args.metrics,
        encoder_checkpoint=args.encoder_checkpoint,
        config_path=args.config,
        context_length=args.context_length,
        rollout_steps=args.rollout_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "observed_flux_rmse": payload["observed_flux_persistence_flux_rmse"],
                "oracle_flux_rmse": payload["diagnostic_head_oracle_flux_rmse"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
