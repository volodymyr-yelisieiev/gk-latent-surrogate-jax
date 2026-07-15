"""Frozen-latent flux head evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gk_surrogate.data.latent_cache import LatentCacheDataset


@dataclass(frozen=True)
class LinearFluxHead:
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float32) @ self.weights + self.bias


def fit_ridge_flux_head(z: np.ndarray, flux: np.ndarray, *, alpha: float = 1e-3) -> LinearFluxHead:
    """Fit a linear ridge head from frozen latents to flux targets."""

    features = _as_feature_matrix(z, "z")
    targets = _as_target_matrix(flux)
    if features.shape[0] != targets.shape[0]:
        msg = f"z and flux sample counts differ: {features.shape[0]} != {targets.shape[0]}"
        raise ValueError(msg)
    if features.shape[0] == 0:
        raise ValueError("cannot fit a flux head with zero samples")

    design = np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float32)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float32) * float(alpha)
    penalty[-1, -1] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ targets
    try:
        params = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.pinv(lhs) @ rhs
    return LinearFluxHead(weights=params[:-1].astype(np.float32), bias=params[-1].astype(np.float32))


def evaluate_flux_head(
    cache: LatentCacheDataset,
    *,
    train_ids: tuple[str, ...],
    eval_ids: tuple[str, ...],
    eval_split: str,
    alpha: float = 1e-3,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Fit on train trajectories and report flux RMSE on the requested split."""

    z_train, flux_train = _stack_cache_flux(cache, train_ids)
    z_eval, flux_eval = _stack_cache_flux(cache, eval_ids)
    head = fit_ridge_flux_head(z_train, flux_train, alpha=alpha)
    pred = head.predict(z_eval)
    error = pred - flux_eval
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    relative_error = float(np.mean(np.abs(error) / (np.abs(flux_eval) + eps)))
    return {
        "primary_metric": f"{eval_split}_flux_rmse",
        "eval_split": eval_split,
        "flux_head": "ridge_linear",
        "ridge_alpha": float(alpha),
        "train_trajectories": list(train_ids),
        "eval_trajectories": list(eval_ids),
        "num_train_samples": int(z_train.shape[0]),
        "num_eval_samples": int(z_eval.shape[0]),
        "latent_dim": int(z_train.shape[1]),
        "flux_dim": int(flux_train.shape[1]),
        "flux_mse": mse,
        "flux_rmse": rmse,
        "flux_mae": mae,
        "flux_relative_error": relative_error,
        "flux_rmse_by_dim": np.sqrt(np.mean(np.square(error), axis=0)).astype(np.float32),
        "flux_pred": pred.astype(np.float32),
        "flux_target": flux_eval.astype(np.float32),
    }


def _stack_cache_flux(cache: LatentCacheDataset, trajectory_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not trajectory_ids:
        raise ValueError("at least one trajectory is required")
    features = []
    targets = []
    missing = []
    for trajectory_id in trajectory_ids:
        z = cache.get_trajectory_latents(trajectory_id)
        flux = cache.get_trajectory_flux(trajectory_id)
        if flux is None:
            missing.append(trajectory_id)
            continue
        flux = _as_target_matrix(flux)
        if z.shape[0] != flux.shape[0]:
            msg = f"trajectory {trajectory_id!r} has {z.shape[0]} latents but {flux.shape[0]} flux rows"
            raise ValueError(msg)
        features.append(z)
        targets.append(flux)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"flux targets are missing for trajectories: {joined}")
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _as_feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        msg = f"{name} must have shape [N, D], got {array.shape}"
        raise ValueError(msg)
    return array


def _as_target_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        msg = f"flux must have shape [N, F], got {array.shape}"
        raise ValueError(msg)
    return array
