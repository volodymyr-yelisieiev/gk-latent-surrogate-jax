"""Validation reports for latent cache artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gk_surrogate.data.split import split_trajectory_ids
from gk_surrogate.utils.paths import ensure_dir


def validate_latent_cache(
    cache_path: str | Path,
    *,
    context_length: int = 8,
    prediction_length: int = 1,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Return a finite/statistical validation report for a latent cache."""

    path = Path(cache_path)
    trajectory_timesteps: dict[str, int] = {}
    sequence_windows: dict[str, int] = {}
    spectra_shapes: dict[str, list[int]] = {}
    has_flux = True
    finite_latents = True
    latent_sum: np.ndarray | None = None
    latent_sumsq: np.ndarray | None = None
    latent_count = 0

    with h5py.File(path, "r") as handle:
        latent_dim = int(handle["metadata"].attrs["latent_dim"])
        encoder_checkpoint_path = str(handle["metadata"].attrs.get("encoder_checkpoint_path", ""))
        config_yaml_present = bool(handle["metadata"].attrs.get("config_yaml", ""))
        trajectories = handle["trajectories"]
        for _group_name, group in trajectories.items():
            trajectory_id = str(group.attrs.get("trajectory_id", _group_name))
            z = np.asarray(group["z"], dtype=np.float64)
            trajectory_timesteps[trajectory_id] = int(z.shape[0])
            sequence_windows[trajectory_id] = max(0, int(z.shape[0]) - context_length - prediction_length + 1)
            finite_latents = finite_latents and bool(np.isfinite(z).all())
            latent_sum = np.sum(z, axis=0) if latent_sum is None else latent_sum + np.sum(z, axis=0)
            latent_sumsq = np.sum(z * z, axis=0) if latent_sumsq is None else latent_sumsq + np.sum(z * z, axis=0)
            latent_count += int(z.shape[0])
            has_flux = has_flux and "flux" in group and group["flux"].shape[0] == z.shape[0]
            if "spectra" in group:
                for name, dataset in group["spectra"].items():
                    spectra_shapes.setdefault(name, list(dataset.shape[1:]))

    if latent_count == 0 or latent_sum is None or latent_sumsq is None:
        msg = f"latent cache contains no latents: {path}"
        raise ValueError(msg)

    ids = tuple(trajectory_timesteps)
    splits = split_trajectory_ids(ids, seed=split_seed) if len(ids) >= 2 else {"train": ids, "val": (), "test": ()}
    split_windows = {
        split_name: int(sum(sequence_windows[trajectory_id] for trajectory_id in split_ids))
        for split_name, split_ids in splits.items()
    }
    mean = latent_sum / latent_count
    variance = np.maximum(latent_sumsq / latent_count - mean * mean, 0.0)
    std = np.sqrt(variance)
    timesteps = list(trajectory_timesteps.values())
    return {
        "cache_path": str(path),
        "latent_dim": latent_dim,
        "encoder_checkpoint_path": encoder_checkpoint_path,
        "config_yaml_present": config_yaml_present,
        "num_trajectories": len(trajectory_timesteps),
        "timesteps_per_trajectory": trajectory_timesteps,
        "min_timesteps": int(min(timesteps)),
        "max_timesteps": int(max(timesteps)),
        "total_latents": latent_count,
        "finite_latents": finite_latents,
        "latent_mean_mean": float(np.mean(mean)),
        "latent_mean_std": float(np.std(mean)),
        "latent_std_mean": float(np.mean(std)),
        "latent_std_min": float(np.min(std)),
        "latent_std_max": float(np.max(std)),
        "flux_available": has_flux,
        "spectra_keys": sorted(spectra_shapes),
        "spectra_shapes": spectra_shapes,
        "split_seed": split_seed,
        "split_counts": {name: len(split_ids) for name, split_ids in splits.items()},
        "sequence_context_length": context_length,
        "sequence_prediction_length": prediction_length,
        "sequence_windows_total": int(sum(sequence_windows.values())),
        "sequence_windows_by_split": split_windows,
    }


def write_latent_cache_report(report: dict[str, Any], out: str | Path) -> Path:
    path = Path(out)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return path
