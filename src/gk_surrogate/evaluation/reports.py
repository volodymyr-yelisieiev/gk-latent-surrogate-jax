"""Rollout report writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gk_surrogate.utils.paths import ensure_dir
from gk_surrogate.utils.pretty import scalarize


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    arr = np.asarray(value)
    if arr.shape == ():
        scalar = arr.item()
        if isinstance(scalar, np.generic):
            scalar = scalar.item()
        if isinstance(scalar, bool | int | float | str) or scalar is None:
            return scalar
        return str(scalar)
    return arr.tolist()


def save_metrics_json(metrics: Mapping[str, Any], output_dir: str | Path, filename: str = "metrics.json") -> Path:
    out_dir = ensure_dir(output_dir)
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump({key: _jsonable(value) for key, value in metrics.items()}, f, indent=2, sort_keys=True)
    return path


def save_metrics_by_step(
    metrics: Mapping[str, Any], output_dir: str | Path, filename: str = "metrics_by_step.csv"
) -> Path:
    out_dir = ensure_dir(output_dir)
    by_step = {
        key: np.asarray(value)
        for key, value in metrics.items()
        if key.endswith("_by_step") and np.asarray(value).ndim == 1
    }
    if not by_step:
        raise ValueError("metrics contain no one-dimensional '*_by_step' entries")
    steps = max(values.shape[0] for values in by_step.values())
    path = out_dir / filename
    fieldnames = ["step", *by_step.keys()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(steps):
            # Step 1 is the first forecast after the supplied context.
            row = {"step": step + 1}
            for key, values in by_step.items():
                row[key] = scalarize(values[step]) if step < values.shape[0] else ""
            writer.writerow(row)
    return path


def save_rollout_report(metrics: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    return {
        "metrics_json": save_metrics_json(metrics, output_dir),
        "metrics_by_step_csv": save_metrics_by_step(metrics, output_dir),
    }


def save_rollout_plots(metrics: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write compact horizon plots for every available primary rollout error."""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    out_dir = ensure_dir(output_dir) / "plots"
    ensure_dir(out_dir)
    specifications = {
        "latent_mse": ("mse_by_step", "mse_std_by_step", "Latent rollout MSE", "Latent MSE"),
        "flux_mse": ("flux_mse_by_step", "flux_mse_std_by_step", "Flux diagnostic rollout MSE", "Flux MSE"),
        "spectra_mse": (
            "spectra_mse_by_step",
            "spectra_mse_std_by_step",
            "Spectral diagnostic rollout MSE",
            "Spectra MSE",
        ),
    }
    paths: dict[str, Path] = {}
    for name, (metric_key, std_key, title, ylabel) in specifications.items():
        if metric_key not in metrics:
            continue
        values = np.asarray(metrics[metric_key], dtype=np.float64)
        steps = np.arange(1, values.shape[0] + 1)
        path = out_dir / f"{name}_by_step.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        (line,) = ax.plot(steps, values, marker="o", label="mean")
        if std_key is not None and std_key in metrics:
            std = np.asarray(metrics[std_key], dtype=np.float64)
            ax.fill_between(
                steps,
                np.maximum(values - std, 0.0),
                values + std,
                color=line.get_color(),
                alpha=0.2,
                label="±1 SD",
            )
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Forecast horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(steps)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[name] = path
    return paths


def save_basic_rollout_plot(metrics: Mapping[str, Any], output_dir: str | Path) -> Path | None:
    """Compatibility wrapper returning the latent-MSE plot."""

    return save_rollout_plots(metrics, output_dir).get("latent_mse")
