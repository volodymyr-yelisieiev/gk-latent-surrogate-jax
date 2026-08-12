"""Build thesis result figures from retained JSON evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs" / "medium_seed52_reproduction"
FIGURES = ROOT / "thesis" / "figures"
MODEL = RESULTS / "test_transformer_cache_normalized_locked" / "metrics.json"
DECODED_PERSISTENCE = RESULTS / "test_persistence_locked" / "metrics.json"


def load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def horizon_plot(
    model: dict[str, object],
    decoded_persistence: dict[str, object],
    *,
    key: str,
    ylabel: str,
    output: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    horizon = np.arange(1, len(model[key]) + 1)
    for metrics, label, color, marker, linestyle in (
        (decoded_persistence, "Decoded latent persistence", "#5F5F5F", "s", "--"),
        (model, "Cache-normalized Transformer", "#0072B2", "o", "-"),
    ):
        values = np.asarray(metrics[key], dtype=float)
        std = np.asarray(metrics[f"{key.removesuffix('_by_step')}_std_by_step"], dtype=float)
        ax.plot(
            horizon,
            values,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            color=color,
            label=label,
        )
        ax.fill_between(horizon, np.maximum(0.0, values - std), values + std, color=color, alpha=0.13)
    ax.set(xlabel="Rollout horizon", ylabel=ylabel, xticks=horizon)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / output, dpi=220)
    plt.close(fig)


def spectra_horizon_plot(model: dict[str, object], decoded_persistence: dict[str, object]) -> None:
    """Plot each spectrum separately so target scale does not hide one series."""

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for ax, target, title in zip(axes, ("kyspec", "fluxspec"), ("kyspec", "fluxspec"), strict=True):
        key = f"spectra_{target}_mse_by_step"
        horizon = np.arange(1, len(model[key]) + 1)
        for metrics, label, color, marker, linestyle in (
            (decoded_persistence, "Decoded latent persistence", "#5F5F5F", "s", "--"),
            (model, "Cache-normalized Transformer", "#0072B2", "o", "-"),
        ):
            values = np.asarray(metrics[key], dtype=float)
            ax.plot(
                horizon,
                values,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                color=color,
                label=label,
            )
        ax.set(title=title, xlabel="Rollout horizon", ylabel="MSE", xticks=horizon)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURES / "verified_spectra_mse_by_step.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    model = load(MODEL)
    decoded_persistence = load(DECODED_PERSISTENCE)
    horizon_plot(
        model,
        decoded_persistence,
        key="mse_by_step",
        ylabel="Latent MSE",
        output="verified_latent_mse_by_step.png",
    )
    horizon_plot(
        model,
        decoded_persistence,
        key="flux_mse_by_step",
        ylabel="Flux MSE",
        output="verified_flux_mse_by_step.png",
    )
    spectra_horizon_plot(model, decoded_persistence)


if __name__ == "__main__":
    main()
