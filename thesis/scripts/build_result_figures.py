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
BASELINE = RESULTS / "test_persistence_locked" / "metrics.json"


def load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def horizon_plot(
    model: dict[str, object],
    baseline: dict[str, object],
    *,
    key: str,
    ylabel: str,
    output: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    horizon = np.arange(1, len(model[key]) + 1)
    for metrics, label, color in (
        (baseline, "Persistence", "#6B7280"),
        (model, "Normalized Transformer", "#006C93"),
    ):
        values = np.asarray(metrics[key], dtype=float)
        std = np.asarray(metrics[f"{key.removesuffix('_by_step')}_std_by_step"], dtype=float)
        ax.plot(horizon, values, marker="o", linewidth=2, color=color, label=label)
        ax.fill_between(horizon, np.maximum(0.0, values - std), values + std, color=color, alpha=0.13)
    ax.set(xlabel="Rollout horizon", ylabel=ylabel, xticks=horizon)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / output, dpi=220)
    plt.close(fig)


def paired_flux_plot(model: dict[str, object], baseline: dict[str, object]) -> None:
    model_values = np.asarray(model["flux_rmse_by_trajectory"], dtype=float)
    baseline_values = np.asarray(baseline["flux_rmse_by_trajectory"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for index, (before, after) in enumerate(zip(baseline_values, model_values, strict=True), start=1):
        color = "#248A3D" if after < before else "#B45309"
        ax.plot([0, 1], [before, after], marker="o", color=color, alpha=0.9)
        ax.annotate(str(index), (1.03, after), va="center", fontsize=8)
    ax.set(
        xlim=(-0.15, 1.22),
        xticks=[0, 1],
        xticklabels=["Persistence", "Normalized\nTransformer"],
        ylabel="Flux RMSE per trajectory",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "paired_test_flux_rmse.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    model = load(MODEL)
    baseline = load(BASELINE)
    horizon_plot(
        model,
        baseline,
        key="mse_by_step",
        ylabel="Latent MSE",
        output="verified_latent_mse_by_step.png",
    )
    horizon_plot(
        model,
        baseline,
        key="flux_mse_by_step",
        ylabel="Flux MSE",
        output="verified_flux_mse_by_step.png",
    )
    horizon_plot(
        model,
        baseline,
        key="spectra_mse_by_step",
        ylabel="Mean spectra MSE",
        output="verified_spectra_mse_by_step.png",
    )
    paired_flux_plot(model, baseline)


if __name__ == "__main__":
    main()
