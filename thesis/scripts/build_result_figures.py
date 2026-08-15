"""Build thesis figures from the accepted multi-seed aggregate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = ROOT / "outputs" / "multiseed-v1" / "aggregate_results.json"
RELEASE = ROOT / "experiment_protocols" / "multiseed_v1_results.json"
FIGURES = ROOT / "thesis" / "figures"

COLORS = {
    "learned": "#0072B2",
    "observed": "#D55E00",
    "oracle": "#009E73",
    "latent": "#5F5F5F",
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _fold_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = aggregate.get("outer_fold_summary")
    if not isinstance(rows, list) or len(rows) != 5 or not all(isinstance(row, dict) for row in rows):
        raise ValueError("aggregate must contain five outer-fold summaries")
    return rows


def _mean_and_std(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    if values.ndim != 2 or values.shape[0] != 5:
        raise ValueError(f"{key} must contain one equal-length series per outer fold")
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def horizon_plot(
    rows: list[dict[str, Any]],
    series: tuple[tuple[str, str, str, str], ...],
    *,
    ylabel: str,
    output: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    horizon = np.arange(1, len(rows[0][series[0][1]]) + 1)
    for label, key, color_key, linestyle in series:
        mean, std = _mean_and_std(rows, key)
        color = COLORS[color_key]
        ax.plot(horizon, mean, linewidth=2.2, color=color, linestyle=linestyle, label=label)
        ax.fill_between(
            horizon,
            np.maximum(0.0, mean - std),
            mean + std,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax.set(xlabel="Forecast horizon (step)", ylabel=ylabel, xticks=horizon)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(FIGURES / output, dpi=220)
    plt.close(fig)


def primary_by_fold_plot(rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(rows))
    means = np.asarray([row["mean_difference"] for row in rows], dtype=float)
    lower = np.asarray([row["ci_lower"] for row in rows], dtype=float)
    upper = np.asarray([row["ci_upper"] for row in rows], dtype=float)
    yerr = np.vstack((means - lower, upper - means))

    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        color=COLORS["learned"],
        ecolor=COLORS["learned"],
        capsize=4,
        linewidth=1.5,
        markersize=6,
        label="Selected learned $-$ observed persistence",
    )
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", label="Equal error")
    ax.set(
        xlabel="Outer fold",
        ylabel="Flux RMSE difference (preprocessed units)",
        xticks=x,
        xticklabels=[str(row["outer_fold"]) for row in rows],
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIGURES / "verified_primary_by_fold.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    release = load(RELEASE)
    aggregate = release
    if AGGREGATE.is_file():
        expected_digest = release.get("provenance", {}).get("raw_aggregate_sha256")
        actual_digest = hashlib.sha256(AGGREGATE.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError("raw aggregate does not match the release provenance digest")
        aggregate = load(AGGREGATE)
    if aggregate.get("status") != "accepted":
        raise ValueError("aggregate result is not accepted")
    rows = _fold_rows(aggregate)
    horizon_plot(
        rows,
        (
            ("Selected learned model", "learned_latent_mse_by_step", "learned", "-"),
            ("Latent persistence", "latent_persistence_latent_mse_by_step", "latent", "--"),
        ),
        ylabel="Latent MSE",
        output="verified_latent_mse_by_step.png",
    )
    horizon_plot(
        rows,
        (
            ("Selected learned model", "learned_flux_rmse_by_step", "learned", "-"),
            ("Observed persistence", "observed_flux_rmse_by_step", "observed", "--"),
            ("Diagnostic-head oracle", "diagnostic_head_oracle_flux_rmse_by_step", "oracle", ":"),
        ),
        ylabel="Flux RMSE (preprocessed units)",
        output="verified_flux_rmse_by_step.png",
    )
    primary_by_fold_plot(rows)


if __name__ == "__main__":
    main()
