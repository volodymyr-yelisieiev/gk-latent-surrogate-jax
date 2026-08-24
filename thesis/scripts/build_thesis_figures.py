"""Build deterministic methodology and qualitative latent-space thesis figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "server_encoder_simsiam_medium.yaml"
DEFAULT_FIGURES = ROOT / "thesis" / "figures"
DEFAULT_RELEASE_MANIFEST = ROOT / "experiment_protocols" / "multiseed_v1_results.json"
DEFAULT_FOLD_MANIFEST = ROOT / "experiment_protocols" / "multiseed_v1_folds.outer_fold_0.json"

INK = "#172033"
MUTED = "#4B5565"
BLUE = "#DCEAFB"
GREEN = "#DDF3E7"
ORANGE = "#FCE8D8"
PURPLE = "#E9E2F6"


def pointwise_shape_trace(
    input_shape: Sequence[int],
    channels: Sequence[int],
    strides: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Return channel-first shapes for pointwise slicing/projection stages."""

    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 6 or any(value <= 0 for value in shape):
        raise ValueError("input_shape must be [C, S1, S2, S3, S4, S5] with positive values")
    widths = tuple(int(value) for value in channels)
    steps = tuple(tuple(int(value) for value in stride) for stride in strides)
    if not widths or len(widths) != len(steps):
        raise ValueError("channels and strides must contain the same non-zero number of stages")
    if any(width <= 0 for width in widths):
        raise ValueError("channels must be positive")
    trace = [shape]
    spatial = shape[1:]
    for width, stride in zip(widths, steps, strict=True):
        if len(stride) != 5 or any(value <= 0 for value in stride):
            raise ValueError("each stride must contain five positive values")
        spatial = tuple(math.ceil(size / step) for size, step in zip(spatial, stride, strict=True))
        trace.append((width, *spatial))
    return tuple(trace)


def load_methodology_contract(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    encoder = payload["model"]["encoder"]
    extra = encoder["extra"]
    trace = pointwise_shape_trace(
        (4, 32, 8, 16, 85, 32),
        extra["channels"],
        extra["strides"],
    )
    return {
        "trace": trace,
        "latent_dim": int(encoder["latent_dim"]),
        "kernel_size": tuple(int(value) for value in extra["kernel_size"]),
        "context_length": int(payload["data"]["context_length"]),
        "simsiam_weight": float(payload["loss"]["simsiam_weight"]),
        "flux_weight": float(payload["loss"]["flux_weight"]),
        "spectra_weight": float(payload["loss"]["spectra_weight"]),
    }


def build_methodology_figure(config_path: Path, output_path: Path) -> Path:
    contract = load_methodology_contract(config_path)
    trace = contract["trace"]
    fig = plt.figure(figsize=(13.6, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(1.05, 1.0, 1.0))
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _panel_title(axes[0], "A", "Five-dimensional snapshot encoder")
    labels = [
        _shape_label(trace[0], "input"),
        _shape_label(trace[1], "stage 1"),
        _shape_label(trace[2], "stage 2"),
        _shape_label(trace[3], "stage 3"),
        "global mean\npool [B,32]",
        f"latent state\n[B,{contract['latent_dim']}]",
    ]
    colors = [BLUE, BLUE, BLUE, BLUE, GREEN, GREEN]
    _horizontal_flow(axes[0], labels, colors, y=0.42)
    axes[0].text(
        0.02,
        0.08,
        r"Pointwise $1^5$ channel mixing; spatial slicing strides: "
        r"$(2,2,2,4,2)$, $(2,1,2,4,2)$, $(2,1,1,2,2)$.",
        color=MUTED,
        fontsize=9.5,
    )

    _panel_title(axes[1], "B", "Task-aware SimSiam representation learning")
    _box(axes[1], 0.03, 0.51, 0.16, 0.24, "view a\naugmentation", BLUE)
    _box(axes[1], 0.03, 0.14, 0.16, 0.24, "view b\naugmentation", BLUE)
    _box(axes[1], 0.27, 0.33, 0.16, 0.24, "shared 5D\nencoder", GREEN)
    _box(axes[1], 0.51, 0.53, 0.17, 0.22, "projection +\npredictor", PURPLE)
    _box(axes[1], 0.51, 0.14, 0.17, 0.22, "diagnostic\nheads", ORANGE)
    _box(axes[1], 0.76, 0.53, 0.20, 0.22, "symmetric cosine\n+ stop-gradient", PURPLE)
    _box(axes[1], 0.76, 0.14, 0.20, 0.22, "flux + two spectra\nMSE", ORANGE)
    for start, end in (
        ((0.19, 0.63), (0.27, 0.48)),
        ((0.19, 0.26), (0.27, 0.42)),
        ((0.43, 0.48), (0.51, 0.64)),
        ((0.43, 0.42), (0.51, 0.25)),
        ((0.68, 0.64), (0.76, 0.64)),
        ((0.68, 0.25), (0.76, 0.25)),
    ):
        _arrow(axes[1], start, end)
    axes[1].text(
        0.50,
        0.04,
        r"$\mathcal{L}_{enc}=\mathcal{L}_{flux}+\mathcal{L}_{spectra}+0.5\,\mathcal{L}_{Siam}$",
        ha="center",
        color=INK,
        fontsize=11,
    )

    _panel_title(axes[2], "C", "Latent sequence modeling and diagnostic evaluation")
    labels = [
        f"true context\n{contract['context_length']} × z[128]",
        "GRU or causal\nTransformer",
        "next latent\nprediction",
        "recursive\n8-step rollout",
        "frozen diagnostic\nheads",
        "flux + spectra\nmetrics",
    ]
    _horizontal_flow(axes[2], labels, [BLUE, PURPLE, GREEN, GREEN, ORANGE, ORANGE], y=0.43)
    axes[2].text(
        0.02,
        0.08,
        "Observed-diagnostic persistence is evaluated directly; latent persistence and the oracle "
        "pass through the same frozen heads.",
        color=MUTED,
        fontsize=9.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def build_latent_space_figure(points_path: Path, metrics_path: Path, output_path: Path) -> Path:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_representation_metadata(metrics)
    with np.load(points_path) as points:
        required = {"flux", "split", "pca", "tsne_perplexity_5", "tsne_perplexity_30"}
        missing = required.difference(points.files)
        if missing:
            raise ValueError(f"representation points are missing arrays: {', '.join(sorted(missing))}")
        flux = np.asarray(points["flux"], dtype=np.float32)[:, 0]
        split = np.asarray(points["split"]).astype(str)
        projections = (
            ("PCA", np.asarray(points["pca"], dtype=np.float32)),
            ("t-SNE, perplexity 5", np.asarray(points["tsne_perplexity_5"], dtype=np.float32)),
            ("t-SNE, perplexity 30", np.asarray(points["tsne_perplexity_30"], dtype=np.float32)),
        )
    if flux.shape[0] != int(metrics["num_points"]):
        raise ValueError("representation points count does not match metrics metadata")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), constrained_layout=True)
    markers = {"train": "o", "val": "^", "test": "s"}
    names = {"train": "train", "val": "validation", "test": "test"}
    mappable = None
    for ax, (title, projection) in zip(axes, projections, strict=True):
        for role in ("train", "val", "test"):
            selected = split == role
            if not np.any(selected):
                continue
            mappable = ax.scatter(
                projection[selected, 0],
                projection[selected, 1],
                c=flux[selected],
                cmap="viridis",
                vmin=float(np.min(flux)),
                vmax=float(np.max(flux)),
                marker=markers[role],
                s=16,
                alpha=0.72,
                linewidths=0.22,
                edgecolors="white",
                label=names[role],
            )
        ax.set_title(title, color=INK, weight="bold")
        ax.set_xlabel("dimension 1")
        ax.set_ylabel("dimension 2")
        ax.grid(alpha=0.2)
    if mappable is None:
        raise ValueError("representation points contain no recognized split labels")
    axes[0].legend(frameon=False, loc="best")
    fig.colorbar(mappable, ax=axes, label="flux (preprocessed target units)", shrink=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def write_latent_provenance(points_path: Path, metrics_path: Path, output_path: Path) -> Path:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_representation_metadata(metrics)
    release = json.loads(DEFAULT_RELEASE_MANIFEST.read_text(encoding="utf-8"))
    payload = {
        "status": "qualitative_protocol_run",
        "protocol_id": "multiseed-v1",
        "source_tag": release["source_tag"],
        "source_commit": release["source_commit"],
        "outer_fold": 0,
        "training_seed": 52,
        "num_points": int(metrics["num_points"]),
        "latent_dim": int(metrics["latent_dim"]),
        "split_manifest_sha256": metrics["split_manifest_sha256"],
        "latent_cache_sha256": metrics["latent_cache_sha256"],
        "encoder_checkpoint_sha256": metrics["encoder_checkpoint_sha256"],
        "embed_config_resolved_sha256": metrics["embed_config_resolved_sha256"],
        "encoder_config_resolved_sha256": metrics["encoder_config_resolved_sha256"],
        "points_sha256": _sha256(points_path),
        "metrics_sha256": _sha256(metrics_path),
        "claim_boundary": "single protocol-bound run; not an aggregate performance estimate",
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _validate_representation_metadata(metrics: Mapping[str, Any]) -> None:
    release = json.loads(DEFAULT_RELEASE_MANIFEST.read_text(encoding="utf-8"))
    if release.get("status") != "accepted" or release.get("source_tag") != "protocol/multiseed-v1":
        raise ValueError("canonical release manifest is not the accepted multiseed-v1 release")
    stage_hashes = release.get("stage_config_resolved_sha256", {})
    expected = {
        "num_points": 1173,
        "latent_dim": 128,
        "split_source": "explicit_manifest",
        "split_fold_id": "outer-0",
        "split_manifest_sha256": _sha256(DEFAULT_FOLD_MANIFEST),
        "protocol_version": 1,
        "data_split_seed": 52,
        "training_seed": 52,
        "embed_config_resolved_sha256": stage_hashes.get("outer_fold_0/seed_52/embed"),
        "encoder_config_resolved_sha256": stage_hashes.get("outer_fold_0/seed_52/encoder"),
    }
    mismatches = {key: (metrics.get(key), value) for key, value in expected.items() if metrics.get(key) != value}
    if tuple(float(value) for value in metrics.get("perplexities", ())) != (5.0, 30.0):
        mismatches["perplexities"] = (metrics.get("perplexities"), [5.0, 30.0])
    for key in ("latent_cache_sha256", "encoder_checkpoint_sha256"):
        digest = metrics.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            mismatches[key] = (digest, "64-character SHA-256")
    if mismatches:
        details = "; ".join(
            f"{key}={actual!r}, expected {expected_value!r}" for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"representation metadata is not the canonical thesis run: {details}")


def _shape_label(shape: Sequence[int], stage: str) -> str:
    return f"{stage}\n[B,{','.join(str(value) for value in shape)}]"


def _panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        0.01,
        0.91,
        letter,
        fontsize=13,
        weight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.28", "fc": INK, "ec": INK},
    )
    ax.text(0.055, 0.91, title, fontsize=13, weight="bold", color=INK, va="center")


def _horizontal_flow(ax: plt.Axes, labels: Sequence[str], colors: Sequence[str], *, y: float) -> None:
    width = 0.135
    gap = (0.96 - len(labels) * width) / (len(labels) - 1)
    xs = [0.02 + index * (width + gap) for index in range(len(labels))]
    for index, (x, label, color) in enumerate(zip(xs, labels, colors, strict=True)):
        _box(ax, x, y, width, 0.28, label, color, fontsize=8.7)
        if index < len(labels) - 1:
            _arrow(ax, (x + width, y + 0.14), (xs[index + 1], y + 0.14))


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
    *,
    fontsize: float = 9.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.25,
        edgecolor=INK,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=INK, fontsize=fontsize)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.35, "color": MUTED})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--representation-points", type=Path)
    parser.add_argument("--representation-metrics", type=Path)
    args = parser.parse_args()
    build_methodology_figure(args.config, args.figures_dir / "methodology_overview.png")
    if (args.representation_points is None) != (args.representation_metrics is None):
        parser.error("--representation-points and --representation-metrics must be supplied together")
    if args.representation_points is not None and args.representation_metrics is not None:
        build_latent_space_figure(
            args.representation_points,
            args.representation_metrics,
            args.figures_dir / "latent_space_projection.png",
        )
        write_latent_provenance(
            args.representation_points,
            args.representation_metrics,
            args.figures_dir / "latent_space_projection.provenance.json",
        )


if __name__ == "__main__":
    main()
