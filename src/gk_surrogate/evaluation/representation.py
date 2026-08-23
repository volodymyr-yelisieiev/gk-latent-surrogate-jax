"""Latent representation projection plots."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.manifold import TSNE

from gk_surrogate.data.latent_cache import LatentCacheDataset
from gk_surrogate.data.split import TrajectorySplits, split_trajectory_ids
from gk_surrogate.utils.paths import ensure_dir


def evaluate_representation(
    cache: LatentCacheDataset,
    output_dir: str | Path,
    *,
    split_seed: int = 42,
    trajectory_ids: tuple[str, ...] | None = None,
    trajectory_splits: TrajectorySplits | None = None,
    perplexities: tuple[float, ...] = (5.0, 30.0),
    tsne_max_iter: int = 1000,
    max_points: int | None = 2000,
) -> dict[str, Any]:
    z, flux, point_trajectory_ids, timestep_indices, split_labels = _collect_latent_points(
        cache,
        split_seed=split_seed,
        trajectory_ids=trajectory_ids,
        trajectory_splits=trajectory_splits,
    )
    z, flux, point_trajectory_ids, timestep_indices, split_labels = _subsample_points(
        z,
        flux,
        point_trajectory_ids,
        timestep_indices,
        split_labels,
        max_points=max_points,
        seed=split_seed,
    )
    valid_perplexities = tuple(dict.fromkeys(float(value) for value in perplexities if 0.0 < float(value) < z.shape[0]))
    if len(valid_perplexities) < 2:
        raise ValueError("representation plots require at least two distinct t-SNE perplexities below the sample count")
    held_out_points = int(np.sum(np.isin(split_labels, ("val", "test"))))
    if held_out_points == 0:
        raise ValueError("representation plots require non-empty validation or test split points")

    out = ensure_dir(output_dir)
    plot_dir = ensure_dir(out / "plots")
    flux_color = flux[:, 0]
    pca = pca_project(z)
    projections: dict[str, np.ndarray] = {"pca": pca}
    plot_paths = {
        "pca": str(
            _save_projection_plot(
                pca,
                flux_color,
                split_labels,
                plot_dir / "pca_flux.png",
                title="PCA Projection",
            )
        ),
    }
    tsne_paths = []
    for perplexity in valid_perplexities:
        projection = tsne_project(z, perplexity=perplexity, max_iter=tsne_max_iter, seed=split_seed)
        key = _projection_key("tsne", perplexity)
        projections[key] = projection
        path = _save_projection_plot(
            projection,
            flux_color,
            split_labels,
            plot_dir / f"{key}_flux.png",
            title=f"t-SNE Projection (perplexity {perplexity:g})",
        )
        plot_paths[key] = str(path)
        tsne_paths.append(str(path))

    points_path = out / "representation_points.npz"
    np.savez_compressed(
        points_path,
        z=z,
        flux=flux,
        trajectory_id=point_trajectory_ids.astype(str),
        timestep_index=timestep_indices,
        split=split_labels.astype(str),
        **projections,
    )
    csv_path = _write_points_csv(
        out / "representation_points.csv", projections["pca"], flux_color, point_trajectory_ids, split_labels
    )
    split_counts = {split: int(np.sum(split_labels == split)) for split in sorted(set(split_labels.tolist()))}
    selected_ids = tuple(cache.trajectory_ids()) if trajectory_ids is None else tuple(trajectory_ids)
    return {
        "num_points": int(z.shape[0]),
        "latent_dim": int(z.shape[1]),
        "flux_dim": int(flux.shape[1]),
        "flux_color_component": 0,
        "configured_trajectories": list(selected_ids),
        "num_configured_trajectories": len(selected_ids),
        "split_counts": split_counts,
        "split_source": trajectory_splits.strategy if trajectory_splits else "seeded",
        "split_manifest_path": trajectory_splits.manifest_path if trajectory_splits else None,
        "split_manifest_sha256": trajectory_splits.manifest_sha256 if trajectory_splits else None,
        "split_fold_id": trajectory_splits.fold_id if trajectory_splits else None,
        "held_out_points": held_out_points,
        "perplexities": list(valid_perplexities),
        "pca_plot": plot_paths["pca"],
        "tsne_plots": tsne_paths,
        "plot_paths": plot_paths,
        "points_npz": str(points_path),
        "points_csv": str(csv_path),
    }


def pca_project(z: np.ndarray) -> np.ndarray:
    values = _as_latent_matrix(z)
    centered = values - np.mean(values, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] < 2:
        components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
    return (centered @ components[:, :2]).astype(np.float32)


def tsne_project(z: np.ndarray, *, perplexity: float, max_iter: int, seed: int) -> np.ndarray:
    values = _as_latent_matrix(z)
    if not 0.0 < perplexity < values.shape[0]:
        raise ValueError("t-SNE perplexity must be positive and below the sample count")
    projection = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit_transform(values)
    return np.asarray(projection, dtype=np.float32)


def _collect_latent_points(
    cache: LatentCacheDataset,
    *,
    split_seed: int,
    trajectory_ids: tuple[str, ...] | None = None,
    trajectory_splits: TrajectorySplits | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = tuple(cache.trajectory_ids()) if trajectory_ids is None else tuple(trajectory_ids)
    duplicates = tuple(dict.fromkeys(trajectory_id for trajectory_id in ids if ids.count(trajectory_id) > 1))
    if duplicates:
        joined = ", ".join(duplicates)
        msg = f"configured trajectories must be distinct for held-out validation: {joined}"
        raise ValueError(msg)
    available = set(cache.trajectory_ids())
    missing = tuple(trajectory_id for trajectory_id in ids if trajectory_id not in available)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        msg = f"requested trajectories are missing from latent cache: {preview}{suffix}"
        raise ValueError(msg)
    if len(ids) < 2:
        raise ValueError("representation plots require at least two cached trajectories")
    split_by_id = _split_labels(ids, split_seed=split_seed, trajectory_splits=trajectory_splits)
    z_rows = []
    flux_rows = []
    trajectory_rows = []
    timestep_rows = []
    split_rows = []
    missing_flux = []
    for trajectory_id in ids:
        z = cache.get_trajectory_latents(trajectory_id)
        flux = cache.get_trajectory_flux(trajectory_id)
        if flux is None:
            missing_flux.append(trajectory_id)
            continue
        flux = _as_flux_matrix(flux)
        if z.shape[0] != flux.shape[0]:
            msg = f"trajectory {trajectory_id!r} has {z.shape[0]} latents but {flux.shape[0]} flux rows"
            raise ValueError(msg)
        z_rows.append(z)
        flux_rows.append(flux)
        trajectory_rows.extend([trajectory_id] * z.shape[0])
        timestep_rows.extend(range(z.shape[0]))
        split_rows.extend([split_by_id.get(trajectory_id, "unknown")] * z.shape[0])
    if missing_flux:
        joined = ", ".join(missing_flux)
        raise ValueError(f"flux targets are missing for trajectories: {joined}")
    return (
        np.concatenate(z_rows, axis=0).astype(np.float32),
        np.concatenate(flux_rows, axis=0).astype(np.float32),
        np.asarray(trajectory_rows, dtype=str),
        np.asarray(timestep_rows, dtype=np.int32),
        np.asarray(split_rows, dtype=str),
    )


def _subsample_points(
    z: np.ndarray,
    flux: np.ndarray,
    trajectory_ids: np.ndarray,
    timestep_indices: np.ndarray,
    split_labels: np.ndarray,
    *,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_points is None or z.shape[0] <= max_points:
        return z, flux, trajectory_ids, timestep_indices, split_labels
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(z.shape[0], size=max_points, replace=False))
    return z[selected], flux[selected], trajectory_ids[selected], timestep_indices[selected], split_labels[selected]


def _split_labels(
    trajectory_ids: tuple[str, ...],
    *,
    split_seed: int,
    trajectory_splits: TrajectorySplits | None,
) -> dict[str, str]:
    if trajectory_splits is None:
        seeded = split_trajectory_ids(trajectory_ids, seed=split_seed)
        return {trajectory_id: name for name, split_ids in seeded.items() for trajectory_id in split_ids}

    split_by_id: dict[str, str] = {}
    for role, split_ids in trajectory_splits.as_dict().items():
        for trajectory_id in split_ids:
            if trajectory_id in split_by_id:
                raise ValueError(f"trajectory split assignments overlap for {trajectory_id!r}")
            split_by_id[trajectory_id] = role
    missing = tuple(trajectory_id for trajectory_id in trajectory_ids if trajectory_id not in split_by_id)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"trajectory split assignments are missing configured trajectories: {preview}{suffix}")
    configured = set(trajectory_ids)
    extra = tuple(trajectory_id for trajectory_id in split_by_id if trajectory_id not in configured)
    if extra:
        preview = ", ".join(extra[:5])
        suffix = "" if len(extra) <= 5 else f", ... ({len(extra)} total)"
        raise ValueError(f"trajectory split assignments contain extra trajectories: {preview}{suffix}")
    return split_by_id


def _save_projection_plot(
    projection: np.ndarray,
    flux: np.ndarray,
    split_labels: np.ndarray,
    path: Path,
    *,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    points = None
    markers = {"train": "o", "val": "^", "test": "s"}
    labels = {"train": "train", "val": "validation", "test": "test"}
    for role in ("train", "val", "test"):
        selected = split_labels == role
        if not np.any(selected):
            continue
        points = ax.scatter(
            projection[selected, 0],
            projection[selected, 1],
            c=flux[selected],
            cmap="viridis",
            vmin=float(np.min(flux)),
            vmax=float(np.max(flux)),
            marker=markers[role],
            s=18,
            alpha=0.78,
            linewidths=0.25,
            edgecolors="white",
            label=labels[role],
        )
    if points is None:
        raise ValueError("representation plot has no split-labelled points")
    ax.set_title(title)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.colorbar(points, ax=ax, label="flux")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_points_csv(
    path: Path,
    pca: np.ndarray,
    flux: np.ndarray,
    trajectory_ids: np.ndarray,
    split_labels: np.ndarray,
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("trajectory_id", "split", "flux", "pca_1", "pca_2"))
        writer.writeheader()
        for idx in range(pca.shape[0]):
            writer.writerow(
                {
                    "trajectory_id": trajectory_ids[idx],
                    "split": split_labels[idx],
                    "flux": float(flux[idx]),
                    "pca_1": float(pca[idx, 0]),
                    "pca_2": float(pca[idx, 1]),
                }
            )
    return path


def _projection_key(prefix: str, value: float) -> str:
    return f"{prefix}_perplexity_{value:g}".replace(".", "p")


def _as_latent_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        msg = f"latents must have shape [N, Z], got {array.shape}"
        raise ValueError(msg)
    if array.shape[0] < 3:
        raise ValueError("representation plots require at least three points")
    return array


def _as_flux_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        msg = f"flux must have shape [N, F], got {array.shape}"
        raise ValueError(msg)
    return array
