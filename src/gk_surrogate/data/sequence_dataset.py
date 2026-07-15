"""Sequence window helpers for latent trajectory datasets."""

from __future__ import annotations

from gk_surrogate.data.latent_cache import LatentCacheDataset


def valid_sequence_starts(
    dataset: LatentCacheDataset,
    trajectory_id: str,
    *,
    context_length: int,
    prediction_length: int,
) -> tuple[int, ...]:
    total = dataset.num_timesteps(trajectory_id)
    window = context_length + prediction_length
    if total < window:
        return ()
    return tuple(range(total - window + 1))
