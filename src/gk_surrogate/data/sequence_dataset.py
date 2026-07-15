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
    if context_length < 1 or prediction_length < 1:
        msg = "context_length and prediction_length must be positive"
        raise ValueError(msg)
    total = dataset.num_timesteps(trajectory_id)
    window = context_length + prediction_length
    if total < window:
        return ()
    return tuple(range(total - window + 1))
