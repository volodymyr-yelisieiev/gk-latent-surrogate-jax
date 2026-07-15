"""Patch-grid utilities for 5D snapshot encoders."""

from __future__ import annotations

from collections.abc import Sequence


def as_5d_tuple(value: int | Sequence[int], *, name: str) -> tuple[int, int, int, int, int]:
    """Normalize an integer or sequence into a positive 5D tuple."""

    result = (value, value, value, value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(result) != 5:
        raise ValueError(f"{name} must have length 5, got {result}")
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result  # type: ignore[return-value]


def patch_grid_shape(
    spatial_shape: Sequence[int],
    patch_size: int | Sequence[int],
) -> tuple[int, int, int, int, int]:
    """Return the 5D patch grid shape for a valid spatial shape and patch size."""

    spatial = as_5d_tuple(spatial_shape, name="spatial_shape")
    patch = as_5d_tuple(patch_size, name="patch_size")
    if any(dim % size != 0 for dim, size in zip(spatial, patch, strict=True)):
        raise ValueError(f"spatial_shape={spatial} must be divisible by patch_size={patch}")
    return tuple(dim // size for dim, size in zip(spatial, patch, strict=True))  # type: ignore[return-value]


def patch_token_count(spatial_shape: Sequence[int], patch_size: int | Sequence[int]) -> int:
    """Return number of non-overlapping 5D patch tokens."""

    count = 1
    for dim in patch_grid_shape(spatial_shape, patch_size):
        count *= dim
    return count


def validate_token_count(
    spatial_shape: Sequence[int],
    patch_size: int | Sequence[int],
    *,
    max_token_count: int,
    allow_large_token_count: bool = False,
) -> int:
    """Validate token-count guardrails and return the token count."""

    if max_token_count <= 0:
        raise ValueError(f"max_token_count must be positive, got {max_token_count}")
    count = patch_token_count(spatial_shape, patch_size)
    if count > max_token_count and not allow_large_token_count:
        raise ValueError(
            f"Patch token count {count} exceeds max_token_count={max_token_count}; "
            "increase patch_size or set allow_large_token_count=True."
        )
    return count
