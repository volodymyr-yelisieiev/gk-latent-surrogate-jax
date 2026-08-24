"""Content-addressed manifests for the direct Cyclone/KvikIO dataset view."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from gk_surrogate.config.schema import CycloneKvikIOConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _trajectory_id(path: Path) -> str:
    name = path.name
    for suffix in ("_ifft_realpotens", "_ifft"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _metadata_path(path: Path) -> Path:
    light = path / "metadata_light.pkl"
    return light if light.is_file() else path / "metadata.pkl"


def _sample_indices(metadata: Mapping[str, Any], config: CycloneKvikIOConfig) -> tuple[int, ...]:
    timesteps = np.asarray(metadata.get("timesteps", ()))
    if timesteps.ndim == 0 or timesteps.shape[0] == 0:
        return ()
    stop = timesteps.shape[0] - config.tail_offset if config.tail_offset else timesteps.shape[0]
    available = np.arange(config.offset, max(config.offset, stop), dtype=np.int64)[:: config.subsample]
    sample_count = max(0, available.shape[0] - config.bundle_seq_length * 2 + 1)
    return tuple(int(index) for index in available[:sample_count])


def _input_paths(path: Path, indices: Sequence[int], fields: Sequence[str]) -> tuple[Path, ...]:
    files: list[Path] = []
    for index in indices:
        if "df" in fields:
            files.append(path / "data" / f"timestep_{index:05d}.bin")
        if any(field in {"phi", "poten", "potential"} for field in fields):
            files.append(path / "data" / f"poten_{index:05d}.bin")
    missing = [item.name for item in files if not item.is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Cyclone manifest inputs are missing under {path.name}: {preview}")
    return tuple(files)


def _trajectory_record(path: Path, config: CycloneKvikIOConfig) -> dict[str, Any]:
    metadata_path = _metadata_path(path)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Cyclone metadata must be a mapping: {metadata_path}")
    indices = _sample_indices(metadata, config)
    inputs = _input_paths(path, indices, config.fields_to_load)
    file_records = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in inputs
    ]
    return {
        "trajectory_id": _trajectory_id(path),
        "metadata_file": metadata_path.name,
        "metadata_sha256": _sha256(metadata_path),
        "input_file_count": len(file_records),
        "input_bytes": sum(item.stat().st_size for item in inputs),
        "input_content_sha256": hashlib.sha256(_canonical_bytes(file_records)).hexdigest(),
    }


def build_cyclone_universe_manifest(
    root: str | Path,
    config: CycloneKvikIOConfig,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Hash every input byte consumed by one configured direct-dataset view."""

    root_path = Path(root).expanduser().resolve()
    trajectories = tuple(
        sorted(
            path
            for path in root_path.iterdir()
            if path.is_dir() and (path / "metadata.pkl").is_file() and (path / "data").is_dir()
        )
    )
    if not trajectories:
        raise FileNotFoundError(f"no Cyclone trajectories found under {root_path}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(lambda path: _trajectory_record(path, config), trajectories))
    records.sort(key=lambda item: str(item["trajectory_id"]))
    revision = hashlib.sha256(_canonical_bytes(records)).hexdigest()
    return {
        "schema_version": "2.0.0",
        "dataset_revision": f"cyclone-consumed-bytes-sha256:{revision}",
        "hash_algorithm": "sha256",
        "sampling_contract": {
            "fields_to_load": list(config.fields_to_load),
            "bundle_seq_length": config.bundle_seq_length,
            "offset": config.offset,
            "tail_offset": config.tail_offset,
            "subsample": config.subsample,
            "spatial_ifft": config.spatial_ifft,
            "real_potens": config.real_potens,
            "prefer_dtype": config.prefer_dtype,
        },
        "trajectory_ids": [str(item["trajectory_id"]) for item in records],
        "trajectories": records,
    }


def verify_cyclone_universe_manifest(
    expected: Mapping[str, Any],
    root: str | Path,
    config: CycloneKvikIOConfig,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Recompute the configured dataset view and fail on any byte-level difference."""

    actual = build_cyclone_universe_manifest(root, config, workers=workers)
    if actual != dict(expected):
        raise ValueError("Cyclone dataset bytes or sampling contract differ from the frozen universe manifest")
    return actual
