"""Filesystem reconnaissance helpers for Cyclone/KvikIO binary datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CycloneTrajectoryLayout:
    trajectory_id: str
    path: str
    metadata_pkl: bool
    metadata_light_pkl: bool
    timestep_bin_count: int
    timestep_bf16_bin_count: int
    potential_bin_count: int
    first_timestep_bin: str | None
    first_timestep_bf16_bin: str | None
    first_potential_bin: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "path": self.path,
            "metadata_pkl": self.metadata_pkl,
            "metadata_light_pkl": self.metadata_light_pkl,
            "timestep_bin_count": self.timestep_bin_count,
            "timestep_bf16_bin_count": self.timestep_bf16_bin_count,
            "potential_bin_count": self.potential_bin_count,
            "first_timestep_bin": self.first_timestep_bin,
            "first_timestep_bf16_bin": self.first_timestep_bf16_bin,
            "first_potential_bin": self.first_potential_bin,
        }


@dataclass(frozen=True)
class CycloneLayoutReport:
    root: str
    trajectory_count: int
    sample_count_estimate: int
    quantized_shards_available: bool
    metadata_pkl_count: int
    metadata_light_pkl_count: int
    trajectories: tuple[CycloneTrajectoryLayout, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "trajectory_count": self.trajectory_count,
            "sample_count_estimate": self.sample_count_estimate,
            "quantized_shards_available": self.quantized_shards_available,
            "metadata_pkl_count": self.metadata_pkl_count,
            "metadata_light_pkl_count": self.metadata_light_pkl_count,
            "trajectories": [item.as_dict() for item in self.trajectories],
            "warnings": self.warnings,
        }


def inspect_cyclone_layout(root: str | Path, *, max_trajectories: int = 8) -> CycloneLayoutReport:
    """Inspect the Cyclone binary directory layout without loading samples."""

    root_path = Path(root).expanduser()
    if max_trajectories < 1:
        msg = "max_trajectories must be positive"
        raise ValueError(msg)
    try:
        root_exists = root_path.exists()
    except PermissionError as exc:
        return _blocked_report(root_path, f"permission denied while accessing Cyclone dataset root: {exc}")
    if not root_exists:
        msg = f"Cyclone dataset root does not exist: {root_path}"
        raise FileNotFoundError(msg)

    try:
        candidates = _trajectory_candidates(root_path)
    except PermissionError as exc:
        return _blocked_report(root_path, f"permission denied while inspecting Cyclone dataset root: {exc}")
    selected = candidates[:max_trajectories]
    trajectories = tuple(_trajectory_layout(path, root_path) for path in selected)
    total_samples = sum(item.timestep_bf16_bin_count or item.timestep_bin_count for item in trajectories)
    warnings: list[str] = []
    if not candidates:
        warnings.append("no trajectory directories with timestep binary files were found")
    if len(candidates) > len(selected):
        warnings.append(f"layout report truncated to {len(selected)} of {len(candidates)} trajectories")
    if trajectories and not any(item.metadata_pkl or item.metadata_light_pkl for item in trajectories):
        warnings.append("no metadata.pkl or metadata_light.pkl files found in selected trajectories")
    if trajectories and not any(item.timestep_bf16_bin_count for item in trajectories):
        warnings.append("no quantized BF16 timestep shards found in selected trajectories")

    return CycloneLayoutReport(
        root=str(root_path),
        trajectory_count=len(candidates),
        sample_count_estimate=total_samples,
        quantized_shards_available=any(item.timestep_bf16_bin_count > 0 for item in trajectories),
        metadata_pkl_count=sum(1 for item in trajectories if item.metadata_pkl),
        metadata_light_pkl_count=sum(1 for item in trajectories if item.metadata_light_pkl),
        trajectories=trajectories,
        warnings=tuple(warnings),
    )


def _blocked_report(root: Path, warning: str) -> CycloneLayoutReport:
    return CycloneLayoutReport(
        root=str(root),
        trajectory_count=0,
        sample_count_estimate=0,
        quantized_shards_available=False,
        metadata_pkl_count=0,
        metadata_light_pkl_count=0,
        trajectories=(),
        warnings=(warning,),
    )


def _trajectory_candidates(root: Path) -> tuple[Path, ...]:
    if _has_timestep_files(root):
        return (root,)
    direct = [path for path in root.iterdir() if path.is_dir() and _has_timestep_files(path)]
    if direct:
        return tuple(sorted(direct, key=_manifest_order_key))
    nested = [path for path in root.glob("*/*") if path.is_dir() and _has_timestep_files(path)]
    return tuple(sorted(nested, key=_manifest_order_key))


def _manifest_order_key(path: Path) -> tuple[str, str]:
    name = path.name
    for suffix in ("_ifft_realpotens", "_ifft"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name, path.name


def _has_timestep_files(path: Path) -> bool:
    data_dir = path / "data"
    return (
        _has_plain_timestep_files(path)
        or any(path.glob("timestep_*.bf16.bin"))
        or (data_dir.exists() and (_has_plain_timestep_files(data_dir) or any(data_dir.glob("timestep_*.bf16.bin"))))
    )


def _trajectory_layout(path: Path, root: Path) -> CycloneTrajectoryLayout:
    timestep_bins = _sorted_relative(path, "data/timestep_*.bin", "timestep_*.bin", exclude_bf16=True)
    timestep_bf16_bins = _sorted_relative(path, "data/timestep_*.bf16.bin", "timestep_*.bf16.bin")
    potential_bins = _sorted_relative(path, "data/poten_*.bin", "poten_*.bin", exclude_bf16=True)
    return CycloneTrajectoryLayout(
        trajectory_id=_trajectory_id(path, root),
        path=str(path),
        metadata_pkl=(path / "metadata.pkl").exists(),
        metadata_light_pkl=(path / "metadata_light.pkl").exists(),
        timestep_bin_count=len(timestep_bins),
        timestep_bf16_bin_count=len(timestep_bf16_bins),
        potential_bin_count=len(potential_bins),
        first_timestep_bin=timestep_bins[0] if timestep_bins else None,
        first_timestep_bf16_bin=timestep_bf16_bins[0] if timestep_bf16_bins else None,
        first_potential_bin=potential_bins[0] if potential_bins else None,
    )


def _has_plain_timestep_files(path: Path) -> bool:
    return any(not item.name.endswith(".bf16.bin") for item in path.glob("timestep_*.bin"))


def _sorted_relative(path: Path, *patterns: str, exclude_bf16: bool = False) -> tuple[str, ...]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    if exclude_bf16:
        files = [item for item in files if not item.name.endswith(".bf16.bin")]
    return tuple(str(item.relative_to(path)) for item in sorted(set(files)))


def _trajectory_id(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.name
    return path.name if rel == Path(".") else str(rel)
