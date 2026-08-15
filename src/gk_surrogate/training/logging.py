"""Metrics and run metadata logging."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import numbers
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gk_surrogate.utils.paths import ensure_dir
from gk_surrogate.utils.pretty import scalarize


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    return out


def append_csv(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    row = {key: scalarize(value) for key, value in payload.items()}
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out


class MetricsLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        wandb_config: Mapping[str, Any] | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self._wandb_run: Any | None = None
        self._wandb_module: Any | None = None
        self._wandb_log_artifacts = False
        self._wandb_run_name: str | None = None
        self._wandb_status = self._init_wandb(wandb_config or {}, run_config=run_config)
        # Keep a durable, sanitized record even when W&B is deliberately
        # disabled. This prevents a missing file from being mistaken for a
        # missing provenance decision in multi-stage experiments.
        write_json(self.output_dir / "wandb_status.json", self._wandb_status)

    def log(self, metrics: Mapping[str, Any], *, prefix: str | None = None) -> None:
        flat = _flatten_mapping(metrics)
        scalar_metrics = {
            key: scalar for key, value in flat.items() if (scalar := _finite_metric_scalar(value)) is not None
        }
        if scalar_metrics:
            append_jsonl(self.jsonl_path, scalar_metrics)
            append_csv(self.csv_path, scalar_metrics)
        if self._wandb_run is not None:
            step = scalar_metrics.get("step")
            payload = self._wandb_payload(scalar_metrics, prefix=prefix)
            if payload:
                self._wandb_run.log(
                    payload,
                    step=int(step) if isinstance(step, int | float) else None,
                )

    def write_summary(self, metrics: Mapping[str, Any]) -> Path:
        summary = {key: _json_ready(value) for key, value in _flatten_mapping(metrics).items()}
        summary_path = write_json(self.output_dir / "metrics.json", summary)
        self.finish(metrics, artifact_paths=(summary_path, self.jsonl_path, self.csv_path))
        return summary_path

    def finish(
        self,
        summary: Mapping[str, Any] | None = None,
        *,
        artifact_paths: Iterable[str | Path | None] = (),
    ) -> None:
        if self._wandb_run is None:
            return
        if summary:
            self._wandb_run.summary.update(self._wandb_payload(_flatten_mapping(summary), prefix=None))
        if self._wandb_log_artifacts:
            candidates = tuple(
                Path(path) for path in artifact_paths if path is not None and Path(path).is_file()
            )
            self._log_wandb_previews(candidates)
            self._log_wandb_artifact(candidates)
        self._wandb_run.finish()
        self._wandb_run = None

    def wandb_status(self) -> dict[str, Any]:
        return dict(self._wandb_status)

    def _wandb_payload(self, metrics: Mapping[str, Any], *, prefix: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in metrics.items():
            scalar = _finite_metric_scalar(value)
            if scalar is None:
                continue
            name = key if prefix is None or key == "step" or "/" in key else f"{prefix}/{key}"
            payload[name] = scalar
        return payload

    def _log_wandb_previews(self, paths: Iterable[Path]) -> None:
        if self._wandb_module is None or self._wandb_run is None:
            return
        for path in paths:
            if path.suffix.lower() == ".csv" and hasattr(self._wandb_module, "Table"):
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.reader(handle))
                if rows:
                    table = self._wandb_module.Table(
                        columns=rows[0], data=[[_parse_csv_cell(cell) for cell in row] for row in rows[1:]]
                    )
                    self._wandb_run.log({f"tables/{path.stem}": table})
            elif path.suffix.lower() in {".png", ".jpg", ".jpeg"} and hasattr(self._wandb_module, "Image"):
                self._wandb_run.log({f"plots/{path.stem}": self._wandb_module.Image(str(path))})

    def _log_wandb_artifact(self, paths: Iterable[Path]) -> None:
        candidates = tuple(paths)
        if not candidates or self._wandb_run is None:
            return
        if self._wandb_module is not None and hasattr(self._wandb_module, "Artifact") and hasattr(
            self._wandb_run, "log_artifact"
        ):
            base_name = self._wandb_run_name or "gk-surrogate-run"
            artifact_name = f"{_wandb_safe_name(base_name)}-outputs"
            artifact = self._wandb_module.Artifact(name=artifact_name, type="run-output")
            for path in candidates:
                artifact.add_file(str(path), name=path.name)
            self._wandb_run.log_artifact(artifact)
            return
        if hasattr(self._wandb_run, "save"):
            for path in candidates:
                self._wandb_run.save(str(path))

    def _init_wandb(
        self,
        wandb_config: Mapping[str, Any],
        *,
        run_config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        enabled = bool(wandb_config.get("enabled", False))
        mode = str(wandb_config.get("mode", "disabled"))
        requested = enabled and mode != "disabled"
        if not requested:
            return {"enabled": False, "requested": False, "mode": mode}

        self._wandb_log_artifacts = bool(wandb_config.get("log_artifacts", False))
        directory = str(wandb_config.get("directory") or self.output_dir)
        try:
            wandb = importlib.import_module("wandb")
        except Exception as exc:
            status = {
                "enabled": False,
                "requested": True,
                "available": False,
                "mode": mode,
                "warning": f"wandb import failed: {exc}",
            }
            write_json(self.output_dir / "wandb_status.json", status)
            return status

        run_name = wandb_config.get("name")
        if run_name is None and isinstance(run_config, Mapping):
            run_name = run_config.get("name")
        self._wandb_run_name = str(run_name) if run_name else None
        job_type = str(wandb_config.get("job_type") or _infer_job_type(self._wandb_run_name))
        try:
            run = wandb.init(
                project=str(wandb_config.get("project", "gk-latent-surrogate")),
                entity=wandb_config.get("entity"),
                name=run_name,
                group=wandb_config.get("group"),
                tags=list(wandb_config.get("tags", ())),
                job_type=job_type,
                mode=mode,
                dir=directory,
                config=run_config,
                reinit=True,
            )
        except Exception as exc:
            status = {
                "enabled": False,
                "requested": True,
                "available": True,
                "mode": mode,
                "warning": f"wandb init failed: {exc}",
            }
            write_json(self.output_dir / "wandb_status.json", status)
            return status

        self._wandb_run = run
        self._wandb_module = wandb
        status = {
            "enabled": True,
            "requested": True,
            "available": True,
            "mode": mode,
            "project": str(wandb_config.get("project", "gk-latent-surrogate")),
            "run_name": run_name,
            "group": wandb_config.get("group"),
            "job_type": job_type,
            "tags": list(wandb_config.get("tags", ())),
            "run_url": getattr(run, "url", None),
            "run_dir": getattr(run, "dir", directory),
        }
        write_json(self.output_dir / "wandb_status.json", status)
        return status


def _finite_metric_scalar(value: Any) -> bool | int | float | None:
    """Return a dashboard-safe scalar; paths, labels, and arrays are artifacts."""

    if isinstance(value, str | bytes | list | tuple | Mapping):
        return None
    if hasattr(value, "shape") and tuple(value.shape) != ():
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        scalar = scalarize(value)
    except Exception:
        return None
    if scalar is value or isinstance(scalar, str | bytes | list | tuple | dict):
        return None
    return _finite_metric_scalar(scalar)


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten metric namespaces without collapsing arrays into misleading means."""

    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, name))
        else:
            flat[name] = value
    return flat


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        return bool(value) if isinstance(value, bool) else int(value) if isinstance(value, numbers.Integral) else number
    if isinstance(value, str) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    return str(value)


def _wandb_safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return normalized or "gk-surrogate-run"


def _parse_csv_cell(value: str) -> str | int | float | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _infer_job_type(run_name: str | None) -> str:
    name = (run_name or "").lower().replace("_", "-")
    if "evaluate-rollout" in name or "rollout" in name:
        return "rollout-evaluation"
    if "flux-head" in name:
        return "diagnostic-evaluation"
    if "representation" in name or "plot" in name:
        return "representation-evaluation"
    if "embed" in name:
        return "dataset-embedding"
    if "sequence" in name:
        return "sequence-training"
    if "encoder" in name:
        return "encoder-training"
    return "experiment"


def collect_git_info(cwd: str | Path = ".") -> dict[str, Any]:
    def _run_bytes(args: list[str]) -> bytes | None:
        try:
            result = subprocess.run(args, cwd=cwd, check=True, capture_output=True)
        except Exception:
            return None
        return result.stdout

    def _run(args: list[str]) -> str | None:
        output = _run_bytes(args)
        return output.decode("utf-8", errors="surrogateescape").strip() if output is not None else None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    tracked_diff = _run_bytes(["git", "diff", "--binary", "HEAD", "--"])
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"])
    untracked_paths = untracked.splitlines() if untracked else []
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "git_available": commit is not None,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest() if tracked_diff is not None else None,
        "has_untracked_paths": bool(untracked_paths) if untracked is not None else None,
        "untracked_path_count": len(untracked_paths) if untracked is not None else None,
        "untracked_paths": untracked_paths if untracked is not None else None,
    }


def write_run_metadata(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    argv: list[str] | None = None,
    cwd: str | Path = ".",
) -> None:
    out = ensure_dir(output_dir)
    if config is not None:
        write_json(out / "config_resolved.json", config)
    write_json(out / "git_info.json", {"argv": argv or [], **collect_git_info(cwd)})
