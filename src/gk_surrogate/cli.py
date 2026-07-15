"""Command-line entrypoint for the GK latent surrogate pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.pretty import Pretty

from gk_surrogate.config.load import config_to_yaml, load_config
from gk_surrogate.data.cyclone_kvikio import MissingCycloneDependencyError
from gk_surrogate.data.h5_loader import write_synthetic_h5
from gk_surrogate.data.inspect import inspect_dataset
from gk_surrogate.pipeline import (
    benchmark_step_time,
    embed_dataset,
    evaluate_flux_head,
    evaluate_rollout,
    plot_representation,
    train_encoder,
    train_sequence,
)
from gk_surrogate.training.logging import write_json
from gk_surrogate.utils.paths import ensure_dir

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "inspect-data",
        "train-encoder",
        "embed-dataset",
        "train-sequence",
        "evaluate-flux-head",
        "plot-representation",
        "evaluate-rollout",
        "make-synthetic-h5",
        "benchmark-step-time",
    ):
        subparser = subparsers.add_parser(command)
        _add_common_args(subparser)
        subparser.set_defaults(func=_dispatch)
    subparsers.choices["inspect-data"].add_argument("--max-trajectories", type=int, default=2)
    subparsers.choices["inspect-data"].add_argument("--max-depth", type=int, default=3)
    subparsers.choices["inspect-data"].add_argument("--max-target-samples", type=int, default=256)
    subparsers.choices["benchmark-step-time"].add_argument("--measured-steps", type=int, default=3)
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Dotted key override, key=value.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and inspect without training writes.")
    parser.add_argument("--seed", type=int, default=None, help="Override data and training seed.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")


def _dispatch(args: argparse.Namespace) -> int:
    overrides = list(args.override)
    if args.seed is not None:
        overrides.extend([f"data.seed={args.seed}", f"training.seed={args.seed}"])
    if args.output_dir is not None:
        overrides.append(f"output_dir={args.output_dir}")
        if args.command == "embed-dataset":
            overrides.append(f"latent_cache.path={Path(args.output_dir) / 'latent_cache.h5'}")
    config = load_config(args.config, overrides=overrides, command=args.command)
    if args.dry_run:
        console.print("Resolved config:")
        console.print(config_to_yaml(config), markup=False, highlight=False)

    if args.command == "inspect-data":
        try:
            inspection = inspect_dataset(
                config.data,
                max_trajectories=args.max_trajectories,
                max_depth=args.max_depth,
                max_target_samples=args.max_target_samples,
                log_spectra=config.loss.use_log_spectra,
            )
        except (MissingCycloneDependencyError, FileNotFoundError, PermissionError) as exc:
            if args.dry_run and config.data.backend == "cyclone_kvikio":
                console.print(Pretty(_cyclone_inspect_dry_run(config.data.model_dump(mode="json"), exc)))
                return 0
            raise
        payload = inspection.as_dict()
        if args.output_dir is not None and not args.dry_run:
            output_dir = ensure_dir(args.output_dir)
            report = write_json(output_dir / "data_inspection.json", payload)
            (output_dir / "config_resolved.yaml").write_text(config_to_yaml(config), encoding="utf-8")
            payload["inspection_json"] = str(report)
        console.print(Pretty(payload))
        return 0

    if args.command == "make-synthetic-h5":
        if config.data.synthetic is None:
            msg = "make-synthetic-h5 requires data.synthetic"
            raise ValueError(msg)
        output_dir = Path(config.output_dir if args.output_dir is None else args.output_dir)
        if args.dry_run:
            console.print(
                {
                    "dry_run": True,
                    "planned_output_dir": str(output_dir),
                    "num_trajectories": config.data.synthetic.num_trajectories,
                    "timesteps": config.data.synthetic.timesteps,
                    "snapshot_shape": [
                        config.data.synthetic.channels,
                        *config.data.synthetic.spatial_shape,
                    ],
                }
            )
            return 0
        written = write_synthetic_h5(
            output_dir,
            config.data.synthetic,
            seed=config.data.seed,
            schema=config.data.h5_schema,
        )
        console.print({"written": [str(path) for path in written]})
        return 0

    if args.command == "train-encoder":
        console.print(train_encoder(config, dry_run=args.dry_run))
        return 0
    if args.command == "embed-dataset":
        console.print(embed_dataset(config, dry_run=args.dry_run))
        return 0
    if args.command == "train-sequence":
        console.print(train_sequence(config, dry_run=args.dry_run))
        return 0
    if args.command == "evaluate-flux-head":
        console.print(evaluate_flux_head(config, dry_run=args.dry_run))
        return 0
    if args.command == "plot-representation":
        console.print(plot_representation(config, dry_run=args.dry_run))
        return 0
    if args.command == "evaluate-rollout":
        console.print(evaluate_rollout(config, dry_run=args.dry_run))
        return 0
    if args.command == "benchmark-step-time":
        console.print(benchmark_step_time(config, dry_run=args.dry_run, measured_steps=args.measured_steps))
        return 0
    return 0


def _cyclone_inspect_dry_run(data_config: dict[str, object], exc: Exception) -> dict[str, object]:
    return {
        "dry_run": True,
        "backend": "cyclone_kvikio",
        "validated": True,
        "root": data_config.get("root"),
        "target_flux": data_config.get("target_flux"),
        "target_spectra": data_config.get("target_spectra"),
        "warning": str(exc),
    }


if __name__ == "__main__":
    raise SystemExit(main())
