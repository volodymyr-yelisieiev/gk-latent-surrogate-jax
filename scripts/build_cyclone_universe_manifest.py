"""Build or verify a content-addressed Cyclone dataset-universe manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gk_surrogate.config.load import load_config
from gk_surrogate.data.universe_manifest import (
    build_cyclone_universe_manifest,
    verify_cyclone_universe_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, overrides=[f"data.root={args.root}"], command="embed-dataset")
    if config.data.cyclone is None:
        raise SystemExit("config must use the Cyclone backend")
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        manifest = verify_cyclone_universe_manifest(
            expected,
            args.root,
            config.data.cyclone,
            workers=args.workers,
        )
    else:
        manifest = build_cyclone_universe_manifest(
            args.root,
            config.data.cyclone,
            workers=args.workers,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "dataset_revision": manifest["dataset_revision"],
                "trajectories": len(manifest["trajectory_ids"]),
            }
        )
    )


if __name__ == "__main__":
    main()
