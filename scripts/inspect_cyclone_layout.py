#!/usr/bin/env python
"""Inspect Cyclone/KvikIO binary layout without loading full samples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from gk_surrogate.utils.paths import ensure_dir


def _load_inspector():
    try:
        from gk_surrogate.data.cyclone_layout import inspect_cyclone_layout

        return inspect_cyclone_layout
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().parents[1] / "src" / "gk_surrogate" / "data" / "cyclone_layout.py"
        spec = importlib.util.spec_from_file_location("gk_surrogate_cyclone_layout_standalone", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.inspect_cyclone_layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("GK_CYCLONE_DATA_ROOT"))
    parser.add_argument("--max-trajectories", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("--root or GK_CYCLONE_DATA_ROOT is required")
    inspect_cyclone_layout = _load_inspector()
    report = inspect_cyclone_layout(args.root, max_trajectories=args.max_trajectories).as_dict()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        ensure_dir(output.parent)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
