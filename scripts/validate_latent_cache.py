#!/usr/bin/env python
"""Validate latent cache schema, finite values, splits, and sequence windows."""

from __future__ import annotations

import argparse

from gk_surrogate.data.latent_cache_report import validate_latent_cache, write_latent_cache_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    report = validate_latent_cache(
        args.cache,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        split_seed=args.split_seed,
    )
    write_latent_cache_report(report, args.out)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
