# GK Latent Surrogate JAX

[![CI](https://github.com/volodymyr-yelisieiev/gk-latent-surrogate-jax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/volodymyr-yelisieiev/gk-latent-surrogate-jax/actions/workflows/ci.yml)
![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![JAX/Flax](https://img.shields.io/badge/JAX%2FFlax-GPU--first-2e7d32)
[![License: MIT](https://img.shields.io/badge/license-MIT-4c566a)](LICENSE)

## Abstract

This repository implements a GPU-first JAX/Flax pipeline for latent time-series
surrogates of channel-first gyrokinetic plasma snapshots. It covers synthetic and
Cyclone/KvikIO data access, encoder training, latent caching, sequence modeling, frozen
diagnostic heads, and short-horizon rollout evaluation.

`PRD.md` defines project intent. Committed code, configurations, tests, and operational
documentation define current behavior. Generated data, caches, checkpoints, plots, and
run state remain outside version control.

## Method scope

- snapshot contract: `[B, C, S1, S2, S3, S4, S5]`;
- encoders: MLP, convolutional, and patch-based JAX/Flax variants;
- sequence models: persistence, GRU, MLP-delta, and causal latent transformers;
- diagnostics: latent error, flux error, spectra error, stability, and representation
  projections;
- execution: single-host GPU path with a portable CPU fallback.

Full 5D reconstruction, multi-node training, mandatory real-data access, and a PyTorch
runtime dependency are outside the implemented scope.

## Verified result

The current medium comparison uses one latent cache, the same held-out test split, an
8-step horizon, 40 rollout windows, and equal weighting across five trajectories.

| method | flux RMSE |
| --- | ---: |
| Persistence | 23.6794 |
| GRU | 24.3575 |
| Transformer | 9.9499 |
| GRU, cache-normalized | 23.2162 |
| Transformer, cache-normalized | **9.2270** |

Each model row is produced by its own checkpoint or baseline under the same evaluation
protocol; results from different protocols are not combined. Detailed provenance and
limitations are recorded in `docs/medium_guppy_experiment_report.md` and
`docs/final_claims.md`.

## Installation and verification

Python 3.12 is preferred; Python 3.11 and newer are supported.

```bash
make install-dev
make check
make smoke-all
uv build
```

The synthetic smoke pipeline runs with `JAX_PLATFORM_NAME=cpu` and requires neither real
data nor GPU hardware. Its artifacts are written under ignored `outputs/smoke_*` paths.

## Command-line interface

```bash
gks inspect-data --config configs/data/tiny_dummy.yaml
gks train-encoder --config configs/experiment/smoke_encoder_supervised.yaml
gks embed-dataset --config configs/experiment/smoke_embed_dataset.yaml
gks evaluate-flux-head --config configs/experiment/smoke_evaluate_flux_head.yaml
gks plot-representation --config configs/experiment/smoke_plot_representation.yaml
gks train-sequence --config configs/experiment/smoke_sequence.yaml
gks evaluate-rollout --config configs/experiment/smoke_evaluate_rollout.yaml
```

All commands support resolved configuration validation through `--dry-run`, targeted
configuration overrides, deterministic seeds, and explicit output directories.

## Reproducibility

- `configs/` contains portable smoke and server-oriented experiment definitions.
- `docs/data_contract.md` and `docs/metrics.md` define data and metric semantics.
- `docs/thesis_result_set.md` separates engineering, validation, and medium-scale evidence.
- `CONTRIBUTING.md` defines provenance, commit, pull-request, and merge conventions.

Raw data, generated HDF5/NPZ files, latent caches, checkpoints, W&B state, package
artifacts, and thesis outputs must not be committed.

## Citation and license

Software citation metadata is provided in `CITATION.cff`. The implementation is released
under the [MIT License](LICENSE).
