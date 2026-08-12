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

## Result status

No learned-model superiority claim is currently accepted. The retained seed-52 comparison is
retrospective and uses a frozen diagnostic head; decoded latent persistence and the Transformer
are not substitutes for direct persistence of the observed flux. The audit gives flux RMSE
`3.7208` for observed-flux persistence, `11.4255` for the selected Transformer, `11.9889` for
decoded latent persistence, and `12.0712` for the diagnostic-head oracle on true future latents.
The oracle result shows that the current head, rather than temporal forecasting alone, limits the
diagnostic comparison. The multi-seed protocol is planned but has not been run; see
`docs/result_status.md` and `experiment_protocols/multiseed_v1.json`.

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
gks train-direct-diagnostics --config configs/experiment/smoke_encoder_supervised.yaml
gks embed-dataset --config configs/experiment/smoke_embed_dataset.yaml
gks evaluate-flux-head --config configs/experiment/smoke_evaluate_flux_head.yaml
gks plot-representation --config configs/experiment/smoke_plot_representation.yaml
gks train-sequence --config configs/experiment/smoke_sequence.yaml
gks evaluate-rollout --config configs/experiment/smoke_evaluate_rollout.yaml
gks-protocol --protocol experiment_protocols/multiseed_v1.json
```

All commands support resolved configuration validation through `--dry-run`, targeted
configuration overrides, deterministic seeds, and explicit output directories.

## Reproducibility

- `configs/` contains portable smoke and server-oriented experiment definitions.
- `docs/data_contract.md` and `docs/metrics.md` define data and metric semantics.
- `docs/result_status.md` separates engineering smoke output from thesis-facing evidence.
- `CONTRIBUTING.md` defines provenance, commit, pull-request, and merge conventions.

Raw data, generated HDF5/NPZ files, latent caches, checkpoints, W&B state, package
artifacts, and thesis outputs must not be committed.

## Citation and license

Software citation metadata is provided in `CITATION.cff`. The implementation is released
under the [MIT License](LICENSE).
