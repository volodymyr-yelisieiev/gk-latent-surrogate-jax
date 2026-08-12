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

The previous medium-comparison table has been withdrawn: encoder/cache and downstream
evaluation used different trajectory-split seeds, and fitted normalization was not
consistently restricted to training trajectories. Those values are not accepted as thesis
evidence.

The corrected comparison uses split seed 52 throughout, fits cache normalization on the
training split, and selects the learned candidate on validation. Its five-trajectory test
evaluation is a retrospective single realization, not pristine locked-test evidence: the
same manifest was inspected in more than one retained evaluation record. The selected
cache-normalized Transformer has a lower trajectory-balanced flux RMSE point estimate
(`11.425478` versus `11.988926`, a `4.70%` reduction). The paired five-trajectory interval
includes zero, so the result does not establish statistical significance, repeatability, or
general superiority. Detailed evidence is recorded in
`docs/medium_guppy_experiment_report.md` and `docs/final_claims.md`.

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
- `docs/thesis_result_set.md` separates engineering, validation, and medium-scale evidence.
- `CONTRIBUTING.md` defines provenance, commit, pull-request, and merge conventions.

Raw data, generated HDF5/NPZ files, latent caches, checkpoints, W&B state, package
artifacts, and thesis outputs must not be committed.

## Citation and license

Software citation metadata is provided in `CITATION.cff`. The implementation is released
under the [MIT License](LICENSE).
