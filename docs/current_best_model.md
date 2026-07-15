# Current Best Model

Evidence reviewed: 2026-07-15.

## Selection

Current best model: cache-normalized from-scratch Guppy-style latent transformer.

| field | value |
| --- | --- |
| encoder | `outputs/server_encoder_simsiam_medium/checkpoints/step_000500` |
| latent cache | `outputs/latent_cache/server_medium/latent_cache.h5` |
| sequence checkpoint | `outputs/server_sequence_transformer_medium_normalized/checkpoints/step_000500` |
| split | test |
| rollout horizon | 8 |
| primary metric | flux RMSE |
| flux RMSE | `9.2270` |
| evidence scope | locally re-evaluated artifact set; exact W&B mirror unavailable |
| source metrics | `outputs/verified_medium/transformer_normalized/metrics.json` |

## Why It Is Current Best

It has the lowest flux RMSE in the comparable verified medium-cache table documented in
`docs/medium_guppy_experiment_report.md`. Selection uses flux RMSE; latent MSE remains a
secondary diagnostic.

## Scope

This model is the best current thesis-scale latent sequence model in this repository. It
is not a production surrogate and does not prove full-field reconstruction quality.

The selection has the following caveats:

- The local rerun uses trajectory-balanced aggregation and 1-based forecast horizons. The
  former W&B run for this exact result is unavailable.
- The comparison is internal to the current latent cache and 8-step horizon.
- Public GyroSwin artifacts are related work, not a directly comparable baseline yet.
