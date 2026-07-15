# Experiment Lifecycle

1. Validate a config with `--dry-run`; dry-runs print the full resolved config and a
   command summary without training/evaluation writes.
2. Inspect data shapes and target availability with `gks inspect-data`.
3. Run smoke training on synthetic data with `gks train-encoder`.
4. Embed trajectories into an HDF5 latent cache.
5. Evaluate validation flux RMSE with `gks evaluate-flux-head`.
6. Generate PCA/t-SNE representation plots with `gks plot-representation`.
7. Train latent sequence models from cache windows.
8. Evaluate rollout metrics and diagnostic quality.

Before real-data training, follow `docs/real_data_binding_checklist.md`. The first
Cyclone/KvikIO pass should be inspection-only:

```bash
export GK_CYCLONE_DATA_ROOT=/path/to/preprocessed_kvikio
JAX_PLATFORM_NAME=cpu uv run gks inspect-data \
  --config configs/data/cyclone_kvikio_template.yaml \
  --dry-run \
  --max-trajectories 1 \
  --max-depth 4 \
  --max-target-samples 64
```

For the currently validated server schema, the first real smoke configs already bind
`iteration_0`, `offset: 80`, `subsample: 32`, and stored spectra
`[kyspec, fluxspec]`. Use `configs/experiment/smoke_real_encoder_flux.yaml` for the first
two-step supervised run, followed by
`configs/experiment/smoke_real_embed_dataset.yaml`,
`configs/experiment/smoke_real_sequence.yaml`, and
`configs/experiment/smoke_real_evaluate_rollout.yaml`.

Use the same real smoke config for hardware comparisons across MacBook CPU, RTX 5070 WSL2,
and the student server. Record steps/sec, memory pressure or maximum fitting batch size,
KvikIO/BF16 settings, and any data-loading throughput bottleneck before changing model
scale.

Outputs, checkpoints, runs, raw data, and generated HDF5 files are gitignored.
Full resolved configs are written as `config_resolved.yaml` by normal commands that have
an output directory; dry-runs intentionally do not create that artifact.
