# Experiment Lifecycle

1. Freeze a versioned protocol under `experiment_protocols/` as described in
   `docs/experiment_provenance.md`. Record source and data-universe hashes before training.
2. Validate a config with `--dry-run`; dry-runs print the full resolved config and a
   command summary without training/evaluation writes.
3. Inspect data shapes and target availability with `gks inspect-data`.
4. Run smoke training on synthetic data with `gks train-encoder`.
5. Train the same-time direct diagnostic control with `gks train-direct-diagnostics`; use it to
   distinguish target learnability from latent representation and temporal-forecast quality.
6. Embed trajectories into an HDF5 latent cache.
7. Evaluate validation flux RMSE with `gks evaluate-flux-head`.
8. Generate PCA/t-SNE representation plots with `gks plot-representation`.
9. Train latent sequence models from cache windows and preserve original telemetry.
10. Select sequence checkpoints by validation latent RMSE and model families by validation flux
    RMSE, both on validation only, then evaluate the frozen final protocol.
11. Link accepted raw metrics and figure-source tables through an accepted-run manifest.

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

The next accepted comparison uses matched learned-model training seeds `52`--`56`. Use at least
ten previously uninspected trajectories for a final test. If that is impossible, run the frozen
five-fold nested group cross-validation fallback and describe the result as retrospective. Do not
reuse the known seed-52 test manifest as new confirmation evidence.
