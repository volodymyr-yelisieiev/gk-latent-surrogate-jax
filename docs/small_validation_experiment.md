# Small Validation Experiment

This experiment evaluates 3-5 trajectories with a held-out validation split. It reports
validation `flux_rmse` as the primary metric and retains latent MSE as a secondary
diagnostic.

## Trajectory Selection

Use three to five real trajectories. If the server dataset has more, pin the set at
runtime and keep the exact IDs in the run notes:

```bash
export GK_SMALL_VALIDATION_TRAJ_0=traj_a
export GK_SMALL_VALIDATION_TRAJ_1=traj_b
export GK_SMALL_VALIDATION_TRAJ_2=traj_c
export GK_SMALL_VALIDATION_TRAJ_3=traj_d
```

The committed small validation configs use these four environment variables instead of
storing server-specific trajectory IDs. Training, embedding, sequence, flux-head, rollout,
and representation commands all use the same configured subset. The validation split is
trajectory-held-out through the shared split helper; evaluation commands reject caches
that cannot produce held-out validation evidence.

## Commands

Run these on the server after setting `GK_CYCLONE_DATA_ROOT`:

```bash
gks train-encoder --config configs/experiment/server_encoder_simsiam_small.yaml
gks embed-dataset --config configs/experiment/server_embed_dataset_small.yaml
gks train-sequence --config configs/experiment/server_sequence_transformer_small.yaml
gks evaluate-flux-head --config configs/experiment/server_evaluate_flux_head_small.yaml
gks evaluate-rollout --config configs/experiment/server_evaluate_persistence_baseline_small.yaml
gks evaluate-rollout --config configs/experiment/server_evaluate_rollout_small_transformer.yaml
gks plot-representation --config configs/experiment/server_plot_representation_small.yaml
```

The existing GRU small rollout config is retained as a baseline model, while the
transformer config is the current from-scratch Guppy-style sequence path.

## Evidence

Primary metric:

- `outputs/server_evaluate_flux_head_small/metrics.json`: validation `flux_rmse`.

Baseline comparison:

- `outputs/baselines/server_persistence_small/metrics.json`: persistence baseline.
- `outputs/server_evaluate_rollout_small_transformer/metrics.json`: learned transformer
  rollout. Use `flux_rmse` first; report latent MSE only as secondary diagnostic context.

Plots:

- `outputs/server_plot_representation_small/plots/pca_flux.png`
- `outputs/server_plot_representation_small/plots/tsne_perplexity_5_flux.png`
- `outputs/server_plot_representation_small/plots/tsne_perplexity_30_flux.png`

## Reporting Contract

Report the validation split, trajectory count, frozen-latent ridge-head `flux_rmse`, and
the matching persistence and transformer rollout metrics. Treat PCA/t-SNE plots as
qualitative representation evidence and latent MSE as a secondary diagnostic.
