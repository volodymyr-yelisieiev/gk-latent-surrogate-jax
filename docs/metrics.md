# Metrics

Implemented latent metrics:

- MSE, MAE, relative L2, cosine similarity;
- rollout MSE and cosine similarity by prediction step;
- finite-output stability and threshold horizon.

Implemented diagnostic metrics:

- flux MSE, RMSE, MAE, relative error, and time-average flux error;
- spectra MSE, log-MSE, relative L2, Pearson/shape correlation, and mean absolute
  relative error.

Relative L2 uses one global reduction over the evaluated tensor:

```text
relative_l2 = ||prediction - target||_2 / (||target||_2 + epsilon)
```

For rollout curves, the pipeline applies this definition independently at each horizon,
reducing over batch and feature dimensions for that step. This reduction is part of the
artifact contract and must remain stable when comparing runs.

`gks evaluate-flux-head` fits a frozen-latent ridge head on train trajectories and reports
flux RMSE on the requested split. For nested-CV model selection, use the validation split and
treat `trajectory_balanced_flux_rmse` (the arithmetic mean of per-trajectory RMSE values) as the
primary number. The `flux_rmse` field is an exact alias for that selection metric in rollout
evidence; the pooled square-root-of-mean-MSE scalar is retained as
`headline_sqrt_mean_trajectory_mse`/`flux_rmse_pooled`. Latent MSE is a secondary training
diagnostic.

For trajectory-balanced rollout evaluation, the declared selection flux RMSE is

```text
mean over trajectories of sqrt(mean over horizons and features of per-trajectory flux MSE)
```

The pooled square-root-of-mean-MSE scalar is reported separately and is not used for architecture
selection. Paired analyses operate on the per-trajectory RMSE values and therefore match the
selection estimand. Report both when using paired uncertainty. The current Cyclone artifacts do
not establish a physical flux unit, so
their flux errors must be labelled `preprocessed target units` unless the data owner supplies
and verifies a physical unit.

Between-trajectory standard deviation bands are dispersion, not confidence intervals. The accepted
report uses a paired hierarchical bootstrap over five outer folds, five matched training seeds, and
trajectories; horizon ribbons remain between-fold dispersion rather than confidence intervals.
Spectral aggregate metrics keep `kyspec` and `fluxspec` separate because they have different
numerical scales; report per-target relative L2 and shape correlation before making any
spectral-fidelity interpretation.

`gks plot-representation` writes PCA and t-SNE plots from the latent cache, colored by
flux and marked by train/validation/test role. When `data.split_manifest` is set, the command
uses those exact trajectory assignments and records the manifest hash and fold ID; it never
creates a second seeded split over a manifest-bound cache. The split labels are also stored
alongside the projection points. The result metadata includes the data/training seeds plus the
latent-cache and encoder-checkpoint SHA-256 values, so a projection remains tied to one
held-out-aware artifact lineage. When colocated resolved configs are present, their SHA-256 values
are recorded as well; the thesis figure generator compares the canonical fold-0/seed-52 values to
the accepted `multiseed-v1` release manifest before producing a manuscript asset.

Rollout evaluation writes `metrics.json`, `metrics_by_step.csv`, and a latent MSE plot.
Exact GyroSwin comparison tables require confirmed real-dataset and reference-metric
definitions.

Implementation entry points:

- `src/gk_surrogate/metrics/latent.py`
- `src/gk_surrogate/metrics/diagnostics.py`
- `src/gk_surrogate/metrics/rollout.py`
- `src/gk_surrogate/evaluation/flux_head.py`
- `src/gk_surrogate/evaluation/representation.py`
- `src/gk_surrogate/evaluation/reports.py`
