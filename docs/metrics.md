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

Between-trajectory standard deviation bands are dispersion, not confidence intervals. The
five-trajectory paired t interval in the retained medium report is descriptive and contains no
training-seed uncertainty. Spectral aggregate metrics average `kyspec` and `fluxspec`, which have
different numerical scales; report per-target relative L2 and shape correlation before making any
spectral-fidelity interpretation.

`gks plot-representation` writes PCA and t-SNE plots from the latent cache, colored by
flux. It records train/validation/test split labels alongside the projection points so the
plots are tied to held-out-aware experiment artifacts.

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
