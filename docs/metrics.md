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
flux RMSE on the requested split. For model validation, use the validation split and treat
`flux_rmse` as the primary number; latent MSE is a secondary training diagnostic.

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
