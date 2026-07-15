# Thesis Result Set

Evidence reviewed: 2026-07-15.

This file separates result tiers so thesis writing does not mix engineering validation,
small validation, medium runs, and protocol-level comparisons.

## Tier 1: Thesis-Scale Medium Evidence

Use this as the main quantitative sequence-model comparison.

| evidence | role | trace |
| --- | --- | --- |
| Medium rollout comparison | main result table | `docs/medium_guppy_experiment_report.md` |
| Best model | current model selection | `docs/current_best_model.md` |
| W&B group | current external experiment evidence | `medium-scale-latent-surrogate` in project `gk-latent-surrogate` |

Main result:

- cache-normalized from-scratch latent transformer;
- test split, 8-step rollout horizon;
- flux RMSE `9.2270`;
- source `outputs/verified_medium/transformer_normalized/metrics.json`;
- evidence scope: locally re-evaluated artifact set; no active W&B run mirrors this exact
  main-cache table.

Current W&B evidence is listed separately in `docs/medium_guppy_experiment_report.md`.
The validation sensitivity comparison and pretrained-SFT test run use different protocols
and do not replace the `9.2270` verified main-cache result.

## Tier 2: Small Validation Evidence

Use this tier for validation flux RMSE and representation-structure evidence.

| evidence | role | trace |
| --- | --- | --- |
| validation flux-head run | validation flux RMSE | `docs/small_validation_experiment.md` |
| PCA plot colored by flux | representation inspection | `outputs/server_plot_representation_small/plots/pca_flux.png` |
| t-SNE plots colored by flux | representation inspection | `outputs/server_plot_representation_small/plots/tsne_perplexity_*_flux.png` |
| persistence and transformer rollouts | small-run baseline comparison | `docs/small_validation_experiment.md` |

Small validation is not the medium thesis-scale table. Keep it as a separate validation
evidence tier.

## Tier 3: Engineering Validation Evidence

Use only for implementation confidence:

- synthetic smoke pipeline;
- one-trajectory pipeline validation;
- real-data smoke configs;
- server/KvikIO inspection outputs.

These establish that the pipeline runs. They should not be used as headline scientific
results.

## Tier 4: Protocol-Level Related Work

GyroSwin remains protocol-level unless matching materials arrive.

Direct comparison requires the same or explicitly documented dataset, split, horizon,
metric definitions, normalization, and checkpoint/config provenance. Until then, cite
GyroSwin as related work or a comparison target, not as a numeric baseline.

Claim boundaries and limitations are maintained in `docs/final_claims.md`; this file only
defines evidence tiers.
