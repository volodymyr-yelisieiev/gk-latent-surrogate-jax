# Thesis Claims

Evidence reviewed: 2026-07-15.

This ledger ties thesis claims to reproducible evidence and states the boundaries of the
current results.

## Research Question

Can a compact JAX/Flax latent surrogate learn short-horizon gyrokinetic dynamics from
preprocessed 5D field snapshots, and do latent sequence models improve flux-predictive
rollouts over persistence baselines on held-out trajectories?

## Supported Claims

| claim | evidence |
| --- | --- |
| The repository implements a portable latent-surrogate pipeline with CPU smoke coverage. | `make smoke-all`, `docs/verification_matrix.md`, GitHub Actions CI |
| The data contract is channel-first snapshots `[B, C, S1, S2, S3, S4, S5]` with latent-cache sequence windows. | `docs/data_contract.md`, tests under `tests/test_*data*`, `tests/test_train_sequence_step.py` |
| Validation and thesis-facing metrics prioritize flux RMSE over latent MSE. | `docs/metrics.md`, `docs/small_validation_experiment.md`, validated experiment protocol |
| Small validation includes flux-head evaluation and PCA/t-SNE plots colored by flux. | `docs/small_validation_experiment.md`, `configs/experiment/server_evaluate_flux_head_small.yaml`, `configs/experiment/server_plot_representation_small.yaml` |
| The current Guppy-style sequence model is trained from scratch, not from a public pretrained Guppy checkpoint. | `docs/pretrained_guppy_sft_feasibility.md` |
| The optional GuppyLM trunk-transfer/SFT experiment completed without silent initialization fallback and reported flux RMSE `17.8415` on validation and `11.5816` on test. | `docs/pretrained_guppy_sft_feasibility.md`, active W&B runs `eh4xe1sg` and `iq92tggu` |
| The best current medium model is the cache-normalized latent transformer: flux RMSE `9.2270` on the verified medium test rollout set. | `docs/medium_guppy_experiment_report.md`, `outputs/verified_medium/transformer_normalized/metrics.json` |
| GRU variants do not beat persistence on medium flux RMSE in the current artifact set. | `docs/medium_guppy_experiment_report.md` |

## Do Not Claim

- Do not claim full 5D reconstruction quality.
- Do not claim a direct GyroSwin win/loss without matching data, split, horizon, and
  metrics.
- Do not use latent MSE as the headline result.
- Do not mix smoke, one-trajectory, small validation, and medium test numbers in one
  headline table.
- Do not imply that active W&B runs mirror the verified local `9.2270` main-cache table; they
  cover distinct validation, representation, comparison, and pretrained-SFT protocols.
- Treat one-trajectory runs as engineering validation and log-spectra single-batch runs as
  sensitivity evidence, not as rows in the main comparison table.

## Limitations

- The evidence is bounded by the current preprocessed Cyclone/KvikIO cache and split.
- The main-cache result was re-evaluated locally with trajectory-balanced aggregation,
  1-based horizons, and finite diagnostic metrics, but has no active exact W&B mirror;
  current W&B runs cover separate experiment protocols.
- Generalization beyond the tested trajectories is not proven.
- GyroSwin remains protocol-level until matching comparison materials arrive.
- The implementation is a latent surrogate, not a full 5D reconstruction model.
