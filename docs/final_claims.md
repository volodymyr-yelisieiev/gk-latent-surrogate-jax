# Thesis Claims

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
| Under the clean seed-52 protocol, the cache-normalized Transformer is the best learned validation candidate and lowers locked-test flux RMSE relative to persistence (`11.425478` versus `11.988926`). | `docs/current_best_model.md`, `docs/thesis_result_set.md`, `outputs/medium_seed52_reproduction/` |
| The selected Transformer also lowers locked-test flux MAE, latent MSE, and spectra relative L2; the paired five-trajectory interval includes zero, so no significance claim is made. | `docs/current_best_model.md`, `docs/medium_guppy_experiment_report.md` |
| Historical mixed-seed medium and transfer-validation runs are invalidated; the separate seed-62 transfer test is internally consistent but not comparable to the accepted seed-52 result. | `docs/medium_guppy_experiment_report.md`, `docs/pretrained_guppy_sft_feasibility.md` |

## Do Not Claim

- Do not claim full 5D reconstruction quality.
- Do not claim a direct GyroSwin win/loss without matching data, split, horizon, and
  metrics.
- Do not use latent MSE as the headline result.
- Do not mix smoke, one-trajectory, small validation, and medium test numbers in one
  headline table.
- Do not present `9.2270` or `17.8415` as held-out scientific evidence; the representation encoder
  had seen part of the downstream evaluation split under a different split seed.
- Do not compare the standalone seed-62 `11.5816` transfer value with seed-52 rows or present it as
  a baseline win; it has no matched persistence row or validation selection in that protocol.
- Treat one-trajectory runs as engineering validation and log-spectra single-batch runs as
  sensitivity evidence, not as rows in the main comparison table.

## Limitations

- The historical medium cache used encoder split seed 52 while downstream sequence/evaluation
  stages used seeds 53 or 54. Four of five seed-53 test trajectories occurred in the seed-52
  encoder training split, so the old headline table is not an end-to-end held-out result.
- The accepted clean result uses split seed 52 from encoder training through embedding, sequence
  training, validation selection, and locked testing. Its primary flux-RMSE point estimate favors
  the selected Transformer by 4.70% under this fixed protocol.
- The clean comparison contains five validation and five test trajectories and one training seed;
  it does not support a significance or equivalence claim.
- Generalization beyond the tested trajectories is not proven.
- GyroSwin remains protocol-level until matching comparison materials arrive.
- The implementation is a latent surrogate, not a full 5D reconstruction model.
