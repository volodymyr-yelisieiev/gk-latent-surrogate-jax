# Result status

The retained seed-52 run is a retrospective audit record, not a prospective locked-test
result. The old comparison used a decoded latent-state persistence reference and reported a
small Transformer advantage. That reference is useful for latent dynamics, but it is not the
appropriate baseline for forecasting an observed diagnostic.

The baseline audit on the retained five-trajectory cache gives:

| method | flux RMSE | interpretation |
| --- | ---: | --- |
| observed-flux persistence | 3.7208 | primary diagnostic baseline; copies the last observed flux |
| selected Transformer | 11.4255 | learned latent rollout followed by the frozen diagnostic head |
| latent persistence, decoded | 11.9889 | latent-state baseline followed by the same head |
| diagnostic-head oracle | 12.0712 | frozen head applied to true future latents; not a forecast |

The oracle's error is close to both latent-rollout rows. This means the present comparison is
dominated by diagnostic decoding error and does not support a claim that the learned sequence model
beats observed-flux persistence. The Transformer is also not presented as a general winner: the
record contains one training seed, five trajectories, and a retrospectively inspected manifest.

The five-fold, five-seed protocol in `experiment_protocols/multiseed_v1.json` is planned and
requires a byte-verified dataset universe before execution. It is the only route for a new
thesis-facing comparison. Engineering smoke outputs under `outputs/` remain useful for checking
the pipeline but are not empirical evidence.

The three historical W&B records and their tags were deleted during the audit. No live W&B URL or
public evidence-release URL is claimed. A future accepted run must create fresh, protocol-linked
records and retain a sanitized offline evidence manifest if public access is not allowed.
