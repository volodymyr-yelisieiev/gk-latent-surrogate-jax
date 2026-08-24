# Result status

## Accepted thesis evidence

The accepted result is `experiment_protocols/multiseed_v1_results.json`, generated from the
ignored aggregate at the frozen source tag `protocol/multiseed-v1` (commit
`be976808582239da201896bd20ef95ff91d97128`). It is a retrospective five-fold nested group
cross-validation estimate on a manifest-/byte-verified 51-trajectory universe, not a pristine
locked test. The release records 255 planned stage slots: 230 accepted ledger slots (225 metric
stages and five selection barriers) and 25 skipped because the
corresponding learned family was not selected in validation.

| quantity | value |
| --- | ---: |
| selected learned flux RMSE | 14.6729 |
| observed-flux persistence flux RMSE | 10.0207 |
| learned minus observed difference | +4.6522 |
| 95% paired hierarchical bootstrap interval | [+2.5788, +6.6470] |
| decoded latent-persistence flux RMSE | 14.8049 |
| diagnostic-head oracle flux RMSE | 12.3732 |
| learned minus latent-persistence latent MSE | +0.0189 |
| secondary 95% interval | [-0.0699, +0.1332] |

The primary interval lies entirely above zero, meaning that the learned model has larger error
than direct observed persistence under the declared estimand. The correct thesis claim is therefore
that this implementation does not demonstrate an advantage over the strong baseline. The selected
family changes by fold (Transformer in 0, 1, and 3; GRU in 2 and 4), so no global architecture
winner is claimed. The diagnostic-head oracle is an analysis control, not a forecast, and does not
isolate a single representation or decoder bottleneck.

## Integrity and visibility

All accepted metric stages are finite and stable, have matching fold/seed manifests and artifact
lineage, and state that test evidence was not opened for selection. The aggregate also checks paired
trajectory identities, stage hashes, checkpoint/cache hashes, and the fixed bootstrap seed. W&B is
disabled by protocol for all 225 accepted metric stages (`enabled=false`, `requested=false`,
`mode=disabled`, with configuration verification); there are no live W&B URLs to cite. Raw data, checkpoints, caches,
and trajectory-level rows remain on the authorized server and are not committed.

Engineering smoke outputs remain useful for regression testing but are not thesis evidence. Any
future scientific change requires a new protocol ID and source/data binding; the frozen protocol
must not be edited in place.
