# W&B tracking

Weights \& Biases is optional and disabled by default. Local resolved configurations, metrics,
checkpoint hashes, stage evidence, and protocol manifests are the source of truth; W\&B is only an
optional telemetry mirror and is never required for tests or CI.

## Accepted result

The frozen multi-seed protocol requested W\&B disabled for every accepted metric stage. The aggregate independently
validated each local `wandb_status.json` and the release manifest records:

```json
{"enabled": false, "requested": false, "mode": "disabled", "config_verified": true}
```

No accepted stage has a live W&B run ID or URL. The thesis therefore does not cite a dashboard or
claim public online evidence. The ignored repository-root `wandb/` directory contained stale
historical telemetry and is pruned during packaging. The accepted experiment's 225 tiny local
status files are retained in the compact ignored `outputs/multiseed-v1/wandb_status/` archive for
audit; model checkpoints and telemetry payloads are not.

post-hoc runs are not evidence for the accepted result and must not be substituted for the frozen
stage ledger or used to select a more favorable seed, fold, or architecture.

## If a future protocol enables W&B

Create the frozen protocol before the first run and use its immutable ID as the W&B group. Every
accepted record must include the source tag and tracked-diff hash, dataset revision and split
hashes, training seed, resolved configuration hash, model/checkpoint lineage, backend metadata,
checkpoint-selection metric, and raw per-trajectory metric hash. Retries are separate records and
are excluded from accepted comparisons unless the protocol records the failure and replacement
rationale.

Do not publish a W&B link until it has been checked while signed out and anonymous access is
permitted. If public access is inappropriate, retain a sanitized offline manifest without raw data,
checkpoints, latent caches, private paths, usernames, or credentials. Invalidated records must be
marked as such and removed from thesis-facing prose.
