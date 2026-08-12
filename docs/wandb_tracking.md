# W&B tracking

Weights & Biases is optional and disabled by default. Local resolved configurations, metrics,
checkpoint hashes, and protocol manifests remain the source of truth; W&B is only a telemetry
mirror. Tests and CI never require an account, network access, or a mutable remote project.

## Current status

The project `gk-latent-surrogate` was emptied during the scientific audit. The historical
post-hoc runs and their tags were deleted, so this repository intentionally contains no live W&B
run IDs or result URLs. The old records must not be recreated or cited as if they were original
training telemetry.

## New accepted runs

Create the frozen protocol before the first run. Use its immutable `protocol_id` as the W&B group
and record the run ID in the local accepted-run manifest only after the local metrics and artifact
hashes have been checked. Every run should include:

- source commit/tag and tracked-diff hash;
- dataset revision, universe/split hashes, and training seed;
- resolved-config hash, stage, model family, normalization, context, horizon, and units;
- upstream encoder, diagnostic-head, latent-cache, and sequence-checkpoint hashes;
- device/backend metadata and the actual checkpoint-selection metric;
- raw per-trajectory metrics and the exact figure/table source data.

Retries are separate records and are excluded from an accepted comparison unless the protocol
records the failure and replacement rationale. Persistence has no training telemetry, but it must
use the same trajectory universe, horizon, diagnostic lineage, and aggregation as the learned
model.

## Visibility and deletion

Do not publish a W&B link until it has been checked while signed out and the project policy permits
anonymous read access. If public access is not appropriate, retain a sanitized offline manifest
without raw data, checkpoints, latent caches, private paths, usernames, or credentials. When a
record is invalidated or a project is pruned, mark it deleted in the local audit notes and remove
all public-facing URLs; do not leave stale links in thesis prose.
