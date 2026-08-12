# W&B Tracking and Evidence Rules

Weights & Biases is optional and disabled by default. Local checks and CI must not require an
account, login, network access, or a mutable remote record. W&B supplements the local artifact
manifest; it is not the source of truth.

## Run identity

Create the protocol manifest before training. Use its immutable `protocol_id` as the W&B group.
Name original runs with the pattern `<model>/seed-<seed>/<stage>`, where stage is one of
`encoder-train`, `embed`, `sequence-train`, `validation`, or `test`. Retries add `/retry-<n>` and
must be excluded from the accepted comparison unless the manifest records why the original run was
invalid.

```bash
uv run gks train-sequence \
  --config configs/experiment/server_sequence_transformer_medium.yaml \
  --override logging.wandb.enabled=true \
  --override logging.wandb.mode=online \
  --override logging.wandb.project=gk-latent-surrogate \
  --override logging.wandb.group=multiseed-v1 \
  --override logging.wandb.name=transformer/seed-52/sequence-train \
  --override logging.wandb.tags='[cyclone,medium,sequence-train,transformer,protocol-v1]'
```

Use offline mode when the server cannot reach W&B, then sync the original run directory without
creating a second logical run:

```bash
uv run gks train-sequence \
  --config configs/experiment/server_sequence_transformer_medium.yaml \
  --override logging.wandb.enabled=true \
  --override logging.wandb.mode=offline \
  --override logging.wandb.directory='${GK_FAST_ROOT}/wandb'
```

When tracking is active, commands write `wandb_status.json` under the output directory. Preserve
the W&B run ID in the local accepted-run manifest.

## Required original telemetry

An accepted learned-model result must link to the original training run, not only to a post-hoc
summary run. The run and its local artifact record must contain:

- protocol ID and schema version;
- source commit, source tag or release, tracked-diff SHA-256, and dirty-state explanation;
- dataset revision, universe-manifest hash, split-manifest hashes, split seed, and training seed;
- resolved configuration SHA-256, command/stage, model family, normalization protocol, context,
  rollout horizon, aggregation semantics, and metric units;
- upstream encoder, diagnostic-head, latent-cache, and sequence-checkpoint hashes;
- system metadata: Python, JAX, Flax, device model/count, backend, precision, and host profile;
- actual scheduled learning rate, train loss components, validation primary metric, checkpoint
  selection step/rationale, runtime, and failure or retry state;
- raw per-seed/per-trajectory evaluation table, accepted metric JSON, and figure-source table.

Log training and validation curves at their actual step. Do not relabel the last minibatch as an
epoch aggregate or best validation result. Persistence has no training telemetry, but its evaluation
must use the same manifest, horizons, diagnostic lineage, and aggregation as the learned model.

All stages in one comparison must use the same declared data universe and split rules. A result is
not comparable when its encoder, cache, diagnostic heads, sequence checkpoint, normalization fit,
or evaluation has incompatible lineage. Mark such runs `protocol_status=invalidated`; do not delete
or silently reuse them.

## Public evidence and fallback

Before citing W&B, test every run and report URL in a signed-out browser. If anonymous read access
is not permitted by the data policy, publish a sanitized GitHub release bundle containing the
protocol manifest, accepted-run manifest, resolved configs, environment metadata, artifact hashes,
metric JSON, raw per-seed/per-trajectory tables, and figure-source tables. Never include raw data,
checkpoints, latent caches, server paths, API keys, usernames, or private project URLs.

The release manifest must map:

```text
release/tag -> source commit -> protocol manifest -> data/split hashes -> resolved config
            -> training seed -> checkpoint/cache hashes -> W&B run ID or sanitized record
            -> raw metric table -> reported table/figure
```

## Status of the retained seed-52 records

The group `medium-seed52-reproduction` contains three post-hoc evidence records:

- validation comparison: [`m52rval1`](https://wandb.ai/v-yelisieiev-johannes-kepler-universit-t-linz/gk-latent-surrogate/runs/m52rval1);
- persistence test: [`m52rper1`](https://wandb.ai/v-yelisieiev-johannes-kepler-universit-t-linz/gk-latent-surrogate/runs/m52rper1);
- cache-normalized Transformer test: [`m52rtrn1`](https://wandb.ai/v-yelisieiev-johannes-kepler-universit-t-linz/gk-latent-surrogate/runs/m52rtrn1).

They reference local artifacts under `outputs/medium_seed52_reproduction`, split seed `52`, the
stored train/cache/evaluation manifests, horizon `8`, trajectory-balanced aggregation, source
commit `f97a0257d7627c8ff8960433aed30c750a9f90d5`, and tracked-diff digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

These records are evidence mirrors, not complete original training telemetry. Their anonymous
access has not been established by repository evidence. The test manifest is already known and has
two retained method evaluations, so this group supports only a retrospective single-realization
comparison. It must not be labelled untouched, opened once, or prospective locked-test evidence.
Earlier `medium-seed52-heldout-evaluation` and mixed-split medium records remain invalidated or
descriptive-only. None of these groups satisfies the planned multi-seed protocol by itself.
