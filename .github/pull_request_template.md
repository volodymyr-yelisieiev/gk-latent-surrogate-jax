## Context

State the problem, research question, or maintenance requirement.

## Changes

-

## Scientific/provenance impact

State whether metrics, datasets, caches, checkpoints, splits, horizons, normalization, or
aggregation semantics change. Use `None` when the change has no scientific-result impact.

## Verification

- [ ] `make check`
- [ ] `make smoke-all` when pipeline behavior changes
- [ ] `uv build` when packaging or CI changes
- [ ] Required GitHub Actions check is green

## Risks

Describe compatibility limits, unresolved evidence gaps, or rollback considerations.

### Repository checks

- [ ] PR title follows `type(scope): imperative summary` and is at most 72 characters.
- [ ] Generated artifacts and secrets are not committed.
- [ ] Results from different evaluation protocols are not combined.
