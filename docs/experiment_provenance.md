# Experiment Protocol and Provenance

`experiment_protocols/multiseed_v1.json` is the versioned plan for the next accepted experiment.
It is deliberately `planned`: no new result, test manifest, source tag, or W&B run is claimed yet.
Its structure is defined by `experiment_protocols/protocol.schema.json`.

## Freeze before execution

Before the first training run:

1. Record the source commit/tag and the SHA-256 of the tracked diff.
2. Bind the dataset revision and ordered universe-manifest hash.
3. Confirm whether at least ten trajectories exist that have not been inspected during model
   development. If they do, hash that final-test manifest and do not open it until model families,
   normalization, budgets, seeds, and checkpoints are fixed from validation.
4. If fewer than ten unseen trajectories exist, use five-fold nested group cross-validation on the
   available universe. Perform all model and checkpoint selection in each outer fold's inner
   validation data and call the result retrospective cross-validation, never a locked test.
5. Change `status` from `planned` to `frozen` in the same commit as the protocol values. Any later
   scientific change requires a new protocol ID; do not edit the frozen record in place.

The matched learned-model seeds are `52`, `53`, `54`, `55`, and `56`. The development split seed
remains `52`; training-seed variation measures initialization and optimization variability without
changing the development trajectories. Persistence is evaluated on every matching trajectory and
horizon but has no training seed dependence.

## Accepted-run linkage

For every accepted stage, create a local record that supplies this chain:

```text
protocol_id + source commit/tag + tracked-diff hash
  -> dataset revision + universe/split manifest hashes
  -> resolved-config hash + training seed
  -> encoder/head/cache/checkpoint hashes
  -> original W&B run ID or sanitized offline record
  -> raw per-seed/per-trajectory metrics hash
  -> table/figure source-data hash
```

The local record and W&B metadata must agree exactly. A path alone is not provenance because paths
can be reused. Hash files or deterministic directory manifests; record the hashing method and sort
directory entries before hashing. Keep credentials, usernames, private server paths, raw data,
checkpoints, and latent caches out of the public bundle.

## Analysis and reporting decision rule

Each sequence trainer selects its checkpoint by trajectory-balanced validation latent RMSE, the
training objective available without reopening the test split. The learned model family is then
selected within each outer fold by trajectory-balanced validation flux RMSE. The primary final
estimand is the paired selected-model-minus-observed-diagnostic-persistence difference in
per-trajectory flux RMSE, averaged over outer folds, training seeds, and trajectories. Compute a
paired hierarchical bootstrap that
resamples training seeds and trajectories; also report variability between seeds and the fraction
of trajectories improved.

The rollout `flux_rmse` field is the trajectory-balanced selection metric (mean per-trajectory
RMSE); `headline_sqrt_mean_trajectory_mse`/`flux_rmse_pooled` retain the pooled square-root
headline as a secondary diagnostic. Report `kyspec` and `fluxspec` separately.
Decoded latent-state persistence is a secondary latent-dynamics reference; applying the diagnostic
head to true future latents is an analysis control, not a forecast ceiling.
If the primary interval contains zero, write that the study found no convincing advantage. A
negative result is accepted protocol evidence; changing selection after seeing it is not.

## Completion and release

The protocol JSON is an immutable pre-run snapshot: it remains `frozen` with an empty
`accepted_runs` array so its source hash and tag cannot change after test evidence is opened. After
all accepted runs finish, publish the separate tracked `experiment_protocols/multiseed_v1_results.json`
release manifest generated from the ignored aggregate output. It records accepted/skipped stage
counts, hashes, fold-level summaries, and sanitized W&B status without trajectory rows, server paths,
or credentials. The raw stage ledger and source tables remain locally available and are referenced by
hash. Check W&B links while signed out; the canonical run uses W&B disabled and therefore has no live
run IDs or URLs to cite.
