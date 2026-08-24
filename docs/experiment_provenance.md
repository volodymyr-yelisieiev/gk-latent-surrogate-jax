# Experiment Protocol and Provenance

`experiment_protocols/multiseed_v1.json` is the immutable pre-run snapshot for the accepted
comparison. Its status is `frozen`, its `accepted_runs` array is empty by design, and its source
tag is `protocol/multiseed-v1`. The release manifest
`experiment_protocols/multiseed_v1_results.json` is the sanitized post-run evidence record; it does
not edit the frozen protocol.

The frozen snapshot leaves its optional `source.commit` field null because the protocol was sealed
before the evidence run; the release resolves the immutable source tag to the full commit digest
shown below. This keeps the pre-run protocol unchanged while making the post-run source binding
explicit.

## Frozen evidence binding

The run and aggregate are bound to source commit
`be976808582239da201896bd20ef95ff91d97128`, an empty tracked diff, the universe manifest digest
`96c85a70119ca790ef46ba6ccbee1f75e2b37a400eed48b900194650ef35c68e`, and outer-fold manifest
digest `da5d95b87985e5fb6c32c880d15627aa17eccd3d262c61f723d1044cef34ff87`. The data revision is
`cyclone-consumed-bytes-sha256:ff2867e9eb8e9ed74dd1ed92d347b02a703368e634f86cce983c07b0754e3d7a`.
The consumed universe has 51 trajectories and 1,173 sampled timestep bundles. The manifest-/byte-
verification routine checked the enumerated consumed files in both the durable subset and the
operational public-data alias; the package publishes the sanitized binding report without private
paths in `experiment_protocols/multiseed_v1_dataset_binding.json`.

The stage ledger contains 255 slots: 230 accepted ledger slots (225 metric stages and five
selection barriers) and 25 `skipped_unselected`. Each accepted metric row
has a resolved configuration hash, training seed, fold manifest hash, and checkpoint/cache lineage.
The five selection barriers use only validation data and all state `test_evidence_opened=false`.
The model family is Transformer in folds 0, 1, and 3 and GRU in folds 2 and 4. Sequence checkpoint
selection is by validation latent RMSE; family selection is by five-seed validation flux RMSE.
When post-run maintenance is present, `provenance.postflight_generation_commit` identifies the
clean checkout used to generate the sanitized manifest. The release commit necessarily follows
that checkout, so this field is intentionally not expected to equal the final package `HEAD`.

## Analysis rule

The primary estimand is the paired selected-model-minus-observed-persistence trajectory-balanced
flux RMSE difference, weighted equally by outer fold, training seed, and trajectory. The secondary
estimand is the corresponding latent-MSE difference against decoded latent persistence. The
10,000-replicate hierarchical bootstrap seeds, scalar metrics, per-fold summaries, per-seed
variability, and separate spectra metrics are all recorded in the aggregate and release manifests.
The protocol requires the conclusion ``no convincing advantage'' whenever the primary interval
contains zero; on this universe the strictly positive interval supports a clear negative result.

During the final independent pass, one scalar validation metric differed from the mean of its JSON
trajectory values by 1.62e-6 because the stage scalar was accumulated in JAX float32 and the
recomputation used NumPy float64. The validator now has a narrow, tested float32-rounding tolerance
(`rel_tol=1e-6`, `abs_tol=3e-6`); it rejects larger discrepancies. This is a post-run maintenance
fix to the analysis checker, not a change to model code, configs, evidence values, selection, or
the frozen source binding. The accepted aggregate remains explicitly tied to the frozen tag.

## W&B and publication boundaries

W&B is disabled and configuration-verified for all 225 accepted metric stages. Local `wandb_status.json`
files are the source of truth and state `enabled=false`, `requested=false`, and `mode=disabled`.
There are no live run IDs or URLs in the thesis. Raw trajectories, checkpoints, latent caches, and
per-trajectory metric rows stay outside Git; the tracked release contains hashes and aggregate
summaries only. A future scientific change must receive a new protocol ID, source binding, and
dataset revision rather than modifying this record in place.
