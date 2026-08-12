# Thesis Result Set

## Tier 1: Internally Consistent Retrospective Result

The retained medium result uses one trajectory protocol across encoder, cache, sequence
training, validation selection, and retrospective test evaluation:

- data split seed `52`;
- validation manifest `94fa0f9184913ba30a99778e4cc5916fdae748d355fed26988e9fb97a0f293df`;
- test manifest `519a2b02e5f9bff8604a3cd28d12264333b95306be64b7619803d0e96772d012`;
- five trajectories and 40 rollout windows per evaluation split;
- horizon `8`, trajectory-balanced aggregation.

Validation selected the cache-normalized causal transformer as the strongest learned
candidate (`18.179277` flux RMSE), although persistence was lower (`17.755674`). On the
test realization, persistence reached flux RMSE `11.988926`; the selected transformer reached
`11.425478`, a `4.70%` reduction in the primary point estimate.

The transformer also improves flux MAE (`10.257084` versus `10.761940`), spectra relative
L2 (`19.168613` versus `19.525700`), and latent MSE (`0.073825` versus `0.101896`). It
improves per-trajectory flux RMSE on three of five trajectories. The headline values are
the square root of the trajectory-balanced mean squared error; the arithmetic mean of the
five per-trajectory RMSE values is `10.671522` versus `11.499391`. The descriptive paired
95% t interval for their mean difference includes zero, so the result is not evidence of
significance, repeatability, or general superiority. Flux is reported in preprocessed target
units because the artifact does not establish a physical unit.

Sources: `docs/medium_guppy_experiment_report.md`, local artifacts under
`outputs/medium_seed52_reproduction/`. W&B links are recorded in `docs/wandb_tracking.md`.

## Tier 2: Standalone Diagnostic Evidence

Representation plots, frozen diagnostic heads, and the seed-62 pretrained-SFT run are
separate evidence. They do not replace or combine with the seed-52 retrospective test.

## Tier 3: Engineering Validation

Synthetic smoke runs, one-trajectory checks, real-data smoke configurations, and
historical mixed-seed outputs establish implementation behavior only.

The former flux-RMSE claim `9.2270` remains invalidated: its encoder and downstream
trajectory splits differed, causing representation leakage. It must not appear as an
accepted thesis result.

## Tier 4: Related Work

GyroSwin remains related work unless dataset, split, horizon, metric definitions,
normalization, checkpoint provenance, and aggregation are made directly comparable.

## Provenance Limitation

Training artifacts record clean tracked source commit
`f97a0257d7627c8ff8960433aed30c750a9f90d5`; final test evaluations record
`280540f54e67c0dbcae253327596bfaf7cbf9307`. The latter has the SHA-256 of an empty
tracked diff. Its dirty flag is solely due to the separately packaged untracked thesis tree.
The test manifest appears in two retained evaluation records; this evidence is therefore a
single retrospective realization, not pristine locked-test evidence.
