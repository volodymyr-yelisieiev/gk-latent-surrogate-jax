# Current Model Selection

## Selection

The cache-normalized causal transformer is the selected learned sequence model. It had
the lowest validation flux RMSE among the four learned candidates under the clean
split-seed-52 protocol.

| method | validation flux RMSE |
| --- | ---: |
| GRU | 19.927494 |
| GRU, cache-normalized | 19.479990 |
| causal transformer | 18.682791 |
| causal transformer, cache-normalized | **18.179277** |
| persistence reference | 17.755674 |

Persistence still had the lowest validation primary metric overall. The transformer was
selected only as the strongest learned candidate for the retained seed-52 test comparison.

## Retrospective Test Result

Both rows use test manifest
`519a2b02e5f9bff8604a3cd28d12264333b95306be64b7619803d0e96772d012`.

| method | flux RMSE ↓ | flux MAE ↓ | latent MSE ↓ | spectra relative L2 ↓ |
| --- | ---: | ---: | ---: | ---: |
| persistence | 11.988926 | 10.761940 | 0.101896 | 19.525700 |
| selected transformer | **11.425478** | **10.257084** | **0.073825** | **19.168613** |

The learned model lowers the primary flux-RMSE point estimate by `0.563448` (`4.70%`).
This headline RMSE is `sqrt(mean trajectory MSE)` over trajectories and horizons; it is
not the arithmetic mean of the five per-trajectory RMSE values. The latter is `10.671522`
for the transformer and `11.499391` for persistence.
It also lowers flux MAE by `4.69%`, latent MSE by `27.55%`, and spectra relative L2 by
`1.83%`. Per-trajectory flux RMSE improves on three of five test trajectories; the paired
mean difference is `-0.8279` with a descriptive 95% t interval `[-2.2249, 0.5692]`.
The interval describes this five-trajectory realization and includes zero; no significance,
equivalence, repeatability, or general-superiority claim is made. Flux values are in the
preprocessed target units because a physical unit is not established by the retained artifact.

## Provenance

- cache: `outputs/medium_seed52_reproduction/embed/latent_cache.h5`;
- encoder: `outputs/medium_seed52_reproduction/encoder/checkpoints/step_000500`;
- sequence checkpoint:
  `outputs/medium_seed52_reproduction/sequence_transformer_cache_normalized/checkpoints/step_000500`;
- split seed: `52`; horizon: `8`; aggregation: trajectory-balanced mean with
  between-trajectory standard deviation;
- encoder/cache/sequence source commit: `f97a0257d7627c8ff8960433aed30c750a9f90d5`;
- final test-evaluation source commit: `280540f54e67c0dbcae253327596bfaf7cbf9307`.

The test records have no tracked diff (`tracked_diff_sha256` is the SHA-256 of an
empty byte string). They record `git_dirty=true` only because the separately packaged
untracked thesis tree was present; all executable tracked source is identified by commit.

The test manifest was evaluated in two retained runs and was known when this report was
written. It must not be described as untouched, opened once, or a prospective locked test.
