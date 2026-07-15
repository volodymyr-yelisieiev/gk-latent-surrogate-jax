# Medium Latent-Sequence Experiment Report

Status: clean split-seed-52 validation selection and locked test.

## Protocol

| field | value |
| --- | --- |
| latent cache | `outputs/medium_seed52_reproduction/embed/latent_cache.h5` |
| encoder | `outputs/medium_seed52_reproduction/encoder/checkpoints/step_000500` |
| data split seed | `52` |
| validation manifest | `94fa0f9184913ba30a99778e4cc5916fdae748d355fed26988e9fb97a0f293df` |
| test manifest | `519a2b02e5f9bff8604a3cd28d12264333b95306be64b7619803d0e96772d012` |
| rollout horizon | 8 |
| aggregation | trajectory-balanced mean; between-trajectory standard deviation |
| evaluation size | 5 trajectories, 40 windows per split |
| primary metric | flux RMSE |

All encoder/cache/sequence/evaluation stages use the same split seed. Model and
normalization selection used validation only; the selected learned method was then
evaluated once on the locked test manifest.

## Validation Selection

| method | flux RMSE ↓ | flux MAE ↓ | latent MSE ↓ | spectra relative L2 ↓ |
| --- | ---: | ---: | ---: | ---: |
| persistence | **17.755674** | **12.768312** | **0.434076** | 1.411545 |
| GRU | 19.927494 | 14.336292 | 0.614362 | 1.426967 |
| causal transformer | 18.682791 | 14.010544 | 0.482307 | 1.422305 |
| GRU, cache-normalized | 19.479990 | 14.796824 | 0.547818 | 1.452332 |
| causal transformer, cache-normalized | 18.179277 | 12.965019 | 0.455987 | **1.405970** |

Persistence wins the validation primary metric. The cache-normalized causal transformer
is the strongest learned candidate and was selected for the locked learned-model test.

## Locked Test

| method | flux RMSE ↓ | flux MAE ↓ | flux relative error ↓ | latent MSE ↓ | spectra relative L2 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 11.988926 | 10.761940 | 2.247385 | 0.101896 | 19.525700 |
| selected cache-normalized transformer | **11.425478** | **10.257084** | **2.208325** | **0.073825** | **19.168613** |

The selected transformer lowers the primary flux-RMSE point estimate by `0.563448`
(`4.70%`). It lowers flux MAE by `4.69%`, flux relative error by `1.74%`, spectra relative
L2 by `1.83%`, and latent MSE by `27.55%`. It improves per-trajectory flux RMSE on three
of five trajectories. The paired mean difference is `-0.8279`, with a descriptive 95% t
interval `[-2.2249, 0.5692]`; this small sample does not support a significance claim.

## Evidence

| record | local source | W&B |
| --- | --- | --- |
| validation selection | `outputs/medium_seed52_reproduction/validation_*/metrics.json` | see `docs/wandb_tracking.md` |
| locked persistence test | `outputs/medium_seed52_reproduction/test_persistence_locked/metrics.json` | see `docs/wandb_tracking.md` |
| locked transformer test | `outputs/medium_seed52_reproduction/test_transformer_cache_normalized_locked/metrics.json` | see `docs/wandb_tracking.md` |

The final W&B group is listed in `docs/wandb_tracking.md`. Each run records the cache,
checkpoints, split seed, manifest, horizon, aggregation, selected trajectory/window
counts, Git state, concise scalar metrics, and resolved-config/metrics artifacts.

## Boundaries

- The former mixed-seed `9.2270` result is invalidated and excluded.
- The seed-62 pretrained-SFT run is a separate protocol and is not in this comparison.
- Results cover five validation and five test trajectories; no multi-seed confidence
  interval is available.
- Encoder, cache, and sequence artifacts identify commit `f97a0257d7627c8ff8960433aed30c750a9f90d5`.
- Final test evaluation identifies commit `280540f54e67c0dbcae253327596bfaf7cbf9307`
  and the SHA-256 of an empty tracked diff; the untracked thesis source is packaged separately.
