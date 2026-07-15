# W&B Tracking

Weights & Biases tracking is optional and disabled by default. Local checks and CI must
not require a W&B account, login, or network access.

Enable tracking only for explicit experiment runs:

```bash
uv run gks train-sequence \
  --config configs/experiment/server_sequence_transformer_small.yaml \
  --override logging.wandb.enabled=true \
  --override logging.wandb.mode=online \
  --override logging.wandb.project=gk-latent-surrogate \
  --override logging.wandb.group=small-scale-latent-surrogate \
  --override logging.wandb.name=cyclone-small-causal-transformer-training \
  --override logging.wandb.tags='[cyclone,small,sequence-training,causal-transformer]'
```

Use offline mode when the server cannot reach W&B:

```bash
uv run gks train-sequence \
  --config configs/experiment/server_sequence_transformer_small.yaml \
  --override logging.wandb.enabled=true \
  --override logging.wandb.mode=offline \
  --override logging.wandb.directory='${GK_FAST_ROOT}/wandb'
```

When tracking is active, commands write `wandb_status.json` under the run output
directory. CLI summaries include either the run URL, the offline run directory, or an
explicit warning if W&B could not be initialized.

The project treats W&B as run evidence, not as the only artifact store. Metrics,
checkpoints, resolved configs, device reports, plots, and cache paths must still be
available from normal output directories.

All stages in one comparison must use the same `data.seed`; this seed defines the
trajectory split and is distinct from `training.seed`. A run is not comparable when its
encoder, cache, sequence checkpoint, or evaluation uses another split seed. Record the
split label, split seed, selected-trajectory manifest digest, rollout horizon,
normalization protocol, and aggregation semantics in every result. Mark superseded or
protocol-invalid runs explicitly instead of silently reusing their metrics.

Use scientific names derived from dataset, scale, model, and stage. Groups should identify
the comparison set; run names should identify the evaluated method and split or stage;
tags should be stable facets such as `cyclone`, `medium`, `rollout`, or `gru`. Do not use
people, meetings, review events, implementation phases, dates, or informal status labels
in groups, names, tags, plot filenames, or report titles.

## Published Medium Reproduction

The published encoder-to-evaluation reproduction uses group
`medium-seed52-reproduction`. It contains exactly three records:

- five-candidate validation selection: [`m52rval1`](https://wandb.ai/v-yelisieiev-johannes-kepler-universit-t-linz/gk-latent-surrogate/runs/m52rval1);
- locked persistence test: [`m52rper1`](https://wandb.ai/v-yelisieiev-johannes-kepler-universit-t-linz/gk-latent-surrogate/runs/m52rper1);
- locked cache-normalized Transformer test: [`m52rtrn1`](https://wandb.ai/v-yelisieiev-johannes-kepler-universit-t-linz/gk-latent-surrogate/runs/m52rtrn1).

These runs originate from `outputs/medium_seed52_reproduction`. They record split seed
`52`, train/cache/evaluation manifests, the reproduced encoder/cache/checkpoints, horizon
`8`, trajectory-balanced aggregation, evaluation sizes, commit
`f97a0257d7627c8ff8960433aed30c750a9f90d5`, and the empty tracked-diff digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Each run has one
evaluation artifact containing its metrics, resolved configuration, and Git metadata.

Validation selected the cache-normalized Transformer as the strongest learned candidate;
persistence retained the lowest validation flux RMSE. On their shared locked-test
manifest, the selected Transformer reports flux RMSE `11.425478` versus persistence
`11.988926`, a `4.70%` lower point estimate.

The earlier `medium-seed52-heldout-evaluation` records are tagged and marked superseded
by this full reproduction. Historical records in `medium-scale-latent-surrogate` retain
their invalidated, descriptive-only, or standalone status.

The reproduction still covers one split seed, five validation trajectories, five test
trajectories, and 40 rollout windows per evaluation split; it does not provide a
multi-seed confidence interval. Git metadata reports untracked paths, including the
separate thesis tree, but the tracked source diff is empty.
