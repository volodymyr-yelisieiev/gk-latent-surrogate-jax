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

Use scientific names derived from dataset, scale, model, and stage. Groups should identify
the comparison set; run names should identify the evaluated method and split or stage;
tags should be stable facets such as `cyclone`, `medium`, `rollout`, or `gru`. Do not use
people, meetings, review events, implementation phases, dates, or informal status labels
in groups, names, tags, plot filenames, or report titles.

The verified medium-scale evidence group is `medium-scale-latent-surrogate`. Its active
runs and protocol boundaries are recorded in `docs/medium_guppy_experiment_report.md`.
