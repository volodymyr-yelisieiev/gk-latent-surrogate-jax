# Real Data Binding Checklist

Current state: the Cyclone/KvikIO binding is implemented for server-side experiments.
This checklist remains as a maintenance guide for future dataset variants and schema
checks.

Use this checklist after the real dataset location and access policy are confirmed. Keep
all paths in environment variables or config overrides; do not hardcode PC, server, or
personal filesystem paths in source code.

## 1. Confirm The Root And Schema

Record these facts before changing training configs:

- canonical dataset root, exported as `GK_CYCLONE_DATA_ROOT`;
- upstream `neugk_jax` or `neugk` checkout/package path that provides `CycloneDataset`;
- trajectory directory layout and whether samples are addressed by trajectory/timestep or flat index;
- presence of `metadata.pkl`, `metadata_light.pkl`, `data/timestep_*.bin`,
  `data/timestep_*.bf16.bin`, and `poten_*.bin`;
- exact upstream sample keys for `df`, optional `phi`, flux, timestep, file index, and conditioning;
- channel order and channel meanings;
- scalar flux dataset key, shape, and physical meaning;
- required spectra keys, shapes, and whether spectra are stored or must be computed;
- physical-time key and timestep spacing, if available.

Example validated server state from 2026-05-18:

- dataset root: exported as `GK_CYCLONE_DATA_ROOT`;
- optional AE checkpoint dir: exported as a local environment variable if needed;
- optional upstream package path: set `GK_NEUGK_UPSTREAM` only when testing the alternate
  upstream loader;
- first tiny trajectory: `iteration_0`, with `offset: 80` and `subsample: 32`.

Use the lightweight layout scanner before loading samples:

```bash
export GK_CYCLONE_DATA_ROOT=/path/to/preprocessed_kvikio
uv run python scripts/inspect_cyclone_layout.py \
  --max-trajectories 8 \
  --output outputs/real_data_inspection_tiny/layout_report.json
```

## 2. Inspect Without Training

Start with bounded inspection:

```bash
export GK_CYCLONE_DATA_ROOT=/path/to/preprocessed_kvikio
JAX_PLATFORM_NAME=cpu uv run gks inspect-data \
  --config configs/data/cyclone_kvikio_template.yaml \
  --dry-run \
  --max-trajectories 1 \
  --max-depth 4 \
  --max-target-samples 64
```

Then inspect a slightly larger slice:

```bash
JAX_PLATFORM_NAME=cpu uv run gks inspect-data \
  --config configs/data/cyclone_kvikio_template.yaml \
  --max-trajectories 1 \
  --max-depth 4 \
  --max-target-samples 5 \
  --output-dir outputs/real_data_inspection_tiny \
  --override 'data.cyclone.trajectories=["iteration_0"]' \
  --override data.cyclone.offset=80 \
  --override data.cyclone.subsample=32 \
  --override 'data.target_spectra=["kyspec","fluxspec"]'
```

The inspection output should include channel-first snapshot shape
`[B, C, S1, S2, S3, S4, S5]`, metadata keys, geometry keys, target shapes, flux
statistics, estimated sample count, whether KvikIO is enabled, whether BF16 shards are
detected, and warnings for missing or unconfirmed spectra targets.

## 3. Bind A Tiny Real Subset

The confirmed server dataset state is:

```text
snapshot_shape: (4, 32, 8, 16, 85, 32)
flux_shape: (1,)
stored spectra: kyspec (32), fluxspec (32)
geometry key count: 24
```

After schema inspection passes:

- keep the real dataset root read-only;
- keep `output_dir` and `latent_cache.path` outside the raw dataset root;
- start with `configs/experiment/smoke_real_encoder_flux.yaml`;
- use the confirmed stored spectra keys `kyspec` and `fluxspec` for this validated
  server schema;
- run encoder dry-runs and a two-step overfit test before full training;
- keep `JAX_PLATFORM_NAME=cpu` for shape validation unless explicitly benchmarking GPU.

The first real smoke sequence is:

```bash
JAX_PLATFORM_NAME=cpu uv run gks train-encoder \
  --config configs/experiment/smoke_real_encoder_flux.yaml

JAX_PLATFORM_NAME=cpu uv run gks embed-dataset \
  --config configs/experiment/smoke_real_embed_dataset.yaml

JAX_PLATFORM_NAME=cpu uv run gks train-sequence \
  --config configs/experiment/smoke_real_sequence.yaml

JAX_PLATFORM_NAME=cpu uv run gks evaluate-rollout \
  --config configs/experiment/smoke_real_evaluate_rollout.yaml
```

## 4. Benchmark Hardware With The Same Config

Use one config and one measured benchmark command across machines:

```bash
JAX_PLATFORM_NAME=cpu uv run gks benchmark-step-time \
  --config configs/experiment/smoke_real_encoder_flux.yaml \
  --measured-steps 3
```

For PC/server GPU tests, install the environment-specific JAX wheel in that environment
only, then run the same command and record:

- backend and visible devices;
- model config and batch size;
- mean, min, and max step time;
- maximum batch size that fits;
- dataloader or HDF5 throughput if real data are involved;
- whether KvikIO is active and whether BF16 shards are present.

Do not update `main` based on hardware-specific work until local gates and GitHub Actions
have both completed successfully, or an external Actions blocker has been reported.

## 5. Current Limitations To Record

- Tiny smoke uses CPU fallback and `use_kvikio: false`.
- The current direct reader trains from plain float32 `.bin` shards; BF16 shards are
  detected during inspection but not consumed by the training path.
- Flux is mapped from upstream `y_flux`, which is the next-timestep target for
  `bundle_seq_length=1`.
- Optional autoencoder checkpoint probing is separate from the main SimSiam/latent
  sequence path; it is loadable, but not required.
- Thesis-scale training target is still a measured hardware decision.

Keep current implementation notes in this checklist and in ignored inspection outputs.
Promote a separate report only when a later experiment/reporting ticket needs it.
