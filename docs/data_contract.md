# Data Contract

Snapshots use channel-first layout:

```text
x: float32[C, S1, S2, S3, S4, S5]
batch x: float32[B, C, S1, S2, S3, S4, S5]
```

Trajectory datasets expose IDs, per-trajectory timesteps, a snapshot shape, and
`get_snapshot(trajectory_id, timestep_index)`. Model and training code should depend on
that interface rather than HDF5 internals.

Snapshot collation requires consistent diagnostic availability across a batch. A mixture
of present and missing flux targets, or inconsistent spectra keys, is rejected instead of
silently dropping targets. Dataset and trajectory normalization statistics pool all
sampled elements into numerically stable population moments, including variation between
snapshots, while retaining only running scalar state in memory. Training-time dataset
statistics must be restricted to the declared training trajectory IDs; validation and
test trajectories must not contribute to fitted preprocessing statistics.
Fixed normalization accepts either scalar statistics or one value per channel. Channel
lists are reshaped against the channel-first axis as `[C, 1, 1, 1, 1, 1]`; mismatched
channel counts are rejected before training.

The generic HDF5 adapter assumes one HDF5 file per trajectory and config-driven dataset
paths under `data_group`, `metadata_group`, `flux_key`, and `spectra_keys`. `input_fields`
selects which field templates are stacked on the channel axis; `df` uses
`timestep_key_template`, while `phi` uses `phi_key_template`. HDF5 read conversion accepts
NumPy `float16`, `float32`, or `float64`; samples are converted to the internal float32
contract before leaving the adapter. NumPy-backed HDF5 ingestion does not accept
`bfloat16` as a configured conversion dtype.

The Cyclone/KvikIO adapter is optional and config-gated behind `data.backend:
cyclone_kvikio`. The primary server path reads the confirmed directory layout directly;
an upstream `CycloneDataset` package can still be provided for compatibility checks. Both
paths map samples into the same internal contract:

```text
upstream df             -> SnapshotSample.x
upstream flux/avg_flux/y_flux -> DiagnosticTargets.flux
upstream timestep/time  -> SnapshotSample.physical_time
upstream file_index     -> SnapshotSample.trajectory_index
upstream timestep_index -> SnapshotSample.timestep_index
upstream conditioning   -> SnapshotSample.metadata["conditioning"]
```

The validated server dataset path uses `input_fields: [df]`. Potential-field `phi`
loading remains supported for generic HDF5/upstream-adapter experiments, but it is not
part of the selected Cyclone/KvikIO evidence unless a future config explicitly
enables and verifies it.

Cyclone snapshots must still resolve to channel-first
`float32[C, S1, S2, S3, S4, S5]`; batched tensors remain
`float32[B, C, S1, S2, S3, S4, S5]`. Quantized shards may be detected during inspection, but
training receives float32 arrays. Spectra are not guessed. The adapter has explicit
stored, computed-placeholder, and no-spectra providers. Stored spectra can come from
sample keys or time-aligned metadata arrays. On the validated server schema, `kyspec` and
`fluxspec` are stored in metadata with shape `(T, 32)`, and real smoke configs request
both targets. Computed spectra raise a clear not-implemented error until the physics
definition is requested. Inspection can still write a flux/input report with a
missing-spectra warning, while training fails clearly instead of silently treating absent
spectra as successful targets.

Generated synthetic HDF5 fixtures follow:

```text
data/timestep_00000
metadata/timesteps
metadata/fluxes
metadata/ky_spectrum
metadata/q_spectrum
```

When an alternate HDF5 schema is supplied, the fixture writer uses its configured time,
flux, and spectra paths. Synthetic dataset adapters expose only the diagnostic targets
requested by `data.target_flux` and `data.target_spectra`, matching the real-data adapter
contract.

Latent caches are HDF5 files written after encoder training:

```text
metadata/created_at
metadata/latent_dim
metadata/config_yaml
metadata/encoder_checkpoint_path
trajectories/<trajectory_id>/z
trajectories/<trajectory_id>/timestep_index
trajectories/<trajectory_id>/physical_time
trajectories/<trajectory_id>/flux
trajectories/<trajectory_id>/spectra/<name>
```

Sequence training and rollout evaluation consume `latent_cache.path`. Rollout evaluation
uses `latent_cache.sequence_checkpoint_path` unless `use_persistence_baseline` is set.
