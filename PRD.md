# PRD / Technical Specification: JAX Latent Time-Series Surrogate for 5D Gyrokinetic Plasma Turbulence

**Document status:** Scientific and technical scope reference
**Target repository:** `gk-latent-surrogate-jax`
**Target package:** `gk_surrogate`
**Primary language:** Python 3.11+
**Primary ML stack:** JAX, Flax, Optax
**Execution mode:** Server GPU thesis workflow with portable local fallback checks
**Source of truth:** This specification defines the original scientific and technical scope

> Status note: this PRD records the initial thesis scope and constraints.
> Current behavior is defined by the local code, configs, tests, and the lean operational
> docs under `docs/`.

---

## 0. Executive summary

This PRD defines a complete implementation plan for a new clean repository for a bachelor thesis project on **machine-learning latent time-series surrogates for 5D gyrokinetic plasma turbulence**.

The implementation targets the server GPU thesis workflow while keeping a portable local
fallback path for validation. The repository must be designed so that the same codebase
can run on:

- shared compute server with the existing 5D dataset and 4x GTX 1080 Ti for dataset inspection,
  preprocessing, and training;
- PC with RTX 5070 under WSL2 for optional local GPU training;
- MacBook or CI fallback runner for development, tests, and tiny smoke runs.

The project direction defined by this specification is:

```text
5D time-slice x_t
  -> representation / embedding model
  -> latent vector z_t
  -> latent sequence model
  -> autoregressive rollout z_{t+1}, z_{t+2}, ...
  -> diagnostic heads for scalar flux and 1D spectra
  -> evaluation against latent and physics-diagnostic metrics
  -> comparison framework for GyroSwin-style metrics
```

The implementation must not depend on the previous NDSwin-JAX practical-work repository. That repository may be mined later for ideas or reusable fragments, but the bachelor codebase should be a focused, clean, thesis-specific system.

At the current stage the repository must be fully implementable with:

- synthetic 5D trajectory data;
- generated mini-HDF5 fixtures for tests;
- hardware-agnostic JAX training steps;
- complete model/training/evaluation infrastructure;
- placeholders and generic adapters for the real dataset schema.

The only expected future work after this PRD is implemented should be:

1. Bind the real dataset schema once the server path, HDF5/KvikIO layout, field shapes, and spectra target definitions are known.
2. Decide final training hardware by benchmarking the same code on PC and server.
3. Tune model scale, batch size, and dataset-specific augmentations.

---

## 1. Product definition

### 1.1 Product name

Recommended repository name:

```text
gk-latent-surrogate-jax
```

Rationale:

- `gk` = gyrokinetic;
- `latent-surrogate` = core thesis objective;
- `jax` = non-negotiable implementation framework;
- the name does not overcommit to Swin, GPT-2, GyroSwin, or a specific architecture.

Python package name:

```text
gk_surrogate
```

CLI entrypoint:

```text
gks
```

Example commands:

```bash
gks inspect-data --config configs/data/tiny_dummy.yaml
gks train-encoder --config configs/experiment/smoke_encoder_supervised.yaml
gks train-encoder --config configs/experiment/smoke_encoder_simsiam.yaml
gks embed-dataset --config configs/experiment/smoke_embed_dataset.yaml
gks train-sequence --config configs/experiment/smoke_sequence.yaml
gks evaluate-rollout --config configs/experiment/smoke_evaluate_rollout.yaml
```

### 1.2 Product goal

Build a JAX/Flax research codebase that supports the full thesis pipeline:

1. Load or generate 5D gyrokinetic time-slice data.
2. Train an encoder that maps high-dimensional 5D snapshots to compact latent vectors.
3. Train diagnostic heads that predict scalar flux and 1D spectra from the latent representation.
4. Embed entire trajectories into latent space.
5. Train a latent sequence model that autoregressively predicts future latents.
6. Evaluate latent rollout stability and downstream physics diagnostic quality.
7. Provide a clean route for comparison to GyroSwin-style metrics once the real dataset and reference metrics are available.

### 1.3 Product non-goals

The initial repository implementation must **not** attempt to solve these items immediately:

- no hard dependency on PC, server, CUDA version, or GPU model;
- no required multi-GPU JAX implementation in v0;
- no immediate full 5D reconstruction objective, because the scientific goal is to avoid 5D reconstruction;
- no full GyroSwin reimplementation;
- no mandatory GPT-2 fine-tuning in the first implementation;
- no mandatory use of the old NDSwin-JAX repo;
- no mandatory local generation of the full dataset before the codebase exists;
- no assumption about real dataset schema until exact paths and shapes are verified.

### 1.4 Key thesis deliverables represented in code

The repository must make the project deliverables concrete:

| Project deliverable | Repository implementation |
|---|---|
| Embedding model / SimSiam / contrastive encoder | `gk_surrogate.models.encoders`, `gk_surrogate.models.simsiam`, `gk_surrogate.training.train_encoder` |
| Dataset embedding into latent space | `gk_surrogate.training.embed_dataset`, latent cache format |
| Sequence model with stable time rollouts | `gk_surrogate.models.sequence`, `gk_surrogate.training.train_sequence`, `gk_surrogate.evaluation.rollout` |
| Diagnostic prediction from latent | `gk_surrogate.models.diagnostics`, diagnostic losses and metrics |
| Metrics for latent rollout, flux, spectra | `gk_surrogate.metrics.*`, `gk_surrogate.evaluation.*` |
| Compare to GyroSwin | metric compatibility layer and result tables, not a GyroSwin clone |
| Setup local or server | config-based data roots and environment-specific profiles |
| Positive/negative pair definitions | augmentation module and SimSiam pair builder |
| Pretrained autoencoder question | adapter interface, not required in v0 |

---

## 2. Core design principles

### 2.1 GPU-first server path with portable fallback

The primary experiment path targets the server GPU environment. JAX/Flax runs should
resolve to GPU execution and single-host `pmap` automatically when multiple devices are
visible, while local and CI fallback checks must still run without GPU hardware.

Rules:

- no hardcoded CUDA wheel, GPU count, `pjit`, or device count assumptions in core modules;
- no hardcoded absolute dataset paths;
- all path and hardware behavior must come from config or CLI overrides;
- portable smoke tests must pass in CI and on a local CPU workstation;
- GPU-specific environment setup must live in documentation and optional environment files, not core code;
- server experiment configs should use automatic parallel mode unless a constrained
  fallback run is explicitly documented.

### 2.2 Shape explicitness

All modules must document and validate tensor shapes. This is critical because the real data are 5D and will likely be expensive to debug.

Global shape convention:

```text
Single snapshot:
  x: float32[C, S1, S2, S3, S4, S5]

Batch of snapshots:
  x: float32[B, C, S1, S2, S3, S4, S5]

Trajectory batch for encoder training:
  x: float32[B, C, S1, S2, S3, S4, S5]
  diagnostics:
    flux: float32[B, F]
    spectra: dict[str, float32[B, K_name]]

Latent batch:
  z: float32[B, Z]

Latent sequence batch:
  z_context: float32[B, T_context, Z]
  z_target: float32[B, T_target, Z]

Predicted latent rollout:
  z_hat: float32[B, T_rollout, Z]
```

The repository must default to channel-first layout because the referenced neural-gyrokinetics loader returns fields in a channel-oriented layout and because channel-first is natural for N-dimensional scientific arrays.

### 2.3 Two-level data abstraction

The implementation must separate:

1. **Raw trajectory data access**: reading snapshots and diagnostics from synthetic arrays, HDF5 fixtures, or real server data.
2. **Training dataset construction**: sampling individual snapshots, SimSiam pairs, latent sequences, or rollout windows.

This prevents the model/training code from depending on raw dataset layout.

### 2.4 Complete synthetic pipeline before real data

Before the real dataset is known, the repo must already support:

- synthetic 5D trajectory generation;
- deterministic train/validation/test splits;
- encoder training on synthetic diagnostics;
- SimSiam loss computation on synthetic augmentations;
- latent caching from synthetic trajectories;
- latent sequence training;
- autoregressive rollout evaluation;
- shape/unit tests for all modules.

This synthetic pipeline is not a scientific result. It is an engineering fixture proving that the full code path is correct.

### 2.5 Small baseline before complex model

The first complete implementation must include simple baselines:

- direct diagnostic baseline: `x_t -> flux/spectra`;
- latent persistence: `z_{t+1} = z_t`;
- linear or MLP latent dynamics;
- small GRU/LSTM-like sequence model;
- small causal Transformer sequence model.

Large Swin/GPT-2-style models can be added later, but the repository must be scientifically usable even if those are not implemented.

### 2.6 JAX purity boundaries

Data loading is allowed to use normal Python, NumPy, and h5py. JAX should be used for model computation, losses, train steps, and evaluation kernels.

Rules:

- do not JIT file I/O;
- do not load HDF5 inside a `jax.jit` function;
- data loaders should return NumPy arrays or JAX arrays at the boundary;
- train steps should accept JAX arrays and be `jax.jit`-able.

---

## 3. Repository structure

The repository must be created with this structure:

```text
gk-latent-surrogate-jax/
  README.md
  AGENTS.md
  PRD.md
  pyproject.toml
  Makefile
  .gitignore
  .pre-commit-config.yaml

  configs/
    data/
      tiny_dummy.yaml
      local_pc_template.yaml
      student_server_template.yaml
      h5_template.yaml
    model/
      encoder_mlp_tiny.yaml
      encoder_patch_transformer_tiny.yaml
      simsiam_tiny.yaml
      sequence_mlp_delta.yaml
      sequence_gru_tiny.yaml
      sequence_transformer_tiny.yaml
    experiment/
      smoke_encoder_supervised.yaml
      smoke_encoder_simsiam.yaml
      smoke_embed_dataset.yaml
      smoke_sequence.yaml
      smoke_evaluate_rollout.yaml

  scripts/
    inspect_cyclone_layout.py
    validate_latent_cache.py
    verify_agent_setup.py

  src/
    gk_surrogate/
      __init__.py
      cli.py

      config/
        __init__.py
        schema.py
        load.py
        validate.py

      data/
        __init__.py
        types.py
        base.py
        synthetic.py
        h5_schema.py
        h5_loader.py
        inspect.py
        normalization.py
        split.py
        latent_cache.py
        sequence_dataset.py
        augmentations.py
        collate.py

      models/
        __init__.py
        encoders.py
        patching.py
        diagnostics.py
        simsiam.py
        sequence.py
        full_models.py

      losses/
        __init__.py
        simsiam.py
        diagnostics.py
        latent.py
        total.py

      metrics/
        __init__.py
        diagnostics.py
        latent.py
        rollout.py
        aggregate.py

      training/
        __init__.py
        state.py
        optimizer.py
        train_encoder.py
        embed_dataset.py
        train_sequence.py
        loops.py
        checkpointing.py
        logging.py
        rng.py

      evaluation/
        __init__.py
        rollout.py
        diagnostics.py
        reports.py

      utils/
        __init__.py
        tree.py
        arrays.py
        paths.py
        timing.py
        pretty.py

  tests/
    conftest.py
    test_config.py
    test_synthetic_data.py
    test_h5_schema_loader.py
    test_data_inspection.py
    test_augmentations.py
    test_encoder_shapes.py
    test_diagnostic_heads.py
    test_simsiam_loss.py
    test_sequence_models.py
    test_latent_cache.py
    test_train_encoder_step.py
    test_train_sequence_step.py
    test_rollout_eval.py
    test_cli_smoke.py
    test_reproducibility.py
    test_no_hardware_assumptions.py

  docs/
    data_contract.md
    hardware_profiles.md
    experiment_lifecycle.md
    metrics.md
    real_data_binding_checklist.md
    server_gpu_setup.md
    verification_matrix.md

  data/
    README.md

  outputs/
    .gitkeep
```

The `data/`, `outputs/`, `checkpoints/`, `runs/`, `wandb/`, and raw data files must be gitignored.

---

## 4. Dependency and tooling specification

### 4.1 Required runtime dependencies

The default installation must be CPU-safe and MacBook-safe.

Minimum runtime dependencies:

```text
jax
flax
optax
numpy
h5py
PyYAML
pydantic or dataclasses-based validation
rich
matplotlib
```

Notes:

- Use JAX CPU installation by default.
- Do not pin CUDA-specific JAX wheels in the base install.
- GPU-specific installation instructions should be documented in `docs/hardware_profiles.md` and validated later on PC/server.
- Do not make PyTorch a required dependency.

### 4.2 Development dependencies

```text
pytest
pytest-cov
ruff
mypy
pre-commit
hypothesis optional
```

### 4.3 Optional dependencies

```text
wandb          optional experiment tracking
orbax-checkpoint optional checkpointing backend
zarr           optional scalable latent cache backend
```

Orbax can be included if checkpointing is clean and lightweight. If it creates version pain, use msgpack/NumPy checkpointing first and add Orbax later.

### 4.4 Makefile contract

The Makefile must expose:

```bash
make install
make install-dev
make test
make test-fast
make lint
make format
make type-check
make check
make smoke-encoder
make smoke-simsiam
make smoke-sequence
make smoke-all
make clean
```

Expected behavior:

- `make test-fast` runs CPU-safe unit tests excluding slow tests.
- `make check` runs lint, type-check, and tests.
- `make smoke-all` executes the complete synthetic pipeline with tiny configs.

### 4.5 CLI contract

The package must expose a CLI entrypoint `gks` through `pyproject.toml`.

CLI subcommands:

```text
gks inspect-data
gks train-encoder
gks embed-dataset
gks train-sequence
gks evaluate-rollout
gks make-synthetic-h5
gks benchmark-step-time
```

Each CLI command must support:

```bash
--config path/to/config.yaml
--override key=value
--dry-run
--seed 42
--output-dir outputs/...
```

Dry-run requirements:

- validate config;
- print resolved config;
- instantiate dataset/model if possible;
- print expected shapes;
- do not train or write checkpoints.

---

## 5. Configuration system

### 5.1 General requirements

The config system must be explicit, validated, and easy to override from CLI.

Recommended approach:

- YAML files for user-facing configs;
- Python dataclasses or Pydantic models for validation;
- resolved config printed at start of every run;
- no hidden defaults that affect scientific results without logging.

### 5.2 Core config objects

Implement these config schemas in `src/gk_surrogate/config/schema.py`:

```python
@dataclass
class DataConfig:
    backend: Literal["synthetic", "h5"]
    root: str | None
    split: Literal["train", "val", "test", "all"]
    input_fields: tuple[str, ...]
    target_flux: bool
    target_spectra: tuple[str, ...]
    context_length: int
    prediction_length: int
    batch_size: int
    shuffle: bool
    num_workers: int
    seed: int
    normalization: NormalizationConfig
    h5_schema: H5SchemaConfig | None
    synthetic: SyntheticDataConfig | None
```

```python
@dataclass
class ModelConfig:
    encoder: EncoderConfig
    diagnostics: DiagnosticHeadConfig
    simsiam: SimSiamConfig | None
    sequence: SequenceModelConfig | None
```

```python
@dataclass
class TrainingConfig:
    max_steps: int
    epochs: int | None
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_clip_norm: float | None
    log_every: int
    eval_every: int
    checkpoint_every: int
    dtype: Literal["float32", "bfloat16", "float16"]
    jit: bool
    seed: int
```

```python
@dataclass
class LossConfig:
    simsiam_weight: float
    flux_weight: float
    spectra_weight: float
    latent_weight: float
    use_log_spectra: bool
    spectra_epsilon: float
    latent_loss: Literal["mse", "huber", "cosine"]
```

```python
@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
```

### 5.3 Config validation rules

The config loader must reject invalid configs before training starts:

- `batch_size > 0`;
- `context_length >= 1`;
- `prediction_length >= 1`;
- if `backend == "h5"`, then `root` and `h5_schema` must be present;
- if `backend == "synthetic"`, then `synthetic` must be present;
- encoder `latent_dim > 0`;
- spectra loss enabled only if spectra targets are present;
- SimSiam weight enabled only if SimSiam model config is present;
- sequence model config required for `train-sequence`;
- output directory must not be inside raw dataset directory.

### 5.4 Environment-specific config files

`configs/data/tiny_dummy.yaml`:

```yaml
data:
  backend: synthetic
  root: null
  split: train
  input_fields: [df]
  target_flux: true
  target_spectra: [ky, q]
  context_length: 4
  prediction_length: 1
  batch_size: 2
  shuffle: true
  num_workers: 0
  seed: 42
  normalization:
    mode: dataset
    mean: null
    std: null
  synthetic:
    num_trajectories: 4
    timesteps: 16
    channels: 2
    spatial_shape: [4, 4, 4, 4, 4]
    flux_dim: 1
    spectra_dims:
      ky: 8
      q: 8
```

`configs/data/local_pc_template.yaml`:

```yaml
data:
  backend: h5
  root: ${GK_LOCAL_PC_DATA_ROOT}
  split: train
  input_fields: [df]
  target_flux: true
  target_spectra: [ky, q]
  context_length: 8
  prediction_length: 1
  batch_size: 1
  shuffle: true
  num_workers: 0
  seed: 42
  h5_schema:
    trajectory_glob: "*.h5"
    data_group: "data"
    timestep_key_template: "timestep_{t:05d}"
    phi_key_template: "poten_{t:05d}"
    metadata_group: "metadata"
    flux_key: "fluxes"
    timestep_key: "timesteps"
    spectra_keys: {}
```

`configs/data/student_server_template.yaml`:

```yaml
data:
  backend: h5
  root: /path/to/server/dataset
  split: train
  input_fields: [df]
  target_flux: true
  target_spectra: [ky, q]
  context_length: 8
  prediction_length: 1
  batch_size: 1
  shuffle: true
  num_workers: 0
  seed: 42
  h5_schema:
    trajectory_glob: "*.h5"
    data_group: "data"
    timestep_key_template: "timestep_{t:05d}"
    phi_key_template: "poten_{t:05d}"
    metadata_group: "metadata"
    flux_key: "fluxes"
    timestep_key: "timesteps"
    spectra_keys: {}
```

---

## 6. Data layer specification

### 6.1 Data types

Create `src/gk_surrogate/data/types.py`.

Required dataclasses:

```python
@dataclass(frozen=True)
class DiagnosticTargets:
    flux: np.ndarray | jax.Array | None
    spectra: Mapping[str, np.ndarray | jax.Array]
```

```python
@dataclass(frozen=True)
class SnapshotSample:
    x: np.ndarray
    targets: DiagnosticTargets
    trajectory_id: str
    trajectory_index: int
    timestep_index: int
    physical_time: float | None
    metadata: Mapping[str, Any]
```

```python
@dataclass(frozen=True)
class SnapshotBatch:
    x: jax.Array
    flux: jax.Array | None
    spectra: Mapping[str, jax.Array]
    trajectory_index: jax.Array
    timestep_index: jax.Array
```

```python
@dataclass(frozen=True)
class LatentSample:
    z: np.ndarray
    targets: DiagnosticTargets
    trajectory_id: str
    timestep_index: int
    physical_time: float | None
```

```python
@dataclass(frozen=True)
class LatentSequenceBatch:
    z_context: jax.Array
    z_target: jax.Array
    flux_target: jax.Array | None
    spectra_target: Mapping[str, jax.Array]
    trajectory_index: jax.Array
    start_timestep_index: jax.Array
```

Rules:

- All arrays must use `float32` unless explicitly configured otherwise.
- Metadata must not be required by models in v0.
- Conditioning parameters can be added later through a separate `conditioning` field.

### 6.2 Base dataset interfaces

Create `src/gk_surrogate/data/base.py`.

Required protocols/abstract classes:

```python
class TrajectoryDataset(Protocol):
    def trajectory_ids(self) -> Sequence[str]: ...
    def num_trajectories(self) -> int: ...
    def num_timesteps(self, trajectory_id: str) -> int: ...
    def snapshot_shape(self) -> tuple[int, ...]: ...
    def get_snapshot(self, trajectory_id: str, timestep_index: int) -> SnapshotSample: ...
```

```python
class SnapshotBatchIterator(Protocol):
    def __iter__(self) -> Iterator[SnapshotBatch]: ...
    def __len__(self) -> int: ...
```

```python
class LatentTrajectoryDataset(Protocol):
    def trajectory_ids(self) -> Sequence[str]: ...
    def get_latent(self, trajectory_id: str, timestep_index: int) -> LatentSample: ...
```

The training code must depend only on these interfaces, not on HDF5 internals.

### 6.3 Synthetic dataset

Create `src/gk_surrogate/data/synthetic.py`.

Purpose:

- enable full pipeline on MacBook without real data;
- provide deterministic tests;
- give training loops something to overfit.

Implementation requirements:

- Generate trajectories with configurable shape:

```text
num_trajectories
num_timesteps
channels
spatial_shape length = 5
flux_dim
spectra_dims mapping
seed
```

- Use deterministic pseudo-physics so diagnostics are learnable:

```text
x_t = smooth random latent process projected into 5D tensor + noise
flux_t = weighted mean / energy-like statistic of x_t + nonlinear term
ky_spectrum_t = binned power-like statistic or synthetic function of latent state
q_spectrum_t = another correlated 1D target
```

- The synthetic target should not be pure random noise. Smoke training must be able to reduce loss over a few dozen steps.

- Store no data on disk unless `make-synthetic-h5` is called.

Required tests:

- reproducibility with same seed;
- different seeds produce different data;
- output shapes match config;
- diagnostics are finite;
- split generation is deterministic.

### 6.4 HDF5 schema abstraction

Create `src/gk_surrogate/data/h5_schema.py`.

Purpose:

Real dataset schema is not known yet, but the code can support generic HDF5 layouts through a config-driven schema.

Required config dataclass:

```python
@dataclass(frozen=True)
class H5SchemaConfig:
    trajectory_glob: str
    data_group: str
    timestep_key_template: str
    phi_key_template: str | None
    metadata_group: str
    flux_key: str | None
    timestep_key: str | None
    spectra_keys: Mapping[str, str]
    geometry_group: str | None
    channel_indices: tuple[int, ...] | None
    dtype: str
```

Example mapping:

```yaml
h5_schema:
  trajectory_glob: "*.h5"
  data_group: "data"
  timestep_key_template: "timestep_{t:05d}"
  phi_key_template: "poten_{t:05d}"
  metadata_group: "metadata"
  flux_key: "fluxes"
  timestep_key: "timesteps"
  spectra_keys:
    ky: "metadata/ky_spectrum"
    q: "metadata/q_spectrum"
  geometry_group: "geometry"
  channel_indices: [0, 1]
  dtype: float32
```

### 6.5 HDF5 loader

Create `src/gk_surrogate/data/h5_loader.py`.

Required behavior:

- Discover files via `root / trajectory_glob`.
- Treat each file as one trajectory unless config later says otherwise.
- Read snapshot keys via `timestep_key_template`.
- Read flux via metadata key if present.
- Read spectra via configured paths if present.
- If spectra keys are missing, raise a clear error only when spectra are requested.
- Support channel selection through `channel_indices`.
- Return channel-first arrays.
- Avoid loading the whole dataset into memory.
- Use context-managed `h5py.File` access.

Important: The first HDF5 implementation must support generated mini-HDF5 fixtures in tests. It does not need to perfectly match the real dataset until real schema is known.

Required tests:

- create temporary HDF5 trajectory file;
- verify file discovery;
- verify snapshot loading;
- verify flux loading;
- verify spectra loading;
- verify missing spectra error;
- verify channel selection;
- verify dtype conversion.

### 6.6 Dataset inspection

Create `src/gk_surrogate/data/inspect.py` and expose it through `gks inspect-data`.

The inspection tool must print:

```text
Dataset root
Backend
Number of trajectory files
Trajectory IDs
Timesteps per selected trajectories
Available HDF5 groups and datasets
First snapshot key
First snapshot shape
First snapshot dtype
Flux key availability
Flux shape and basic stats if available
Spectra key availability
Spectra shapes and basic stats if available
Metadata keys
Geometry keys if available
Estimated bytes per snapshot
Estimated bytes per trajectory
Estimated bytes per batch for configured batch size
Recommended initial batch size estimate
Warnings for missing targets
```

Command:

```bash
gks inspect-data --config configs/data/student_server_template.yaml --max-trajectories 2 --max-depth 3
```

Acceptance criteria:

- Must work on synthetic data.
- Must work on generated mini-HDF5 fixtures.
- Must not require GPU.
- Must not load full dataset unless explicitly requested.

### 6.7 Normalization

Create `src/gk_surrogate/data/normalization.py`.

Supported modes:

```text
none
sample
trajectory
dataset
fixed
```

Initial implementation requirements:

- `none`: return input unchanged.
- `sample`: normalize each snapshot by its own mean/std over spatial dimensions and channels.
- `fixed`: use provided mean/std from config.
- `dataset`: compute mean/std over a bounded sample of snapshots, not full dataset by default.
- store computed stats in output directory as JSON or NPZ.

Do not implement expensive full-dataset stats by default.

### 6.8 Data splitting

Create `src/gk_surrogate/data/split.py`.

Splitting strategy:

- Default split by trajectory, not by random timestep, to avoid temporal leakage.
- Configurable ratios: train/val/test.
- Deterministic with seed.
- Allow explicit trajectory ID lists later.

Required tests:

- deterministic split;
- no overlap between train/val/test trajectory IDs;
- single-trajectory edge case handled with warning or explicit failure.

### 6.9 SimSiam augmentations

Create `src/gk_surrogate/data/augmentations.py`.

Purpose:

Produce two positive views of the same 5D snapshot.

Required augmentations:

1. Gaussian noise:

```text
x' = x + sigma * std(x) * epsilon
```

2. Random element masking:

```text
mask selected elements or blocks with probability p
```

3. Channel dropout:

```text
randomly zero or attenuate channels with probability p
```

4. Periodic spatial shift:

```text
roll along selected spatial axes
```

5. Amplitude jitter:

```text
x' = scale * x + bias
```

Initial safety rules:

- Augmentations must preserve shape.
- Augmentations must be optional and configured individually.
- Default smoke config should use mild noise and mild amplitude jitter only.
- Periodic shifts should be disabled by default until the physical validity of each spatial axis is confirmed.
- Adjacent timesteps should not be used as positive pairs by default.

Required tests:

- deterministic with PRNG key;
- shape preservation;
- finite outputs;
- no mutation of input arrays;
- identity behavior when all augmentation probabilities are zero.

### 6.10 Latent cache format

Create `src/gk_surrogate/data/latent_cache.py`.

Purpose:

After encoder training, embed full trajectories into a compact latent dataset for sequence model training.

Initial cache backend:

```text
HDF5 latent cache
```

Recommended file layout:

```text
latent_cache.h5
  metadata/
    config_yaml
    encoder_checkpoint_path
    latent_dim
    created_at
  trajectories/
    <trajectory_id>/
      z                  float32[T, latent_dim]
      timestep_index     int32[T]
      physical_time      float32[T] optional
      flux               float32[T, F] optional
      spectra/
        ky               float32[T, K_ky]
        q                float32[T, K_q]
```

Requirements:

- support writing one trajectory at a time;
- support reading sequence windows;
- verify latent dimension consistency;
- preserve diagnostic targets.

Required tests:

- write/read roundtrip;
- multiple trajectories;
- missing diagnostics handled cleanly;
- sequence window extraction.

---

## 7. Model architecture specification

### 7.1 Encoder interface

Create `src/gk_surrogate/models/encoders.py`.

All encoders must implement:

```python
class Encoder(nn.Module):
    latent_dim: int

    def __call__(self, x: Array, *, train: bool) -> Array:
        """Return z with shape [B, latent_dim]."""
```

Input shape:

```text
x: [B, C, S1, S2, S3, S4, S5]
```

Output shape:

```text
z: [B, latent_dim]
```

### 7.2 Required encoder models

#### 7.2.1 FlattenMLPEncoder

Purpose:

- simplest baseline;
- useful for tiny synthetic shapes;
- not intended for real large 5D fields.

Architecture:

```text
x -> flatten -> Dense -> activation -> Dense -> activation -> Dense(latent_dim)
```

Config:

```yaml
encoder:
  type: flatten_mlp
  latent_dim: 64
  hidden_dims: [256, 128]
  activation: gelu
  dropout_rate: 0.0
```

#### 7.2.2 ConvNDEncoder

Purpose:

- simple N-dimensional local encoder;
- more realistic than flatten MLP;
- should work with Flax `nn.Conv` over 5 spatial dimensions.

Architecture:

```text
x channel-first -> move channels last
ConvND -> activation -> optional pooling/downsample
ConvND -> activation -> global average pooling -> Dense(latent_dim)
```

Config:

```yaml
encoder:
  type: conv_nd
  latent_dim: 128
  channels: [16, 32, 64]
  kernel_size: [3, 3, 3, 3, 3]
  strides: [[1,1,1,1,1], [2,2,2,2,2], [2,2,2,2,2]]
  activation: gelu
```

#### 7.2.3 PatchTransformerEncoder

Purpose:

- modern architecture without implementing full Swin;
- feasible starting point for 5D snapshots;
- can serve as the first serious representation-learning backbone.

Architecture:

```text
x [B,C,S1..S5]
  -> N-D patch embedding via Conv with kernel_size=patch_size and strides=patch_size
  -> tokens [B,N_tokens,D]
  -> learned or sinusoidal position embedding
  -> Transformer encoder blocks
  -> global average pool or CLS token
  -> Dense(latent_dim)
```

Config:

```yaml
encoder:
  type: patch_transformer
  latent_dim: 128
  patch_size: [2, 2, 2, 2, 2]
  embed_dim: 64
  depth: 2
  num_heads: 4
  mlp_ratio: 4.0
  dropout_rate: 0.0
  attention_dropout_rate: 0.0
  use_cls_token: false
```

Memory guardrails:

- Reject configs where token count is too high unless `allow_large_token_count: true`.
- Print token count during dry-run.
- For 5D data, token count can explode quickly.

#### 7.2.4 OptionalEncoderAdapter

Create an optional adapter interface for future reuse of NDSwin or pretrained autoencoders, but do not require implementation in v0.

```python
class ExternalEncoderAdapter(nn.Module):
    name: str
    latent_dim: int
```

Initial behavior:

- raise `NotImplementedError` with a clear message;
- include config placeholder but do not use in smoke tests.

### 7.3 Diagnostic heads

Create `src/gk_surrogate/models/diagnostics.py`.

Required module:

```python
class DiagnosticHeads(nn.Module):
    flux_dim: int
    spectra_dims: Mapping[str, int]
    hidden_dims: tuple[int, ...]
    dropout_rate: float

    def __call__(self, z: Array, *, train: bool) -> DiagnosticPredictions:
        ...
```

Output dataclass:

```python
@dataclass(frozen=True)
class DiagnosticPredictions:
    flux: Array | None
    spectra: Mapping[str, Array]
```

Architecture:

- shared MLP trunk optional;
- one Dense output for flux;
- separate Dense output per spectrum key.

Config:

```yaml
diagnostics:
  flux_dim: 1
  spectra_dims:
    ky: 8
    q: 8
  hidden_dims: [128]
  dropout_rate: 0.0
```

### 7.4 SimSiam heads

Create `src/gk_surrogate/models/simsiam.py`.

Required modules:

```python
class ProjectionHead(nn.Module):
    output_dim: int
    hidden_dim: int
    num_layers: int
```

```python
class PredictionHead(nn.Module):
    output_dim: int
    hidden_dim: int
```

```python
class SimSiamModel(nn.Module):
    encoder: nn.Module
    projection_head: ProjectionHead
    prediction_head: PredictionHead
```

Behavior:

```text
view1 -> encoder -> z1 -> projection p1 -> prediction q1
view2 -> encoder -> z2 -> projection p2 -> prediction q2
loss = -0.5 * cosine(q1, stop_gradient(p2))
       -0.5 * cosine(q2, stop_gradient(p1))
```

Rules:

- stop-gradient must be used on the target branch;
- normalization before cosine similarity required;
- batch size can be small for smoke tests;
- no negative pairs required for SimSiam.

### 7.5 Combined encoder model

Create `src/gk_surrogate/models/full_models.py`.

Required module:

```python
class EncoderWithDiagnostics(nn.Module):
    encoder: nn.Module
    diagnostic_heads: DiagnosticHeads | None

    def __call__(self, x: Array, *, train: bool) -> EncoderOutput:
        z = self.encoder(x, train=train)
        diagnostics = self.diagnostic_heads(z, train=train) if ... else None
        return EncoderOutput(z=z, diagnostics=diagnostics)
```

Output:

```python
@dataclass(frozen=True)
class EncoderOutput:
    z: Array
    diagnostics: DiagnosticPredictions | None
```

### 7.6 Sequence models

Create `src/gk_surrogate/models/sequence.py`.

All sequence models must implement:

```python
class LatentSequenceModel(nn.Module):
    latent_dim: int

    def __call__(self, z_context: Array, *, train: bool) -> Array:
        """Return next latent(s)."""
```

Input:

```text
z_context: [B, T_context, Z]
```

Output:

```text
For one-step training:
  z_pred: [B, Z]

For multi-step direct training:
  z_pred: [B, T_target, Z]
```

#### 7.6.1 PersistenceBaseline

No trainable parameters.

```text
z_hat_{t+1} = z_t
```

Used only for evaluation baseline.

#### 7.6.2 MLPDeltaSequenceModel

Architecture:

```text
flatten context -> MLP -> delta_z
z_pred = last_z + delta_z
```

Config:

```yaml
sequence:
  type: mlp_delta
  latent_dim: 128
  context_length: 4
  hidden_dims: [256, 256]
```

#### 7.6.3 GRUSequenceModel

Architecture:

```text
z_context -> GRU scan -> final hidden -> Dense(latent_dim)
```

Config:

```yaml
sequence:
  type: gru
  latent_dim: 128
  hidden_dim: 256
  num_layers: 1
```

If Flax GRU implementation causes friction, implement a minimal custom GRU cell in JAX/Flax.

#### 7.6.4 CausalTransformerSequenceModel

Architecture:

```text
z_context -> Dense(embed_dim)
          -> positional embedding
          -> causal self-attention blocks
          -> last token -> Dense(latent_dim)
```

Config:

```yaml
sequence:
  type: causal_transformer
  latent_dim: 128
  context_length: 8
  embed_dim: 128
  depth: 2
  num_heads: 4
  mlp_ratio: 4.0
  dropout_rate: 0.0
```

GPT-2 fine-tuning is not required in v0. Add only an interface placeholder:

```text
sequence.type = gpt2_adapter
```

with `NotImplementedError` until the project scope explicitly requires it.

### 7.7 Rollout function

Create `src/gk_surrogate/evaluation/rollout.py`.

Required function:

```python
def autoregressive_rollout(
    model_apply: Callable,
    params: PyTree,
    z_initial_context: Array,
    rollout_steps: int,
    *,
    train: bool = False,
) -> Array:
    """Return z_hat with shape [B, rollout_steps, Z]."""
```

Behavior:

- use model prediction recursively;
- append prediction to context;
- keep context length fixed by dropping oldest latent;
- support JIT.

Required tests:

- shape correct;
- persistence baseline reproduces last state;
- no mutation;
- deterministic outputs with same params/input.

---

## 8. Loss functions

### 8.1 SimSiam loss

Create `src/gk_surrogate/losses/simsiam.py`.

Function:

```python
def negative_cosine_similarity(p: Array, z_stop: Array, eps: float = 1e-8) -> Array:
    ...
```

```python
def simsiam_loss(p1: Array, z2: Array, p2: Array, z1: Array) -> Array:
    ...
```

Requirements:

- normalize vectors;
- use `jax.lax.stop_gradient` outside or inside loss consistently;
- return scalar mean over batch;
- finite for zero vectors due to epsilon.

### 8.2 Diagnostic losses

Create `src/gk_surrogate/losses/diagnostics.py`.

Flux loss:

```python
def flux_mse(pred: Array, target: Array) -> Array:
    ...
```

Spectra loss:

```python
def spectra_mse(
    pred: Mapping[str, Array],
    target: Mapping[str, Array],
    *,
    log_space: bool,
    eps: float,
) -> Array:
    ...
```

Additional optional losses:

- relative L2;
- MAE;
- Huber.

Rules:

- If a target is missing and its weight is zero, skip it.
- If a target is missing and its weight is nonzero, raise a clear error.
- Use log spectra only with positive clipping: `log(max(x, 0) + eps)`.

### 8.3 Latent dynamics loss

Create `src/gk_surrogate/losses/latent.py`.

Supported losses:

```text
mse
huber
cosine
mse_plus_cosine
```

Function:

```python
def latent_prediction_loss(pred: Array, target: Array, mode: str) -> Array:
    ...
```

### 8.4 Total loss composition

Create `src/gk_surrogate/losses/total.py`.

Encoder training loss:

```text
L_encoder = w_simsiam * L_simsiam
          + w_flux * L_flux
          + w_spectra * L_spectra
```

Sequence training loss:

```text
L_sequence = w_latent * L_latent
           + optional w_flux * L_flux(predicted_latent -> diagnostic_heads)
           + optional w_spectra * L_spectra(predicted_latent -> diagnostic_heads)
```

For v0, sequence diagnostic loss is optional. The main sequence objective is latent prediction.

---

## 9. Metrics specification

### 9.1 Latent metrics

Create `src/gk_surrogate/metrics/latent.py`.

Required metrics:

```text
latent_mse
latent_mae
latent_relative_l2
latent_cosine_similarity
rollout_mse_by_step
rollout_cosine_by_step
```

### 9.2 Diagnostic metrics

Create `src/gk_surrogate/metrics/diagnostics.py`.

Flux metrics:

```text
flux_mse
flux_rmse
flux_mae
flux_relative_error
time_average_flux_error
```

Spectra metrics:

```text
spectra_mse
spectra_log_mse
spectra_relative_l2
spectra_pearson_corr
spectra_shape_corr
spectra_mean_absolute_relative_error
```

### 9.3 Rollout metrics

Create `src/gk_surrogate/metrics/rollout.py`.

Required rollout evaluation:

```text
per-step latent error
per-step diagnostic error
horizon until error threshold
stability flag: finite outputs for all rollout steps
mean and std over trajectories
```

For GyroSwin comparison, exact metric definitions should remain configurable until the final comparison protocol is established.

### 9.4 Metrics aggregation

Create `src/gk_surrogate/metrics/aggregate.py`.

Requirements:

- aggregate per-batch metrics into epoch metrics;
- save JSON metrics;
- save CSV table for plotting;
- support nested metric names like `spectra/ky/relative_l2`.

---

## 10. Training implementation

### 10.1 Train state

Create `src/gk_surrogate/training/state.py`.

Use Flax `train_state.TrainState` or a custom dataclass.

Required fields:

```python
step: int
params: PyTree
opt_state: PyTree
apply_fn: Callable
rng: PRNGKey
model_config: Mapping[str, Any]
```

Optional:

```python
batch_stats
ema_params
```

Do not implement EMA in v0 unless easy.

### 10.2 Optimizer

Create `src/gk_surrogate/training/optimizer.py`.

Use Optax.

Required features:

- AdamW;
- optional gradient clipping;
- optional warmup + cosine decay;
- constant LR fallback.

### 10.3 Encoder training

Create `src/gk_surrogate/training/train_encoder.py`.

Training modes:

```text
supervised_diagnostics
simsiam_only
simsiam_with_diagnostics
```

Required train step signature:

```python
@jax.jit
def train_encoder_step(state: TrainState, batch: SnapshotBatch) -> tuple[TrainState, dict[str, Array]]:
    ...
```

For SimSiam modes:

- generate augmented views outside or inside train step using PRNG;
- if inside train step, store/update PRNG in state;
- return metrics for SimSiam loss, flux loss, spectra loss, total loss.

Validation step:

```python
@jax.jit
def eval_encoder_step(state: TrainState, batch: SnapshotBatch) -> dict[str, Array]:
    ...
```

Requirements:

- works with synthetic data on CPU;
- supports `max_steps` independent of epochs;
- logs every `log_every` steps;
- saves checkpoint every `checkpoint_every` steps;
- writes resolved config to output dir;
- writes metrics JSON/CSV.

### 10.4 Dataset embedding

Create `src/gk_surrogate/training/embed_dataset.py`.

Functionality:

- load trained encoder checkpoint;
- iterate through each trajectory in order;
- compute `z_t` for each timestep;
- write latent cache HDF5;
- include diagnostic targets in cache;
- store encoder config and checkpoint reference.

Command:

```bash
gks embed-dataset --config configs/experiment/smoke_embed_dataset.yaml
```

Acceptance criteria:

- works on synthetic dataset;
- latent cache can be read by sequence training;
- output HDF5 shape documented and validated.

### 10.5 Sequence training

Create `src/gk_surrogate/training/train_sequence.py`.

Inputs:

- latent cache;
- context length;
- prediction length;
- sequence model config.

Required train step:

```python
@jax.jit
def train_sequence_step(state: TrainState, batch: LatentSequenceBatch) -> tuple[TrainState, dict[str, Array]]:
    ...
```

Requirements:

- support MLPDelta, GRU, CausalTransformer;
- support persistence baseline evaluation without training;
- save checkpoint and metrics;
- validate latent shapes before training.

### 10.6 Evaluation

Create `src/gk_surrogate/evaluation/reports.py` and expose rollout evaluation through
`gks evaluate-rollout`.

Evaluation workflow:

1. Load latent cache.
2. Load sequence checkpoint.
3. For each trajectory/window, run autoregressive rollout for configured horizon.
4. Compare predicted latents to true latents.
5. Optionally pass predicted latents through diagnostic heads if available.
6. Compute metrics by rollout step and aggregate.
7. Save:
   - `metrics.json`;
   - `metrics_by_step.csv`;
   - basic plots: latent MSE by step, flux error by step, spectra error by step.

Plots should use matplotlib and avoid hardcoded colors unless explicitly requested.

---

## 11. Checkpointing and outputs

### 11.1 Output directory layout

Each run should create:

```text
outputs/
  <experiment_name>/
    <timestamp_or_run_id>/
      config_resolved.yaml
      git_info.json
      metrics.json
      metrics.csv
      checkpoints/
        step_000001/
        step_000100/
      plots/
      latent_cache.h5 optional
      logs.txt
```

### 11.2 Checkpoint requirements

For v0:

- checkpoint model params;
- checkpoint optimizer state if training can resume;
- checkpoint current step;
- checkpoint model config;
- support loading for inference/evaluation.

Do not overengineer checkpointing. A simple Flax serialization or Orbax-based implementation is acceptable if tested.

### 11.3 Git info

At run start, save:

```text
git commit hash if available
whether working tree is dirty
command-line args
resolved config
```

If git is unavailable, continue and log warning.

---

## 12. CLI command details

The supported user-facing entrypoint is the `gks` CLI. The `scripts/` directory is
reserved for unique maintenance utilities that are not normal pipeline commands.

### 12.1 `gks make-synthetic-h5`

Purpose:

- generate a mini-HDF5 dataset matching the generic schema;
- use in tests and manual HDF5 loader debugging.

Command:

```bash
gks make-synthetic-h5 \
  --config configs/data/tiny_dummy.yaml \
  --output-dir outputs/synthetic_h5
```

Output:

```text
data/synthetic_h5/traj_000.h5
...
```

### 12.2 `gks inspect-data`

Purpose:

- inspect synthetic/HDF5 dataset;
- print shape and target availability;
- estimate memory.

### 12.3 `gks train-encoder`

Purpose:

- train encoder on snapshots;
- supports supervised diagnostics and SimSiam.

### 12.4 `gks embed-dataset`

Purpose:

- create latent cache from trained encoder.

### 12.5 `gks train-sequence`

Purpose:

- train sequence model on latent cache.

### 12.6 `gks evaluate-rollout`

Purpose:

- evaluate autoregressive rollout.

### 12.7 `gks benchmark-step-time`

Purpose:

- later compare PC vs server with same config.

Output:

```text
device list
model name
batch size
input shape
compile time
mean step time after warmup
max memory if available
```

This command is optional for hardware comparison work and is not required for core tests.

---

## 13. Test plan

### 13.1 Test philosophy

Testing must focus on:

- shape correctness;
- deterministic behavior;
- hardware agnosticism;
- no hidden dependency on real data;
- JIT compatibility;
- complete smoke pipeline.

Coverage target:

```text
Minimum: 80% line coverage for core modules
Preferred: 85%+
```

### 13.2 Test categories

#### Unit tests

Fast tests for individual functions/classes.

#### Integration tests

Tiny end-to-end tests using synthetic or mini-HDF5 data.

#### CLI smoke tests

Run CLI commands with temporary output directories.

#### Reproducibility tests

Same seed should produce same synthetic data and same initial model outputs.

#### Portable fallback tests

Ensure fallback tests pass without GPU hardware and no local test requires a server GPU.

### 13.3 Required test files

#### `test_config.py`

Assertions:

- valid config loads;
- invalid backend rejected;
- missing HDF5 schema rejected when backend is h5;
- CLI override changes resolved config;
- config roundtrip serialization works.

#### `test_synthetic_data.py`

Assertions:

- dataset length and shapes;
- deterministic same seed;
- finite diagnostics;
- learnable synthetic flux is not constant;
- train/val split no overlap.

#### `test_h5_schema_loader.py`

Assertions:

- mini-HDF5 generated in temp dir;
- loader discovers trajectories;
- reads snapshot;
- reads flux;
- reads spectra;
- missing spectra error is clear;
- channel selection works.

#### `test_data_inspection.py`

Assertions:

- inspect synthetic returns expected keys;
- inspect HDF5 returns shapes;
- memory estimates positive;
- no full dataset load required.

#### `test_augmentations.py`

Assertions:

- all augmentations preserve shape;
- zero probabilities produce identity;
- same PRNG gives same result;
- outputs finite.

#### `test_encoder_shapes.py`

Assertions:

- FlattenMLPEncoder outputs `[B, Z]`;
- ConvNDEncoder outputs `[B, Z]`;
- PatchTransformerEncoder outputs `[B, Z]`;
- invalid spatial dimensions rejected;
- JIT apply works.

#### `test_diagnostic_heads.py`

Assertions:

- flux output shape;
- spectra output shapes for multiple keys;
- no spectra config returns empty spectra dict;
- JIT apply works.

#### `test_simsiam_loss.py`

Assertions:

- loss finite;
- identical vectors produce lower loss than random vectors;
- stop-gradient path exists by code behavior where practical;
- zero vector safe due to epsilon.

#### `test_sequence_models.py`

Assertions:

- persistence baseline shape;
- MLPDelta shape;
- GRU shape;
- Transformer shape;
- autoregressive rollout shape;
- JIT rollout works.

#### `test_latent_cache.py`

Assertions:

- write/read latent cache;
- metadata persisted;
- sequence windows extracted correctly;
- inconsistent latent dimensions rejected.

#### `test_train_encoder_step.py`

Assertions:

- one train step changes params;
- loss finite;
- eval step no param change;
- supervised and SimSiam modes both run on synthetic batch.

#### `test_train_sequence_step.py`

Assertions:

- one train step changes params;
- latent loss finite;
- sequence model learns one synthetic batch enough to reduce loss over several steps.

#### `test_rollout_eval.py`

Assertions:

- metrics JSON produced;
- by-step metrics shape correct;
- persistence baseline can be evaluated;
- finite outputs.

#### `test_cli_smoke.py`

Assertions:

- `gks inspect-data` works with dummy config;
- `gks train-encoder --max-steps 2` works;
- `gks embed-dataset` works after training;
- `gks train-sequence --max-steps 2` works;
- `gks evaluate-rollout` works.

Use temporary directories. Do not write to repository root.

#### `test_reproducibility.py`

Assertions:

- same seed gives same initialized params where applicable;
- same synthetic data and same config gives same first loss;
- different seed changes data or model init.

#### `test_no_hardware_assumptions.py`

Assertions:

- importing package does not require GPU;
- configs do not hardcode `/mnt/d`, server paths, or CUDA;
- portable fallback smoke tests can run with `JAX_PLATFORM_NAME=cpu`.

### 13.4 CI requirements

GitHub Actions should run:

```bash
make install-dev
make lint
make type-check
make test-fast
```

No GPU CI required.

---

## 14. Documentation requirements

### 14.1 README.md

README must include:

- project summary;
- relation to the scientific project specification;
- installation for Mac CPU dev;
- how to run synthetic smoke pipeline;
- repository structure;
- how to add real dataset config;
- what is not implemented yet;
- next hardware decision steps.

### 14.2 AGENTS.md

AGENTS.md must be written for coding agents and future self.

It must specify:

- use `make` and `gks` entrypoints;
- never commit raw data/checkpoints;
- keep core code hardware-agnostic;
- run tests after changes;
- do not introduce PyTorch dependency unless explicitly needed;
- do not implement multi-GPU before single-GPU baseline works;
- maintain shape comments and tests.

### 14.3 docs/data_contract.md

Must describe:

- raw snapshot shape convention;
- target shape convention;
- HDF5 schema config;
- latent cache format;
- dataset inspection workflow;
- known unknowns for real data.

### 14.4 docs/hardware_profiles.md

Must describe:

- portable fallback development profile;
- PC WSL2 GPU profile template;
- shared-server profile template;
- how to benchmark step time;
- how to choose final hardware based on evidence.

No exact CUDA/JAX wheel prescription should be treated as permanent. The document should say to validate environment-specific JAX installation at setup time.

### 14.5 docs/experiment_lifecycle.md

Must describe:

```text
inspect data
train encoder
embed dataset
train sequence model
evaluate rollout
compare metrics
```

### 14.6 docs/metrics.md

Must define all implemented metrics and how they map to project deliverables.

### 14.7 Lean operational docs

The current repository keeps operational docs rather than draft thesis prose:

- `docs/real_data_binding_checklist.md` records dataset/schema questions and binding
  checks before real-data runs.
- `docs/server_gpu_setup.md` records GPU/KvikIO setup and server execution notes.
- `docs/verification_matrix.md` records the local and CI gates for code, config, data,
  training, evaluation, packaging, and agent-readiness changes.

Open scientific and protocol questions include:

- Are 1D spectra stored in the dataset or must they be computed?
- Which spectra are required: `ky`, `Q`, both, or others?
- Should diagnostic heads predict same-time or future-time quantities?
- Is SimSiam specifically preferred, or is another self-supervised objective acceptable?
- What augmentations are physically valid for each dimension?
- Is a pretrained autoencoder expected or merely optional?
- What exactly counts as comparison to GyroSwin: same metrics, same dataset, same rollout
  horizon, or direct baseline numbers?

---

## 15. Implementation milestones

### Repository bootstrap

Deliverables:

- create repo structure;
- `pyproject.toml`;
- `Makefile`;
- `.gitignore`;
- `README.md`;
- `AGENTS.md`;
- basic CLI skeleton;
- config loader;
- tests import package.

Acceptance:

```bash
make install-dev
make test-fast
```

passes on a portable fallback runner without GPU hardware.

### Synthetic data and HDF5 fixture layer

Deliverables:

- synthetic trajectory dataset;
- mini-HDF5 generator;
- generic HDF5 schema loader;
- data inspection command;
- normalization and splitting utilities.

Acceptance:

```bash
gks inspect-data --config configs/data/tiny_dummy.yaml
gks make-synthetic-h5 --config configs/data/tiny_dummy.yaml --output-dir /tmp/gk_h5
```

works, and data tests pass.

### Encoder and diagnostic heads

Deliverables:

- FlattenMLPEncoder;
- ConvNDEncoder;
- PatchTransformerEncoder;
- DiagnosticHeads;
- supervised encoder training step;
- metrics and logging.

Acceptance:

```bash
gks train-encoder --config configs/experiment/smoke_encoder_supervised.yaml
```

runs for a few steps on CPU and decreases or at least reports finite loss.

### SimSiam representation learning

Deliverables:

- augmentations;
- SimSiam heads;
- SimSiam loss;
- SimSiam training mode;
- diagnostic auxiliary loss mode.

Acceptance:

```bash
gks train-encoder --config configs/experiment/smoke_encoder_simsiam.yaml
```

runs on CPU, returns finite SimSiam and diagnostic metrics.

### Latent cache

Deliverables:

- encoder checkpoint loading;
- latent cache HDF5 writer/reader;
- embed dataset command;
- sequence window dataset.

Acceptance:

```bash
gks embed-dataset --config configs/experiment/smoke_embed_dataset.yaml
```

produces `latent_cache.h5` with correct shape.

### Sequence models and rollout

Deliverables:

- persistence baseline;
- MLPDeltaSequenceModel;
- GRUSequenceModel;
- CausalTransformerSequenceModel;
- latent train step;
- autoregressive rollout.

Acceptance:

```bash
gks train-sequence --config configs/experiment/smoke_sequence.yaml
gks evaluate-rollout --config configs/experiment/smoke_evaluate_rollout.yaml
```

works on synthetic latent cache.

### Full smoke pipeline

Deliverables:

- `make smoke-all` runs:
  1. synthetic data inspection;
  2. encoder training;
  3. embedding;
  4. sequence training;
  5. rollout evaluation.

Acceptance:

```bash
make smoke-all
```

passes on a portable fallback runner without real data or GPU hardware.

### Real dataset readiness

Deliverables:

- `student_server_template.yaml` prepared;
- `local_pc_template.yaml` prepared;
- inspect tool ready;
- open scientific and protocol questions documented;
- benchmark tool ready.

Acceptance:

Once dataset path is known, only config/schema edits should be required before first real data inspection.

---

## 16. Real dataset integration plan

When server dataset details are available and verified, execute:

### Step 1 - Inspect raw file structure

On server:

```bash
gks inspect-data --config configs/data/student_server_template.yaml --dry-run
```

If schema is wrong, inspect manually:

```python
import h5py
with h5py.File(path, "r") as f:
    print(list(f.keys()))
```

Update `h5_schema` config only. Avoid changing model/training code unless absolutely necessary.

### Step 2 - Confirm shapes

Record:

```text
df shape
phi shape
flux shape
spectra target names and shapes
number of trajectories
timesteps per trajectory
bytes per snapshot
```

### Step 3 - Create tiny real subset

Options:

- symbolic config selecting 1-2 trajectories and 20-100 timesteps;
- copied subset to PC;
- generated HDF5 subset if allowed.

### Step 4 - Run overfit test

Train encoder on tiny subset until it overfits. This verifies loader, targets, and training.

### Step 5 - Benchmark hardware

Run identical benchmark on:

```text
PC RTX 5070 WSL2
server 1x1080 Ti
```

Only consider 4-GPU server training after single-GPU baseline works.

---

## 17. Hardware decision protocol

The repository must not encode the final hardware decision. It must provide tools to make the decision empirically.

### 17.1 Benchmark criteria

For each candidate hardware setup, measure:

```text
JAX installation difficulty
whether all tests pass
whether data loading works
first compile time
steady-state training step time
maximum stable batch size
GPU memory usage if available
data loading throughput
checkpoint write speed
stability over 100-1000 steps
```

### 17.2 Expected likely outcome

Do not encode this as a requirement, but the likely decision path is:

```text
Develop on a portable local workstation
Inspect data on the shared server
Train initial real models on a compatible local GPU if data access permits
Use the shared server for the data archive, preprocessing, repeat runs, and optional parallel sweeps
```

### 17.3 Server multi-GPU caution

The repository must not require multi-GPU training in v0 because:

- four 1080 Ti GPUs do not provide one shared memory pool;
- JAX multi-GPU requires explicit parallelism/sharding decisions;
- compatibility issues are more likely on old GPUs;
- thesis progress should not depend on distributed training infrastructure.

A later milestone may add `pmap`/`pjit` if needed.

---

## 18. Scientific experiment plan encoded by the repo

The codebase should support the following experiment ladder.

### Experiment A - Direct diagnostic baseline

Question:

```text
Can simple supervised models predict flux/spectra from 5D snapshots?
```

Pipeline:

```text
x_t -> encoder -> z_t -> diagnostic heads -> flux/spectra
```

Purpose:

- validate data and targets;
- establish baseline;
- provide auxiliary loss for representation learning.

### Experiment B - SimSiam encoder with diagnostic heads

Question:

```text
Can self-supervised representation learning produce a latent space that preserves physics diagnostics?
```

Pipeline:

```text
view1(x_t), view2(x_t) -> shared encoder -> SimSiam loss
z_t -> diagnostic heads -> flux/spectra
```

### Experiment C - Latent sequence model

Question:

```text
Can latent dynamics predict future embeddings over multiple timesteps?
```

Pipeline:

```text
z_{t-k+1}, ..., z_t -> sequence model -> z_{t+1}
```

### Experiment D - Autoregressive rollout

Question:

```text
How stable is the latent surrogate when recursively rolled forward?
```

Pipeline:

```text
initial latent context -> sequence model rollout -> z_hat_{future}
```

Metrics:

- per-step latent error;
- diagnostic error from predicted latent;
- stability horizon;
- comparison to persistence baseline.

### Experiment E - Architecture comparison

Once real data work:

```text
FlattenMLP / ConvND / PatchTransformer / optional NDSwin-style encoder
MLPDelta / GRU / CausalTransformer sequence model
```

Do not run broad sweeps until the end-to-end pipeline is stable.

---

## 19. Risk register

### Risk 1 - Real dataset schema unknown

Mitigation:

- generic HDF5 schema config;
- inspection script;
- synthetic and mini-HDF5 tests;
- data contract documentation.

### Risk 2 - 5D data too large for 12 GB / 11 GB GPU

Mitigation:

- small encoders;
- patching/downsampling configs;
- batch size 1 support;
- latent cache stage;
- no full reconstruction objective;
- memory estimates during dry-run.

### Risk 3 - SimSiam collapse or unhelpful representation

Mitigation:

- diagnostic auxiliary heads;
- supervised diagnostic baseline;
- representation variance metrics optional;
- compare to simple encoders.

### Risk 4 - Sequence rollout unstable

Mitigation:

- persistence baseline;
- one-step and multi-step metrics;
- small sequence models first;
- teacher-forcing training before autoregressive evaluation;
- rollout horizon curves.

### Risk 5 - JAX/CUDA compatibility problems

Mitigation:

- portable fallback tests;
- no hardcoded GPU behavior;
- benchmark script;
- PC/server profiles separated;
- do not require 4-GPU training.

### Risk 6 - Scope creep into GyroSwin clone or GPT-2 project

Mitigation:

- GyroSwin comparison is metric-level initially;
- GPT-2 only adapter placeholder in v0;
- main deliverable remains latent surrogate pipeline.

### Risk 7 - Old NDSwin repo drags in irrelevant assumptions

Mitigation:

- new repo from scratch;
- copy only small, reviewed pieces if needed;
- no dependency on old repo.

---

## 20. Acceptance criteria for the full current-stage implementation

When this PRD is fully implemented, the following must be true.

### 20.1 Development acceptance

```bash
make install-dev
make lint
make type-check
make test-fast
make smoke-all
```

passes on a portable fallback runner without GPU hardware.

### 20.2 Code acceptance

- Package imports without GPU.
- All public modules have docstrings.
- All model classes validate shapes or fail clearly.
- All CLI commands support `--config` and `--dry-run`.
- No raw data paths are hardcoded.
- No old practical-work repo is required.
- No PyTorch dependency is required.

### 20.3 Data acceptance

- Synthetic dataset supports full pipeline.
- Mini-HDF5 fixture supports loader and inspect tests.
- Generic HDF5 schema config exists for real data binding.
- Latent cache read/write works.

### 20.4 Model acceptance

- At least three encoder options implemented:
  - FlattenMLP;
  - ConvND;
  - PatchTransformer.
- Diagnostic heads implemented.
- SimSiam heads and loss implemented.
- At least three sequence options implemented:
  - persistence baseline;
  - MLPDelta;
  - GRU or CausalTransformer.
- Autoregressive rollout implemented.

### 20.5 Training acceptance

- Encoder training step JITs and runs.
- SimSiam training step JITs and runs.
- Dataset embedding creates latent cache.
- Sequence training step JITs and runs.
- Evaluation creates metrics and plots.

### 20.6 Testing acceptance

- All required test files exist.
- CLI smoke tests use temp dirs.
- Portable fallback tests can be run with `JAX_PLATFORM_NAME=cpu`.
- No test requires server or PC GPU.

---

## 21. Dependency-ordered implementation sequence

Use this exact order to avoid getting blocked.

### Repository structure and configuration

1. Create repo.
2. Add `pyproject.toml`, `Makefile`, `.gitignore`.
3. Add package skeleton and `cli.py`.
4. Add config schema and loader.
5. Add first tests for config and import.

### Data layer

1. Implement data types.
2. Implement synthetic dataset.
3. Implement synthetic batch iterator.
4. Implement mini-HDF5 generator.
5. Implement HDF5 schema loader.
6. Implement inspect command.
7. Add tests.

### Model layer

1. Implement FlattenMLPEncoder.
2. Implement DiagnosticHeads.
3. Implement supervised encoder train step.
4. Add ConvNDEncoder.
5. Add PatchTransformerEncoder.
6. Add tests.

### SimSiam representation learning

1. Implement augmentations.
2. Implement projection/prediction heads.
3. Implement SimSiam loss.
4. Implement SimSiam train mode.
5. Add tests.

### Latent cache and sequence modeling

1. Implement latent cache.
2. Implement embed dataset command.
3. Implement sequence window dataset.
4. Implement sequence models.
5. Implement train sequence.
6. Implement rollout evaluation.
7. Add tests.

### Integration and verification

1. Add docs.
2. Add full smoke pipeline.
3. Add benchmark script.
4. Add CI.
5. Run `make check`.

---

## 22. Suggested initial configs

### 22.1 Smoke supervised encoder

```yaml
name: smoke_encoder_supervised
output_dir: outputs

data:
  backend: synthetic
  split: train
  input_fields: [df]
  target_flux: true
  target_spectra: [ky, q]
  context_length: 1
  prediction_length: 1
  batch_size: 2
  shuffle: true
  num_workers: 0
  seed: 42
  synthetic:
    num_trajectories: 4
    timesteps: 16
    channels: 2
    spatial_shape: [4, 4, 4, 4, 4]
    flux_dim: 1
    spectra_dims:
      ky: 8
      q: 8

model:
  encoder:
    type: conv_nd
    latent_dim: 32
    channels: [8, 16]
    kernel_size: [3, 3, 3, 3, 3]
    strides:
      - [1, 1, 1, 1, 1]
      - [2, 2, 2, 2, 2]
  diagnostics:
    flux_dim: 1
    spectra_dims:
      ky: 8
      q: 8
    hidden_dims: [64]

training:
  max_steps: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_steps: 0
  gradient_clip_norm: 1.0
  log_every: 1
  eval_every: 5
  checkpoint_every: 10
  dtype: float32
  jit: true
  seed: 42

loss:
  simsiam_weight: 0.0
  flux_weight: 1.0
  spectra_weight: 1.0
  latent_weight: 0.0
  use_log_spectra: true
  spectra_epsilon: 1.0e-6
```

### 22.2 Smoke SimSiam encoder

```yaml
name: smoke_encoder_simsiam
output_dir: outputs

data:
  backend: synthetic
  split: train
  input_fields: [df]
  target_flux: true
  target_spectra: [ky, q]
  context_length: 1
  prediction_length: 1
  batch_size: 2
  shuffle: true
  num_workers: 0
  seed: 42
  synthetic:
    num_trajectories: 4
    timesteps: 16
    channels: 2
    spatial_shape: [4, 4, 4, 4, 4]
    flux_dim: 1
    spectra_dims:
      ky: 8
      q: 8
  augmentations:
    gaussian_noise_std: 0.01
    amplitude_jitter_std: 0.02
    mask_probability: 0.0
    channel_dropout_probability: 0.0
    periodic_shift: false

model:
  encoder:
    type: conv_nd
    latent_dim: 32
    channels: [8, 16]
  simsiam:
    projection_dim: 64
    projection_hidden_dim: 64
    projection_layers: 2
    prediction_hidden_dim: 32
  diagnostics:
    flux_dim: 1
    spectra_dims:
      ky: 8
      q: 8
    hidden_dims: [64]

training:
  max_steps: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_steps: 0
  gradient_clip_norm: 1.0
  log_every: 1
  eval_every: 5
  checkpoint_every: 10
  dtype: float32
  jit: true
  seed: 42

loss:
  simsiam_weight: 1.0
  flux_weight: 0.1
  spectra_weight: 0.1
  latent_weight: 0.0
  use_log_spectra: true
  spectra_epsilon: 1.0e-6
```

### 22.3 Smoke sequence

```yaml
name: smoke_sequence
output_dir: outputs

latent_cache:
  path: outputs/smoke_embed_dataset/latent_cache.h5

data:
  context_length: 4
  prediction_length: 1
  batch_size: 2
  shuffle: true
  seed: 42

model:
  sequence:
    type: mlp_delta
    latent_dim: 32
    context_length: 4
    hidden_dims: [64, 64]

training:
  max_steps: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_steps: 0
  gradient_clip_norm: 1.0
  log_every: 1
  eval_every: 5
  checkpoint_every: 10
  dtype: float32
  jit: true
  seed: 42

loss:
  latent_weight: 1.0
  latent_loss: mse
```

---

## 23. Definition of done for the MacBook stage

The current stage is done when the repository can be cloned on a MacBook and the following sequence works without real data:

```bash
git clone git@github.com:volodymyr-yelisieiev/gk-latent-surrogate-jax.git
cd gk-latent-surrogate-jax
make install-dev
make check
make smoke-all
```

Expected outputs:

```text
- synthetic dataset inspection printed;
- encoder trained for tiny number of steps;
- checkpoint written;
- latent_cache.h5 written;
- sequence model trained for tiny number of steps;
- rollout metrics written;
- plots written;
- all tests passed.
```

At that point, the project is ready for:

```text
- real dataset schema binding;
- subset copy from server to PC;
- PC/server benchmark;
- thesis experiments.
```

---

## 24. Final concise implementation directive

Build a new clean repository named `gk-latent-surrogate-jax`. Implement the complete
latent surrogate pipeline in JAX/Flax for the server GPU thesis workflow, with synthetic
and mini-HDF5 fallback coverage. The code must be heavily tested and able to run
end-to-end on a portable fallback runner. Do not depend on the previous NDSwin-JAX repo or
hardcoded machine paths. After implementation, only environment-specific dataset schema
and hardware profile details should need adjustment.
