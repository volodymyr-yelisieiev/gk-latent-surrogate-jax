# Server GPU Setup

This repo is GPU-first for the server thesis workflow. Server GPU execution is controlled
through the runtime environment and the experiment-level `parallel` config block, while
local fallback checks stay portable.

## Environment

Use `uv` for the project environment, not the system Python:

```bash
cd gk-latent-surrogate-jax
uv sync --python 3.12 --extra dev
```

On GTX 1080 Ti-class GPUs, use the CUDA 12 JAX wheel path. The current JAX installation
guide says CUDA 12 supports NVIDIA GPUs with SM 5.2+, while CUDA 13 supports SM 7.5+ and
newer because older GPUs were dropped from CUDA 13. See the official JAX installation
guide: https://docs.jax.dev/en/latest/installation.html.

Install GPU/KvikIO packages in the server environment, not in `pyproject.toml`:

```bash
uv pip install -U "jax[cuda12]"
uv pip install -U cupy-cuda12x kvikio-cu12 ml-dtypes
```

Verify JAX before running training:

```bash
JAX_PLATFORM_NAME=gpu uv run python - <<'PY'
import jax
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
print("local_device_count", jax.local_device_count())
PY
```

## Parallel Config

The default is automatic device resolution:

```yaml
parallel:
  mode: auto
```

For server runs, keep the full parallel block explicit when documenting a run:

```yaml
parallel:
  mode: auto
  axis_name: devices
  min_devices: 1
  require_all_visible_devices: false
  drop_remainder: true
  auto_scale_learning_rate: false
  log_device_summary: true
```

`data.batch_size` is the global batch size. With four devices and `data.batch_size: 8`,
each device receives a per-device batch of 2. If the batch size is not divisible by the
visible device count and `require_all_visible_devices` is false, the resolver selects the
largest usable device count. Forced `parallel.mode=pmap` errors when the requested setup
cannot be resolved.

The implementation uses single-host `jax.pmap` for synchronous data parallelism. JAX
documents `pmap` as SPMD execution over local XLA devices; mapped axis size must be no
larger than `jax.local_device_count()`, and participating devices must be identical:
https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html.

## Fast Storage

Keep raw data read-only and write heavy generated artifacts under an environment-selected
fast root:

```bash
export GK_FAST_ROOT=/path/to/fast/local/gk-latent-surrogate-jax
```

Configs may use `${GK_FAST_ROOT}` for `output_dir` and `latent_cache.path`. The value is
site-specific and should stay in shell setup or scheduler configuration, not in Python
source or committed run reports.

## Server Smoke

```bash
export GK_CYCLONE_DATA_ROOT=/path/to/preprocessed_kvikio
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The direct KvikIO reader used by the current server path does not need an upstream checkout
on `PYTHONPATH`. Only set one when deliberately validating an alternate upstream loader:

```bash
export GK_NEUGK_UPSTREAM=/path/to/neural-gyrokinetics
export PYTHONPATH=src:${GK_NEUGK_UPSTREAM}:${PYTHONPATH}
```

```bash
uv run gks inspect-data \
  --config configs/data/cyclone_kvikio_template.yaml \
  --max-trajectories 1 \
  --max-depth 4 \
  --max-target-samples 5 \
  --output-dir outputs/server_gpu_inspection \
  --override 'data.cyclone.trajectories=["iteration_0"]' \
  --override data.cyclone.offset=80 \
  --override data.cyclone.subsample=32 \
  --override 'data.target_spectra=["kyspec","fluxspec"]'

uv run gks train-encoder \
  --config configs/experiment/smoke_real_encoder_flux.yaml \
  --override parallel.mode=auto \
  --override data.batch_size=4 \
  --override training.max_steps=2 \
  --output-dir outputs/server_gpu_smoke_encoder
```

Training commands write `device_report.json` next to metrics and checkpoints when
`parallel.log_device_summary` is true.

On shared GTX 1080 Ti runs, keep `XLA_PYTHON_CLIENT_PREALLOCATE=false`. JAX may otherwise
try to reserve large chunks of GPU memory before compiling the first step.

The real-data encoder configs use `kernel_size: [1, 1, 1, 1, 1]` for the `conv_nd`
encoder. That path is implemented as pointwise channel mixing plus strided spatial
slicing, because CUDNN convolution on this stack does not support five spatial dimensions.

`data.cyclone.use_kvikio=true` is supported on the current GTX 1080 Ti server through the
repo's direct Cyclone/KvikIO directory reader. That path reads `.bin` shards with
`kvikio.CuFile`/CuPy and then runs preprocessing in NumPy, bypassing the optional upstream
CUDA dataset path that lacks Pascal `sm_61` kernels. Use `use_kvikio=false` only as the
portable CPU fallback.

## Observed Device Pattern

For server experiments, record the visible device count, resolved parallel mode, global
batch size, per-device batch size, and output directory in the run notes or generated
`device_report.json`. Those records belong in ignored `outputs/` paths unless a later
reporting ticket explicitly promotes a summary into docs.

On shared GPUs, canonical pmap runs can be blocked by external memory pressure. The
fallback is to document the blocker and run a smaller `parallel.mode=single` or lower
batch-size check. Do not claim constrained runs are hardware-normalized comparisons.
