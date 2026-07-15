# Hardware Profiles

The base install is CPU-safe:

```bash
make install-dev
```

Do not pin CUDA wheels in `pyproject.toml`. For WSL2 or server GPU work, install the
JAX build that matches the host driver/CUDA stack in that environment, then run the same
configs with environment-specific dataset roots such as `GK_LOCAL_PC_DATA_ROOT` and
`GK_CYCLONE_DATA_ROOT`.

Profiles:

- MacBook CPU: local tests, dry-runs, tiny synthetic data.
- RTX 5070 under WSL2: optional local GPU training after JAX wheel compatibility is known.
- Shared Cyclone server: Cyclone/KvikIO inspection, tiny real smoke, optional AE probing,
  measured server benchmarks, and opt-in data-parallel training.

## Hardware Decision Protocol

Compare hardware only after the same config runs locally. Use the same model, seed, and
batch size where possible, then record the largest fitting batch size separately.

Baseline command:

```bash
JAX_PLATFORM_NAME=cpu uv run gks benchmark-step-time \
  --config configs/experiment/smoke_real_encoder_flux.yaml \
  --measured-steps 3
```

Benchmark results should be recorded in ignored run outputs. Add a short repository
summary only after the measurement is repeatable with a committed config.

For PC or server GPU runs, install the matching JAX wheel outside `pyproject.toml`, remove
the CPU override only for that benchmark shell, and record:

- machine profile: MacBook CPU, RTX 5070 WSL2, or shared Cyclone server GTX 1080 Ti;
- backend and devices printed by the benchmark;
- config file, overrides, batch size, and measured steps;
- first-step compile time and mean/min/max measured step time;
- maximum batch size that fits without changing model architecture;
- HDF5 throughput notes if real data are used.
- Cyclone/KvikIO backend settings, including `use_kvikio`, `prefer_dtype`, and whether
  BF16 shards are present.

Do not introduce `pjit`, hardcoded CUDA wheels, or device-count assumptions as part of
the comparison. Use the repository `parallel` config for optional single-host `pmap`
data parallelism and record the generated `device_report.json`.
