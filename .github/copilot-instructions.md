# Repository instructions

Follow `AGENTS.md` for implementation constraints and `CONTRIBUTING.md` for branches,
commits, pull requests, provenance, and merge policy.

- Treat `PRD.md` as project intent and current code/tests as implemented behavior.
- Preserve the GPU-first JAX/Flax design and channel-first snapshot contract
  `[B, C, S1, S2, S3, S4, S5]`.
- Keep `JAX_PLATFORM_NAME=cpu` as the portable fallback for local checks and CI.
- Use `uv`, `make`, and `gks`; run `make check`, `make smoke-all`, and `uv build` before
  publishing behavior or workflow changes.
- Never commit data, caches, checkpoints, generated results, W&B state, or build outputs.
- Use neutral scientific naming and keep results from different protocols separate.
