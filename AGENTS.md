# Agent Contract

This repository is a GPU-first JAX/Flax implementation of the latent surrogate pipeline
specified in `PRD.md`, centered on the server GPU/KvikIO thesis workflow. Treat the PRD as
the source of truth for project intent and use the local code, configs, and tests as the
source of truth for current behavior. `CONTRIBUTING.md` is the authoritative workflow for
branches, commits, pull requests, scientific provenance, and merge policy.

## Working Rules

- Use `uv`, `make`, and the `gks` CLI entrypoints for normal development and verification.
- Prefer Python 3.12 locally when available; the project supports Python `>=3.11`.
- Keep local and CI checks portable; use `JAX_PLATFORM_NAME=cpu` only as the fallback
  runner setting when a GPU is not available.
- Keep the stack free of brittle hardware assumptions: no hardcoded CUDA wheel, GPU count,
  `pjit`, server paths, PC paths, or dependencies on the old NDSwin repository.
- Do not introduce PyTorch, mandatory real-data requirements, multi-node execution, GPT-2
  fine-tuning, or full 5D reconstruction unless the PRD or user explicitly changes the
  scope.
- Never commit raw data, checkpoints, latent caches, generated HDF5 files, run outputs,
  `wandb`, or generated package/build artifacts.
- Maintain channel-first snapshot shape comments when documenting tensors:
  `[B, C, S1, S2, S3, S4, S5]`.
- Add or update tests whenever shapes, configs, loaders, training steps, metrics, CLI
  contracts, or generated artifact schemas change.
- Use neutral technical naming. Do not put contributor identity, tools, meetings, dates,
  internal phases, or informal progress labels in repository or experiment metadata.

## Repository Map

- `src/gk_surrogate/config`: Pydantic schemas, YAML loading, overrides, and consistency
  validation.
- `src/gk_surrogate/data`: synthetic trajectories, HDF5 loading/inspection, normalization,
  collation, splits, augmentations, latent cache, and sequence-window datasets.
- `src/gk_surrogate/models`: encoders, diagnostic heads, SimSiam heads, sequence models,
  and external adapter placeholders.
- `src/gk_surrogate/training`: train states, optimizers, train/eval steps, checkpointing,
  embedding, and training loops.
- `src/gk_surrogate/evaluation` and `src/gk_surrogate/metrics`: rollout execution,
  diagnostics, aggregation, JSON/CSV reports, and plots.
- `configs`: tiny synthetic data/model/experiment configs used by local and CI smoke runs.
- `tests`: required coverage for PRD behavior; no local/CI test may require real data,
  GPU hardware, server access, PC access, PyTorch, or old repository code.
- `docs`: data contract, lifecycle, metrics, hardware profiles, and agent runbooks.

## Verification Gates

Run the smallest gate that proves the change, then run the full gate before publishing.

- Agent/readiness docs: `make agent-check`.
- Code changes: `make check`.
- Pipeline/config/training/evaluation changes: `make smoke-all`.
- Packaging or CI changes: `uv build`.
- Final local gate before any CI verification branch: `make check && make smoke-all && uv build`.

Expected smoke outputs are ignored by git and include synthetic inspection output,
encoder checkpoints, `outputs/smoke_embed_dataset/latent_cache.h5`, sequence checkpoints,
rollout metrics, CSV reports, and plots.

## PRD Compliance Checklist

Before marking work complete, confirm:

- synthetic fallback pipeline still works without real data;
- CLI dry-runs validate configs and write resolved config output where applicable;
- data loaders preserve channel-first 5D snapshot contracts;
- encoder, SimSiam, latent cache, sequence training, and rollout paths are covered;
- diagnostic flux/spectra metrics remain finite in smoke runs;
- line coverage remains at or above the configured gate;
- docs and templates describe the changed workflow accurately.

## Subagent Boundaries

When parallel agents are used, split ownership by subsystem and avoid overlapping writes:

- config/data/CLI/docs;
- models/losses/metrics;
- training/checkpointing/evaluation;
- tests/CI/coverage.

Workers must not revert edits from other agents. Integrators must review the combined diff,
run the full verification gate, and resolve interface mismatches before publishing.

## Stop Conditions

- Stop before publishing if local gates fail.
- After bootstrap, update `main` only through a pull request with the required GitHub
  Actions check green.
- If GitHub Actions is blocked by account billing or spending limits, report that external
  blocker and do not update `main`.
- If real dataset details are missing, keep work on generic HDF5 schema/templates and
  synthetic smoke coverage.
