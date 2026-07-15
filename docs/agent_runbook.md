# Agent Runbook

Use this runbook when an AI agent needs to inspect, modify, verify, or hand off work in
this repository. `PRD.md` defines the product intent; current code, configs, tests, and
this runbook define the operational contract.

## Repository Map

- Configs live under `configs` and are loaded through `gk_surrogate.config`.
- CLI commands are routed through `src/gk_surrogate/cli.py` and high-level pipeline helpers
  in `src/gk_surrogate/pipeline.py`.
- Data contracts are implemented under `src/gk_surrogate/data` and documented in
  `docs/data_contract.md`.
- Models live under `src/gk_surrogate/models`; training/evaluation code lives under
  `src/gk_surrogate/training` and `src/gk_surrogate/evaluation`.
- Tests mirror PRD behavior under `tests`; smoke configs are tiny, synthetic, and
  portable.
- Generated data, checkpoints, latent caches, package artifacts, and smoke outputs are
  intentionally ignored by git.

## Task Handoff

Every handoff should include:

- goal and PRD section or behavior being changed;
- files or subsystems touched;
- commands run and their pass/fail status;
- smoke artifacts produced, if any;
- known blockers, especially external GitHub Actions billing or unavailable real data;
- whether `main` is safe to update.

Use concise, concrete status. Do not say a verification passed unless the command actually
completed successfully in the current workspace or in GitHub Actions.

## Subagent Ownership

Split parallel work by write ownership:

- config/data/CLI/docs;
- models/losses/metrics;
- training/checkpointing/evaluation;
- tests/CI/coverage.

Subagents should edit only their assigned subsystem, should not revert other agents'
changes, and should report changed files plus validation results. The integrating agent
owns final interface review, `make check`, `make smoke-all`, package build, and CI status.

## Smoke Artifacts

`make smoke-all` writes ignored artifacts under `outputs/smoke_*`:

- synthetic inspection output from `gks inspect-data`;
- supervised and SimSiam encoder checkpoints;
- `outputs/smoke_embed_dataset/latent_cache.h5`;
- sequence-model checkpoint and training metrics;
- rollout `metrics.json`, `metrics_by_step.csv`, and `plots/latent_mse_by_step.png`.

These files are useful for debugging but must not be committed.

## Common Failure Paths

- Config validation failure: rerun the CLI command with `--dry-run` to validate and print
  a compact summary. Use a normal command with `--output-dir` when you need the full
  `config_resolved.yaml` written for inspection.
- Shape failure: confirm channel-first snapshots `[B, C, S1, S2, S3, S4, S5]`, then check
  collate, encoder input shape, and sequence-window dimensions.
- HDF5 failure: inspect `configs/data/h5_template.yaml` and `docs/data_contract.md`; do not
  hardcode external roots.
- Real Data Binding Checklist: follow `docs/real_data_binding_checklist.md` before running
  any real-data training, and keep inspection bounded with `--max-target-samples`.
- Non-finite loss or metric: reproduce with the tiny smoke config and seed before changing
  optimizer, normalization, or rollout logic.
- Coverage failure: add targeted tests for the missing behavior; do not lower the gate.
- GitHub Actions failure before jobs start: check the annotation. Billing/spending-limit
  errors are external account blockers and do not prove code failure.

## Completion Standard

A change is complete only when the relevant local gate passes, docs/templates match the
new workflow, and GitHub Actions has run green before `main` is updated. If Actions cannot
start because of an account-level blocker, stop and report that blocker.
