# Verification Matrix

This matrix defines which deterministic checks prove which class of change. Prefer the
smallest relevant gate while iterating, then run the final local gate before publishing.

| Change type | Command | Expected result |
| --- | --- | --- |
| Agent docs, templates, or CI contract | `make agent-check` | Required agent files exist and mention the mandatory PRD, GPU-first, smoke, and CI gates. |
| Python source, configs, or tests | `make check` | Agent check, ruff, mypy, fast tests, and 95% coverage gate pass on the portable fallback runner. |
| Data, training, embedding, evaluation, or CLI pipeline | `make smoke-all` | Synthetic inspection, encoder training, latent cache, validation flux-head metrics, representation plots, sequence training, rollout metrics, CSV, and plot are produced. |
| Packaging, build metadata, or CI package job | `uv build` | Wheel and source distribution build successfully under `dist/`. |
| GitHub workflow changes | GitHub Actions `CI` workflow | The required `Verify repository` job runs conventions, static checks, tests with coverage, the smoke pipeline, and package build on `ubuntu-latest`. |

## Final Local Gate

Before any temporary CI verification branch commit, run:

```bash
make check
make smoke-all
uv build
git status -sb
```

Only intended tracked files may be staged. Ignored smoke outputs and `dist/` artifacts may
exist locally but must not be committed.

## Remote Gate

Use GitHub Actions only after local verification is green. A temporary verification branch
is allowed to trigger CI; `main` must not be updated until a real Actions run completes
successfully.

If GitHub Actions reports that the job was not started because of account billing or
spending-limit settings, treat that as an external blocker. Do not mark remote CI green and
do not update `main`.
