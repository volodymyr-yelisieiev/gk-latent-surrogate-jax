# Contributing

This document defines the repository workflow for people and automated contributors.
`PRD.md` is the source of truth for project intent; committed code, configuration, tests,
and documentation define implemented behavior.

## Development workflow

Use Python 3.12 when available and the repository entrypoints:

```bash
make install-dev
make check
make smoke-all
uv build
```

Run the smallest relevant check while developing. Before publishing a code, pipeline,
configuration, packaging, or CI change, run the complete sequence above.

## Branches, commits, and pull requests

Branch names use a neutral category and a concise kebab-case subject:

```text
feat/<subject>
fix/<subject>
research/<subject>
refactor/<subject>
docs/<subject>
test/<subject>
ci/<subject>
chore/<subject>
```

Commit messages and pull-request titles use the following form and must not exceed 72
characters:

```text
type(scope): imperative summary
```

Allowed types are `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `ci`, `build`, and
`chore`. The scope is optional, lowercase, and kebab-case. Use `!` before the colon only
for an intentional breaking change.

Pull requests are squash-merged. The pull-request title becomes the commit subject on
`main`, so temporary branch commits may be reorganized before merge.

Names must describe scientific or technical content. Do not encode contributor identity,
tools, meetings, review events, calendar dates, internal phases, or informal progress
labels in branch names, commit subjects, pull-request titles, W&B metadata, plots, or
reports.

## Scientific provenance

Every reported metric must identify:

- dataset or latent cache;
- checkpoint or baseline;
- split and rollout horizon;
- aggregation semantics;
- resolved configuration;
- local artifact or external run record.

Keep results from different datasets, caches, splits, horizons, normalization modes, or
aggregation rules in separate tables. A comparison may use different model checkpoints
only when the evaluation protocol is otherwise identical and that distinction is explicit.

Generated results remain outside git. Do not commit raw data, checkpoints, latent caches,
generated HDF5/NPZ files, run directories, W&B state, plots, or build artifacts.

## Pull-request content

Use the repository template and keep these sections concise:

1. Context
2. Changes
3. Scientific/provenance impact
4. Verification
5. Risks

Update tests whenever shapes, configs, loaders, training steps, metrics, CLI contracts, or
artifact schemas change. Keep the synthetic CPU fallback independent of real data, GPUs,
server access, PyTorch, and external repositories.

## Merge policy

Changes reach `main` through a pull request after the required `Verify repository` check
passes and review conversations are resolved. Direct pushes, force pushes, merge commits,
and deletion of `main` are prohibited after repository bootstrap.
