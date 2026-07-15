---
name: Bug report
about: Report a reproducible defect in the latent surrogate pipeline
title: "fix: "
labels: bug
assignees: ""
---

## Context

Describe the failure and the expected behavior.

## Reproduction

- Command:
- Config:
- Seed:
- Platform:

## Scientific/provenance impact

Paste the relevant traceback, metric regression, or GitHub Actions annotation.

## Verification

- [ ] `make agent-check`
- [ ] `make check`
- [ ] `make smoke-all`
- [ ] `uv build`

## Risks

Confirm whether the issue requires real data, GPU hardware, server access, or changes to
`PRD.md`. Tests and smoke paths should remain portable by default.
