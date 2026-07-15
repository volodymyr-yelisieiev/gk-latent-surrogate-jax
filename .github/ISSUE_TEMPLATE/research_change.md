---
name: Research or engineering change
about: Propose a focused scientific, implementation, documentation, or infrastructure change
title: "feat: "
labels: enhancement
assignees: ""
---

## Context

State the research question, intended behavior, or maintenance requirement.

## Changes

- Subsystem:
- Intended behavior:
- Out of scope:

## Scientific/provenance impact

Identify affected data, caches, checkpoints, splits, horizons, metrics, or artifact schemas.
Use `None` when the proposal has no scientific-result impact.

## Verification

- [ ] Behavior is covered by tests or a synthetic smoke run.
- [ ] Documentation is updated when interfaces or workflows change.
- [ ] Generated artifacts remain outside git.
- [ ] `make check`, relevant smoke checks, and `uv build` pass.

## Risks

List compatibility constraints, missing evidence, and rollback considerations.
