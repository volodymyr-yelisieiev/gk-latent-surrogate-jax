# Documentation Guide

This directory is intentionally small. It keeps only the operational contracts needed to
run, verify, and extend the latent surrogate pipeline.

## Core Contracts

| document | purpose |
| --- | --- |
| `data_contract.md` | channel-first snapshot, latent cache, and diagnostic target contracts |
| `metrics.md` | rollout, latent, flux, and spectra metric definitions |
| `experiment_lifecycle.md` | expected config, output, checkpoint, and report flow |
| `experiment_provenance.md` | frozen protocol, accepted-run linkage, and release evidence rules |
| `verification_matrix.md` | local checks, smoke checks, build gate, and CI expectations |
| `result_status.md` | current evidence status, baseline audit, and explicit non-claims |

## Runtime Notes

| document | purpose |
| --- | --- |
| `server_gpu_setup.md` | GPU/server execution and KvikIO setup notes |
| `wandb_tracking.md` | optional W&B telemetry contract and deletion/status policy |
| `hardware_profiles.md` | portable hardware assumptions and config boundaries |
| `real_data_binding_checklist.md` | real-data adapter requirements and missing-detail policy |

Generated thesis prose, PDFs, exploratory plots, tables, caches, checkpoints, run directories,
W&B state, and package artifacts do not belong in this repository. The six reviewed figures in
`thesis/figures/` are the deliberate source-controlled manuscript-asset exception; regenerate
them with the thesis figure script and verify their release hash before publication. Recreate
other outputs from committed configs and CLI commands after experiment evidence is available.
Keep tracked run reports concise, formal, and tied to reproducible evidence; working notes
belong in Notion, ignored `outputs/`, or external lab notes.
