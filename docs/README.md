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
| `small_validation_experiment.md` | 3-5 trajectory validation run with flux RMSE, baseline, and plots |
| `medium_guppy_experiment_report.md` | medium-scale evidence, W&B links, metrics, and framing |
| `thesis_result_set.md` | separated evidence tiers for thesis writing |
| `current_best_model.md` | current best model selection and traceability |
| `final_claims.md` | supported thesis claims, non-claims, and framing |
| `pretrained_guppy_sft_feasibility.md` | completed GuppyLM trunk-transfer/SFT evaluation |

## Runtime Notes

| document | purpose |
| --- | --- |
| `server_gpu_setup.md` | GPU/server execution and KvikIO setup notes |
| `wandb_tracking.md` | original W&B telemetry, grouping, public evidence, and invalidated-run rules |
| `hardware_profiles.md` | portable hardware assumptions and config boundaries |
| `real_data_binding_checklist.md` | real-data adapter requirements and missing-detail policy |
| `gyroswin_comparison.md` | fair-comparison policy without fabricated numbers |
| `agent_runbook.md` | internal maintenance handoff for coding agents |

Generated thesis prose, PDFs, plots, tables, caches, checkpoints, run directories, W&B
state, and package artifacts do not belong in this repository. Recreate them from
committed configs and CLI commands after experiment evidence is available. Keep tracked
run reports concise, formal, and tied to reproducible evidence; working notes belong in
Notion, ignored `outputs/`, or external lab notes.
