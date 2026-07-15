# GyroSwin-Style Comparison Policy

The repository may compare latent-surrogate diagnostics against GyroSwin-style reference
results, but only when the comparison is technically fair. Do not train a GyroSwin clone
or invent reference numbers inside this project.

## Current Status

No matching GyroSwin checkpoint/config/evaluation bundle is available
for this thesis split. Keep GyroSwin protocol-level unless comparable materials become
available.

Public GyroSwin checkpoints are useful related-work pointers, but they are not a direct
baseline for the current JAX/Flax latent-cache rollout tables. This project predicts in a
learned latent space with diagnostic heads; GyroSwin-style models target full-field
surrogate behavior under their own data and preprocessing contracts.

## Direct Comparison Requirements

A numeric comparison needs all of these facts:

- exact dataset root or trajectory IDs;
- offset, tail offset, and subsampling;
- train/validation/test split seed or explicit IDs;
- rollout context length and prediction horizon;
- flux and spectra target definitions;
- metric definitions for flux, spectra, stability, and any normalization;
- hardware, batch size, runtime, and memory notes.

If any item is missing, keep the comparison protocol-level and state what is missing.

## Allowed Claims

- This repo implements a latent surrogate with diagnostic heads and short-horizon latent
  rollout.
- It can report internal baselines such as persistence when they use the same cache,
  horizon, and metric definitions.
- It can include GyroSwin as a related-work or protocol target when the reference setup is
  not yet comparable.

## Disallowed Claims

- Do not claim full 5D reconstruction quality.
- Do not claim a direct GyroSwin win/loss without matching data, horizon, and metrics.
- Do not mix smoke results, one-trajectory overfit checks, and thesis-scale experiments
  into one evidence table.

## Thesis Wording

Use:

```text
GyroSwin is treated as related work and a possible future comparison target. A direct
numerical comparison is not reported because matching checkpoint/config/split/horizon
materials were not available for the current latent-surrogate evaluation.
```

Do not add a numeric GyroSwin row to `docs/thesis_result_set.md` or
`docs/medium_guppy_experiment_report.md` until the direct-comparison requirements above
are satisfied.
