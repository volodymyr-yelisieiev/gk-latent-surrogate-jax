# Pretrained Guppy/SFT Evaluation

Evidence reviewed: 2026-07-15.

## Outcome

THESIS-19 completed the optional GuppyLM trunk-transfer and supervised fine-tuning
experiment. The transfer mapped 75 source tensors into 99 Flax target tensors and rejected
incompatible mappings instead of silently falling back to random initialization.

The current Guppy-style path in this repository is trained from scratch. The Flax
`guppy_latent_transformer` and `causal_transformer` sequence models initialize local
parameters and predict the next continuous latent vector. No pretrained Guppy latent
checkpoint is wired into the codebase.

`arman-bd/guppylm` is a small MIT-licensed text LLM rather than a gyrokinetic
latent-sequence checkpoint. Its causal Transformer architecture is close to the local
Guppy-style block family, but it is not a drop-in model for continuous latent rollouts.

## Transfer Scope

`arman-bd/guppylm-9M` is a PyTorch/ONNX text model with an 8.7M-parameter causal
Transformer. The reusable portion was limited to compatible attention and MLP trunk
tensors. Token embeddings and the tied language-model head do not map to continuous
latent input/output projections and were reinitialized for the surrogate objective.

Sources checked:

- https://github.com/arman-bd/guppylm at commit `a30df3091cff73a259ce581dd8439271084bca40`
- https://huggingface.co/arman-bd/guppylm-9M
- https://github.com/ml-jku/neural-gyrokinetics

## Results

| split | flux RMSE | external evidence |
| --- | ---: | --- |
| validation | 17.8415 | W&B `rollout-comparison-medium-validation` (`eh4xe1sg`) |
| test | 11.5816 | W&B `rollout-guppylm-pretrained-sft-medium-test` (`iq92tggu`) |

The training comparison is recorded in W&B run `sequence-training-comparison-medium`
(`doy8yhgt`). All three runs belong to `medium-scale-latent-surrogate`.

## Conclusion

The completed adapter feasibility study did not replace the from-scratch main result. The
historical cache-normalized from-scratch transformer remains the selected main-cache model
with test flux RMSE `9.2270`; the SFT result belongs to a distinct protocol and must be
reported separately.

The current tracked main tree contains neither the transfer utility nor the SFT checkpoint.
Reproducing this experiment therefore requires restoring the THESIS-19 conversion artifact
and checkpoint outside the repository; the normal package remains free of a PyTorch
dependency. No claim should imply that `gks` can currently reproduce the transfer from a
clean checkout.
