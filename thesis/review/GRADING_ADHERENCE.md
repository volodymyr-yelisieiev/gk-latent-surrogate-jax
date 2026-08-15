# Grading and Task Adherence Review

This review maps the thesis package to the supplied IML BSc grading guideline. It describes
repository evidence and remaining author-owned checks; it does not represent institutional approval.

## Grading guideline

| component | weight | evidence and present status |
| --- | ---: | --- |
| Scientific work | 50% | A frozen, retrospective five-fold nested group cross-validation study covers 51 manifest-/byte-verified trajectories, five matched seeds, validation-only selection, trajectory-balanced metrics, paired uncertainty, and explicit limitations. The primary interval is strictly positive for learned minus observed persistence, so the negative result is reported without a superiority claim. |
| Scientific documentation and code | 10% | Portable setup, data contract, configs, CLI dry-runs, synthetic fallback, lineage checks, stage ledger, sanitized release, disabled-W&B evidence, tests, and reproducibility commands are present. Final local gates and PDF audits must be rerun from the final worktree before submission. |
| Manuscript | 40% | The source contains the required title/abstract, introduction, method, implementation, experiment, results, discussion, conclusion, story-summary appendix, references, figures, tables, and reproducibility appendix. The rebuilt PDF audit is the final authority for page count, words, links, references, overfull boxes, and visual quality. |
| Plagiarism/originality | pass condition | The author must run the institutionally appropriate originality check, review citations and wording, and complete any required AI-assistance disclosure. Repository automation cannot certify originality. |

## Scientific claim control

- Primary metric: trajectory-balanced observed-flux RMSE.
- Accepted evidence: 230 stages plus 25 explicitly skipped unselected-family test slots; five
  outer folds and five matched seeds.
- Selected family: Transformer in folds 0, 1, and 3; GRU in folds 2 and 4. No global winner is
  claimed.
- Primary estimate: learned 14.6729, observed persistence 10.0207, difference +4.6522, bootstrap
  95% interval [+2.5788, +6.6470].
- Secondary estimate: learned minus decoded latent-persistence latent MSE +0.0189, interval
  [-0.0699, +0.1332].
- Diagnostic-head oracle: 12.3732 flux RMSE; an analysis control, not a forecast ceiling or proof
  of a single bottleneck.
- Generalization beyond this 51-trajectory universe, preprocessing revision, targets, and
  eight-step horizon is not claimed.

## Task-history adherence

- The software, data contract, protocol runner, aggregate, release manifest, thesis source, figures,
  tests, and review materials are present in the repository.
- GyroSwin remains related work because no matched checkpoint/data/metric bundle is available; no
  direct numerical comparison is made.
- W&B is disabled and locally verified for all 225 accepted metric stages; no public run URL is claimed.
- Historical mixed-split and seed-52-only values are withdrawn from thesis evidence and do not
  appear in the current result chapters.

## Author-owned checks before formal submission

- Read every page and verify every number, citation, name, link, and claim boundary.
- Confirm official thesis title, supervisor title, degree wording, submission date, and filename.
- Run the institutional originality check and add any required originality or AI-use statement.
- Obtain supervisor approval and submit through the official channel.
