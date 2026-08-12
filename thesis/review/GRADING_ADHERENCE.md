# Grading and Task Adherence Review

This review maps the thesis package to the supplied IML BSc grading guideline and to the canonical
thesis task history. It describes the current technical state; it does not represent institutional
approval or submission.

## Grading guideline

| component | weight | evidence and present status |
| --- | ---: | --- |
| Scientific work | 50% | The pipeline, real-data medium experiment, persistence baseline, validation-only selection, trajectory-balanced metrics, paired trajectory uncertainty, and limitations are documented. The seed-52 result is one retrospective test realization with one training seed and five trajectories; its interval includes zero. Matched seeds and an unseen test set or nested group cross-validation remain outstanding. |
| Scientific documentation and code | 10% | Setup, data contract, configurations, CLI dry-runs, artifact provenance, synthetic fallback, tests, and reproducibility commands are present. A current clean run of the repository gates, exact coverage, and a source-linked public release remain required; this file does not certify them. |
| Manuscript | 40% | The LaTeX source contains the required sections, abstracts, story summary, figures, tables, and references. Page count, word count, unresolved references, overfull boxes, links, and visual quality must be re-audited on the PDF built from the current source. |
| Plagiarism/originality | pass condition | Automated source preparation cannot certify originality. The author must run the institutionally appropriate check and accept responsibility for citations, wording, and any required AI-assistance disclosure. |

## Scientific claim control

- Primary metric: trajectory-balanced flux RMSE.
- Validation-selected learned model: cache-normalized causal Transformer, flux RMSE `18.179277`.
- Retrospective test point estimates: Transformer `11.425478`; persistence `11.988926` in
  preprocessed target units.
- Primary conclusion: the fixed-protocol point estimate improves by `0.563448` (`4.70%`).
- Per-trajectory flux RMSE improves in three of five cases; the paired mean difference is `-0.8279`
  with descriptive 95% t interval `[-2.2249, 0.5692]`, so no significance claim is made.
- Historical mixed-seed values, including `9.2270`, are retained only as explicitly invalidated
  audit evidence.
- Generalization beyond the 51-trajectory subset, five validation trajectories, five test
  trajectories, one training seed, and eight-step horizon is not claimed.

## Canonical task-history adherence

- `THESIS-01` through `THESIS-21`: task status in an external database is not evidence of technical
  or scientific correctness. Each claimed deliverable must be checked against the repository and
  accepted artifacts.
- `THESIS-22`: remains Waiting because matched external GyroSwin artifacts are unavailable. The
  manuscript therefore treats GyroSwin as related work and makes no direct numerical comparison.
- `THESIS-23`: source, figures, tables, references, reproducibility appendix, and review checklist
  exist. The current-source PDF build and page-level review remain to be recorded.
- `THESIS-24`: package naming and provenance requirements are documented. Public source commits,
  anonymous W&B access or sanitized fallback, release checksums, and evidence links require external
  verification.
- `THESIS-25`: the filename convention and human-only submission checklist are prepared. Formal
  upload, grade request, and any institutional declarations remain external actions.

## Review package

- Intended PDF name: `2026S-12340334-Yelisieiev_Volodymyr-Thesis_BSc-v1-Latent_Surrogates.pdf`.
- PDF and source-archive SHA-256 values must be generated from the final reviewed files; this review
  does not assert that a current release package exists.
- Identity recorded in the PDF: matriculation number `12340334`, supervisor Gianluca Galletti,
  programme Artificial Intelligence.

## Remaining human checks

- Read and accept every page, citation, number, link, and scientific formulation.
- Confirm whether the title page requires the supervisor's academic title or different official
  degree wording.
- Complete the required originality check and any institutional declaration.
- Obtain any required external approval, then submit through the official channel.
