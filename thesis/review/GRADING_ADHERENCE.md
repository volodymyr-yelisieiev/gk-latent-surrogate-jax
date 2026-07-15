# Grading and Task Adherence Review

This review maps the thesis package to the supplied IML BSc grading guideline and to the canonical
thesis task history. It describes the current technical state; it does not represent institutional
approval or submission.

## Grading guideline

| component | weight | evidence and present status |
| --- | ---: | --- |
| Scientific work | 50% | A complete JAX/Flax pipeline, real-data medium experiment, explicit persistence baseline, validation-only model selection, locked test, trajectory-balanced metrics, paired trajectory uncertainty, and limitations are documented. A lineage audit invalidated a mixed-seed result; the full source-to-test seed-52 reproduction is the sole headline evidence and lowers the primary point estimate by 4.70%. |
| Scientific documentation and code | 10% | Setup, data contract, configurations, CLI dry-runs, fail-closed artifact provenance, synthetic fallback, tests, and reproducibility commands are present. The final gate covers 230 tests at 95.00% line coverage, the full synthetic smoke pipeline, Ruff, MyPy, and package build. |
| Manuscript | 40% | The compiled A4 PDF has 38 pages and remains within the 5,000--10,000 main-word requirement. It has a 259-word English abstract, more than 1,000 words in introduction plus related work, more than 500 words in discussion plus conclusion, seven figures, three tables, and all nine story answers within three sentences. Final PDF checks require no unresolved references or overfull boxes. |
| Plagiarism/originality | pass condition | Automated source preparation cannot certify originality. The author must run the institutionally appropriate check and accept responsibility for citations, wording, and any required AI-assistance disclosure. |

## Scientific claim control

- Primary metric: trajectory-balanced flux RMSE.
- Validation-selected learned model: cache-normalized causal Transformer, flux RMSE `18.179277`.
- Locked test: Transformer `11.425478`; persistence `11.988926`.
- Primary conclusion: the fixed-protocol point estimate improves by `0.563448` (`4.70%`).
- Per-trajectory flux RMSE improves in three of five cases; the paired mean difference is `-0.8279`
  with descriptive 95% t interval `[-2.2249, 0.5692]`, so no significance claim is made.
- Historical mixed-seed values, including `9.2270`, are retained only as explicitly invalidated
  audit evidence.
- Generalization beyond the 51-trajectory subset, five validation trajectories, five test
  trajectories, one training seed, and eight-step horizon is not claimed.

## Canonical task-history adherence

- `THESIS-01` through `THESIS-21`: recorded as Done in the canonical Notion task database; their
  implementation, experiment, documentation, and result-framing deliverables were rechecked against
  the repository.
- `THESIS-22`: remains Waiting because matched external GyroSwin artifacts are unavailable. The
  manuscript therefore treats GyroSwin as related work and makes no direct numerical comparison.
- `THESIS-23`: the source, compiled PDF, figures, tables, references, reproducibility appendix, and
  review checklist are prepared. External approval and author review remain outside the technical
  package.
- `THESIS-24`: the correctly named review candidate, source archive, checksums, repository link,
  clean W&B records, and evidence summary are prepared. Delivery remains an external action.
- `THESIS-25`: the filename convention and human-only submission checklist are prepared. Formal
  upload, grade request, and any institutional declarations remain external actions.

## Review package

- PDF: `2026S-12340334-Yelisieiev_Volodymyr-Thesis_BSc-v1-Latent_Surrogates.pdf`.
- PDF SHA-256: recorded in the final package checksum manifest.
- Source archive: `thesis-source-review.tar.gz`; its checksum is recorded beside the package.
- Identity recorded in the PDF: matriculation number `12340334`, supervisor Gianluca Galletti,
  programme Artificial Intelligence.

## Remaining human checks

- Read and accept every page, citation, number, link, and scientific formulation.
- Confirm whether the title page requires the supervisor's academic title or different official
  degree wording.
- Complete the required originality check and any institutional declaration.
- Obtain any required external approval, then submit through the official channel.
