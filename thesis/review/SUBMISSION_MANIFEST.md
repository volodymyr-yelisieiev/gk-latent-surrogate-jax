# Submission Manifest

## Review artifacts

- `build/main-thesis.pdf`: local build output; rebuild from the current source before packaging.
- `main-thesis.tex` and chapter files: editable LaTeX source.
- `references.bib`: bibliography source.
- `figures/`: tracked figures whose source artifacts and generation script must be checked before
  submission.
- `REQUIREMENTS.md`: requirements trace.
- `review/FINAL_CHECKLIST.md`: completion and hand-in checklist.

## Final filename

The review candidate follows the exact course convention:

`2026S-12340334-Yelisieiev_Volodymyr-Thesis_BSc-v1-Latent_Surrogates.pdf`

The short-title segment is two words. Do not use the review PDF's generic filename for formal
submission.

## Provenance

- Training implementation: commit `f97a0257d7627c8ff8960433aed30c750a9f90d5`.
- Final evaluation implementation: commit `280540f54e67c0dbcae253327596bfaf7cbf9307`.
- If a `repository-source.bundle` is included, verify it with
  `git bundle verify repository-source.bundle`; its existence is not asserted by this manifest.
- JKU template: upstream release `v2.2`, commit
  `79d78bd9fd8b69d5635a0d419536827823772b59`.
- Main evidence paths and hashes are recorded in Appendix B of the thesis.

## Submission blockers

The review package must not be submitted until the author has reviewed the full text and figures,
the current PDF and repository gates pass, provenance links are public or have a sanitized fallback,
and any required institutional declarations and approvals are complete.
