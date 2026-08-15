# Submission Manifest

## Review artifacts

- `build/main-thesis.pdf`: local build output; rebuild from current source before packaging.
- `main-thesis.tex` and chapter files: editable LaTeX source.
- `references.bib`: bibliography source.
- `figures/`: tracked figures generated or inspected for the accepted result.
- `REQUIREMENTS.md`: requirements trace.
- `review/FINAL_CHECKLIST.md`: completion and hand-in checklist.
- `../experiment_protocols/multiseed_v1_results.json`: sanitized accepted evidence release.

## Final filename

The review candidate follows the course convention:

`2026S-12340334-Yelisieiev_Volodymyr-Thesis_BSc-v1-Latent_Surrogates.pdf`

The short-title segment is two words. Do not use a generic build filename for formal submission.

## Provenance

- Frozen experiment source tag: `protocol/multiseed-v1`.
- Frozen experiment source commit: `be976808582239da201896bd20ef95ff91d97128`.
- Frozen tracked-diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Dataset revision: `cyclone-consumed-bytes-sha256:ff2867e9eb8e9ed74dd1ed92d347b02a703368e634f86cce983c07b0754e3d7a`.
- Universe manifest SHA-256: `96c85a70119ca790ef46ba6ccbee1f75e2b37a400eed48b900194650ef35c68e`.
- Outer-fold manifest SHA-256: `da5d95b87985e5fb6c32c880d15627aa17eccd3d262c61f723d1044cef34ff87`.
- JKU template: upstream release `v2.2`, with the vendored license files.
- W&B: disabled and verified locally; no public W&B or evidence-release URL is claimed.

The release is a retrospective nested cross-validation estimate, not a pristine locked-test
result. Post-run maintenance fixes are documented in Appendix B and
`docs/experiment_provenance.md`; they do not change the frozen evidence values.

## Submission blockers

The package must not be submitted until the author has reviewed the full text and figures, the
current PDF and repository gates pass, official metadata is confirmed, and institutional
originality/AI-use declarations and approvals are complete.
