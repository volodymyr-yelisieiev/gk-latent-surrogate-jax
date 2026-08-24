# Final Thesis Checklist

Status: manuscript engineering is in progress. AI-assisted English/German abstracts and the
conclusion still require author review and acceptance. The final feedback PDF will be shared by the
author through Mattermost; no automated external communication or formal submission is in scope.

## Present in the source

- [x] JKU report template v2.2 and its license files.
- [ ] Replace the existing English abstract, German abstract, and conclusion with reviewed and
      author-accepted versions; all other manuscript sections are present.
- [x] Nine-question story summary with no answer exceeding three sentences.
- [x] Reproducibility appendix with frozen protocol, hashes, commands, stage counts, result summary,
      W&B status, and data-availability boundaries.
- [x] Retrospective five-fold nested group cross-validation with matched seeds 52--56 and
      validation-only selection.
- [x] Sanitized release manifest with 230 accepted and 25 skipped stage slots.
- [x] Primary negative result reported with point estimate and hierarchical bootstrap interval.
- [x] Historical mixed-split/seed-52-only claims removed from thesis-facing evidence.
- [x] No full-field reconstruction, long-horizon, solver-speed, or matched GyroSwin claim.
- [x] No live W&B URL; all 225 accepted metric-stage status records state W&B disabled and verified.
- [x] Deterministic methodology figure with the exact five-dimensional shape trace, SimSiam and
      diagnostic objectives, and the eight-step sequence rollout.
- [x] Qualitative 1,173-point latent-space figure bound to the canonical outer-fold-0/seed-52
      cache, exact split manifest, release config hashes, and tracked provenance sidecar.

## Current source verification

- [x] `make check` passes (339 tests, 95.02% line coverage).
- [x] `make smoke-all` passes on the CPU fallback.
- [x] `uv build` succeeds for the source distribution and wheel.
- [x] `make -C thesis audit` succeeds and the PDF is rebuilt from current sources.
- [x] `python3 thesis/scripts/audit_pdf.py --pdf thesis/build/main-thesis.pdf --log thesis/build/main-thesis.log --output thesis/build/manuscript-audit.json` passes.
- [x] Inspect every PDF page and all figures at readable scale (contact sheet plus readable-scale
      page/figure spot checks).
- [x] Confirm no unresolved references, overfull boxes, stale numbers, private paths, or generated
      artifacts are tracked.
- [ ] After author-section integration, rerun the thesis audit and visual inspection before creating
      the feedback PDF.

## Required human confirmations

- [x] Matriculation number `12340334`, supervisor `Gianluca Galletti`, and supplied programme/study
      code are recorded.
- [ ] Confirm official title-page wording, supervisor academic title, submission date, semester,
      and filename.
- [ ] Run the institutional originality check and add any required declaration or AI-use statement.
- [ ] Read and accept every page, citation, number, name, and link.
- [ ] Obtain supervisor feedback and incorporate approved corrections.

## External submission sequence

- [ ] Build `Volodymyr_Yelisieiev_Bachelor_Thesis_Draft.pdf` after author-section integration and
      final visual review.
- [ ] The author shares that feedback version with the supervisor through Mattermost.
- [ ] Record supervisor feedback and incorporate approved corrections.
- [ ] Build the formal-submission PDF after author review and metadata confirmation.
- [ ] Upload through the official IML hand-in link.
- [ ] Email the submitted report to the supervisor and complete the grade-request procedure.

Do not share or submit while any author-owned confirmation remains unresolved.
