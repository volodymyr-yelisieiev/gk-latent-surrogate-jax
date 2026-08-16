# Final Thesis Checklist

Status: technically complete evidence package; formal submission and originality checks remain
author-owned actions. No external communication or submission has been performed.

## Present in the source

- [x] JKU report template v2.2 and its license files.
- [x] English and German abstracts, research question, scope, contributions, related work, method,
      implementation, experiments, results, discussion, limitations, and conclusion.
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

## Technical checks to rerun from the final worktree

- [x] `make check` passes with the final source and coverage gate (332 tests, 95.01% line coverage).
- [x] `make smoke-all` passes on the CPU fallback.
- [x] `uv build` succeeds for the source distribution and wheel.
- [x] `make -C thesis audit` succeeds and the PDF is rebuilt from current sources.
- [x] `python3 thesis/scripts/audit_pdf.py --pdf thesis/build/main-thesis.pdf --log thesis/build/main-thesis.log --output thesis/build/manuscript-audit.json` passes.
- [x] Inspect every PDF page and all figures at readable scale (contact sheet plus readable-scale
      page/figure spot checks).
- [x] Confirm no unresolved references, overfull boxes, stale numbers, private paths, or generated
      artifacts are tracked.

## Required human confirmations

- [x] Matriculation number `12340334`, supervisor `Gianluca Galletti`, and supplied programme/study
      code are recorded.
- [ ] Confirm official title-page wording, supervisor academic title, submission date, semester,
      and filename.
- [ ] Run the institutional originality check and add any required declaration or AI-use statement.
- [ ] Read and accept every page, citation, number, name, and link.
- [ ] Obtain supervisor feedback and incorporate approved corrections.

## External submission sequence

- [ ] Build the final PDF after author review.
- [ ] Upload through the official IML hand-in link.
- [ ] Email the submitted report to the supervisor and complete the grade-request procedure.

Do not submit while any author-owned confirmation remains unresolved.
