# Bachelor Thesis Package

This directory contains a draft bachelor thesis built from the JKU report/thesis LaTeX
template v2.2 and the retained evidence in this repository.

Build it with:

```bash
cd thesis
make pdf
```

The PDF is written to `thesis/build/main-thesis.pdf`.

The title page records matriculation number `12340334`, supervisor `Gianluca Galletti`, and
degree programme `Artificial Intelligence` (study code `033 536`). Confirm the accepted thesis
title, submission date, and final semester/version segment before formal upload. The current
evidence and prose must be
reviewed by the author, especially because the IML rules make the author responsible for
the accuracy, originality, citation coverage, and editing of any LLM-assisted text.

The seed-52 test result is a retrospective single realization, not untouched locked-test evidence.
Before submission, rebuild the PDF from the current source, run the manuscript audit, inspect every
page, and complete the external provenance and multi-seed actions listed in
`review/FINAL_CHECKLIST.md`.

The included `jkureport.sty`, fonts, and logo assets come from
`michaelroland/jku-templates-report-latex` tag `v2.2` and remain subject to the licenses
included in `JKU_TEMPLATE_LICENSE` and the font license files.
