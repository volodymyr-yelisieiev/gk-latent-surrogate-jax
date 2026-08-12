"""Audit the rendered thesis against objective submission constraints."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _run(*args: str) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout


def _pdf_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _run("pdfinfo", str(path)).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _page_with(pages: list[str], pattern: str) -> int:
    expression = re.compile(pattern, re.MULTILINE)
    for index, page in enumerate(pages, start=1):
        if expression.search(page):
            return index
    raise ValueError(f"rendered PDF is missing required page marker: {pattern}")


def _source_environment_count(root: Path, environment: str) -> int:
    marker = rf"\begin{{{environment}}}"
    return sum(path.read_text(encoding="utf-8").count(marker) for path in root.glob("[0-9][0-9]-*.tex"))


def audit(pdf: Path, log: Path) -> dict[str, object]:
    info = _pdf_info(pdf)
    rendered_text = _run("pdftotext", "-layout", str(pdf), "-")
    pages = rendered_text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()

    introduction_page = _page_with(pages, r"^\s*Chapter 1\s*$")
    bibliography_page = _page_with(pages, r"^\s*Bibliography\s*$")
    appendix_page = _page_with(pages, r"^\s*Appendix A\s*$")
    main_matter_pages = bibliography_page - introduction_page
    source_root = pdf.parents[1]
    figure_count = _source_environment_count(source_root, "figure")
    table_count = _source_environment_count(source_root, "table")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    lowercase_text = rendered_text.lower()

    checks = {
        "a4": info.get("Page size", "").startswith("595.28 x 841.89 pts"),
        "not_encrypted": info.get("Encrypted") == "no",
        "main_matter_at_most_30_pages": main_matter_pages <= 30,
        "at_least_five_figures_or_tables": figure_count + table_count >= 5,
        "no_overfull_boxes": "Overfull \\hbox" not in log_text,
        "no_undefined_references": "undefined references" not in log_text.lower(),
        "no_unresolved_question_marks": "??" not in rendered_text,
        "no_prospective_test_misstatement": not any(
            phrase in lowercase_text
            for phrase in ("opened once", "untouched test", "the locked test", "on a locked test")
        ),
    }
    report: dict[str, object] = {
        "pdf": str(pdf),
        "total_pages": int(info["Pages"]),
        "main_matter_first_physical_page": introduction_page,
        "bibliography_first_physical_page": bibliography_page,
        "appendix_first_physical_page": appendix_page,
        "main_matter_pages": main_matter_pages,
        "figure_environments": figure_count,
        "table_environments": table_count,
        "overfull_box_warnings": log_text.count("Overfull \\hbox"),
        "underfull_box_warnings": log_text.count("Underfull \\hbox"),
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.pdf, args.log)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
