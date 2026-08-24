from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "CITATION.cff",
    "PRD.md",
    "docs/real_data_binding_checklist.md",
    "docs/verification_matrix.md",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/research_change.md",
    ".github/workflows/ci.yml",
    "scripts/validate_pr_title.py",
)

REQUIRED_TERMS = {
    "AGENTS.md": (
        "PRD.md",
        "CONTRIBUTING.md",
        "make check",
        "make smoke-all",
        "GPU-first",
        "fallback",
        "Subagent Boundaries",
        "GitHub Actions",
    ),
    "CONTRIBUTING.md": (
        "type(scope): imperative summary",
        "Scientific provenance",
        "Verify repository",
        "squash",
    ),
    "PRD.md": (
        "JAX Latent Time-Series Surrogate",
        "GPU-first server path",
        "Definition of done for the MacBook stage",
    ),
    "docs/verification_matrix.md": (
        "make agent-check",
        "make check",
        "make smoke-all",
        "uv build",
        "GitHub Actions",
    ),
    ".github/pull_request_template.md": (
        "Scientific/provenance impact",
        "make check",
        "make smoke-all",
        "GitHub Actions",
    ),
    ".github/workflows/ci.yml": (
        "workflow_dispatch",
        "concurrency",
        "permissions:",
        "ubuntu-latest",
        "JAX_PLATFORM_NAME: cpu",
        "Validate pull-request title",
        "Verify repository",
        "make check",
        "make smoke-all",
        "uv build",
    ),
}


def main() -> int:
    errors: list[str] = []

    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"missing required agent file: {relative_path}")

    for relative_path, terms in REQUIRED_TERMS.items():
        path = ROOT / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for term in terms:
            if term not in text:
                errors.append(f"{relative_path} does not mention required term: {term}")

    prd_path = ROOT / "PRD.md"
    if prd_path.is_file():
        prd_text = prd_path.read_text(encoding="utf-8")
        local_home_marker = "/" + "Users" + "/"
        if len(prd_text.splitlines()) < 500:
            errors.append("PRD.md is too short to be the checked-in PRD")
        if local_home_marker in prd_text or "Latent Surrogate JAX.md" in prd_text:
            errors.append("PRD.md must not point to a local absolute-path source document")

    if errors:
        for error in errors:
            print(error)
        return 1

    print("Agent setup checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
