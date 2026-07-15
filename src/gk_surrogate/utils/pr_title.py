from __future__ import annotations

import re

MAX_TITLE_LENGTH = 72
TITLE_PATTERN = re.compile(
    r"^(feat|fix|refactor|perf|test|docs|ci|build|chore)"
    r"(?:\([a-z0-9][a-z0-9-]*\))?!?: [^\s].*$"
)
DISALLOWED_PATTERNS = (
    re.compile(r"\b(?:codex|agent|supervisor|meeting|wip|tmp)\b", re.IGNORECASE),
    re.compile(r"\bphase[\s_-]*\d+\b", re.IGNORECASE),
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
)


def title_errors(title: str) -> tuple[str, ...]:
    errors: list[str] = []
    if len(title) > MAX_TITLE_LENGTH:
        errors.append(f"title exceeds {MAX_TITLE_LENGTH} characters")
    if not TITLE_PATTERN.fullmatch(title):
        errors.append("title must follow 'type(scope): imperative summary'")
    if title.endswith("."):
        errors.append("title must not end with a period")
    if any(pattern.search(title) for pattern in DISALLOWED_PATTERNS):
        errors.append("title contains process-oriented or informal metadata")
    return tuple(errors)
