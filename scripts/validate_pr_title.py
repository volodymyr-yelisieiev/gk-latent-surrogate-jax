from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gk_surrogate.utils.pr_title import title_errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: validate_pr_title.py '<title>'", file=sys.stderr)
        return 2
    errors = title_errors(args[0])
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Pull-request title is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
