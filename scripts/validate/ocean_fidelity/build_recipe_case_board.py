"""Render the unified ocean case board (legoesm.ocean.fidelity.recipe_case_board)
to a Markdown table.

Source of truth = the ``CASES`` registry in the package module; this only renders.
Tier 0: the rendered table is committed at docs/ocean/fidelity/recipe_case_board.md
and kept fresh by tests/ocean/fidelity/test_recipe_case_board.py (which calls
``--check``).

Usage::

    .venv/bin/python scripts/validate/ocean_fidelity/build_recipe_case_board.py            # write docs/...
    .venv/bin/python scripts/validate/ocean_fidelity/build_recipe_case_board.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legoesm.ocean.fidelity.recipe_case_board import CASES, VERIFIED, WORKS, PARTIAL, BLOCKED, TODO, NA

_BADGE = {VERIFIED: "✅ verified", WORKS: "🟩 works", PARTIAL: "🟡 partial",
          BLOCKED: "⛔ blocked", TODO: "⬜ todo", NA: "— n/a"}
# Sort: assessed-and-interesting first (verified, blocked, partial), then works, todo, n/a; then by name.
_ORDER = {VERIFIED: 0, BLOCKED: 1, PARTIAL: 2, WORKS: 3, TODO: 4, NA: 5}

_DEFAULT_OUT = Path("docs/ocean/fidelity/recipe_case_board.md")


def render() -> str:
    rows = sorted(CASES, key=lambda c: (_ORDER.get(c["status"], 9), c["case"]))
    n = len(rows)
    by = {s: sum(1 for c in CASES if c["status"] == s) for s in _BADGE}
    lines = [
        "# Ocean case board",
        "",
        "**GENERATED — do not edit by hand.** Source: "
        "`packages/ocean/legoesm/ocean/fidelity/recipe_case_board.py`; "
        "regenerate with `scripts/validate/ocean_fidelity/build_recipe_case_board.py`. "
        "Completeness + freshness are enforced by "
        "`tests/ocean/fidelity/test_recipe_case_board.py`.",
        "",
        "One unified inventory of every ocean case (idealized + oracle-comparison, no "
        "distinction): what it tests, whether a true oracle exists in another model, and "
        "whether legoESM reproduces it. Holes (`⬜ todo`) are the roadmap; `⛔ blocked` "
        "links the issue explaining the limitation. The **tests** column is a physics "
        "summary so duplicate/overlapping cases are visible (use it to retire repeats).",
        "",
        f"**{n} cases** — "
        + ", ".join(f"{_BADGE[s]}: {by[s]}" for s in (VERIFIED, BLOCKED, PARTIAL, WORKS, TODO, NA) if by[s]),
        "",
        "| case | tests | grids | oracle | recipe | status | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in rows:
        oracle = c["oracle"] or "_(idealized — none)_"
        grids = ", ".join(c["grids"]) if c["grids"] else "—"
        lines.append(
            f"| `{c['case']}` | {c['tests']} | {grids} | {oracle} | "
            f"`{c['recipe']}` | {_BADGE[c['status']]} | {c['note']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed file is stale (do not write)")
    args = ap.parse_args()
    content = render()
    if args.check:
        existing = args.output.read_text() if args.output.exists() else None
        if existing != content:
            print(f"STALE: {args.output} is out of date — run "
                  "`build_recipe_case_board.py` and commit.", file=sys.stderr)
            return 1
        print(f"OK: {args.output} is up to date.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"wrote {args.output} ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
