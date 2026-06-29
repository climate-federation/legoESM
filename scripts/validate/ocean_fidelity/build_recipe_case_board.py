"""Render the unified ocean case board (legoesm.ocean.fidelity.recipe_case_board)
to Markdown: a detailed flat table (one row per case×recipe) + a pivot matrix
(case rows × recipe columns — the experiments-×-recipes view).

Source of truth = the ``CASES`` registry in the package module; this only renders.
Tier 0: the render is committed at docs/ocean/fidelity/recipe_case_board.md and kept
fresh by tests/ocean/fidelity/test_recipe_case_board.py (which calls ``--check``).

Usage::

    .venv/bin/python scripts/validate/ocean_fidelity/build_recipe_case_board.py            # write docs/...
    .venv/bin/python scripts/validate/ocean_fidelity/build_recipe_case_board.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legoesm.ocean.fidelity.recipe_case_board import (
    CASES, all_recipes, VERIFIED, WORKS, PARTIAL, BLOCKED, TODO, NA)

_BADGE = {VERIFIED: "✅ verified", WORKS: "🟩 works", PARTIAL: "🟡 partial",
          BLOCKED: "⛔ blocked", TODO: "⬜ todo", NA: "— n/a"}
_GLYPH = {VERIFIED: "✅", WORKS: "🟩", PARTIAL: "🟡", BLOCKED: "⛔", TODO: "⬜", NA: "—"}
# Sort cases by their "best" result status, then name.
_ORDER = {VERIFIED: 0, BLOCKED: 1, PARTIAL: 2, WORKS: 3, TODO: 4, NA: 5}

_DEFAULT_OUT = Path("docs/ocean/fidelity/recipe_case_board.md")


def _case_rank(c: dict) -> int:
    return min((_ORDER.get(r["status"], 9) for r in c["results"]), default=9)


def render() -> str:
    cases = sorted(CASES, key=lambda c: (_case_rank(c), c["case"]))
    n_cases = len(cases)
    n_results = sum(len(c["results"]) for c in cases)
    by: dict[str, int] = {s: 0 for s in _BADGE}
    for c in cases:
        for r in c["results"]:
            by[r["status"]] = by.get(r["status"], 0) + 1

    L = [
        "# Ocean case board",
        "",
        "**GENERATED — do not edit by hand.** Source: "
        "`packages/ocean/legoesm/ocean/fidelity/recipe_case_board.py`; "
        "regenerate with `scripts/validate/ocean_fidelity/build_recipe_case_board.py`. "
        "Completeness + freshness enforced by "
        "`tests/ocean/fidelity/test_recipe_case_board.py`.",
        "",
        "One unified inventory of every ocean case (idealized + oracle-comparison, no "
        "distinction): what it tests, whether a true oracle exists in another model, and "
        "— per recipe — whether legoESM reproduces it. One experiment run under many "
        "recipes = many result rows (below) / many filled cells (matrix), NOT many cases. "
        "Holes (`⬜ todo`) are the roadmap; `⛔ blocked` cites the issue. The **tests** "
        "column is a physics summary so duplicate/overlapping cases are visible (use it "
        "to retire repeats).",
        "",
        f"**{n_cases} cases, {n_results} case×recipe results** — "
        + ", ".join(f"{_BADGE[s]}: {by[s]}"
                    for s in (VERIFIED, BLOCKED, PARTIAL, WORKS, TODO, NA) if by[s]),
        "",
        "## Detail (one row per case × recipe)",
        "",
        "| case | tests | grids | oracle | recipe | status | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        oracle = c["oracle"] or "_(idealized — none)_"
        grids = ", ".join(c["grids"]) if c["grids"] else "—"
        for i, r in enumerate(c["results"]):
            # repeat case metadata only on the first result row (blank thereafter)
            cse = f"`{c['case']}`" if i == 0 else ""
            tst = c["tests"] if i == 0 else ""
            grd = grids if i == 0 else ""
            orc = oracle if i == 0 else ""
            L.append(f"| {cse} | {tst} | {grd} | {orc} | `{r['recipe']}` | "
                     f"{_BADGE[r['status']]} | {r['note']} |")

    # --- pivot matrix: case × recipe ---
    recipes = all_recipes()
    L += [
        "",
        "## Matrix (case × recipe)",
        "",
        "Legend: " + " · ".join(f"{_GLYPH[s]} {s}" for s in
                                 (VERIFIED, WORKS, PARTIAL, BLOCKED, TODO)) + " · blank = not run",
        "",
        "| case | " + " | ".join(recipes) + " |",
        "|---|" + "|".join("---" for _ in recipes) + "|",
    ]
    for c in cases:
        status_by_recipe = {r["recipe"]: r["status"] for r in c["results"]}
        cells = [_GLYPH[status_by_recipe[rec]] if rec in status_by_recipe else ""
                 for rec in recipes]
        L.append(f"| `{c['case']}` | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


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
                  "build_recipe_case_board.py and commit.", file=sys.stderr)
            return 1
        print(f"OK: {args.output} is up to date.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"wrote {args.output} ({len(CASES)} cases, "
          f"{sum(len(c['results']) for c in CASES)} results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
