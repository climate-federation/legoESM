"""Render a recipe wiring-comparison table: the dycore-identity config fields of
every named recipe side by side, so the numerical differences between recipes are
obvious at a glance.

100% DERIVED from the recipe catalog (`legoesm.ocean.recipes.get_recipe`) — there is
NO hand-authored data here; the table is computed from the actual recipe definitions,
so it can never drift. Rows where recipes differ are flagged (✏) and sorted first —
those are exactly "what is numerically different between these recipes".

Tier 0: committed at docs/ocean/fidelity/recipe_comparison.md, kept fresh by
tests/ocean/fidelity/test_recipe_comparison.py (``--check``).

Usage::

    .venv/bin/python scripts/validate/ocean_fidelity/build_recipe_comparison.py            # write
    .venv/bin/python scripts/validate/ocean_fidelity/build_recipe_comparison.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legoesm.ocean.recipes import get_recipe, list_recipes

_DEFAULT_OUT = Path("docs/ocean/fidelity/recipe_comparison.md")


def _table(kind: str) -> list[str]:
    recipes = list_recipes(kind)
    cfgs = {name: get_recipe(name, kind) for name in recipes}
    fields = sorted({k for cfg in cfgs.values() for k in cfg})

    def cell(name: str, field: str) -> str:
        return str(cfgs[name].get(field, "—"))

    # split fields into "differs across recipes" vs "common" — differing first.
    def differs(field: str) -> bool:
        vals = {cell(n, field) for n in recipes}
        return len(vals) > 1
    diff_fields = [f for f in fields if differs(f)]
    same_fields = [f for f in fields if not differs(f)]

    L = [f"## {kind} recipes",
         "",
         f"{len(recipes)} recipes × {len(fields)} dycore-identity fields. "
         f"**{len(diff_fields)} fields differ** (✏, listed first) — those are what "
         "distinguishes these recipes; the rest are shared.",
         "",
         "| field | " + " | ".join(f"`{n}`" for n in recipes) + " |",
         "|---|" + "|".join("---" for _ in recipes) + "|"]
    for f in diff_fields:
        L.append(f"| ✏ **{f}** | " + " | ".join(f"`{cell(n, f)}`" for n in recipes) + " |")
    for f in same_fields:
        L.append(f"| {f} _(shared)_ | " + " | ".join(f"`{cell(n, f)}`" for n in recipes) + " |")
    L.append("")
    return L


def render() -> str:
    L = [
        "# Recipe wiring comparison",
        "",
        "**GENERATED — do not edit by hand.** Source: the recipe catalog "
        "`legoesm.ocean.recipes` (`get_recipe`); regenerate with "
        "`scripts/validate/ocean_fidelity/build_recipe_comparison.py`. Freshness "
        "enforced by `tests/ocean/fidelity/test_recipe_comparison.py`.",
        "",
        "Each named recipe is a bundle of dycore-identity numerics choices (the "
        "Veros / NEMO / Oceananigans / MITgcm / legoESM-default dycores). This table "
        "puts them side by side: the **✏ rows differ** between recipes (what makes "
        "each numerically distinct); the _(shared)_ rows are common to all. Compare "
        "two columns to see exactly what changes in the numerics.",
        "",
        "`—` = the recipe does not set this field (uses the model default). A field "
        "where some recipes set a value and others show `—` still **differs** (one "
        "pins it, another inherits the default).",
        "",
    ]
    L += _table("latlon")
    L += _table("mpas")
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
            print(f"STALE: {args.output} — run build_recipe_comparison.py and commit.",
                  file=sys.stderr)
            return 1
        print(f"OK: {args.output} is up to date.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
