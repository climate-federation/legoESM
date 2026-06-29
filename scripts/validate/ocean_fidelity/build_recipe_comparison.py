"""Render a recipe wiring-comparison table: the dycore-identity config fields of
every recipe side by side, so the numerical differences between recipes are obvious.

Latlon table: EFFECTIVE dycore values (read via ``flat_get`` from each config built
by ``from_flat(**get_recipe(name))``) for every ``list_recipes('latlon')`` catalog
recipe — which now INCLUDES the oracle dycores ``oceananigans_v1`` / ``mitgcm_v1``
(their ``*_canonical_ocean_config`` factories source their defaults from these
catalog entries, so they are ordinary recipes here, no special-casing). MPAS table:
the catalog MPAS recipes' declared scheme bundles.

DERIVED from the recipe catalog (``get_recipe``) — no hand-authored data, so it
can't drift. Rows where recipes differ are flagged (✏) and sorted first.

Tier 0: committed at docs/ocean/fidelity/recipe_comparison.md, kept fresh by
tests/ocean/fidelity/test_recipe_comparison.py (``--check``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legoesm.ocean.recipes import get_recipe, list_recipes
from legoesm.ocean.state import LatLonCGridOceanConfig

_DEFAULT_OUT = Path("docs/ocean/fidelity/recipe_comparison.md")

# The dycore-identity fields (the numerics fingerprint). Read via flat_get from the
# effective latlon config; these are flat names #501-from_flat routes correctly.
_DYCORE_FIELDS = (
    "eos", "momentum_advection", "tracer_advection", "coriolis_scheme",
    "barotropic_solver", "barotropic_time_filter", "pgf_scheme",
    "ke_gradient_scheme", "lateral_viscosity_operator", "outer_integrator",
    "tracer_time_integrator", "vertical_momentum_scheme", "ab2_scope",
    "momentum_flux_scheme",
)

def _flat(cfg, field: str) -> str:
    try:
        v = cfg.flat_get(field)
    except (KeyError, AttributeError):
        return "—"
    return "—" if v is None else str(v)


def _latlon_table() -> list[str]:
    # effective config per recipe: every latlon catalog recipe (now incl. the oracle
    # dycores oceananigans_v1 / mitgcm_v1) built via from_flat → read via flat_get.
    cfgs: dict = {f"`{n}`": LatLonCGridOceanConfig.from_flat(**get_recipe(n, "latlon"))
                  for n in list_recipes("latlon")}
    cols = list(cfgs)
    fields = _DYCORE_FIELDS

    def differs(f: str) -> bool:
        return len({_flat(cfgs[c], f) for c in cols}) > 1
    diff = [f for f in fields if differs(f)]
    same = [f for f in fields if not differs(f)]

    L = ["## latlon recipes (effective dycore values)",
         "",
         f"{len(cols)} recipes × {len(fields)} dycore-identity fields. "
         f"**{len(diff)} fields differ** (✏, first). Values are EFFECTIVE (defaults "
         "resolved), so two columns compared = the real numerical difference. The "
         "oracle dycores `oceananigans_v1` / `mitgcm_v1` are now ordinary catalog "
         "recipes (their factories source defaults from them).",
         "",
         "| field | " + " | ".join(cols) + " |",
         "|---|" + "|".join("---" for _ in cols) + "|"]
    for f in diff:
        L.append(f"| ✏ **{f}** | " + " | ".join(f"`{_flat(cfgs[c], f)}`" for c in cols) + " |")
    for f in same:
        L.append(f"| {f} _(shared)_ | " + " | ".join(f"`{_flat(cfgs[c], f)}`" for c in cols) + " |")
    L.append("")
    return L


def _mpas_table() -> list[str]:
    recipes = list_recipes("mpas")
    cfgs = {n: get_recipe(n, "mpas") for n in recipes}
    fields = sorted({k for c in cfgs.values() for k in c})

    def cell(n, f):
        return str(cfgs[n].get(f, "—"))

    def differs(f):
        return len({cell(n, f) for n in recipes}) > 1
    diff = [f for f in fields if differs(f)]
    same = [f for f in fields if not differs(f)]

    L = ["## MPAS recipes (declared scheme bundle)",
         "",
         f"{len(recipes)} recipes × {len(fields)} fields. **{len(diff)} differ** (✏, "
         "first). Declared overrides (`—` = inherits the MPAS model default); no "
         "oracle-recipe factories exist for MPAS.",
         "",
         "| field | " + " | ".join(f"`{n}`" for n in recipes) + " |",
         "|---|" + "|".join("---" for _ in recipes) + "|"]
    for f in diff:
        L.append(f"| ✏ **{f}** | " + " | ".join(f"`{cell(n, f)}`" for n in recipes) + " |")
    for f in same:
        L.append(f"| {f} _(shared)_ | " + " | ".join(f"`{cell(n, f)}`" for n in recipes) + " |")
    L.append("")
    return L


def render() -> str:
    L = [
        "# Recipe wiring comparison",
        "",
        "**GENERATED — do not edit by hand.** Source: the recipe catalog "
        "`legoesm.ocean.recipes` + the oracle-recipe factories "
        "(`oceananigans_recipe`, `mitgcm_recipe`); regenerate with "
        "`scripts/validate/ocean_fidelity/build_recipe_comparison.py`. Freshness "
        "enforced by `tests/ocean/fidelity/test_recipe_comparison.py`.",
        "",
        "Each recipe is a bundle of dycore-identity numerics choices. This table puts "
        "them side by side: the **✏ rows differ** between recipes (what makes each "
        "numerically distinct); _(shared)_ rows are common. Compare two columns to see "
        "exactly what changes in the numerics. The oracle dycores `oceananigans_v1` / "
        "`mitgcm_v1` are ordinary catalog recipes here — their "
        "`*_canonical_ocean_config` factories now source their defaults from the "
        "catalog (single source).",
        "",
    ]
    L += _latlon_table()
    L += _mpas_table()
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
