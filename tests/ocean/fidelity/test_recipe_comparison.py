"""Freshness + completeness ratchet for the generated recipe wiring-comparison table.

The table is 100% derived from the recipe catalog, so the only things to enforce are:
every named recipe appears as a column, and the committed render is fresh.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from legoesm.ocean.recipes import list_recipes

_REPO = Path(__file__).resolve().parents[3]
_GEN = _REPO / "scripts" / "validate" / "ocean_fidelity" / "build_recipe_comparison.py"
_OUT = _REPO / "docs" / "ocean" / "fidelity" / "recipe_comparison.md"


def _gen():
    spec = importlib.util.spec_from_file_location("_build_recipe_cmp", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_recipe_is_a_column():
    rendered = _gen().render()
    for kind in ("latlon", "mpas"):
        for name in list_recipes(kind):
            assert f"`{name}`" in rendered, f"recipe {name!r} missing from comparison table"


def test_committed_markdown_is_fresh():
    assert _OUT.exists(), "render the comparison: build_recipe_comparison.py"
    assert _OUT.read_text() == _gen().render(), (
        "docs/ocean/fidelity/recipe_comparison.md is STALE — "
        "run build_recipe_comparison.py and commit.")
