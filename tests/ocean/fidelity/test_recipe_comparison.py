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


def test_oracle_dycores_are_catalog_recipes():
    """The oracle dycores are now ordinary catalog recipes (single source) — they
    appear in list_recipes() and as columns, no factory special-casing."""
    assert "oceananigans_v1" in list_recipes("latlon")
    assert "mitgcm_v1" in list_recipes("latlon")
    rendered = _gen().render()
    assert "`oceananigans_v1`" in rendered and "`mitgcm_v1`" in rendered


def test_factory_defaults_match_catalog_single_source():
    """The factory's effective dycore values equal the catalog recipe it sources
    from — proves there is ONE source (no drift between factory + catalog)."""
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.ocean.recipes import get_recipe
    from legoesm.ocean.fidelity.oceananigans_recipe import oceananigans_canonical_ocean_config
    from legoesm.ocean.fidelity.mitgcm_recipe import mitgcm_canonical_ocean_config
    eos = LinearEOSConfig(alpha_T=2.0e-4, beta_S=0.0)
    for name, cfg in (("oceananigans_v1", oceananigans_canonical_ocean_config(eos_linear=eos)),
                      ("mitgcm_v1", mitgcm_canonical_ocean_config(eos_linear=eos))):
        for field, val in get_recipe(name, "latlon").items():
            assert cfg.flat_get(field) == val, (
                f"{name}: factory {field}={cfg.flat_get(field)!r} != catalog {val!r} (drift!)")


def test_committed_markdown_is_fresh():
    assert _OUT.exists(), "render the comparison: build_recipe_comparison.py"
    assert _OUT.read_text() == _gen().render(), (
        "docs/ocean/fidelity/recipe_comparison.md is STALE — "
        "run build_recipe_comparison.py and commit.")
