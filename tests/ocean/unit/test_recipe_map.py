"""Verify the recipe x setup grid (#490).

The matrix's ~21 experiments are tagged with the catalog recipe each runs
(``EXPERIMENT_RECIPES``). These tests keep the tags honest:

* every matrix experiment is tagged, and every tag names a real catalog recipe;
* the grid inversion round-trips;
* (the load-bearing one) each experiment's ACTUAL assembled dycore fingerprint
  equals its tagged recipe's scheme bundle — so a tag cannot silently lie.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
from legoesm.ocean.experiments.recipe_map import (
    EXPERIMENT_RECIPES,
    recipe_for,
    recipe_setup_grid,
    setups_for_recipe,
)
from legoesm.ocean.recipes import LATLON_RECIPES, MPAS_RECIPES

# Scheme fields that define a dycore recipe identity.
_SCHEME = ("eos", "momentum_advection", "tracer_advection", "pgf_scheme",
           "ke_gradient_scheme", "barotropic_solver", "coriolis_scheme",
           "outer_integrator", "tracer_time_integrator")


def test_every_matrix_experiment_is_tagged():
    untagged = set(AVAILABLE_EXPERIMENTS) - set(EXPERIMENT_RECIPES)
    assert not untagged, f"experiments missing a recipe tag: {sorted(untagged)}"


def test_no_stale_tags():
    extra = set(EXPERIMENT_RECIPES) - set(AVAILABLE_EXPERIMENTS)
    assert not extra, f"recipe tags for unknown experiments: {sorted(extra)}"


def test_every_tag_names_a_real_recipe():
    known = set(LATLON_RECIPES) | set(MPAS_RECIPES)
    for exp, rec in EXPERIMENT_RECIPES.items():
        assert rec in known, f"{exp} -> unknown recipe {rec!r}"


def test_grid_inversion_roundtrips():
    grid = recipe_setup_grid()
    for rec, setups in grid.items():
        assert setups == setups_for_recipe(rec)
        for s in setups:
            assert recipe_for(s) == rec


def test_recipe_for_unknown_raises():
    with pytest.raises(ValueError, match="no recipe tag"):
        recipe_for("not_an_experiment")


def test_a_handful_of_recipes_cover_the_matrix():
    """The headline finding: ~21 experiments collapse to a handful of recipes.

    bench deliberately joins the NEMO-like recipe (the benchmark measures
    the production step cost, Irrmann et al. 2022 Sect. 2.2.1).
    """
    used = set(EXPERIMENT_RECIPES.values())
    assert used == {"default_wright_v1", "legoesm_linear_v1",
                    "eady_weno5_v1", "nemo_dino_v1",
                    "legoesm_nemo_like_v1"}


@pytest.mark.slow
def test_tags_match_actual_assembled_dycore():
    """Load-bearing: build each experiment's real matrix config and assert its
    scheme fingerprint equals its tagged recipe's bundle. Uses the same scrape +
    factory + eos-linear paths the matrix runner uses."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "matrix"))
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.ocean.recipes import get_recipe
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase

    grid_res = {"latlon": "36x72", "latlon_channel": "48x48",
                "latlon_regional": "36x72", "mpas": "2"}
    primary = ["latlon_channel", "latlon", "latlon_regional", "mpas"]
    # runners that inject a linear EOS (eos_linear_factory in run_ocean_test_matrix)
    eos_linear_runners = {"eady_uniform", "eady_instability", "held_larichev",
                          "global_overturning"}

    def fingerprint(cfg):
        return {f: getattr(cfg, f, None) for f in _SCHEME if hasattr(cfg, f)}

    for name, v in AVAILABLE_EXPERIMENTS.items():
        cfg = v["config_class"]()
        if "create_model_config" in v:
            mc = v["create_model_config"](cfg)
        elif name == "dino":
            from legoesm.grids.latlon import create_mercator_grid
            from legoesm.ocean.experiments.dino import DINOConfig, dino_lat_lon_model_config
            mc, _ = dino_lat_lon_model_config(
                create_mercator_grid(n_lon=16, lat_max_deg=70.0,
                                     lon_west_deg=0.0, lon_east_deg=50.0),
                DINOConfig())
        else:
            gs = v.get("grid_support", {})
            gt = next((g for g in primary if gs.get(g)), None)
            if gt is None:
                continue          # cube/spectral-only — out of this grid set
            kw = {a: getattr(cfg, a) for a in (
                "A_h", "A_v", "K_h", "K_v", "K_bih", "B_h", "C_smag",
                "tracer_advection", "barotropic_diffusion_alpha",
                "barotropic_div_damp") if hasattr(cfg, a)}
            if name in eos_linear_runners:
                kw["eos"] = "linear"
                kw["eos_linear"] = LinearEOSConfig()
            tc = TestCase(case=name, grid_type=gt, resolution=grid_res[gt],
                          duration_days=1.0, quick_days=0.1, run_kwargs={})
            _, _, mc, _, _, _, _ = _create_ocean_setup(
                tc, nlev=15, H_max=4000.0, **kw)

        expected = get_recipe(recipe_for(name),
                              "mpas" if "mpas" in recipe_for(name) else "latlon")
        actual = fingerprint(mc)
        for k, want in expected.items():
            if k in actual:
                assert actual[k] == want, (
                    f"{name} tagged {recipe_for(name)!r} but {k}={actual[k]!r} "
                    f"!= {want!r}")
