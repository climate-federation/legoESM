"""Recipe x setup grid for the ocean test matrix (#490).

The matrix runs ~21 "experiments", but they are really a handful of **recipes**
(dycore identities) paired with many **setups** (geometry / forcing / IC). An
empirical clustering of every experiment's assembled dycore showed only ~4
distinct recipes — most experiments silently share one default dycore.

This module makes that structure explicit: each experiment (a SETUP) is tagged
with the named catalog recipe it actually runs, so the matrix can be viewed and
reported as a ``recipe x setup`` grid instead of 21 flat rows. The tags are
verified against each experiment's real assembled config by
``tests/ocean/unit/test_recipe_map.py`` (so a tag cannot silently drift).

Filling in MORE (recipe, setup) cells — e.g. running a gyre setup on the WENO5
recipe — is the "coverage later" work; this grid is the scaffold for it.
"""

from __future__ import annotations

# experiment (setup) name -> the catalog recipe its dycore matches today.
# Derived from the empirical dycore-fingerprint clustering (see module docstring);
# kept honest by the verification test.
EXPERIMENT_RECIPES: dict[str, str] = {
    # --- the corrected / designed recipes (deliberately chosen dycores) ---
    "eady_uniform": "eady_weno5_v1",
    "dino": "nemo_dino_v1",
    # BENCH (Irrmann et al. 2022) deliberately runs the production-like
    # NEMO-style dycore — the benchmark must measure the production step.
    "bench": "legoesm_nemo_like_v1",
    # --- legoESM default dycore, linear EOS (runner injects eos_linear) ---
    "global_overturning": "legoesm_linear_v1",
    "eady_instability": "legoesm_linear_v1",
    "held_larichev": "legoesm_linear_v1",
    # --- the de-facto default dycore (eos=wright); NOT deliberately chosen ---
    "acc_channel": "default_wright_v1",
    "baroclinic": "default_wright_v1",
    "baroclinic_gyre": "default_wright_v1",
    "barotropic_wave": "default_wright_v1",
    "geostrophic_adjustment": "default_wright_v1",
    "global_barotropic_wind": "default_wright_v1",
    "inertia_gravity_wave": "default_wright_v1",
    "isomip_plus": "default_wright_v1",
    "lock_exchange": "default_wright_v1",
    "munk_gyre": "default_wright_v1",
    "neverworld2_lite": "default_wright_v1",
    "overflow": "default_wright_v1",
    "phillips_two_layer": "default_wright_v1",
    "regional_gyre": "default_wright_v1",
    "rest_state": "default_wright_v1",
    "stommel_gyre_tracer": "default_wright_v1",
}


def recipe_for(experiment: str) -> str:
    """Return the catalog recipe name an experiment (setup) runs."""
    if experiment not in EXPERIMENT_RECIPES:
        raise ValueError(
            f"experiment {experiment!r} has no recipe tag; add it to "
            "EXPERIMENT_RECIPES (legoesm.ocean.experiments.recipe_map)")
    return EXPERIMENT_RECIPES[experiment]


def setups_for_recipe(recipe: str) -> list[str]:
    """Return the sorted experiments (setups) that run a given recipe."""
    return sorted(e for e, r in EXPERIMENT_RECIPES.items() if r == recipe)


def recipe_setup_grid() -> dict[str, list[str]]:
    """Return the recipe -> [setups] grid (the matrix, viewed by recipe)."""
    grid: dict[str, list[str]] = {}
    for exp, rec in EXPERIMENT_RECIPES.items():
        grid.setdefault(rec, []).append(exp)
    return {rec: sorted(exps) for rec, exps in sorted(grid.items())}
