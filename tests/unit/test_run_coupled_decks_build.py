"""Every committed coupled deck in config/cmip/ BUILDS through run_coupled.

The parser-level tests (test_run_coupled_config_yaml.py) only check that deck
keys are argument names.  A deck can still fail to build (ExperimentConfig /
CoupledConfig validation at driver construction) or, for the 3-D ocean deck,
refuse at the first ocean step (decision 90: the RK3 card must state
``barotropic_slow_forcing_depth_evaluation``; the 3-D deck crashed there on
main).  This test builds each deck through ``run_coupled.main(build_only=True)``
and, for a dynamic-ocean deck, traces ONE step of the ocean card the driver
builds with ``jax.eval_shape`` on a tiny rest grid, so the step's own entry
guards run (trace only: no compile, no integration, no data files).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_decks_mod", REPO / "scripts" / "run" / "run_coupled.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

CMIP_DIR = REPO / "config" / "cmip"
# YAML files in config/cmip that are NOT run decks, with the reason.
NOT_RUN_DECKS = {
    "cmip_tuned_physics.yaml": "include base of the run decks",
    "params_tuned.yaml": "--params bundle, not a --config deck",
}
DECKS = sorted(p for p in CMIP_DIR.glob("*.yaml") if p.name not in NOT_RUN_DECKS)

# Launcher-supplied machine paths (CMIP_PATH_FLAGS); build_only reads none.
PATH_FLAGS = [
    "--ozone-file", "ozone.nc", "--solar-file", "solar.nc",
    "--ghg-file", "ghg.nc", "--aerosol-file", "aerosol.nc",
    "--woa-t-path", "woa_t.nc", "--woa-s-path", "woa_s.nc",
]


def test_deck_glob_and_exclusions_are_real():
    assert len(DECKS) >= 2, DECKS
    for name in NOT_RUN_DECKS:
        assert (CMIP_DIR / name).is_file(), f"stale exclusion {name}"


def _trace_one_ocean_step(driver):
    """eval_shape one step of the lat-lon ocean card this driver would run."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import (
        idealized_bathymetry_latlon_cgrid, rest_state_latlon_cgrid_ocean,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    cfg = driver.coupled_cfg
    ocfg = driver._build_latlon_ocean_config()
    grid = create_latlon_grid(8)
    z = create_ocean_z_star(4, H_max=cfg.ocean_H_max_m)
    H, land = idealized_bathymetry_latlon_cgrid(
        grid, H_max=cfg.ocean_H_max_m, land_lat_threshold=90.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=land, H_bathy_override=H)
    model = LatLonCGridOceanModel(grid, z, ocfg)
    out = jax.eval_shape(lambda s: model.step(s, cfg.ocean_dt_s), state)
    assert out.T.data.shape == state.T.data.shape


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_committed_coupled_deck_builds(deck, tmp_path):
    driver = mod.main(
        ["--config", str(deck), *PATH_FLAGS, "--output", str(tmp_path)],
        build_only=True)
    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    ocean = read_yaml_with_includes(deck).get("ocean")
    assert ocean is not None, f"{deck.name} does not state its ocean"
    if ocean == "dynamic":
        # Branch on the deck's intent, not the built mode, so a build that
        # silently dropped the 3-D ocean cannot skip the trace.
        assert driver.coupled_cfg.ocean_mode == "dynamic"
        _trace_one_ocean_step(driver)


def test_three_d_deck_is_a_dynamic_ocean_deck():
    """The trace above is not vacuous: at least one deck runs the 3-D ocean."""
    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    assert any(read_yaml_with_includes(d).get("ocean") == "dynamic"
               for d in DECKS)


def test_sbm_cape_threshold_reaches_the_coupled_atmosphere(tmp_path):
    """The CMIP base deck pins SBM's CAPE threshold at the oracle's 0 (tier-2
    R3) and the flag reaches the atmosphere ExperimentConfig: a CLI value
    overrides the deck, so the wiring is not the default by coincidence."""
    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    assert read_yaml_with_includes(
        CMIP_DIR / "cmip_tuned_physics.yaml")["sbm_cape_threshold"] == 0.0
    deck = DECKS[0]
    base = ["--config", str(deck), *PATH_FLAGS, "--output", str(tmp_path)]
    assert mod.main(base, build_only=True).atm_config.sbm_cape_threshold == 0.0
    over = mod.main(base + ["--sbm-cape-threshold", "50"], build_only=True)
    assert over.atm_config.sbm_cape_threshold == 50.0
